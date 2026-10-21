package tier

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
)

// QSA on the device: Qwen3.8-Flash-Next's block selection
// (nn.LayerPlan.QSA; engine/model/qsa.go is the graph).
//
// The cache row carries the indexer's raw key as one more kv head, as
// MiniMax-M3's does (msa.go), but unrotated: the row is copied whole and the
// rotary then turns k's heads alone. Each call:
//
//	the key: idx_k on the block's normed input, assembled after k's normed
//	  heads into one row
//	the query: idx_q on the same input, each head's RMSNorm, the rotary
//	the scores (mainkernels.QSAScores): each complete block's keys pooled,
//	  normed, rotated at the block's first position and scored against the
//	  query's heads; the tail block forced in
//	the selection, the compaction, the gather and the descriptors (MSA's,
//	  under QSA's forced tail)
//
// and the attention then runs over the gathered pages. Below the budget every
// block is kept, which is the reference's dense attention.
//
// The block-start rotary table is one table for every row: a block starts at
// a multiple of the ratio, which is its rotary position on a sequence with
// no picture. A picture moves the positions after it (M-RoPE's marks, which
// the model hands over before the call, SetRopeMarks): each mark's turn is a
// second table, built per call, that the scores compose with the first.

// qsaMarkCap is how many marks one call's rows may carry between them: a mark
// is a picture in a sequence, and a call past this many is refused naming it.
const qsaMarkCap = 64

// qsaScratch is one batch width's selection buffers and kernels, on the
// paged scratch (pagedScratch.qsa).
type qsaScratch struct {
	h, d, k, blk, nb, ppr, kvRow, attnW, groups int
	qLanes, kLanes                              int
	normQ, normK, copyQ, ropeQ, asmK, asmI      backend.Kernel
	csTab, scores, sel, compact                 backend.Kernel
	gatherK, gatherV, desc                      backend.Kernel
	qRaw, qn, qRot, kRaw, kAll                  backend.Buf
	qOff, allOff, idxOff                        backend.Buf
	cs, csPos, planes, consts                   backend.Buf
	// The marks (mainkernels.QSAScores' pMark) and their turns: a RopeTable over
	// the marks' |delta| (dpos) into dcs.
	marks, dpos, dcs                       backend.Buf
	dTab                                   backend.Kernel
	markStage                              []uint32
	dposStage                              []uint32
	score, flag, list, gk, gv, gtab, gdesc backend.Buf
}

// qsaWhyNot names a QSA shape this tier declines, or "".
func qsaWhyNot(p *nn.LayerPlan) string {
	switch {
	case !p.QSA():
		return ""
	case p.MLA() || p.GeomSplit || p.KVShared || p.MSA():
		return "QSA with latent attention, a split geometry, KV sharing or another selection"
	case p.IdxHeadDim > p.HeadDim || p.IdxHeads < 1:
		return fmt.Sprintf("QSA with %d indexer heads of %d on heads of %d", p.IdxHeads, p.IdxHeadDim, p.HeadDim)
	case p.NoPosEnc || p.NRot == 0 || p.NRot > p.IdxHeadDim:
		return "QSA with no rotary, or a rotary wider than the indexer's head"
	case p.IdxTopK < p.Q4E.QSARatio:
		return "QSA with a budget under one block"
	}
	return ""
}

// SetRopeMarks takes the rows' rotary marks for the next call
// (nn.QSADevice).
func (g *devTier) SetRopeMarks(marks []int32) {
	g.mu.Lock()
	g.qsaMarks = marks
	g.mu.Unlock()
}

// SetRopeMarks keeps the rows' marks for this session's next call, which
// hands them to every device it holds (inputsTo).
func (s *gpuSession) SetRopeMarks(marks []int32) { s.marks = marks }

// qsaStage lays the call's marks out as mainkernels.QSAScores reads them, from
// nn.QSADevice's form: per row in the call's order its count, then its
// (position, delta) pairs. A row past the caller's carries none. Callers hold
// g.mu.
func (g *devTier) qsaStage(m *qsaScratch, rows int) error {
	if m == nil {
		return nil
	}
	st, dp := m.markStage, m.dposStage
	clear(st)
	clear(dp)
	in, at, nm := g.qsaMarks, 0, 0
	for r := 0; r < rows; r++ {
		st[r] = uint32(nm)
		if at >= len(in) {
			continue
		}
		k := int(in[at])
		at++
		for j := 0; j < k; j++ {
			if nm == qsaMarkCap {
				return fmt.Errorf("the call's rows carry more than %d pictures' rotary marks", qsaMarkCap)
			}
			pos, d := in[at], in[at+1]
			at += 2
			w := uint32(pos)
			if d < 0 {
				w |= 1 << 31
				d = -d
			}
			st[rows+1+nm] = w
			dp[1+nm] = uint32(d)
			nm++
		}
	}
	st[rows] = uint32(nm)
	dp[0] = uint32(max(nm, 1))
	return nil
}

// qsaWrites writes the staging qsaStage filled.
func (m *qsaScratch) qsaWrites(w func(backend.Buf, []byte)) {
	if m == nil {
		return
	}
	w(m.marks, u32view(m.markStage))
	w(m.dpos, u32view(m.dposStage))
}

// free releases the selection's kernels and buffers.
func (m *qsaScratch) free(g *devShared) {
	if m == nil {
		return
	}
	for _, k := range []backend.Kernel{m.normQ, m.normK, m.copyQ, m.ropeQ, m.asmK, m.asmI, m.csTab, m.dTab, m.scores,
		m.sel, m.compact, m.gatherK, m.gatherV, m.desc} {
		if k != nil {
			g.closeKern(k)
		}
	}
	for _, b := range []backend.Buf{m.qRaw, m.qn, m.qRot, m.kRaw, m.kAll, m.qOff, m.allOff, m.idxOff, m.cs,
		m.csPos, m.planes, m.consts, m.marks, m.dpos, m.dcs, m.score, m.flag, m.list, m.gk, m.gv, m.gtab, m.gdesc} {
		if b != nil {
			b.Free()
		}
	}
}
