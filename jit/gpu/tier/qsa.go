package tier

import (
	"fmt"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
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
//	the scores (kernels.QSAScores): each complete block's keys pooled,
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
	// The marks (kernels.QSAScores' pMark) and their turns: a RopeTable over
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

// initQSA builds bs's selection: rows rows of P-position pages whose K row is
// kvRow wide, the indexer's raw key its last head. Callers hold g.mu.
func (g *devTier) initQSA(bs *blockScratch, rows, P int,
	comp func(*backend.Kernel, *ir.Kernel, error) error) error {
	p := &bs.p
	if want := kernels.RopeTabBlock(p.NRot / 2); len(p.Q4E.QSARope) != want {
		return fmt.Errorf("QSA's rotary planes are %d words, the layout wants %d", len(p.Q4E.QSARope), want)
	}
	tcap := max(p.MaxSeq, 1)
	blk := p.Q4E.QSARatio
	fault := p.Q4E.Fault
	m := &qsaScratch{h: p.IdxHeads, d: p.IdxHeadDim, blk: blk, nb: tcap/blk + 1, kvRow: p.KVRow(),
		attnW: p.NKVHead * p.HeadDim, groups: p.NKVHead}
	// The kept complete blocks and the tail; under the dense violation every
	// block.
	m.k = min(p.IdxTopK/blk+1, m.nb)
	if fault == nn.Q4EFaultDenseQSA {
		m.k = m.nb
	}
	m.ppr = kernels.MSAGatherPages(m.k, blk, P)
	bs.pkv.qsa = m
	m.qLanes, m.kLanes = bs.kNormL, bs.kNormL
	h, d, rot, kv := m.h, m.d, p.NRot, m.attnW
	noTail := fault == nn.Q4EFaultNoTail
	for _, b := range []struct {
		dst *backend.Kernel
		mk  func() (*ir.Kernel, error)
	}{
		{&m.normQ, func() (*ir.Kernel, error) { return kernels.HeadNormRows(h, d, p.RMSEps, rows, m.qLanes) }},
		{&m.copyQ, func() (*ir.Kernel, error) { return kernels.CopyAtRows(h*d, rows) }},
		{&m.ropeQ, func() (*ir.Kernel, error) { return kernels.RoPERows(h, d, rot, p.RopeNeox, rows) }},
		{&m.asmK, func() (*ir.Kernel, error) { return kernels.CopyAtRows(kv, rows) }},
		{&m.asmI, func() (*ir.Kernel, error) { return kernels.CopyAtRows(d, rows) }},
		{&m.csTab, func() (*ir.Kernel, error) { return kernels.RopeTable(rot/2, m.nb) }},
		{&m.dTab, func() (*ir.Kernel, error) { return kernels.RopeTable(rot/2, qsaMarkCap) }},
		{&m.scores, func() (*ir.Kernel, error) {
			return kernels.QSAScores(rows, h, d, m.nb, blk, m.groups, P, m.kvRow, kv, rot, p.RopeNeox,
				float32(p.RMSEps), fault == nn.Q4EFaultQSAFirstKey, fault == nn.Q4EFaultDenseQSA)
		}},
		{&m.sel, func() (*ir.Kernel, error) { return kernels.QSASelect(rows, m.groups, m.nb*blk, blk, m.k) }},
		{&m.compact, func() (*ir.Kernel, error) { return kernels.MSACompact(rows, m.groups, m.nb*blk, blk, m.k) }},
		{&m.gatherK, func() (*ir.Kernel, error) {
			return kernels.QSAGatherK(rows, p.NKVHead, p.HeadDim, blk, m.k, P, m.kvRow, noTail)
		}},
		{&m.gatherV, func() (*ir.Kernel, error) {
			return kernels.QSAGatherV(rows, p.NKVHead, p.HeadDim, blk, m.k, P, m.kvRow, g.KVF16, noTail)
		}},
		{&m.desc, func() (*ir.Kernel, error) { return kernels.QSADesc(rows, blk, m.k, P, noTail) }},
	} {
		kn, err := b.mk()
		if err := comp(b.dst, kn, err); err != nil {
			return fmt.Errorf("the block selection: %w", err)
		}
	}
	vw := m.ppr * P * kv
	if g.KVF16 {
		vw /= 2
	}
	var err error
	al := func(dst *backend.Buf, n int) {
		if err == nil {
			*dst, err = g.dev.Alloc(n * 4)
		}
	}
	al(&m.qRaw, rows*h*d)
	al(&m.qn, rows*h*d)
	al(&m.qRot, rows*h*d)
	al(&m.kRaw, rows*d)
	al(&m.kAll, rows*m.kvRow)
	al(&m.qOff, rows)
	al(&m.allOff, rows)
	al(&m.idxOff, rows)
	al(&m.cs, m.nb*rot)
	al(&m.csPos, m.nb+1)
	al(&m.marks, rows+1+qsaMarkCap)
	al(&m.dpos, qsaMarkCap+1)
	al(&m.dcs, qsaMarkCap*rot)
	m.markStage = make([]uint32, rows+1+qsaMarkCap)
	m.dposStage = make([]uint32, qsaMarkCap+1)
	al(&m.planes, len(p.Q4E.QSARope))
	al(&m.consts, len(kernels.RopeTabConsts()))
	al(&m.score, rows*m.groups*m.nb)
	al(&m.flag, rows*m.groups*m.nb)
	al(&m.list, rows*m.groups*(m.k+1))
	al(&m.gk, rows*m.ppr*P*kv)
	al(&m.gv, rows*vw)
	al(&m.gtab, rows*m.ppr)
	al(&m.gdesc, rows*kernels.PRowWords)
	if err != nil {
		return err
	}
	for _, x := range []struct {
		b backend.Buf
		v []float32
	}{{m.planes, p.Q4E.QSARope}, {m.consts, kernels.RopeTabConsts()}} {
		if err := x.b.Write(f32b(x.v)); err != nil {
			return err
		}
	}
	qo, ao, io, tab := make([]uint32, rows), make([]uint32, rows), make([]uint32, rows),
		make([]uint32, rows*m.ppr)
	for r := range qo {
		qo[r], ao[r], io[r] = uint32(r*h*d), uint32(r*m.kvRow), uint32(r*m.kvRow+kv)
	}
	for i := range tab {
		tab[i] = uint32(i)
	}
	// The block starts, after their count (kernels.RopeTable's positions).
	pos := make([]uint32, m.nb+1)
	pos[0] = uint32(m.nb)
	for b := 0; b < m.nb; b++ {
		pos[b+1] = uint32(b * blk)
	}
	// The key row's padding past the indexer's width is never read; it is
	// written once so no row carries another allocation's bytes.
	for _, x := range []struct {
		b backend.Buf
		v []uint32
	}{{m.qOff, qo}, {m.allOff, ao}, {m.idxOff, io}, {m.gtab, tab}, {m.csPos, pos},
		{m.kAll, make([]uint32, rows*m.kvRow)}} {
		if err := x.b.Write(u32view(x.v)); err != nil {
			return err
		}
	}
	return nil
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

// qsaStage lays the call's marks out as kernels.QSAScores reads them, from
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
func (m *qsaScratch) free() {
	if m == nil {
		return
	}
	for _, k := range []backend.Kernel{m.normQ, m.normK, m.copyQ, m.ropeQ, m.asmK, m.asmI, m.csTab, m.dTab, m.scores,
		m.sel, m.compact, m.gatherK, m.gatherV, m.desc} {
		if k != nil {
			closeKern(k)
		}
	}
	for _, b := range []backend.Buf{m.qRaw, m.qn, m.qRot, m.kRaw, m.kAll, m.qOff, m.allOff, m.idxOff, m.cs,
		m.csPos, m.planes, m.consts, m.marks, m.dpos, m.dcs, m.score, m.flag, m.list, m.gk, m.gv, m.gtab, m.gdesc} {
		if b != nil {
			b.Free()
		}
	}
}

// qsaKeyQuery runs the indexer's projections while bs.a/bs.ax hold the
// block's normed input, the query's norms and rotary, and assembles k's
// normed heads kn and the raw key into one row each (m.kAll), which the
// pool's writers copy and whose k heads they then turn.
func (g *devTier) qsaKeyQuery(lc *launcher, bs *blockScratch, l *layer, R int, kn, cs backend.Buf,
	mvrun func(mv, *resident, backend.Buf)) error {
	m := bs.pkv.qsa
	mvrun(l.mvIK, l.idxK, m.kRaw)
	mvrun(l.mvMQ, l.idxQ, m.qRaw)
	qg, qw := headNormGeom(m.qLanes, R*m.h)
	if err := lc.launch(m.normQ, qg, qw, m.qRaw, l.nIdxQ, m.qn); err != nil {
		return err
	}
	lc.la(m.copyQ, R*m.h*m.d, m.qn, m.qRot, m.qOff)
	lc.la(m.ropeQ, R*m.h*bs.p.NRot/2, m.qn, cs, m.qOff, m.qRot)
	lc.la(m.asmK, R*m.attnW, kn, m.kAll, m.allOff)
	lc.la(m.asmI, R*m.d, m.kRaw, m.kAll, m.idxOff)
	return nil
}

// qsaAttend selects each row's blocks, gathers them and runs the attention
// over the gathered pages: q the rotated queries, out the heads' results. pl
// is the layer's pool.
func (g *devTier) qsaAttend(s backend.Session, lc *launcher, bs *blockScratch, R int, pk *pagedScratch,
	pl *kvLayerPool, l *layer, q, n, out backend.Buf) error {
	m := pk.qsa
	lc.la(m.csTab, m.nb*bs.p.NRot/2, m.cs, m.planes, m.consts, m.csPos)
	lc.la(m.dTab, qsaMarkCap*bs.p.NRot/2, m.dcs, m.planes, m.consts, m.dpos)
	lc.la(m.scores, R*m.nb, m.qRot, pl.k, pl.tab, pk.pRow, m.cs, l.nIdxK, m.marks, m.dcs, m.score)
	lc.la(m.sel, R*m.groups*m.nb, m.score, pk.pRow, m.flag)
	lc.la(m.compact, R*m.groups*m.nb, m.flag, m.list)
	lc.la(m.gatherK, R*m.attnW*m.k*m.blk, pl.k, pl.tab, pk.pRow, m.list, m.gk)
	vw := m.attnW
	if g.KVF16 {
		vw /= 2
	}
	lc.la(m.gatherV, R*m.k*m.blk*vw, pl.v, pl.tab, pk.pRow, m.list, m.gv)
	lc.la(m.desc, R*kernels.PRowWords, pk.pRow, m.gdesc)
	g.QSASelects++
	return pk.pagedDecode(s, lc, q, m.gk, m.gv, n, out, m.gtab, m.gdesc, nil)
}
