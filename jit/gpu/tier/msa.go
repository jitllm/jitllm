package tier

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
)

// MiniMax Sparse Attention on the device (nn.LayerPlan.MSA;
// engine/model/msa.go has the graph).
//
// A selecting block's cache row carries the indexer's key as one more kv
// head (nn.LayerPlan.KVRow), so it pages, migrates and is stored with K and V.
// The block runs in a scratch set of its own (geoMSA, gemma4.go): every K/V
// writer bakes the row. Each call:
//
//	the key: idx_k on the block's normed input, its per-head RMSNorm, then
//	  assembled after k's heads into one (NKVHead+1)-head row, which the
//	  pool's own writers rotate and store (the indexer's key turns with the
//	  attention's: same table, same width)
//	the query: idx_q on the same input, each head's RMSNorm, the rotary
//	the scores, the block ranking, the selection, the compaction and the
//	  gather (kernels/msa.go): each row's kept positions, per kv group, copied
//	  into scratch pages
//
// and the attention then runs unchanged over the gathered pages, whose
// descriptors give each row exactly its kept count. Gathering every call,
// including where every block is kept, keeps one submission shape for every
// token. A history with evicted pages (streamed) is refused by the call: the
// scores read every position through the table.

// msaScratch is one batch width's selection buffers and kernels, on the paged
// scratch (pagedScratch.msa).
type msaScratch struct {
	h, d, k, blk, tcap, nb, ppr, kvRow, attnW int
	qNorm, kNorm, qLanes, kLanes              int
	normQ, normK                              backend.Kernel
	copyQ, ropeQ, asmK, asmI                  backend.Kernel
	scores, bmax, sel, compact                backend.Kernel
	gatherK, gatherV, desc                    backend.Kernel
	// qRaw/qn/qRot the query through its norm and rotary; kRaw/kn the key
	// through its norm; kAll the assembled row, k's heads then the key.
	qRaw, qn, qRot, kRaw, kn, kAll backend.Buf
	qOff, allOff, idxOff           backend.Buf
	// score each (row, group)'s per-position scores, bmx its blocks' ranks,
	// flag the kept blocks, list their ascending order (k+1 slots, the last a
	// spare), gk/gv the gathered pages, gtab their table and gdesc their
	// descriptors.
	score, bmx, flag, list, gk, gv, gtab, gdesc backend.Buf
}

// msaWhyNot names a block-selection shape this tier declines, or "".
func msaWhyNot(p *nn.LayerPlan) string {
	switch {
	case !p.MSA():
		return ""
	case p.MLA() || p.Recurrent.Conv > 0 || p.GeomSplit || p.KVShared || p.AttnOutGate:
		return "MiniMax Sparse Attention with latent attention, a recurrence, a split geometry, " +
			"KV sharing or an out-gate"
	case p.IdxHeads != p.NKVHead || p.IdxHeadDim != p.HeadDim:
		return fmt.Sprintf("MiniMax Sparse Attention with %d indexer heads of %d on %d kv heads of %d",
			p.IdxHeads, p.IdxHeadDim, p.NKVHead, p.HeadDim)
	case p.IdxLocal < 1 || p.IdxLocal > p.IdxTopK:
		return fmt.Sprintf("MiniMax Sparse Attention forcing %d of %d blocks", p.IdxLocal, p.IdxTopK)
	case p.NoPosEnc || p.NRot == 0 || p.NoPEGlobal:
		return "MiniMax Sparse Attention with no rotary"
	}
	return ""
}

// free releases the selection's kernels and buffers.
func (m *msaScratch) free(g *devShared) {
	if m == nil {
		return
	}
	for _, k := range []backend.Kernel{m.normQ, m.normK, m.copyQ, m.ropeQ, m.asmK, m.asmI, m.scores,
		m.bmax, m.sel, m.compact, m.gatherK, m.gatherV, m.desc} {
		if k != nil {
			g.closeKern(k)
		}
	}
	for _, b := range []backend.Buf{m.qRaw, m.qn, m.qRot, m.kRaw, m.kn, m.kAll, m.qOff, m.allOff,
		m.idxOff, m.score, m.bmx, m.flag, m.list, m.gk, m.gv, m.gtab, m.gdesc} {
		if b != nil {
			b.Free()
		}
	}
}


// msaKeys is the most keys a gathered row attends: the call's, at most the
// kept blocks.
func msaKeys(p *nn.LayerPlan, keys int) int {
	if !p.MSA() {
		return keys
	}
	return min(keys, (p.IdxTopK*p.IdxBlock+scoreGrain-1)/scoreGrain*scoreGrain)
}
