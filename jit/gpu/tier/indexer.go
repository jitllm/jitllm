package tier

import (
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
)

// DeepSeek V3.2's lightning indexer on the device (nn.LayerPlan.IdxHeads;
// engine/model/indexer.go has the graph).
//
// The indexer's key is cached in the MLA row of the paged pool, after the
// latent and the rotary key (nn.LayerPlan.KVRow), so it pages, migrates and
// is stored with them. Each call:
//
//	the key: idx_k on the block input, its LayerNorm, the NEOX rotary on its
//	  first NRot dimensions, into the row's tail at the row's position
//	the query: idx_q_b on the normed query latent, rotated the same way per head
//	the weights: idx_proj on the block input
//	the scores, the selection and the gather (kernels/indexer.go): each row's
//	  IdxTopK best positions' latent rows copied into scratch pages
//
// and the MLA attention then runs unchanged over the gathered pages, whose
// descriptors give each row exactly its kept positions. Gathering every call,
// including below IdxTopK positions where every position is kept, keeps one
// submission shape for every token. A history with evicted pages (streamed)
// is refused by the call: the scores read every position through the table.

// idxScratch is one batch width's indexer buffers and kernels, on the paged
// scratch (pagedScratch.idx).
type idxScratch struct {
	h, d, k, tcap, w, ppr  int
	lnPart, lnVar, lnApply backend.Kernel
	copyK, ropeK, writeK   backend.Kernel
	copyQ, ropeQ           backend.Kernel
	scores, sel, gather    backend.Kernel
	desc                   backend.Kernel
	// kRaw/kn/kRot the key through its norm and rotary, kPart/kVar the norm's
	// partials; qRaw/qRot the query; w the heads' weights.
	kRaw, kPart, kVar, kn, kRot backend.Buf
	qRaw, qRot, wts             backend.Buf
	kOff, qOff                  backend.Buf
	// score each row's per-position scores, selBuf its kept positions (k+1
	// slots, the last a spare), gk the gathered pages, gtab their table and
	// gdesc their descriptors.
	score, selBuf, gk, gtab, gdesc backend.Buf
	parts                          int
}

// free releases the indexer's kernels and buffers.
func (ix *idxScratch) free(g *devShared) {
	if ix == nil {
		return
	}
	for _, k := range []backend.Kernel{ix.lnPart, ix.lnVar, ix.lnApply, ix.copyK, ix.ropeK, ix.writeK,
		ix.copyQ, ix.ropeQ, ix.scores, ix.sel, ix.gather, ix.desc} {
		if k != nil {
			g.closeKern(k)
		}
	}
	for _, b := range []backend.Buf{ix.kRaw, ix.kPart, ix.kVar, ix.kn, ix.kRot, ix.qRaw, ix.qRot, ix.wts,
		ix.kOff, ix.qOff, ix.score, ix.selBuf, ix.gk, ix.gtab, ix.gdesc} {
		if b != nil {
			b.Free()
		}
	}
}

// idxKeys is the most keys a gathered row attends: the call's, at most the
// top-k.
func idxKeys(p *nn.LayerPlan, keys int) int {
	if p.IdxHeads == 0 {
		return keys
	}
	return min(keys, (min(p.IdxTopK, max(p.MaxSeq, 1))+scoreGrain-1)/scoreGrain*scoreGrain)
}
