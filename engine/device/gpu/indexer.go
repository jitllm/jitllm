package gpu

import (
	"github.com/jitllm/jitllm/engine/nn"
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

// idxKeys is the most keys a gathered row attends: the call's, at most the
// top-k.
func idxKeys(p *nn.LayerPlan, keys int) int {
	if p.IdxHeads == 0 {
		return keys
	}
	return min(keys, (min(p.IdxTopK, max(p.MaxSeq, 1))+scoreGrain-1)/scoreGrain*scoreGrain)
}
