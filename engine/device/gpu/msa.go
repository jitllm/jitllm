package gpu

import (
	"github.com/jitllm/jitllm/engine/nn"
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
//	  gather: each row's kept positions, per kv group, copied
//	  into scratch pages
//
// and the attention then runs unchanged over the gathered pages, whose
// descriptors give each row exactly its kept count. Gathering every call,
// including where every block is kept, keeps one submission shape for every
// token. A history with evicted pages (streamed) is refused by the call: the
// scores read every position through the table.

// msaKeys is the most keys a gathered row attends: the call's, at most the
// kept blocks.
func msaKeys(p *nn.LayerPlan, keys int) int {
	if !p.MSA() {
		return keys
	}
	return min(keys, (p.IdxTopK*p.IdxBlock+scoreGrain-1)/scoreGrain*scoreGrain)
}
