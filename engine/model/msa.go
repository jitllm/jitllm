package model

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
)

// MiniMax Sparse Attention: MiniMax-M3's block selection (transformers'
// MiniMaxM3VLIndexer, llama.cpp's minimax-m3.cpp).
//
// Per kv group g the query at p scores every cached t <= p by q_g . k_t, ranks
// the blocks of IdxBlock positions by their best position, forces in the
// IdxLocal blocks ending at its own, and keeps the IdxTopK best; the group's
// heads attend there alone, through a -inf mask added after the scale and
// before the softmax. The key is one more kv head of the attention row
// (Config.KVRowAt), so it pages, relocates and is prefix-cached with
// k: the attention reads heads 0..NKVHead-1, the indexer head NKVHead, and
// the value row's extra head is zero. The ranking is the sampler's ordering
// over positions, keeping each block as it is first seen: the first position
// seen of a block is its maximum.
//
// The full description: docs/engineering-history/model-correctness.md,
// "engine/model/msa.go".

// msaFault is a violation of one block-selection piece, set only by a gate
// (Config.msaFault) to show the fixture can see it.
type msaFault uint8

const (
	msaFaultNone msaFault = iota
	// msaFaultNoLocal forces no block in: the query's own block competes.
	msaFaultNoLocal
	// msaFaultSharedGroups selects once, with group 0's head, for every
	// group.
	msaFaultSharedGroups
	// msaFaultFirstPosition ranks a block by its first position's score
	// instead of its best.
	msaFaultFirstPosition
	// msaFaultOneShort keeps one block fewer than the top-k.
	msaFaultOneShort
)

// loadMSA reads block bi's indexer tensors and checks their shapes.
func loadMSA(l *layer, bi int32, cfg *Config,
	getTo func(*tensor, jlm.Role, int32, int32) error, vec func(jlm.Role, int32) ([]float32, error)) error {
	var err error
	if err = getTo(&l.idxQ, jlm.RoleIdxQ, bi, -1); err != nil {
		return err
	}
	if err = getTo(&l.idxK, jlm.RoleIdxK, bi, -1); err != nil {
		return err
	}
	if _, err = vec(jlm.RoleIdxQNorm, bi); err != nil {
		return err
	}
	if _, err = vec(jlm.RoleIdxKNorm, bi); err != nil {
		return err
	}
	h, d := cfg.IdxHeads, cfg.IdxHeadDim
	if l.idxQ.k != cfg.NEmbd || l.idxQ.rows != h*d || l.idxK.k != cfg.NEmbd || l.idxK.rows != d ||
		len(l.vector(jlm.RoleIdxQNorm)) != d || len(l.vector(jlm.RoleIdxKNorm)) != d {
		return fmt.Errorf("model: block %d's indexer is q %dx%d, k %dx%d, norms %d/%d; "+
			"want %d heads of %d over a %d-wide input", bi, l.idxQ.k, l.idxQ.rows, l.idxK.k,
			l.idxK.rows, len(l.vector(jlm.RoleIdxQNorm)), len(l.vector(jlm.RoleIdxKNorm)), h, d, cfg.NEmbd)
	}
	return nil
}
