package model

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
)

// QSA: Qwen3.8-Flash-Next's block selection (transformers'
// Qwen4ExpTextQSAIndexer, llama.cpp's build_qsa_top_k).
//
// The indexer projects the block's input to IdxHeads queries and one key of
// IdxHeadDim. The key is cached raw -- no norm, no rotary -- as one more kv
// head of the attention row (Config.KVRowAt; its value zero, the head padded
// to HeadDim), so it pages, relocates and is prefix-cached with k. A query
// at p sees its visible positions in blocks of QSARatio: every complete
// block's keys are averaged, RMSNormed and rotated at the block's first
// position, and scored sum_h relu(q_h . key_b), each query head RMSNormed and
// rotated at p; attention reads the IdxTopK/QSARatio best complete blocks and
// the incomplete tail, through a -inf mask added before the softmax. Under
// IdxTopK + QSARatio visible positions every key is read and the mask is
// nil. transformers scales the score by 1/sqrt(IdxHeadDim) and llama.cpp
// does not; a positive scale on a sum of ReLUs selects the same blocks, and
// the engine takes llama.cpp's.

// loadQSA reads block bi's indexer tensors and checks their shapes.
func loadQSA(l *layer, bi int32, cfg *Config,
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
		return fmt.Errorf("model: block %d's QSA indexer is q %dx%d, k %dx%d, norms %d/%d; "+
			"want %d heads of %d over a %d-wide input", bi, l.idxQ.k, l.idxQ.rows, l.idxK.k,
			l.idxK.rows, len(l.vector(jlm.RoleIdxQNorm)), len(l.vector(jlm.RoleIdxKNorm)), h, d, cfg.NEmbd)
	}
	return nil
}
