package model

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
)

// DeepSeek Sparse Attention: the lightning indexer of DeepSeek V3.2
// (transformers' DeepseekV32Indexer, llama.cpp's deepseek32.cpp).
//
// For a query at p the indexer scores every cached t <= p as
// sum_h w_h * relu(q_h . k_t), and the attention reads only the IdxTopK best
// positions -- every position at or below IdxTopK, where the block is exactly
// deepseek2's -- through a -inf mask added after the scale and before the
// softmax. The key is cached in the MLA row itself, after the latent and the
// rotary key (Config.KVDim), so it pages, relocates and is prefix-cached with
// the latent: the MLA kernels read the row's first KVLoraRank+NRot elements
// and the indexer its last IdxHeadDim. Every step is generated code.
//
// The projections, norms and scales: docs/engineering-history/
// model-correctness.md, "model/indexer.go".

// idxFault is a violation of one indexer piece, set only by a gate
// (Config.idxFault) to show the fixture can see it.
type idxFault uint8

const (
	idxFaultNone idxFault = iota
	// idxFaultPairRope rotates adjacent pairs (MLA's own layout) where the
	// indexer rotates NEOX halves.
	IdxFaultPairRope
	// idxFaultNoReLU sums the heads' raw scores.
	IdxFaultNoReLU
)

// idxEps is the indexer key norm's epsilon: nn.LayerNorm(eps=1e-6) in the
// reference, hardcoded in llama.cpp's builder; no GGUF key carries it.
const idxEps = 1e-6

// loadIndexer reads block bi's indexer tensors and checks their shapes.
func loadIndexer(l *Layer, bi int32, cfg *Config,
	getTo func(*Tensor, jlm.Role, int32, int32) error, vec func(jlm.Role, int32) ([]float32, error)) error {
	var err error
	if err = getTo(&l.IdxQB, jlm.RoleIdxQB, bi, -1); err != nil {
		return err
	}
	if err = getTo(&l.IdxK, jlm.RoleIdxK, bi, -1); err != nil {
		return err
	}
	if err = getTo(&l.IdxProj, jlm.RoleIdxProj, bi, -1); err != nil {
		return err
	}
	if _, err = vec(jlm.RoleIdxKNorm, bi); err != nil {
		return err
	}
	if _, err = vec(jlm.RoleIdxKNormB, bi); err != nil {
		return err
	}
	h, d := cfg.IdxHeads, cfg.IdxHeadDim
	if l.IdxQB.K != cfg.QLoraRank || l.IdxQB.Rows != h*d || l.IdxK.K != cfg.NEmbd || l.IdxK.Rows != d ||
		l.IdxProj.K != cfg.NEmbd || l.IdxProj.Rows != h || len(l.Vector(jlm.RoleIdxKNorm)) != d || len(l.Vector(jlm.RoleIdxKNormB)) != d {
		return fmt.Errorf("model: block %d's indexer is q %dx%d, k %dx%d, weights %dx%d, norm %d/%d; "+
			"want %d heads of %d over a %d-wide query latent", bi, l.IdxQB.K, l.IdxQB.Rows, l.IdxK.K,
			l.IdxK.Rows, l.IdxProj.K, l.IdxProj.Rows, len(l.Vector(jlm.RoleIdxKNorm)), len(l.Vector(jlm.RoleIdxKNormB)), h, d, cfg.QLoraRank)
	}
	return nil
}
