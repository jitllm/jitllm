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
// model-correctness.md, "engine/model/indexer.go".

// idxFault is a violation of one indexer piece, set only by a gate
// (Config.idxFault) to show the fixture can see it.
type idxFault uint8

const (
	idxFaultNone idxFault = iota
	// idxFaultPairRope rotates adjacent pairs (MLA's own layout) where the
	// indexer rotates NEOX halves.
	idxFaultPairRope
	// idxFaultNoReLU sums the heads' raw scores.
	idxFaultNoReLU
)

// idxEps is the indexer key norm's epsilon: nn.LayerNorm(eps=1e-6) in the
// reference, hardcoded in llama.cpp's builder; no GGUF key carries it.
const idxEps = 1e-6

// loadIndexer reads block bi's indexer tensors and checks their shapes.
func loadIndexer(l *layer, bi int32, cfg *Config,
	get func(jlm.Role, int32, int32) (tensor, error), vec func(jlm.Role, int32) ([]float32, error)) error {
	var err error
	if l.idxQB, err = get(jlm.RoleIdxQB, bi, -1); err != nil {
		return err
	}
	if l.idxK, err = get(jlm.RoleIdxK, bi, -1); err != nil {
		return err
	}
	if l.idxProj, err = get(jlm.RoleIdxProj, bi, -1); err != nil {
		return err
	}
	if l.idxKNorm, err = vec(jlm.RoleIdxKNorm, bi); err != nil {
		return err
	}
	if l.idxKNormB, err = vec(jlm.RoleIdxKNormB, bi); err != nil {
		return err
	}
	h, d := cfg.IdxHeads, cfg.IdxHeadDim
	if l.idxQB.k != cfg.QLoraRank || l.idxQB.rows != h*d || l.idxK.k != cfg.NEmbd || l.idxK.rows != d ||
		l.idxProj.k != cfg.NEmbd || l.idxProj.rows != h || len(l.idxKNorm) != d || len(l.idxKNormB) != d {
		return fmt.Errorf("model: block %d's indexer is q %dx%d, k %dx%d, weights %dx%d, norm %d/%d; "+
			"want %d heads of %d over a %d-wide query latent", bi, l.idxQB.k, l.idxQB.rows, l.idxK.k,
			l.idxK.rows, l.idxProj.k, l.idxProj.rows, len(l.idxKNorm), len(l.idxKNormB), h, d, cfg.QLoraRank)
	}
	return nil
}

// allocIndexer sizes the decode token's indexer buffers.
func (s *State) allocIndexer() {
	c := s.m.Cfg
	if !c.Indexer() {
		return
	}
	s.idxQ = make([]float32, c.IdxHeads*c.IdxHeadDim)
	s.idxW = make([]float32, c.IdxHeads)
	s.idxTmp = make([]float32, s.attStride)
	s.idxScore = make([]float32, s.attStride)
	s.idxBias = make([]float32, s.attStride)
	s.idxOrd.Reserve(s.maxSeq)
}

// growIndexerRows sizes a chunk's: each row's query, weights and mask.
func (s *State) growIndexerRows(n int) {
	c := s.m.Cfg
	if !c.Indexer() || len(s.bidxW) >= n*c.IdxHeads {
		return
	}
	s.bidxQ = make([]float32, n*c.IdxHeads*c.IdxHeadDim)
	s.bidxW = make([]float32, n*c.IdxHeads)
	s.bidxK = make([]float32, n*c.IdxHeadDim)
	s.bidxBias = make([]float32, n*s.attStride)
	s.idxMasks = make([][]float32, n)
}
