package model

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
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

// qsaScratch is a instance's QSA buffers.
type qsaScratch struct {
	q, key           []float32 // the decode row's query heads and raw key
	pooled           []float32 // every complete block's key
	score, tmp, bias []float32
	cs               []float32 // one block's rotary table
	kept             []bool
	ord              nn.SampleOrder
	dot              *nn.AttnSet
	// per row of a chunk: queries, keys and masks.
	bq, bk, bbias []float32
	masks         [][]float32
	rows          int
}

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

// allocQSA sizes the decode token's selection buffers.
func (s *Instance) allocQSA() {
	c := s.m.Cfg
	if c.QSARatio == 0 {
		return
	}
	g := &qsaScratch{}
	s.q4.qsa = g
	nb := s.maxSeq/c.QSARatio + 1
	g.q = make([]float32, c.IdxHeads*c.IdxHeadDim)
	g.key = make([]float32, c.HeadDim)
	g.pooled = make([]float32, nb*c.IdxHeadDim)
	g.score, g.tmp = make([]float32, nb), make([]float32, nb)
	g.bias = make([]float32, s.attStride)
	g.cs = make([]float32, c.NRot)
	g.kept = make([]bool, nb)
	g.ord.Reserve(nb)
	g.dot = s.attnSetF32(c.IdxHeadDim)
}

// growQSARows sizes a chunk's: each row's query, key and mask.
func (s *Instance) growQSARows(n int) {
	c := s.m.Cfg
	g := s.qsa()
	if g == nil || g.rows >= n {
		return
	}
	g.rows = n
	g.bq = make([]float32, n*c.IdxHeads*c.IdxHeadDim)
	g.bk = make([]float32, n*c.HeadDim)
	g.bbias = make([]float32, n*s.attStride)
	g.masks = make([][]float32, n)
}

func (s *Instance) qsa() *qsaScratch {
	if s.q4 == nil {
		return nil
	}
	return s.q4.qsa
}
