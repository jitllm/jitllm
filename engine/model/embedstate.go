package model

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// A decoder embedding model -- qwen3-embedding, EmbeddingGemma -- is the
// decoder graph with a different readout, so it runs through prefill and not
// through a second implementation of qwen3 or gemma3. Two things change:
//
//	the readout   the final residual is RMSNormed with the model's own output
//	              norm and pooled (mean over positions, or the last one),
//	              instead of being projected onto the vocabulary;
//	the mask      where the container says FlagNonCausal, every position sees
//	              every other one -- which is why the sequence runs as one
//	              chunk (see prefillSrc).

// embedMode is prefill's working state for one embedding run.
type embedMode struct {
	bidir bool
	pool  jlm.Pooling
	total int       // positions in the sequence
	out   []float32 // NEmbd: the pooled vector, before any dense head
	row   []float32 // NEmbd scratch for one normed row
}

// collect folds a chunk's final rows into the pooled vector. It runs once per
// prefill chunk, after the last block and before the chunk's residual is
// dropped, so a causal model longer than one chunk pools every row too.
func (e *embedMode) collect(s *State, base, n int) {
	// A mode with no vector pools nothing: a decision head (Clef's) takes
	// every row through the State's rowTap instead.
	if e.out == nil {
		return
	}
	c, m := s.c, s.m
	d := c.NEmbd
	norm := func(i int) []float32 {
		s.norm(e.row, s.bx[i*d:(i+1)*d], m.outNorm, m.outNormB)
		return e.row
	}
	switch e.pool {
	case jlm.PoolLast:
		if base+n == e.total {
			copy(e.out, norm(n-1))
		}
	case jlm.PoolCLS:
		if base == 0 {
			copy(e.out, norm(0))
		}
	default: // mean
		inv := 1 / float32(e.total)
		for i := 0; i < n; i++ {
			nn.Axpy32JIT(e.out, norm(i), inv)
		}
	}
}

// symmetricWindow is a bidirectional sliding window: the keys within window/2
// of pos on either side, clipped to the sequence. It is llama.cpp's
// LLAMA_SWA_TYPE_SYMMETRIC and transformers' |q - kv| < sliding_window for
// EmbeddingGemma, whose GGUF carries 512 where its config.json carries 257 --
// both are "256 either side".
func symmetricWindow(pos, total, window int) (w0, an int) {
	half := window / 2
	w0 = max(0, pos-half)
	end := min(total, pos+half+1)
	return w0, end - w0
}

// embedPrefill runs ids from position zero and leaves the pooled, normed (but
// not yet L2-normalised) vector in out.
func (s *State) embedPrefill(ids []int32, out []float32) ([]float32, error) {
	c := s.c
	if c.NonCausal && s.devCount() > 0 {
		// The device tier's attention is causal, so a non-causal block is
		// declined rather than run with the wrong mask.
		return nil, fmt.Errorf("model: %s attends bidirectionally, which the device tier does not "+
			"implement; embed it on the host", c.Arch)
	}
	if len(ids) > s.maxSeq {
		return nil, fmt.Errorf("model: %d tokens, and this session holds %d", len(ids), s.maxSeq)
	}
	s.Reset()
	for i := range out {
		out[i] = 0
	}
	s.emb = &embedMode{bidir: c.NonCausal, pool: c.Pooling, total: len(ids), out: out,
		row: make([]float32, c.NEmbd)}
	defer func() { s.emb = nil }()
	if _, err := s.Prefill(ids); err != nil {
		return nil, err
	}
	return out, nil
}

// denseHeads applies an embedding model's projection heads to the pooled
// vector in place: EmbeddingGemma's two sentence-transformers Dense layers,
// NEmbd -> 3072 -> NEmbd with no bias and no activation. A model without them
// returns at once.
func (m *Model) denseHeads(e *Embedder, v []float32) error {
	if m.dense1.e == nil {
		return nil
	}
	if len(e.dense) < m.dense1.rows {
		e.dense = make([]float32, m.dense1.rows)
	}
	mid := e.dense[:m.dense1.rows]
	if err := e.matvec(mid, m.dense1, v); err != nil {
		return err
	}
	return e.matvec(v, m.dense2, mid)
}

// matvec is one generated matrix-vector product with either weight layout.
func (e *Embedder) matvec(out []float32, w tensor, x []float32) error {
	e.jit.NewInput()
	if w.packed != nil {
		if e.jit.MatVecPacked(out, w.typ, w.packed, x, w.rows, w.k) {
			return nil
		}
	} else if e.jit.MatVec(out, w.typ, w.data, x, w.rows, w.k) {
		return nil
	}
	return fmt.Errorf("jitllm: no host kernel reads %s (%s, %dx%d)", w.name(), w.typ, w.rows, w.k)
}
