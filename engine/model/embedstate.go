package model

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
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
	// unit says out is the composer's readout (cpuPool): through the dense
	// heads and at unit length.
	unit bool
	row  []float32 // NEmbd scratch for one normed row
	wins []nn.KeyRun
}

// collect folds a chunk's final rows into the pooled vector. It runs once per
// prefill chunk, after the last block and before the chunk's residual is
// dropped, so a causal model longer than one chunk pools every row too.
func (e *embedMode) collect(s *Instance, base, n int) {
	// A mode with no vector pools nothing: a decision head (Clef's) takes
	// every row through the instance's rowTap instead.
	if e.out == nil {
		return
	}
	if e.composed(s, base, n) {
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

// composed reads the readout from a chunk's n rows from position base on
// the CPU the composer's way (cpuPool), where the instance composed it, and
// reports whether the chunk is done with: the readout served (unit, the
// vector at unit length), or a chunk that holds no row it reads. A mean
// over several chunks is declined by name (DeclinePrefill) and pooled by
// main's.
func (e *embedMode) composed(s *Instance, base, n int) bool {
	if !s.cpuHeadOn() || !s.cpu.pool.form.composed {
		return false
	}
	switch e.pool {
	case jlm.PoolLast:
		if base+n != e.total {
			return true
		}
	case jlm.PoolCLS:
		if base != 0 {
			return true
		}
	default:
		if n != e.total {
			if base == 0 {
				s.cpu.pool.declined[jit.DeclinePrefill]++
			}
			return false
		}
	}
	e.unit = s.cpuPool(s.bx[:n*s.c.NEmbd], n, e.out)
	return e.unit
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

// callTell readies the device for a chunk's call: a bidirectional embedding
// run's windows (embedTell), or the key runs of the prompt's bidirectional
// spans (bidirTell).
func (s *Instance) callTell(p0, n int) bool {
	if s.emb != nil && s.emb.bidir {
		return s.embedTell(n)
	}
	return s.bidirTell(p0, n)
}

// embedTell hands the device a bidirectional embedding run's n rows, the
// whole sequence in one call: the non-causal set sized for them
// (nn.RowsReserver), and each row's symmetric window on the sliding layers
// (nn.RowWindowDevice), every row's keys the call's on the others. false is
// a device that cannot take them; the caller does not run the call.
func (s *Instance) embedTell(n int) bool {
	if rr, ok := s.ld.(nn.RowsReserver); ok && !rr.ReserveRows(n) {
		return false
	}
	rw, ok := s.ld.(nn.RowWindowDevice)
	if !ok {
		return false
	}
	c := s.c
	if cap(s.emb.wins) < n {
		s.emb.wins = make([]nn.KeyRun, n)
	}
	wins := s.emb.wins[:n]
	for i := range wins {
		lo, an := 0, n
		if c.SWAWindow > 0 {
			lo, an = symmetricWindow(i, n, int(c.SWAWindow))
		}
		wins[i] = nn.KeyRun{Lo: lo, Hi: lo + an}
	}
	return rw.SetRowWindows(wins)
}

// embedPrefill runs ids from position zero and leaves the pooled, normed (but
// not yet L2-normalised) vector in out, or, unit, the composer's readout
// through the dense heads at unit length.
func (s *Instance) embedPrefill(ids []int32, out []float32) (_ []float32, unit bool, _ error) {
	c := s.c
	if len(ids) > s.maxSeq {
		return nil, false, fmt.Errorf("model: %d tokens, and this session holds %d", len(ids), s.maxSeq)
	}
	s.Reset()
	for i := range out {
		out[i] = 0
	}
	s.emb = &embedMode{bidir: c.NonCausal, pool: c.Pooling, total: len(ids), out: out,
		row: make([]float32, c.NEmbd)}
	defer func() { s.emb = nil }()
	if _, err := s.Prefill(ids); err != nil {
		return nil, false, err
	}
	return out, s.emb.unit, nil
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
