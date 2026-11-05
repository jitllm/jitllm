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
	total int // positions in the sequence
	// out is the readout (cpuPool): through the dense heads and at unit
	// length, or with raw (Embedder.skipDense) at unit length before them.
	out  []float32
	raw  bool
	wins []nn.KeyRun
	// rowWins is each row's window where the run states them (a windowed
	// tower's, TowerBlock.Window), nil where the sliding window says them.
	rowWins []nn.KeyRun
	// rows keeps every row of the last block's, unnormed, in out (a tower's
	// run: its head reads them), in place of a pooled readout.
	rows bool
	// taps keeps the rows blocks leave where a tower's taps read them
	// (jlm.TowerTap): a device's run ends at such a block (tapsAfter).
	taps []embedTap
}

// embedTap is the rows block li (the instance's) leaves, kept for a tower's
// tap.
type embedTap struct {
	li   int
	rows []float32
}

// tapsAfter says a tap keeps the rows block li leaves.
func (s *Instance) tapsAfter(li int) bool {
	if s.emb == nil {
		return false
	}
	for _, tp := range s.emb.taps {
		if tp.li == li {
			return true
		}
	}
	return false
}

// keepTaps copies the n rows block li left into each tap that keeps them.
func (s *Instance) keepTaps(li, n int) {
	if s.emb == nil {
		return
	}
	for _, tp := range s.emb.taps {
		if tp.li == li {
			copy(tp.rows[:n*s.c.NEmbd], s.bx[:n*s.c.NEmbd])
		}
	}
}

// collect reads the readout from a chunk's n final rows from position base
// on the CPU the composer's way (cpuPool): the last row's on the chunk
// holding it, the first's on the first chunk, and a mean on the one chunk a
// mean-pooled run is (Prefill). A chunk holding no row it reads is passed
// over.
func (e *embedMode) collect(s *Instance, base, n int) error {
	if e.rows {
		w := s.c.NEmbd
		copy(e.out[base*w:(base+n)*w], s.bx[:n*w])
		return nil
	}
	switch e.pool {
	case jlm.PoolLast:
		if base+n != e.total {
			return nil
		}
	case jlm.PoolCLS:
		if base != 0 {
			return nil
		}
	default:
		if n != e.total {
			return fmt.Errorf("model: a mean over %d of %d rows, and a mean-pooled run is one chunk", n, e.total)
		}
	}
	if !s.cpuPool(s.bx[:n*s.c.NEmbd], n, e.out, e.raw) {
		return fmt.Errorf("model: the embedding readout has no composed form on this instance: %v", s.cpu.formErr)
	}
	return nil
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
	if s.emb.rowWins != nil {
		return rw.SetRowWindows(s.emb.rowWins)
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
func (s *Instance) embedPrefill(ids []int32, out []float32, raw bool) error {
	c := s.c
	if len(ids) > s.maxSeq {
		return fmt.Errorf("model: %d tokens, and this session holds %d", len(ids), s.maxSeq)
	}
	s.Reset()
	clear(out)
	s.emb = &embedMode{bidir: c.NonCausal, pool: c.Pooling, total: len(ids), out: out, raw: raw}
	defer func() { s.emb = nil }()
	_, err := s.Prefill(ids)
	return err
}
