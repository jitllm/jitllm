package engine

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/nn"
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
	// whole runs the prompt as one chunk: a marker readout reads rows of
	// every chunk.
	whole bool
	pool  jlm.Pooling
	total int // positions in the sequence
	// out is the readout (Instance.readout): through the dense heads and at unit
	// length, or with raw (Embedder.skipDense) at unit length before them.
	out []float32
	raw bool
	// req is the request words a marker head's readout reads (Instance.req).
	req  []uint32
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
// on the CPU the composer's way (Instance.readout): the last row's on the chunk
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
	if s.readout == nil || !s.readout.Pool(s.bx[:n*s.c.NEmbd], n, e.req, e.out, e.raw) {
		if _, err := s.m.UncoveredAt(-3); err != nil {
			return fmt.Errorf("model: the readout's graph: %w", err)
		}
		if r, ok := s.readout.(nn.PrepReporter); ok && r.PrepErr() != nil {
			return fmt.Errorf("model: the readout has no composed form on the CPU: %w", r.PrepErr())
		}
		return fmt.Errorf("model: the embedding readout has no composed form on this instance: %v", s.prepErr())
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

// embedWins is each of a bidirectional embedding run's n rows' windows: its
// symmetric window on the sliding layers, every row's keys the call's on the
// others, the run's own where it states them.
func (s *Instance) embedWins(n int) []nn.KeyRun {
	if s.emb.rowWins != nil {
		return s.emb.rowWins
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
	return wins
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
