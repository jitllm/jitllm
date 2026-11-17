package engine

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
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
	// out is the readout (Instance.readout): through the dense heads, or with
	// raw (Embedder.skipDense) before them, normalised as norm asks
	// (EmbedNorm's codes; 2 is unit length).
	out  []float32
	raw  bool
	norm float32
	// req is the request words a marker head's readout reads (Instance.req).
	req []uint32
	// rowWins is each row's window where the run states them (a windowed
	// tower's, TowerBlock.Window); nil where the blocks read none.
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
	if s.readout == nil || !s.readout.Pool(s.bx[:n*s.c.NEmbd], n, e.req, e.out, e.raw, e.norm) {
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

// blocksReadRowWindows says any of the instance's text blocks reads its
// rows' windows from the call (readsRowWindows).
func (s *Instance) blocksReadRowWindows() bool {
	for li := s.lo; li < min(s.hi, s.c.NLayer); li++ {
		if readsRowWindows(s.m.Composing().BlockGraph(li)) {
			return true
		}
	}
	return false
}

// readsRowWindows says gr's attention reads each row's window from the call
// (graph.Attrs.RowWindows).
func readsRowWindows(gr *graph.Graph) bool {
	if gr == nil {
		return false
	}
	for _, o := range gr.Ops {
		if o.Kind == graph.OpAttend && o.Attr.RowWindows {
			return true
		}
	}
	return false
}

// embedPrefill runs ids from position zero and leaves the composer's readout
// in out: through the dense heads, or with raw before them, normalised as
// norm asks (EmbedNorm's codes).
func (s *Instance) embedPrefill(ids []int32, out []float32, raw bool, norm int) error {
	c := s.c
	if len(ids) > s.maxSeq {
		return fmt.Errorf("model: %d tokens, and this session holds %d", len(ids), s.maxSeq)
	}
	s.Reset()
	clear(out)
	s.emb = &embedMode{bidir: c.NonCausal, pool: c.Pooling, total: len(ids), out: out, raw: raw, norm: float32(norm)}
	defer func() { s.emb = nil }()
	_, err := s.Prefill(ids)
	return err
}
