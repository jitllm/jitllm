//go:build jitllmbench && amd64 && linux

// This file is a measurement instrument, not a gate, so it is opt-in by build
// tag rather than by a skip in every default run:
//
//	go test -tags jitllmbench -run <Name> ./<pkg>
//
// JITLLM_* variables select its parameters.

package model

import (
	"os"
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/dev/bench"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/engine/sched"
)

// TestPrefillTileDose prices the token tile against prefill, in one process. A
// tok-wide tile makes ceil(n/tok) passes over the weights with the same
// arithmetic either way, so the rate tracking the tile means prefill is
// traffic-bound. See docs/engineering-history/cpu-kernels.md.
//
// Each iteration resets to position zero so every round attends over the same
// history. JITLLM_TILE_A and _B default to 2 and 4; setting them equal is the
// A/A control, to be run first. JITLLM_CHUNK_A/_B cap the pool chunk.
func TestPrefillTileDose(t *testing.T) {
	if testing.Short() {
		t.Skip("opens a model")
	}
	path := benchModel(t)
	m, err := Open(jlmOf(t, path))
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()

	chunk := envInt("JITLLM_TILE_PROMPT", 128)
	rounds := envInt("JITLLM_TILE_ROUNDS", 21)
	ids := make([]int32, chunk)
	for i := range ids {
		ids[i] = int32(1 + i%64)
	}

	// The chunk cap is the second axis: the token loop lives inside the pool
	// chunk, so the chunk decides which cache level serves a re-read. Both arms
	// default to no cap.
	ca, cb := envInt("JITLLM_CHUNK_A", 0), envInt("JITLLM_CHUNK_B", 0)

	// capBytes, not chunk: `chunk` is the prompt length in this function.
	mk := func(tile string, capBytes int) *Instance {
		n, err := strconv.Atoi(tile)
		if err != nil || n < 1 {
			t.Fatalf("tile %q is not a positive integer", tile)
		}
		m.jit = []nn.Option{nn.WithTileTokens(n), nn.WithTune(nn.TuneOff), nn.WithQuietTuner(true)}
		if capBytes > 0 {
			m.jit = append(m.jit, nn.WithChunkBytes(capBytes))
		}
		st := m.NewInstance(chunk + 8)
		// Warm: the tiled kernels are emitted lazily per (type, k, nrows), and
		// that cost must not land inside the measurement.
		st.Reset()
		if _, err := st.Prefill(ids); err != nil {
			t.Fatal(err)
		}
		return st
	}

	av, bv := os.Getenv("JITLLM_TILE_A"), os.Getenv("JITLLM_TILE_B")
	if av == "" {
		av, bv = "2", "4"
	}
	if bv == "" {
		bv = av
	}
	a, b := mk(av, ca), mk(bv, cb)
	defer a.Close()
	defer b.Close()

	// Assert the configuration was selected: the pin narrows silently, so two
	// arms on the same width would gate at 1.00 and mean nothing.
	wa, wb := tileWidths(a), tileWidths(b)
	t.Logf("arm %s ran tiles %v, arm %s ran tiles %v", av, wa, bv, wb)
	// A pin of 1 legitimately tiles nothing (tiledFor's floor is two tokens),
	// which reproduces a pre-VNNI host's prefill. The checks apply only to the
	// knob that varies: on a pre-VNNI host both arms tile nothing, and a
	// chunk-only dose is still valid there.
	switch {
	case av == bv && ca == cb:
		t.Logf("both arms identical -- this is the A/A self-control")
	case av != bv:
		// A tile dose: the widths must differ, and an arm pinned above one
		// token on a host that HAS a tiled kernel must have used it.
		if sameWidths(wa, wb) {
			t.Fatalf("both arms ran the same tile widths %v -- the pin did not "+
				"separate them and this would be an A/A control labelled as a dose", wa)
		}
		if len(wa) == 0 && len(wb) == 0 {
			t.Fatalf("neither arm tiled anything, so a TILE dose measures the " +
				"per-token path against itself on this host")
		}
	}

	step := func(st *Instance) func(int) {
		return func(n int) {
			for i := 0; i < n; i++ {
				st.Reset()
				if _, err := st.Prefill(ids); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
	// BytesPerIter is left zero, disabling bench's physics guard: the arms move
	// different numbers of bytes by design, so one value would be wrong.
	name := func(tile string, cap int) string {
		if cap == 0 {
			return "tile" + tile + "/nocap"
		}
		return "tile" + tile + "/" + strconv.Itoa(cap>>10) + "KiB"
	}
	na, nb := name(av, ca), name(bv, cb)
	// Assert the chunk cap was selected too: BatchedChunkRows reports what the
	// last batched matmul chose.
	ra, rb := a.jit.BatchedChunkRows(), b.jit.BatchedChunkRows()
	t.Logf("last batched chunk: %s %d rows, %s %d rows", na, ra, nb, rb)
	if ca != cb && ra == rb {
		t.Fatalf("the two caps both chose %d rows -- this is an A/A control "+
			"labelled as a chunk dose", ra)
	}
	r := bench.AB(
		bench.Case{Name: na, Threads: len(sched.DecodeCores()), Fn: step(a)},
		bench.Case{Name: nb, Threads: len(sched.DecodeCores()), Fn: step(b)},
		rounds, 1)
	t.Logf("%s", r)
	t.Logf("%s %.1f prefill tok/s   %s %.1f prefill tok/s   (%d-token prompt)",
		na, float64(chunk)/r.AMedian.Seconds(),
		nb, float64(chunk)/r.BMedian.Seconds(), chunk)
}

// tileWidths reports the tile each of layer 0's four matrices runs at, so the
// dose can be shown to have moved something. Layer 0 speaks for the block: the
// census over Llama-3.2-1B's shapes found one width at all four of them.
func tileWidths(s *Instance) map[string]int {
	l := &s.m.layers[0]
	out := map[string]int{}
	for _, w := range []struct {
		name string
		t    tensor
	}{{"q", l.wq}, {"k", l.wk}, {"v", l.wv}, {"o", l.wo}} {
		if w.t.rows == 0 {
			continue
		}
		if n := s.jit.TiledWidth(w.t.typ, w.t.k, w.t.rows); n > 0 {
			out[w.name] = n
		}
	}
	return out
}

func sameWidths(a, b map[string]int) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if b[k] != v {
			return false
		}
	}
	return true
}
