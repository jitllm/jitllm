package model

import (
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestShortChunkRunsAsRaggedStep holds a prompt whose last chunk is a few rows
// (tailrows.go) to the same prompt with the chunk run batched
// (WithTailRows(false)) and to the prompt decoded a token at a time, every
// block and the head on the device: the last row's logits and a continuation
// that reads the history the tail wrote, within the device's reduction-order
// band, and the counters saying which path each arm took. Two shapes: a tail
// after a full chunk, and a prompt that is only a few rows.
func TestShortChunkRunsAsRaggedStep(t *testing.T) {
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.jlm")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	type out struct {
		last  []float32
		toks  []int32
		steps [][]float32
		tails int
	}
	// One arm at a time: a 4 GiB card holds one copy of the model.
	// decode is the third arm: the prompt a token at a time, which the
	// ragged step's matvec is (decode's, over several rows).
	run := func(on, decode, dev bool, n int, force []int32) out {
		m, err := Open(p, WithTailRows(on))
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		ids := m.Vocab.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog. ", 80), true)[:n]
		st := m.NewInstance(n + 16)
		defer st.Close()
		if dev {
			g, err := tier.OpenWith(tier.WithDeviceTune(tier.TuneOff))
			if err != nil || g == nil {
				noDevice(t, "device", err)
			}
			defer g.Close()
			st.SetDevice(g)
			if st.GPULayers() != m.Cfg.NLayer {
				t.Fatalf("placed %d of %d blocks: %q", st.GPULayers(), m.Cfg.NLayer, g.Err())
			}
		}
		var lg []float32
		if decode {
			for _, id := range ids {
				if lg, err = st.Forward(id); err != nil {
					t.Fatal(err)
				}
			}
		} else if lg, err = st.Prefill(ids); err != nil {
			t.Fatal(err)
		}
		o := out{last: append([]float32(nil), lg...), tails: st.TailRowChunks()}
		next := int32(argmax(lg))
		for i := range 8 {
			if force != nil {
				next = force[i]
			}
			o.toks = append(o.toks, next)
			if lg, err = st.Forward(next); err != nil {
				t.Fatal(err)
			}
			o.steps = append(o.steps, append([]float32(nil), lg...))
			next = int32(argmax(lg))
		}
		return o
	}
	for _, n := range []int{512 + 3, 2, 4} {
		dec := run(true, true, true, n, nil)
		on, off := run(true, false, true, n, dec.toks), run(false, false, true, n, dec.toks)
		// The CPU device on the same tokens: how far each device arm sits
		// from it says whether a gap between the arms is one arm's.
		ref := run(true, true, false, n, dec.toks)
		far := func(o out) float64 {
			w := nmseF32(ref.last, o.last)
			for i := range ref.steps {
				w = max(w, nmseF32(ref.steps[i], o.steps[i]))
			}
			return w
		}
		t.Logf("n=%d: from the CPU device, decode %.2e, the ragged tail %.2e, the chunk %.2e", n, far(dec), far(on), far(off))
		if on.tails != 1 || off.tails != 0 || dec.tails != 0 {
			t.Fatalf("n=%d: %d ragged tail(s) with the rows on, %d off, %d decoding; want 1, 0 and 0",
				n, on.tails, off.tails, dec.tails)
		}
		// Teacher-forced on the decode arm's tokens, so a near-tie that one
		// arm breaks the other way cannot send the arms down different
		// texts: every step reads the history the prompt wrote. Each device
		// arm is held to the CPU device, not to another device arm, whose
		// gap adds both arms' errors (the user's decision).
		for _, a := range []struct {
			name string
			o    out
		}{{"decoding the prompt", dec}, {"the ragged tail", on}, {"the batched chunk", off}} {
			if w := far(a.o); !(w < tailBand) {
				t.Fatalf("n=%d: %s reads NMSE %.3e from the CPU device, past %.0e", n, a.name, w, tailBand)
			}
		}
		worst, chunkWorst := nmseF32(dec.last, on.last), nmseF32(dec.last, off.last)
		for i := range dec.steps {
			worst = max(worst, nmseF32(dec.steps[i], on.steps[i]))
			chunkWorst = max(chunkWorst, nmseF32(dec.steps[i], off.steps[i]))
		}
		t.Logf("n=%d: worst NMSE against decoding the prompt %.2e with the ragged tail, %.2e with the chunk", n, worst, chunkWorst)
	}
}

// tailBand is the NMSE a device arm's logits may sit from the CPU device's on
// the same tokens: the reduction-order band, which on a V100 read up to
// 8.3e-3 (decode at n=4).
const tailBand = 1e-2
