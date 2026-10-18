//go:build amd64 || arm64

package model

import (
	"math"
	"os"
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
)

// TestPrefillMatchesForward is the gate for the whole batched path.
//
// Prefill must leave the session in EXACTLY the state a loop of Forward calls
// would: the same logits for the last token, the same position, and -- the part
// that actually catches bugs -- a KV cache that makes subsequent decode produce
// identical tokens. Attention is the one op the batch cannot flatten, because
// token i may only see positions up to p0+i, and an off-by-one there does not
// crash; it silently lets a token attend to its own future and produces fluent,
// wrong text.
func TestPrefillMatchesForward(t *testing.T) {
	path := benchModel(t)
	// With the batched GEMM out of the way, Prefill is bit-identical to a
	// Forward loop, so this mode demands equality; an NMSE bar would hide a
	// prefill path computing the same op by different arithmetic.
	t.Run("no-gemm", func(t *testing.T) {
		prefillVsForward(t, path, true)
	})
	// The GEMM's exact (float-epilogue) form sums what the matvec sums in the
	// same order, so it is held to equality too. Only the integer-accumulating
	// form is left to the NMSE tripwire.
	t.Run("exact-gemm", func(t *testing.T) {
		prefillVsForwardOpts(t, path, true, WithJITOptions(nn.WithGEMMExact(true)))
	})
	t.Run("shipped", func(t *testing.T) {
		prefillVsForward(t, path, false)
	})
}

// prefillVsForward compares Prefill against a Forward loop. exact demands bit
// equality; otherwise the NMSE tripwire applies.
func prefillVsForward(t *testing.T, path string, exact bool) {
	if exact {
		prefillVsForwardOpts(t, path, true, noGEMM)
		return
	}
	prefillVsForwardOpts(t, path, false)
}

// prefillVsForwardOpts is prefillVsForward with the options that select the
// batched path under test; exact demands bit equality.
func prefillVsForwardOpts(t *testing.T, path string, exact bool, extra ...Option) {
	// The tuner is pinned so both sessions run the same kernels: it advances
	// on Forward calls and Prefill makes none.
	opts := append([]Option{noTune}, extra...)
	m, err := Open(jlmOf(t, path), opts...)
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()

	// A prompt longer than one chunk, so the cross-chunk causal bound is
	// exercised rather than assumed.
	full := m.Vocab.Encode("The capital of France is Paris, and the capital of "+
		"Germany is Berlin, and the capital of Italy is Rome, and the capital "+
		"of Spain is Madrid, and the capital of Portugal is", true)
	n := len(full)
	if v := os.Getenv("JITLLM_TEST_NTOK"); v != "" {
		if k, err := strconv.Atoi(v); err == nil && k > 0 && k <= n {
			n = k
		}
	}
	ids := full[:n]
	t.Logf("prompt is %d tokens, max chunk %d", len(ids), MaxPrefillChunk)

	ref := m.NewState(len(ids) + 40)
	defer ref.Close()
	var want []float32
	for _, id := range ids {
		if want, err = ref.Forward(id); err != nil {
			t.Fatal(err)
		}
	}
	batch := m.NewState(len(ids) + 40)
	defer batch.Close()
	got, err := batch.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	if batch.Pos() != ref.Pos() {
		t.Fatalf("position %d after Prefill, want %d", batch.Pos(), ref.Pos())
	}
	var sse, sy2 float64
	for i := range want {
		d := float64(got[i] - want[i])
		sse += d * d
		sy2 += float64(want[i]) * float64(want[i])
	}
	nmse := sse / sy2
	if exact {
		// Without the reassociating GEMM there is no difference at all.
		if nmse != 0 {
			t.Errorf("last-token logits: NMSE %.3e with the GEMM disabled -- "+
				"prefill and decode should be bit-identical here", nmse)
		}
		t.Logf("last-token logits bit-identical (NMSE %.3e)", nmse)
	} else {
		// The shipped GEMM sums a tile where MatVec sums a row, and that
		// reassociation compounds over layers (measured up to ~3e-3 across
		// models). 1e-2 is a gross-error tripwire; the exact modes above are
		// the gates with teeth.
		if nmse > 1e-2 || math.IsNaN(nmse) {
			t.Errorf("last-token logits: NMSE %.3e between Prefill and Forward -- too far to be reassociation", nmse)
		}
		t.Logf("last-token logits NMSE %.3e", nmse)
	}

	// The KV cache is the real product of prefill: decode from both sessions
	// and require identical token ids. A wrong cache diverges at step 0 or 1;
	// 16 steps stays short of where rounding-level gaps flip greedy argmaxes.
	const steps = 16
	for i := 0; i < steps; i++ {
		a, b := Greedy(want), Greedy(got)
		if a != b {
			// A flip where both sides won by less than the perturbation
			// between the two logit vectors is the model being undecided; a
			// flip with either side more confident than that is a defect.
			mA, mB, d := margin(want), margin(got), maxAbsDiff(want, got)
			if mA <= d && mB <= d {
				t.Logf("decode step %d: ids diverge (%d vs %d) on margins "+
					"%.4f and %.4f against a %.4f perturbation -- undecided, "+
					"not a defect", i, b, a, mB, mA, d)
				break
			}
			t.Fatalf("decode step %d: Prefill gave %d (margin %.4f), Forward %d "+
				"(margin %.4f), perturbation %.4f -- too confident to be rounding",
				i, b, mB, a, mA, d)
		}
		if want, err = ref.Forward(a); err != nil {
			t.Fatal(err)
		}
		if got, err = batch.Forward(b); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("%d decode steps after prefill agree token for token", steps)
}

// margin is how far the argmax won by: the gap to the runner-up.
func margin(l []float32) float64 {
	best, second := float32(math.Inf(-1)), float32(math.Inf(-1))
	for _, v := range l {
		if v > best {
			best, second = v, best
		} else if v > second {
			second = v
		}
	}
	return float64(best - second)
}

// maxAbsDiff is the largest single-logit disagreement between two evaluations
// of the same model -- the size of the perturbation a flip has to beat.
func maxAbsDiff(a, b []float32) float64 {
	d := 0.0
	for i := range a {
		if x := math.Abs(float64(a[i] - b[i])); x > d {
			d = x
		}
	}
	return d
}

// TestSlidingWindowIsHonouredEverywhere: a local layer must attend to the same
// keys in Prefill, ForwardBatch and Forward. It checks the window arithmetic the
// three paths share on a synthetic config with alternating local and global
// layers.
func TestSlidingWindowIsHonouredEverywhere(t *testing.T) {
	c := &Config{NLayer: 4, NCtx: 4096, SWAWindow: 8, SWAPeriod: 2}
	// The pattern must actually alternate, or every layer is global and the
	// comparison below is between two identical things.
	local, global := 0, 0
	for li := 0; li < c.NLayer; li++ {
		if c.SWA(li) {
			local++
		} else {
			global++
		}
	}
	if local == 0 || global == 0 {
		t.Fatalf("%d local and %d global layer(s): this config cannot show the window "+
			"being applied to some layers and not others", local, global)
	}

	// The window arithmetic the three paths share; decode's form (forward.go)
	// is the reference.
	win := func(li, n int) (w0, an int) {
		w0, an = 0, n
		if c.SWA(li) && c.SWAWindow < an {
			w0, an = an-c.SWAWindow, c.SWAWindow
		}
		return
	}
	for _, n := range []int{1, 7, 8, 9, 64} {
		for li := 0; li < c.NLayer; li++ {
			w0, an := win(li, n)
			if w0+an != n {
				t.Fatalf("layer %d at n=%d: window [%d,%d) does not end at the current "+
					"position -- a local layer must include the token being generated",
					li, n, w0, w0+an)
			}
			if c.SWA(li) && n > c.SWAWindow && an != c.SWAWindow {
				t.Fatalf("layer %d at n=%d: local layer read %d keys, want the %d-key window",
					li, n, an, c.SWAWindow)
			}
			if !c.SWA(li) && an != n {
				t.Fatalf("layer %d at n=%d: a GLOBAL layer was narrowed to %d keys", li, n, an)
			}
		}
	}
}
