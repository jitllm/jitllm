package model

import (
	"fmt"
	"math"
	"sort"
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// lpCheck holds a step's reported logprobs to the oracle's log-softmax of the
// raw logits: the chosen token's value, and the top-n ids against a sort of
// the oracle (higher value first, lower id on a tie). The ids must match
// exactly unless two neighbours are within the kernel's rounding of each
// other, which on these rows they are not.
func lpCheck(logits []float32, chosen int32, got float32, top []TopLogprob, n int) error {
	want := oracle.LogSoftmax32(logits)
	for _, v := range want {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("the oracle is not finite")
		}
	}
	// The engine's exp is good to ~5e-5 relative (the softmax kernel's own
	// band), which is ~5e-5 absolute on a log; the ordering is exact.
	tol := func(w float64) float64 { return 1e-4 + 2e-6*math.Abs(w) }
	if d := math.Abs(float64(got) - want[chosen]); !(d <= tol(want[chosen])) {
		return fmt.Errorf("chosen %d: logprob %v, the oracle %v", chosen, got, want[chosen])
	}
	ids := make([]int32, len(want))
	for i := range ids {
		ids[i] = int32(i)
	}
	sort.SliceStable(ids, func(a, b int) bool {
		if want[ids[a]] != want[ids[b]] {
			return want[ids[a]] > want[ids[b]]
		}
		return ids[a] < ids[b]
	})
	if len(top) != n {
		return fmt.Errorf("%d alternatives, want %d", len(top), n)
	}
	for i, tp := range top {
		if tp.ID != ids[i] {
			return fmt.Errorf("alternative %d is token %d, the oracle's %d", i, tp.ID, ids[i])
		}
		w := want[tp.ID]
		if d := math.Abs(float64(tp.Logprob) - w); !(d <= tol(w)) {
			return fmt.Errorf("alternative %d (token %d): %v, the oracle %v", i, tp.ID, tp.Logprob, w)
		}
	}
	return nil
}

// TestLogprobsMatchTheOracle runs stories15M through a prompt and a sampled
// continuation and holds every step past position 0 to the oracle's
// log-softmax of the raw logits -- before temperature, which the request
// samples at 0.8 so the claim "raw" is load-bearing. The same check is run
// against the temperature-scaled distribution at every step and must reject
// it. (The kernel's own violations -- the maximum not subtracted, ln(sum)
// dropped -- fail nn.TestLogSoftmaxMatchesTheOracle.)
func TestLogprobsMatchTheOracle(t *testing.T) {
	mp, ok := existingModel(models["stories15M"])
	if !ok {
		testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS to the model directory) -- this gate proved nothing",
			models["stories15M"])
	}
	m, err := Open(jlmOf(t, mp))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.Vocab.Encode("Once upon a time, there was a little dog", true)
	const gen, n = 12, 20
	st := m.NewInstance(len(ids) + gen + 1)
	defer st.Close()
	logits, err := st.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	sm := &Sampler{Temp: 0.8, Seed: 7}
	lp := &Logprobs{N: n}
	checked, rejected := 0, 0
	for i := 0; i < gen; i++ {
		next := sm.Sample(logits)
		got := lp.Take(logits, next)
		if err := lpCheck(logits, next, got, lp.Top, n); err != nil {
			t.Fatalf("step %d (position %d): %v", i, st.Pos(), err)
		}
		checked++

		// Violation 1: the tempered distribution claimed as raw.
		hot := append([]float32(nil), logits...)
		for j := range hot {
			hot[j] /= 0.8
		}
		vl := &Logprobs{N: n}
		if lpCheck(logits, next, vl.Take(hot, next), vl.Top, n) != nil {
			rejected++
		}
		if logits, err = st.Forward(next); err != nil {
			t.Fatal(err)
		}
	}
	if checked != gen || rejected != gen {
		t.Fatalf("checked %d steps and the tempered violation was rejected at %d, want %d each",
			checked, rejected, gen)
	}
}

// TestLogprobsAllocateNothingWarm: a warm Take, top-20 at a vocabulary width,
// makes no heap allocation.
func TestLogprobsAllocateNothingWarm(t *testing.T) {
	logits := allocProbeLogits(32000)
	lp := &Logprobs{N: MaxTopLogprobs}
	lp.Take(logits, 5)
	if a := testing.AllocsPerRun(20, func() { lp.Take(logits, 5) }); a != 0 {
		t.Fatalf("%v allocations per warm Take, want none", a)
	}
}
