package engine

import (
	"os"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/model"
)

// TestGreedyReadsThePenalties: at temperature 0 a draw is the largest of the
// row the penalties leave, as llama.cpp's chain penalizes before its
// temperature sampler keeps the largest (llama_sampler_temp_ext at 0). Over
// a copying prompt on stories15M, the sampler having seen the prompt (Prime),
// Sample, ForwardSample and a lookup Speculator each give at every step the
// greedy token of oracle.Penalize's row over the last RepeatLastN tokens,
// two near-equal logits excepted; the penalty moves the token on some steps,
// or the gate sees nothing.
func TestGreedyReadsThePenalties(t *testing.T) {
	if _, err := os.Stat(models()["stories15M"]); err != nil {
		t.Fatalf("MODEL MISSING: %v", err)
	}
	m, err := model.Open(jlmOf(t, models()["stories15M"]))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	prompt := m.Vocab.Encode(copyPrompt, true)
	const n = 32
	pen := Sampler{RepeatPen: 1.3, PresencePen: 0.8, FreqPen: 0.2, RepeatLastN: 32}
	window := func(seen []int32) []int {
		var h []int
		for _, x := range seen[max(0, len(seen)-pen.RepeatLastN):] {
			h = append(h, int(x))
		}
		return h
	}
	// The reference: each step's greedy token of the penalized row, and how
	// near its two best are.
	st := testInstance(t, m, len(prompt)+n+8)
	lg, err := st.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	seen := slices.Clone(prompt)
	var want []int32
	var margins []float32
	moved := 0
	for range n {
		v := slices.Clone(lg)
		oracle.Penalize(v, window(seen), float32(pen.RepeatPen), float32(pen.PresencePen), float32(pen.FreqPen))
		y := Greedy(v)
		best, next := v[y], float32(-1e30)
		for i, x := range v {
			if int32(i) != y && x > next {
				next = x
			}
		}
		if y != Greedy(lg) {
			moved++
		}
		want, margins = append(want, y), append(margins, best-next)
		seen = append(seen, y)
		if lg, err = st.Forward(y); err != nil {
			t.Fatal(err)
		}
	}
	st.Close()
	if moved == 0 {
		t.Fatalf("the penalties moved none of %d greedy tokens: the gate sees nothing", n)
	}
	check := func(arm string, got []int32) {
		for i := range want {
			if i >= len(got) {
				t.Fatalf("%s: %d tokens of %d", arm, len(got), n)
			}
			if got[i] != want[i] {
				if margins[i] < 1e-4 {
					t.Logf("%s: step %d parts from the reference at a near tie (%.2e): compared to there", arm, i, margins[i])
					return
				}
				t.Fatalf("%s: step %d drew %d, the penalized row's largest is %d (the raw row's in the reference moved %d of %d)",
					arm, i, got[i], want[i], moved, n)
			}
		}
	}

	sampleArm := func(forward bool) []int32 {
		st := testInstance(t, m, len(prompt)+n+8)
		defer st.Close()
		lg, err := st.Prefill(prompt)
		if err != nil {
			t.Fatal(err)
		}
		sm := pen
		sm.Prime(prompt)
		y, err := st.Sample(&sm, lg)
		if err != nil {
			t.Fatal(err)
		}
		out := []int32{y}
		sm.Observe(y)
		for len(out) < n {
			if forward {
				y, err = st.ForwardSample(y, &sm)
			} else {
				if lg, err = st.Forward(y); err == nil {
					y, err = st.Sample(&sm, lg)
				}
			}
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, y)
			sm.Observe(y)
		}
		return out
	}
	check("Sample", sampleArm(false))
	check("ForwardSample", sampleArm(true))

	sst := testInstance(t, m, len(prompt)+5*n+8)
	defer sst.Close()
	sp, err := sst.Speculate(WithSpecDrafter(SpecDraftLookup))
	if err != nil {
		t.Fatal(err)
	}
	defer sp.Close()
	sm := pen
	sm.Prime(prompt)
	y, err := sp.Start(prompt, &sm)
	if err != nil {
		t.Fatal(err)
	}
	out := []int32{y}
	for len(out) < n {
		ys, err := sp.Next(&sm)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, ys...)
	}
	check("a lookup Speculator", out[:n])
	t.Logf("the penalties moved %d of %d greedy tokens; Sample, ForwardSample and the Speculator drew the penalized row's", moved, n)
}
