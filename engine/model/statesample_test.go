package model

import (
	"math"
	"os"
	"sort"
	"testing"
)

// TestStateSampleIsComposed requires every token Instance.Sample draws on the
// CPU device to come from the composed draws (cpuDraw: the top-k walk, and
// with no top-k the row and the cut, graph.SampleRow and graph.SampleCut),
// counted served and none declined, over the logits of a real model, the
// history growing so the repeat penalty reaches ids it has drawn, at no top-k
// with no cut (the OpenAI default), top-p, min-p, both, a top-k and one past
// the walk's bound (graph.SampleCut's top-k variant). A draw
// with a cut or a top-k is the host sampler's token for the same seed; one
// with neither is the walk of its definition in float64 (rowMargin), which
// the host sampler's own walk over the row does not always reach (it drew
// 14332 where the definition lands on 476, 5.3e-5 of the mass from an edge),
// unless its point sits closer to an edge than float32 rounding reaches; nine
// in ten must be decisive. The graphs' exactness is jit's
// TestComposedCPUSampleRow/Cut/Walk; this gate says Instance.Sample runs them
// with the inputs the sampler draws from.
func TestStateSampleIsComposed(t *testing.T) {
	if _, err := os.Stat(models()["stories15M"]); err != nil {
		t.Fatalf("MODEL MISSING: %v", err)
	}
	m, err := Open(jlmOf(t, models()["stories15M"]), noTune())
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for _, c := range []struct {
		name string
		smp  Sampler
	}{
		{"no top-k", Sampler{Temp: 1, TopP: 1, Seed: 7}},
		{"top-p", Sampler{Temp: 0.8, TopP: 0.9, RepeatPen: 1.3, RepeatLastN: 32, Seed: 11}},
		{"min-p", Sampler{Temp: 1.2, MinP: 0.05, RepeatPen: 1.2, Seed: 13}},
		{"top-p and min-p", Sampler{Temp: 1, TopP: 0.7, MinP: 0.02, RepeatPen: 1.5, RepeatLastN: 16, Seed: 17}},
		{"top-k", Sampler{Temp: 1.1, TopK: 40, TopP: 0.95, RepeatPen: 1.3, Seed: 19}},
		{"top-k past the walk", Sampler{Temp: 0.9, TopK: 1000, TopP: 0.97, RepeatPen: 1.2, Seed: 23}},
	} {
		t.Run(c.name, func(t *testing.T) {
			st := m.NewInstance(96)
			defer st.Close()
			composed, host, peek := c.smp, c.smp, c.smp
			lg, err := st.Prefill([]int32{1, 450, 3014, 471, 263})
			if err != nil {
				t.Fatal(err)
			}
			served0, declined0 := st.CPUSampleSteps()
			const draws = 48
			near := 0
			for i := range draws {
				hist := append([]int32(nil), host.hist...)
				u := peek.uniform()
				got, want := st.Sample(&composed, lg), host.Sample(lg)
				if c.smp.TopK > 0 || c.smp.MinP > 0 || c.smp.TopP > 0 && c.smp.TopP < 1 {
					if got != want {
						t.Fatalf("draw %d: the composed draw is %d, the host sampler's %d", i, got, want)
					}
				} else if ref, mg := rowMargin(lg, hist, float32(c.smp.RepeatPen), float32(1/c.smp.Temp), u); got != ref {
					if mg >= 1e-6 {
						t.Fatalf("draw %d: the composed draw is %d, its definition's %d (u %.9f, margin %.3g)", i, got, ref, u, mg)
					}
					near++
				} else {
					want = ref
				}
				composed.Observe(want)
				host.Observe(want)
				peek.Observe(want)
				if lg, err = st.Forward(want); err != nil {
					t.Fatal(err)
				}
			}
			served, declined := st.CPUSampleSteps()
			for i := range declined {
				if declined[i] != declined0[i] {
					t.Fatalf("the composed draws declined %v (before %v)", declined, declined0)
				}
			}
			if served-served0 != draws {
				t.Fatalf("the composed draws served %d of %d", served-served0, draws)
			}
			if near > draws/10 {
				t.Fatalf("%d of %d draws within rounding of an edge: the comparison sees too little", near, draws)
			}
			t.Logf("%d draws composed, each the reference's token but %d within rounding of an edge", draws, near)
		})
	}
}

// rowMargin is how far the walk's point at u sits from the nearest edge of
// the cumulative mass of the penalized row's softmax at 1/T in its order (the
// larger first, the lower id of two equal), as a share of the mass, in
// float64: the draw with no top-k and no cut, as the host sampler draws it;
// and the token the walk lands on.
func rowMargin(logits []float32, hist []int32, pen, invT float32, u float64) (int32, float64) {
	v := append([]float32(nil), logits...)
	if pen > 1 {
		seen := map[int32]bool{}
		for _, h := range hist {
			if seen[h] {
				continue
			}
			seen[h] = true
			if v[h] > 0 {
				v[h] /= pen
			} else {
				v[h] *= pen
			}
		}
	}
	m := math.Inf(-1)
	for _, x := range v {
		m = max(m, float64(x*invT))
	}
	q := make([]float64, len(v))
	mass := 0.0
	for i, x := range v {
		q[i] = math.Exp(float64(x*invT) - m)
		mass += q[i]
	}
	order := make([]int, len(v))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return q[order[a]] > q[order[b]] })
	target, upto, margin := u*mass, 0.0, math.Inf(1)
	tok := int32(-1)
	for _, i := range order {
		upto += q[i]
		margin = min(margin, math.Abs(upto-target)/mass)
		if tok < 0 && upto > target {
			tok = int32(i)
		}
	}
	return tok, margin
}
