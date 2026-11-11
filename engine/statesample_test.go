package engine

import (
	"math"
	"os"
	"testing"

	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"

	"github.com/jitllm/jitllm/internal/oracle"
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
	m, err := model.Open(jlmOf(t, models()["stories15M"]))
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
		{"typical-p", Sampler{Temp: 1, TypicalP: 0.8, Seed: 29}},
		{"top-k and typical-p", Sampler{Temp: 1.1, TopK: 40, TypicalP: 0.7, TopP: 0.95, RepeatPen: 1.3, Seed: 31}},
		{"top-k past the walk and typical-p", Sampler{Temp: 0.9, TopK: 1000, TypicalP: 0.9, MinP: 0.01, Seed: 37}},
		{"XTC", Sampler{Temp: 1, XTCProbability: 0.6, XTCThreshold: 0.05, Seed: 41}},
		{"top-k and XTC", Sampler{Temp: 1.1, TopK: 40, XTCProbability: 0.5, XTCThreshold: 0.1, MinP: 0.02, Seed: 43}},
		{"top-n-sigma", Sampler{Temp: 1, TopNSigma: 1.5, Seed: 47}},
		{"top-k and top-n-sigma", Sampler{Temp: 1.2, TopK: 40, TopNSigma: 1, TopP: 0.9, Seed: 53}},
	} {
		t.Run(c.name, func(t *testing.T) {
			st := testInstance(t, m, 96)
			defer st.Close()
			composed, peek := c.smp, c.smp
			lg, err := st.Prefill([]int32{1, 450, 3014, 471, 263})
			if err != nil {
				t.Fatal(err)
			}
			served0, declined0 := st.Account(nn.CPU).Steps.Of(nn.ComposedSample)
			const draws = 48
			near := 0
			for i := range draws {
				var hist []int
				for _, h := range peek.hist {
					hist = append(hist, int(h))
				}
				u := peek.uniform()
				xtc := math.Inf(1)
				if t := peek.xtcDraw(); t > 0 {
					xtc = float64(t)
				}
				got, err := st.Sample(&composed, lg)
				if err != nil {
					t.Fatal(err)
				}
				if ref, mg := sampleRef(lg, hist, c.smp, u, xtc); got != ref {
					if mg >= 1e-6 {
						t.Fatalf("draw %d: the composed draw is %d, its definition's %d (u %.9f, margin %.3g)", i, got, ref, u, mg)
					}
					near++
				}
				composed.Observe(got)
				peek.Observe(got)
				if lg, err = st.Forward(got); err != nil {
					t.Fatal(err)
				}
			}
			served, declined := st.Account(nn.CPU).Steps.Of(nn.ComposedSample)
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

// sampleRef is the draw s makes over logits at u by its definition
// (internal/oracle): the walk over a top-k's candidates, else the cut, which
// is the row's where no cut fires; and how far the draw is from flipping.
func sampleRef(logits []float32, hist []int, s Sampler, u, xtc float64) (int32, float64) {
	pen := float32(1)
	if s.RepeatPen > 1 && len(hist) > 0 {
		pen = float32(s.RepeatPen)
	}
	invT := float32(1 / s.Temp)
	topp, minp := s.cuts()
	sig := math.Inf(1)
	if s.TopNSigma > 0 {
		v := append([]float32(nil), logits...)
		oracle.PenalizeOnce(v, hist, pen)
		logits, sig = oracle.TopNSigma(v, s.TopNSigma)
		hist, pen = nil, 1
	}
	if s.TopK > 0 && s.TopK < len(logits) {
		v := append([]float32(nil), logits...)
		oracle.PenalizeOnce(v, hist, pen)
		tok, mg := oracle.SampleWalk(v, s.TopK, s.TopK, float64(invT), float64(topp), float64(minp), float64(s.typical()), xtc, u)
		return int32(tok), min(mg, sig)
	}
	tok, mg := oracle.SampleCut(logits, hist, pen, invT, float64(topp), float64(minp), float64(s.typical()), xtc, u)
	return int32(tok), min(mg, sig)
}
