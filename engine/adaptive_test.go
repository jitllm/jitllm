package engine

import (
	"math"
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// TestAdaptivePIsComposed: llama.cpp's adaptive-p sampler through
// Instance.Sample on stories15M, over a top-k draw's candidates and over the
// whole row's cut: every draw the definition's token (oracle.AdaptiveP's
// logits walked as the row's draw walks them, oracle.SampleRow) with the
// moving average carried draw to draw in float64 beside it, and
// SampledProbs the transformed distribution. adaptive-p moves the token
// against the plain draw on a share of draws, or the comparison cannot see
// it; a target below 0 is the plain draw.
func TestAdaptivePIsComposed(t *testing.T) {
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
		{"top-k", Sampler{Temp: 0.9, TopK: 40, Adaptive: true, AdaptiveTarget: 0.25, AdaptiveDecay: 0.9, Seed: 41}},
		{"top-p", Sampler{Temp: 1.1, TopP: 0.97, Adaptive: true, AdaptiveTarget: 0.4, AdaptiveDecay: 0.5, Seed: 43}},
		{"off", Sampler{Temp: 0.9, TopK: 40, Adaptive: true, AdaptiveTarget: -1, AdaptiveDecay: 0.9, Seed: 47}},
	} {
		t.Run(c.name, func(t *testing.T) {
			st := testInstance(t, m, 96)
			defer st.Close()
			smp := c.smp
			peek := smp
			lg, err := st.Prefill([]int32{1, 450, 3014, 471, 263})
			if err != nil {
				t.Fatal(err)
			}
			decay := min(max(smp.AdaptiveDecay, 0), 0.99)
			sum, weight := float64(float32(smp.AdaptiveTarget)/(1-float32(decay))), float64(1/(1-float32(decay)))
			const draws = 32
			near, moved := 0, 0
			for i := range draws {
				u := peek.uniform()
				p, mg := distRef(lg, &smp, false)
				plain, _ := sampleRef(m, lg, nil, Sampler{Temp: smp.Temp, TopK: smp.TopK, TopP: smp.TopP}, u, math.Inf(1))
				got, err := st.Sample(&smp, lg)
				if err != nil {
					t.Fatal(err)
				}
				if c.name == "off" {
					if got != plain && mg >= 1e-4 {
						t.Fatalf("draw %d: a target below 0 drew %d, the plain draw %d", i, got, plain)
					}
					if lg, err = st.Forward(got); err != nil {
						t.Fatal(err)
					}
					continue
				}
				logits, km := oracle.AdaptiveP(p, smp.AdaptiveTarget, sum, weight)
				ref, wm := oracle.SampleRow(logits, nil, 1, 1, u)
				if int32(ref) != plain {
					moved++
				}
				if got != int32(ref) {
					if min(mg, km, wm) >= 1e-5 {
						t.Fatalf("draw %d: the composed draw is %d, its definition's %d (u %.6f, average %.5f of %.5f)", i, got, ref, u, sum, weight)
					}
					near++
				}
				l := Logprobs{N: 20}
				pc, err := st.SampledProbs(&l, &smp, lg, got)
				if err != nil {
					t.Fatal(err)
				}
				q := make([]float64, len(logits))
				top, mass := math.Inf(-1), 0.0
				for _, x := range logits {
					top = max(top, float64(x))
				}
				for j, x := range logits {
					q[j] = math.Exp(float64(x) - top)
					mass += q[j]
				}
				if w := q[got] / mass; math.Abs(float64(pc)-w) > 1e-4*w+1e-6 && min(mg, km) >= 1e-5 {
					t.Fatalf("draw %d: the drawn token %d sampled at %.7g, its definition's %.7g", i, got, pc, w)
				}
				sum, weight = p[got]+decay*sum, 1+decay*weight
				if lg, err = st.Forward(got); err != nil {
					t.Fatal(err)
				}
			}
			if c.name != "off" && (near > draws/8 || moved == 0) {
				t.Fatalf("%d of %d draws within rounding of an edge, %d moved by adaptive-p: the comparison sees too little", near, draws, moved)
			}
			served, _ := st.Account(nn.CPU).Steps.Of(nn.ComposedSample)
			t.Logf("%d draws, %d near an edge, %d moved against the plain draw, %d composed samples served", draws, near, moved, served)
		})
	}
}
