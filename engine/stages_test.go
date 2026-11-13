package engine

import (
	"math"
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// TestSamplerStagesRunInTheirOrder: llama.cpp's samplers in an order of
// their own (Sampler.Stages) through Instance.Sample on stories15M: every
// draw its definition's token -- the stages run in their order over the
// row, each one llama.cpp's (oracle.Chain), then the row's walk at
// temperature 1 (oracle.SampleRow) at the draw's uniform, XTC at its
// chance as it fell -- and SampledProbs the chain's distribution. The
// order moves the token against the fused draw, llama.cpp's own order over
// the same parameters, on a share of the draws, or the comparison cannot
// see it.
func TestSamplerStagesRunInTheirOrder(t *testing.T) {
	if _, err := os.Stat(models()["stories15M"]); err != nil {
		t.Fatalf("MODEL MISSING: %v", err)
	}
	m, err := model.Open(jlmOf(t, models()["stories15M"]))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	names := map[Stage]string{StagePenalties: "penalties", StageTopNSigma: "top_n_sigma", StageTopK: "top_k",
		StageTypical: "typ_p", StageTopP: "top_p", StageMinP: "min_p", StageXTC: "xtc", StageTemperature: "temperature"}
	for _, c := range []struct {
		name string
		smp  Sampler
	}{
		{"temperature first", Sampler{Temp: 2.5, TopK: 30, TopP: 0.5, MinP: 0.03, Seed: 51,
			Stages: []Stage{StageTemperature, StageTopK, StageTopP, StageMinP}}},
		{"min-p after the temperature", Sampler{Temp: 2, TopK: 40, MinP: 0.15, Seed: 53,
			Stages: []Stage{StageTopK, StageTemperature, StageMinP}}},
		{"xtc first, the penalties last", Sampler{Temp: 1.3, TopP: 0.9, XTCProbability: 0.7, XTCThreshold: 0.05,
			RepeatPen: 1.5, PresencePen: 0.4, FreqPen: 0.2, RepeatLastN: 16, Seed: 57,
			Stages: []Stage{StageXTC, StageTopP, StageTemperature, StagePenalties}}},
		{"typical and top-n-sigma after a dynamic temperature", Sampler{Temp: 1.6, TypicalP: 0.7, TopNSigma: 1.1,
			DynatempRange: 0.5, DynatempExponent: 1.3, Seed: 59,
			Stages: []Stage{StageTemperature, StageTypical, StageTopNSigma}}},
	} {
		t.Run(c.name, func(t *testing.T) {
			st := testInstance(t, m, 96)
			defer st.Close()
			// Copies before the first draw: each seeds a stream of its own
			// from the same seed, and draws as many uniforms a token.
			smp, peek, fused := c.smp, c.smp, c.smp
			fused.Stages = nil
			lg, err := st.Prefill([]int32{1, 450, 3014, 471, 263})
			if err != nil {
				t.Fatal(err)
			}
			const draws = 32
			near, moved := 0, 0
			for i := range draws {
				u := peek.uniform()
				xtc := math.Inf(1)
				if thr := peek.xtcDraw(); thr > 0 {
					xtc = float64(thr)
				}
				hist := make([]int, len(smp.hist))
				for j, h := range smp.hist {
					hist[j] = int(h)
				}
				a := st.drawArgs(&smp, 0, 0)
				rng, ex := smp.dyn()
				var chain []oracle.ChainStage
				for _, s := range smp.Stages {
					cs := oracle.ChainStage{Name: names[s], K: smp.TopK, InvT: float64(a.InvTemp), Dyn: [2]float64{float64(rng), float64(ex)}}
					switch s {
					case StagePenalties:
						pen := smp.penalties()
						cs.Hist, cs.Rep, cs.Presence, cs.Frequency = hist, pen[0], pen[1], pen[2]
					case StageTopNSigma:
						cs.P = smp.TopNSigma
					case StageTypical:
						cs.P = smp.TypicalP
					case StageTopP:
						cs.P = float64(a.TopP)
					case StageMinP:
						cs.P = smp.MinP
					case StageXTC:
						cs.P = xtc
					}
					chain = append(chain, cs)
				}
				row, cm := oracle.Chain(lg, chain)
				ref, wm := oracle.SampleRow(row, nil, 1, 1, u)
				plain, err := st.Sample(&fused, lg)
				if err != nil {
					t.Fatal(err)
				}
				got, err := st.Sample(&smp, lg)
				if err != nil {
					t.Fatal(err)
				}
				if plain != got {
					moved++
				}
				if got != int32(ref) {
					if min(cm, wm) >= 1e-5 {
						t.Fatalf("draw %d: the staged draw is %d, its definition's %d (u %.6f)", i, got, ref, u)
					}
					near++
				}
				l := Logprobs{N: 20}
				pc, err := st.SampledProbs(&l, &smp, lg, got)
				if err != nil {
					t.Fatal(err)
				}
				top, mass := math.Inf(-1), 0.0
				for _, x := range row {
					top = max(top, float64(x))
				}
				for _, x := range row {
					mass += math.Exp(float64(x) - top)
				}
				if w := math.Exp(float64(row[got])-top) / mass; math.Abs(float64(pc)-w) > 1e-4*w+1e-6 && min(cm, wm) >= 1e-5 {
					t.Fatalf("draw %d: the drawn token %d sampled at %.7g, its definition's %.7g", i, got, pc, w)
				}
				smp.Observe(got)
				fused.Observe(got)
				if lg, err = st.Forward(got); err != nil {
					t.Fatal(err)
				}
			}
			if near > draws/8 || moved == 0 {
				t.Fatalf("%d of %d draws within rounding of an edge, %d moved against the fused order: the comparison sees too little", near, draws, moved)
			}
			served, _ := st.Account(nn.CPU).Steps.Of(nn.ComposedSample)
			t.Logf("%d draws, %d near an edge, %d moved against the fused order, %d composed samples served", draws, near, moved, served)
		})
	}
}
