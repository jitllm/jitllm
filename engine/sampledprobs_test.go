package engine

import (
	"math"
	"os"
	"sort"
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// TestSampledProbsAreTheDrawsDistribution holds Instance.SampledProbs --
// llama.cpp's post_sampling_probs -- to the distribution each draw drew
// from, in float64: over stories15M's logits through a generation, the
// row's draw and its cuts with a top-k and without (distRef), XTC as the
// draw took it or did not (oracle.SampleProb), Mirostat at the mu the draw
// read (oracle.MirostatKeep), the infill sampler (oracle.InfillProbs) and a
// greedy draw. Every listed token's probability is its definition's, the
// list is the survivors alone, the likeliest first, as many as asked or as
// survive, and the drawn token's probability is its own. Reading Mirostat at
// the mu the draw leaves, or XTC at its chance rather than as it fell, must
// read off the definition somewhere.
func TestSampledProbsAreTheDrawsDistribution(t *testing.T) {
	if _, err := os.Stat(models()["stories15M"]); err != nil {
		t.Fatalf("MODEL MISSING: %v", err)
	}
	m, err := model.Open(jlmOf(t, models()["stories15M"]))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	piece := func(id int) string {
		p, _ := m.Vocab.Piece(int32(id))
		return p
	}
	eog := func(id int) bool { return m.Vocab.IsEOG(int32(id)) }
	const asked, draws = 12, 16
	for _, c := range []struct {
		name string
		smp  Sampler
	}{
		{"greedy", Sampler{}},
		{"top-p", Sampler{Temp: 0.7, TopP: 0.85, Seed: 3}},
		{"min-p", Sampler{Temp: 1.3, MinP: 0.05, Seed: 5}},
		{"top-k", Sampler{Temp: 0.8, TopK: 30, TopP: 0.9, Seed: 7}},
		{"top-k past the walk", Sampler{Temp: 0.9, TopK: 600, TopP: 0.95, Seed: 11}},
		{"xtc", Sampler{Temp: 1, TopK: 40, XTCProbability: 0.5, XTCThreshold: 0.05, Seed: 13}},
		{"mirostat v1", Sampler{Temp: 0.9, Mirostat: 1, MirostatTau: 4, MirostatEta: 0.2, Seed: 17}},
		{"mirostat v2", Sampler{Temp: 0.9, Mirostat: 2, MirostatTau: 3, MirostatEta: 0.3, Seed: 19}},
		{"infill", Sampler{Temp: 1, TopK: 40, TopP: 0.99, Infill: true, Seed: 23}},
	} {
		t.Run(c.name, func(t *testing.T) {
			st := testInstance(t, m, 64)
			defer st.Close()
			smp := c.smp
			lg, err := st.Prefill([]int32{1, 450, 3014, 471, 263})
			if err != nil {
				t.Fatal(err)
			}
			checked, fired := 0, 0
			for i := range draws {
				mu := float64(smp.mu)
				if smp.Mirostat != 0 && !smp.muSet {
					mu = 2 * smp.MirostatTau
				}
				tok, err := st.Sample(&smp, lg)
				if err != nil {
					t.Fatal(err)
				}
				if smp.drew.XTC > 0 {
					fired++
				}
				served0, _ := st.Account(nn.CPU).Steps.Of(nn.ComposedSample)
				l := Logprobs{N: asked}
				pc, err := st.SampledProbs(&l, &smp, lg, tok)
				if err != nil {
					t.Fatal(err)
				}
				served, _ := st.Account(nn.CPU).Steps.Of(nn.ComposedSample)
				if smp.Temp > 0 && served <= served0 {
					t.Fatalf("draw %d: no composed form served the distribution (%d to %d)", i, served0, served)
				}
				want, margin := sampledRef(m, lg, &smp, mu, piece, eog)
				if smp.Temp <= 0 {
					want = map[int32]float64{tok: 1}
				}
				if margin < 1e-4 {
					if lg, err = st.Forward(tok); err != nil {
						t.Fatal(err)
					}
					continue
				}
				checked++
				if w := want[tok]; math.Abs(float64(pc)-w) > 1e-5*w+1e-6 {
					t.Fatalf("draw %d: the drawn token %d at %.9g, its definition's %.9g", i, tok, pc, w)
				}
				live := 0
				for _, p := range want {
					if p > 0 {
						live++
					}
				}
				if len(l.Top) != min(asked, live) {
					t.Fatalf("draw %d: %d listed, of %d survivors and %d asked", i, len(l.Top), live, asked)
				}
				for j, a := range l.Top {
					w := want[a.ID]
					if math.Abs(float64(a.Logprob)-w) > 1e-5*w+1e-6 {
						t.Fatalf("draw %d: token %d listed at %.9g, its definition's %.9g", i, a.ID, a.Logprob, w)
					}
					if j > 0 && a.Logprob > l.Top[j-1].Logprob {
						t.Fatalf("draw %d: token %d at %.9g after one at %.9g", i, a.ID, a.Logprob, l.Top[j-1].Logprob)
					}
				}
				if lg, err = st.Forward(tok); err != nil {
					t.Fatal(err)
				}
			}
			if checked < draws/2 || c.smp.XTCProbability > 0 && (fired == 0 || fired == draws) {
				t.Fatalf("%d of %d draws checked, XTC taken on %d: the comparison sees too little", checked, draws, fired)
			}
		})
	}
}

// sampledRef is the distribution smp's last draw over logits drew from by
// its definition (internal/oracle), each surviving token's probability, the
// penalty off, Mirostat at mu; and how far it is from moving.
func sampledRef(m *model.Model, logits []float32, smp *Sampler, mu float64, piece func(int) string, eog func(int) bool) (map[int32]float64, float64) {
	out := map[int32]float64{}
	if smp.Temp <= 0 {
		return out, math.Inf(1)
	}
	invT := float32(1 / smp.Temp)
	topp, minp := smp.cuts()
	rng, ex := smp.dyn()
	dyn := [2]float64{float64(rng), float64(ex)}
	xtc := math.Inf(1)
	if smp.drew.XTC > 0 {
		xtc = float64(smp.drew.XTC)
	}
	switch {
	case smp.Mirostat != 0:
		keep, margin := oracle.MirostatKeep(logits, smp.Mirostat == 1, invT, mu)
		order := make([]int, len(logits))
		for i := range order {
			order[i] = i
		}
		sort.SliceStable(order, func(a, b int) bool { return logits[order[a]] > logits[order[b]] })
		top := float64(logits[order[0]] * invT)
		mass := 0.0
		for _, i := range order[:keep] {
			mass += math.Exp(float64(logits[i]*invT) - top)
		}
		for _, i := range order[:keep] {
			out[int32(i)] = math.Exp(float64(logits[i]*invT)-top) / mass
		}
		return out, margin
	case smp.Infill:
		k := smp.TopK
		if k <= 0 || k > nn.InfillMaxK {
			k = nn.InfillMaxK
		}
		ids, p, _, margin := oracle.InfillProbs(logits, nn.InfillMaxK, k, float64(invT), float64(topp), float64(minp), float64(smp.typical()), xtc, 0,
			dyn, piece, eog, int(m.Vocab.Infill().EOT))
		for j, id := range ids {
			if p[j] > 0 {
				out[int32(id)] = p[j]
			}
		}
		return out, margin
	case smp.XTCProbability > 0:
		order := make([]int, len(logits))
		for i := range order {
			order[i] = i
		}
		sort.SliceStable(order, func(a, b int) bool { return logits[order[a]] > logits[order[b]] })
		margin := math.Inf(1)
		for _, i := range order[:smp.TopK] {
			p, mg := oracle.SampleProb(logits, nil, 1, invT, smp.TopK, float64(topp), float64(minp), float64(smp.typical()), xtc, 1, dyn, i)
			margin = min(margin, mg)
			if p > 0 {
				out[int32(i)] = p
			}
		}
		return out, margin
	}
	p, margin := distRef(logits, smp, false)
	for i, x := range p {
		if x > 0 {
			out[int32(i)] = x
		}
	}
	return out, margin
}
