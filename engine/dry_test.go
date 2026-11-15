package engine

import (
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// dryPrompt is a prompt stories15M goes on repeating, so DRY has runs to
// lower.
const dryPrompt = "Tom had a red ball. Tom had a red ball. Tom had a red ball. Tom had a red ball. Tom had a"

// TestDryRunsInTheChain: llama.cpp's DRY sampler in the composed draw, on
// stories15M over a prompt that repeats itself, the sampler primed with it.
// With no order named the draw runs llama.cpp's own (penalties, DRY,
// top-n-sigma, top-k, typical-p, top-p, min-p, XTC, temperature); in an
// order of its own DRY runs where it is put. Each draw is the chain's token
// (oracle.Chain, its DRY stage oracle.Dry over the sampler's history) at
// the draw's uniform, then the row's walk at temperature 1, and
// SampledProbs its distribution; DRY moves the token against the same draw
// without it on some draws, or the comparison cannot see it. At
// temperature 0 the draw is the largest of the row the penalties and DRY
// leave.
func TestDryRunsInTheChain(t *testing.T) {
	if _, err := os.Stat(models()["stories15M"]); err != nil {
		t.Fatalf("MODEL MISSING: %v", err)
	}
	m, err := model.Open(jlmOf(t, models()["stories15M"]))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	prompt := m.Vocab.Encode(dryPrompt, true)
	breakers := breakersOf(m.Vocab, dryBreakers())
	names := map[Stage]string{StagePenalties: "penalties", StageDRY: "dry", StageTopNSigma: "top_n_sigma", StageTopK: "top_k",
		StageTypical: "typ_p", StageTopP: "top_p", StageMinP: "min_p", StageXTC: "xtc", StageTemperature: "temperature"}
	const draws = 32
	for _, c := range []struct {
		name string
		smp  Sampler
	}{
		{"llama.cpp's own order", Sampler{Temp: 1, TopK: 40, TopP: 0.95, MinP: 0.05, DryMultiplier: 0.8, DryBase: 1.75,
			DryAllowedLength: 2, DryPenaltyLastN: -1, Seed: 71}},
		{"DRY after top-k, the penalties last", Sampler{Temp: 1.1, TopK: 30, RepeatPen: 1.2, RepeatLastN: 64, DryMultiplier: 1.5,
			DryBase: 1.5, DryAllowedLength: 3, DryPenaltyLastN: 128, Seed: 73,
			Stages: []Stage{StageTopK, StageDRY, StageTemperature, StagePenalties}}},
	} {
		t.Run(c.name, func(t *testing.T) {
			st := testInstance(t, m, len(prompt)+draws+8)
			defer st.Close()
			lg, err := st.Prefill(prompt)
			if err != nil {
				t.Fatal(err)
			}
			// Copies before the first draw, each its own stream from the seed.
			smp, peek, plain := c.smp, c.smp, c.smp
			plain.DryMultiplier = 0
			smp.Prime(prompt)
			plain.Prime(prompt)
			n, stage := smp.chain()
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
				for k := range n {
					s := stage(k)
					cs := oracle.ChainStage{Name: names[s], K: smp.TopK, InvT: float64(a.InvTemp), Dyn: [2]float64{float64(rng), float64(ex)}}
					switch s {
					case StagePenalties:
						pen := smp.penalties()
						cs.Hist, cs.Rep, cs.Presence, cs.Frequency = hist, pen[0], pen[1], pen[2]
					case StageDRY:
						cs.DryMult, cs.DryBase, cs.DryAllowed, cs.DryLastN = float32(smp.DryMultiplier), float32(smp.DryBase),
							smp.DryAllowedLength, smp.DryPenaltyLastN
						cs.DryCtx, cs.DryHist, cs.DryBreakers = st.MaxSeq(), slices.Clone(smp.dhist), breakers
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
				without, err := st.Sample(&plain, lg)
				if err != nil {
					t.Fatal(err)
				}
				got, err := st.Sample(&smp, lg)
				if err != nil {
					t.Fatal(err)
				}
				if got != without {
					moved++
				}
				if got != int32(ref) {
					if min(cm, wm) >= 1e-5 {
						t.Fatalf("draw %d: the draw is %d, the chain's %d (u %.6f)", i, got, ref, u)
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
					t.Fatalf("draw %d: the drawn token %d sampled at %.7g, the chain's %.7g", i, got, pc, w)
				}
				smp.Observe(got)
				plain.Observe(got)
				if lg, err = st.Forward(got); err != nil {
					t.Fatal(err)
				}
			}
			if near > draws/8 || moved == 0 {
				t.Fatalf("%d of %d draws within rounding of an edge, %d moved by DRY: the comparison sees too little", near, draws, moved)
			}
			served, _ := st.Account(nn.CPU).Steps.Of(nn.ComposedSample)
			t.Logf("%d draws, %d near an edge, %d moved by DRY, %d composed samples served", draws, near, moved, served)
		})
	}
	t.Run("greedy", func(t *testing.T) {
		st := testInstance(t, m, len(prompt)+draws+8)
		defer st.Close()
		lg, err := st.Prefill(prompt)
		if err != nil {
			t.Fatal(err)
		}
		smp := Sampler{DryMultiplier: 1, DryBase: 1.75, DryAllowedLength: 2, DryPenaltyLastN: -1}
		smp.Prime(prompt)
		moved := 0
		for i := range draws {
			v := slices.Clone(lg)
			oracle.Dry(v, smp.dhist, 1, 1.75, 2, -1, st.MaxSeq(), breakers)
			want := Greedy(v)
			got, err := st.Sample(&smp, lg)
			if err != nil {
				t.Fatal(err)
			}
			if got != want {
				t.Fatalf("draw %d: the greedy token is %d, the largest of the row DRY leaves %d", i, got, want)
			}
			if want != Greedy(lg) {
				moved++
			}
			smp.Observe(got)
			if lg, err = st.Forward(got); err != nil {
				t.Fatal(err)
			}
		}
		if moved == 0 {
			t.Fatalf("DRY moved none of %d greedy tokens: the gate sees nothing", draws)
		}
		t.Logf("DRY moved %d of %d greedy tokens", moved, draws)
	})
}

// TestDryBreakersAreTheVocabularys: llama.cpp's default breakers read off
// stories15M's vocabulary: a token is a breaker alone exactly where its text
// holds one of them, and none of these one-byte breakers heads a tail.
func TestDryBreakersAreTheVocabularys(t *testing.T) {
	if _, err := os.Stat(models()["stories15M"]); err != nil {
		t.Fatalf("MODEL MISSING: %v", err)
	}
	m, err := model.Open(jlmOf(t, models()["stories15M"]))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	br := breakersOf(m.Vocab, dryBreakers())
	alone := 0
	for id := range int32(m.Vocab.Size()) {
		w := m.Vocab.Literal(id)
		holds := slices.ContainsFunc(dryBreakers(), func(s string) bool { return strings.Contains(w, s) })
		tails, ok := br[id]
		if holds != ok {
			t.Fatalf("token %d (%q): holds a breaker %v, read as one %v", id, w, holds, ok)
		}
		for _, tail := range tails {
			if len(tail) != 0 {
				t.Fatalf("token %d (%q) heads a tail %v of a one-byte breaker", id, w, tail)
			}
		}
		if ok {
			alone++
		}
	}
	if alone == 0 {
		t.Fatal("no token of the vocabulary is a breaker: the gate sees nothing")
	}
	t.Logf("%d of %d tokens are a breaker alone", alone, m.Vocab.Size())
}
