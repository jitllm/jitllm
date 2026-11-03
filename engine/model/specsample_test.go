//go:build amd64 || arm64

package model

import (
	"math"
	"testing"
)

// TestSpecSamplingKeepsTheDistribution is the sampled contract: speculative
// rejection sampling must leave every token distributed as the trunk's own
// sampler would draw it, whatever the draft proposes. A token-equality gate
// cannot say that -- the two arms consume the random stream differently -- so
// this draws the second and third generated tokens many times from each arm
// and compares the empirical distributions by total variation.
//
// The fixture's prediction block has random weights, so its own proposals are
// far from the trunk's distribution and mostly reject: the residual draw, the
// acceptance ratio and the rollback all run. The violation -- taking every
// draft without the ratio test -- leaves the block's distribution in the third
// token, and must read far above the noise floor there. It runs at
// temperature 1 and at 0.6 with a top-p that cuts. Its floor is too wide to
// see a distribution cut at the wrong temperature (the host's dist read 0.13
// under a 0.48 floor): TestSpecDistIsTheDrawsDistribution holds that.
func TestSpecSamplingKeepsTheDistribution(t *testing.T) {
	m := openSpecModel(t, "synth-qwen35-hybrid-mtp.gguf", noTune(), WithKVF16(false))
	defer m.Close()
	for _, c := range []struct {
		name       string
		temp, topP float64
	}{{"T1", 1, 0.95}, {"T0.6 top-p 0.8", 0.6, 0.8}} {
		t.Run(c.name, func(t *testing.T) { specKeepsTheDistribution(t, m, c.temp, c.topP) })
	}
}

func specKeepsTheDistribution(t *testing.T, m *Model, temp, topP float64) {
	prompt := m.Vocab.Encode("The capital of France is", true)
	// top-k keeps each distribution on a handful of tokens, so a few thousand
	// draws measure it; min-p and top-p run in the dist path too.
	const draws, topK = 1500, 6
	samp := func(seed int64) *Sampler {
		return &Sampler{Temp: temp, TopK: topK, TopP: topP, Seed: seed}
	}
	plain := func(seed int64) []int32 {
		st := m.NewInstance(len(prompt) + 8)
		defer st.Close()
		sm := samp(seed)
		lg, err := st.Prefill(prompt)
		if err != nil {
			t.Fatal(err)
		}
		var out []int32
		for len(out) < 3 {
			y, err := st.Sample(sm, lg)
			if err != nil {
				t.Fatal(err)
			}
			sm.Observe(y)
			out = append(out, y)
			if lg, err = st.Forward(y); err != nil {
				t.Fatal(err)
			}
		}
		return out
	}
	// The trunk's own logits after the prompt and ctx: what the forced draft
	// distribution proposes from, so a draft is accepted with probability
	// one where it is not corrupted (Speculator.qOracle). The first round
	// drafts two: the first from the trunk (accepted, and drawn exactly as
	// plain sampling draws the second token, so that comparison is exact by
	// construction), the second from the block itself (corrupt 2), whose
	// acceptance ratio and residual decide the third token -- the one a
	// violation must move.
	trunkAfter := func(ctx []int32) []float32 {
		st := m.NewInstance(len(prompt) + len(ctx) + 2)
		defer st.Close()
		lg, err := st.Prefill(append(append([]int32(nil), prompt...), ctx...))
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), lg...)
	}
	spec := func(seed int64, fault specFault) ([]int32, SpecStats) {
		st := m.NewInstance(len(prompt) + 16)
		defer st.Close()
		sp, err := st.Speculate(WithSpecDraft(2))
		if err != nil {
			t.Fatal(err)
		}
		defer sp.Close()
		var out []int32
		sp.fault, sp.corrupt = fault, 2
		sp.qOracle = func(drafts []int32) []float32 {
			return trunkAfter(append(append([]int32(nil), out...), drafts...))
		}
		sm := samp(seed)
		y, err := sp.Start(prompt, sm)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, y)
		for len(out) < 3 {
			ys, err := sp.Next(sm)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, ys...)
		}
		return out[:3], sp.Stats()
	}
	// The distributions of the second and third tokens, as counts.
	hist := func(f func(int64) []int32) [2]map[int32]int {
		h := [2]map[int32]int{{}, {}}
		for i := int64(0); i < draws; i++ {
			out := f(i + 1)
			h[0][out[1]]++
			h[1][out[2]]++
		}
		return h
	}
	tv := func(a, b map[int32]int) float64 {
		keys := map[int32]bool{}
		for k := range a {
			keys[k] = true
		}
		for k := range b {
			keys[k] = true
		}
		var d float64
		for k := range keys {
			d += math.Abs(float64(a[k]-b[k])) / draws
		}
		return d / 2
	}
	want := hist(plain)
	var rounds, drafted, accepted int64
	got := hist(func(seed int64) []int32 {
		out, st := spec(seed, specFaultNone)
		rounds, drafted, accepted = rounds+st.Rounds, drafted+st.Drafted, accepted+st.Accepted
		return out
	})
	bad := hist(func(seed int64) []int32 {
		out, _ := spec(seed, faultAcceptUnchecked)
		return out
	})
	// Two empirical distributions of the same law over k outcomes differ in
	// total variation by about sqrt(k/(2*pi*n)) each way; three times the
	// combined spread is the noise floor. k is the support the draws found:
	// the third token mixes the second's six continuations.
	for i := 0; i < 2; i++ {
		support := map[int32]bool{}
		for _, h := range []map[int32]int{got[i], want[i]} {
			for k := range h {
				support[k] = true
			}
		}
		floor := 3 * math.Sqrt(2*float64(len(support))/(2*math.Pi*draws))
		clean, broken := tv(got[i], want[i]), tv(bad[i], want[i])
		t.Logf("token %d: TV %.4f against plain sampling (floor %.4f); accepting unchecked %.4f",
			i+2, clean, floor, broken)
		if clean > floor {
			t.Errorf("token %d: speculative sampling is %.4f from plain sampling in total variation, "+
				"over the %.4f noise floor", i+2, clean, floor)
		}
		if i == 1 && broken <= floor {
			t.Errorf("token %d: accepting drafts unchecked reads %.4f, inside the floor -- the gate "+
				"cannot see the violation", i+2, broken)
		}
	}
	if drafted == 0 || accepted == 0 || accepted == drafted {
		t.Fatalf("%d drafted, %d accepted over %d rounds: the rejection path must run and so must "+
			"acceptance", drafted, accepted, rounds)
	}
	t.Logf("%d rounds, %d drafted, %d accepted", rounds, drafted, accepted)
}
