package model

import (
	"math"
	"os"
	"sort"
	"testing"
)

// TestSpecDistIsTheDrawsDistribution holds Instance.dist -- the distribution
// speculative sampling accepts and resamples by -- to the one Sample draws
// from, in float64 (distRef): over a real model's logits, at temperatures
// either side of 1 with top-p and min-p cutting, with a top-k (within the
// walk's bound and past it) and without, every token's probability within
// float32's rounding. Cutting at the temperature instead (the rule the host's
// dist kept) is the violation: it must read off the definition somewhere, or
// this gate cannot tell the two apart.
func TestSpecDistIsTheDrawsDistribution(t *testing.T) {
	if _, err := os.Stat(models()["stories15M"]); err != nil {
		t.Fatalf("MODEL MISSING: %v", err)
	}
	m, err := Open(jlmOf(t, models()["stories15M"]))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	st := m.NewInstance(32)
	defer st.Close()
	lg, err := st.Prefill([]int32{1, 450, 3014, 471, 263})
	if err != nil {
		t.Fatal(err)
	}
	apart := false
	for _, smp := range []Sampler{
		{Temp: 0.6, TopK: 6, TopP: 0.8},
		{Temp: 1.4, TopK: 40, TopP: 0.9, MinP: 0.02},
		{Temp: 0.7, TopP: 0.85},
		{Temp: 1.3, MinP: 0.05},
		{Temp: 0.5, TopK: 600, TopP: 0.9},
	} {
		served0, _ := st.CPUSampleSteps()
		var d specDist
		st.dist(&smp, lg, &d)
		if served, _ := st.CPUSampleSteps(); served != served0+1 {
			t.Fatalf("%+v: the composed distribution was not served (%d to %d)", smp, served0, served)
		}
		want, margin := distRef(lg, &smp, false)
		if margin < 1e-4 {
			t.Fatalf("%+v: a cut within %.3g of flipping: pick another setting", smp, margin)
		}
		wrong, _ := distRef(lg, &smp, true)
		for x := range int32(m.Cfg.NVocab) {
			if got := float64(d.prob(x)); math.Abs(got-want[x]) > 1e-5*want[x]+1e-7 {
				t.Fatalf("%+v: p(%d) %.9g, the draw's %.9g", smp, x, got, want[x])
			}
			if math.Abs(wrong[x]-want[x]) > 1e-3 {
				apart = true
			}
		}
	}
	if !apart {
		t.Fatal("cutting at the temperature reads as cutting at 1 everywhere: the gate cannot see it")
	}
}

// distRef is the distribution Sample's draw draws from over logits, every
// token's probability in float64, the penalty off: the candidates (the k
// best, or the row), min-p and top-p over their softmax at temperature 1 (at
// the temperature with atT, the violation), the best kept, then each
// survivor's share of the survivors' softmax at the temperature, or of the
// row's where no top-k is set and no cut fires. It returns it and how far
// the cuts are from flipping, which is the whole margin of every
// probability.
func distRef(logits []float32, smp *Sampler, atT bool) (p []float64, margin float64) {
	n := len(logits)
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return logits[order[a]] > logits[order[b]] })
	cand := order
	if smp.TopK > 0 && smp.TopK < n {
		cand = order[:smp.TopK]
	}
	invT := float64(float32(1 / smp.Temp))
	cutT := 1.0
	if atT {
		cutT = invT
	}
	v0 := float64(logits[cand[0]])
	q := make([]float64, len(cand))
	sum := 0.0
	for j, i := range cand {
		q[j] = math.Exp((float64(logits[i]) - v0) * cutT)
		sum += q[j]
	}
	topp, minp := math.Inf(1), smp.MinP
	if smp.TopP > 0 && smp.TopP < 1 {
		topp = smp.TopP
	}
	keep, fired, acc := len(cand), false, 0.0
	margin = math.Inf(1)
	for j := range cand {
		q[j] /= sum
		if minp > 0 {
			margin = min(margin, math.Abs(q[j]-minp*q[0])/q[0])
		}
		if q[j] < minp*q[0] {
			keep, fired = j, true
			break
		}
		acc += q[j]
		if !math.IsInf(topp, 1) {
			margin = min(margin, math.Abs(acc-topp))
		}
		if acc >= topp {
			keep, fired = j+1, true
			break
		}
	}
	surv := cand[:max(keep, 1)]
	if len(cand) == n && !fired {
		surv = order
	}
	p = make([]float64, n)
	mass := 0.0
	for _, i := range surv {
		p[i] = math.Exp((float64(logits[i]) - v0) * invT)
		mass += p[i]
	}
	for _, i := range surv {
		p[i] /= mass
	}
	return p, margin
}
