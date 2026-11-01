package jit

import (
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// SampleWalkRef is graph.SampleWalk's definition with no penalty, in
// float64, which is the host sampler's top-k draw (model.Sampler): the K best
// logits (the lower id on a tie), the first k of them, top-p and then min-p
// over their softmax at temperature 1 with the best always kept, then the
// walk at u over the survivors' softmax at 1/T -- the first whose mass up to
// and including it passes u times their mass, the last where none does. It
// returns the token and how far u's point is from the nearest edge, as a
// share of the mass.
func SampleWalkRef(logits []float32, K, k int, invT, topp, minp, u float64) (tok uint32, margin float64) {
	order := make([]int, len(logits))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return logits[order[a]] > logits[order[b]] })
	K = min(K, len(logits))
	v0 := float64(logits[order[0]])
	p := make([]float64, K)
	sum := 0.0
	for i := range min(k, K) {
		p[i] = math.Exp(float64(logits[order[i]]) - v0)
		sum += p[i]
	}
	var kept []int
	before := 0.0
	for i := range K {
		p[i] /= sum
		if i == 0 || i < k && before < topp && p[i] >= minp*p[0] {
			kept = append(kept, i)
		}
		before += p[i]
	}
	et := make([]float64, len(kept))
	mass := 0.0
	for j, i := range kept {
		et[j] = math.Exp((float64(logits[order[i]]) - v0) * invT)
		mass += et[j]
	}
	target, upto := u*mass, 0.0
	margin = math.Inf(1)
	tok = uint32(order[kept[len(kept)-1]])
	landed := false
	for j, i := range kept {
		upto += et[j]
		margin = min(margin, math.Abs(upto-target)/mass)
		if !landed && upto > target {
			tok, landed = uint32(order[i]), true
		}
	}
	return tok, margin
}

// TestComposedCPUSampleWalk holds graph.SampleWalk on the CPU device to its
// definition, the host sampler's top-k draw: the token of every draw whose
// point clears each edge of the survivors' cumulative mass by a margin
// float32 cannot cross, over random logits, k, temperatures, cuts and
// uniforms, u at 0 and just under 1 among them (u at 0 held whatever its
// margin: it lands on the best).
func TestComposedCPUSampleWalk(t *testing.T) {
	const n, K, H = 300, 16, 8
	g := (&graph.SampleWalk{SampleTopK: graph.SampleTopK{NVocab: n, K: K, Hist: H}}).Build()
	if err := graph.Verify(g, graph.Facts{}); err != nil {
		t.Fatal(err)
	}
	in := map[string]graph.ValueID{}
	for _, v := range g.In {
		in[g.Values[v].Name] = v
	}
	tokV := g.Out[0]
	_, codes, err := Default().Form(Request{Graph: g, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: 1, Page: 64, Keys: 1}, Device: cpuDevice()})
	if err != nil {
		t.Fatal(err)
	}
	args := make([]uint32, 2+H)
	args[1] = math.Float32bits(1)
	for i := 2; i < len(args); i++ {
		args[i] = n
	}
	walk := func(logits []float32, k int, invT, topp, minp, u float32) uint32 {
		lw := make([]uint32, n)
		for i, v := range logits {
			lw[i] = math.Float32bits(v)
		}
		vals := map[graph.ValueID][]uint32{
			in["logits"]: lw, in["sample.args"]: append([]uint32(nil), args...), in["sample.penalty"]: {math.Float32bits(1)},
			in["sample.walk"]: {math.Float32bits(float32(k)), math.Float32bits(invT), math.Float32bits(topp), math.Float32bits(minp), math.Float32bits(u)},
			tokV:              {0},
		}
		if err := runCPUValues(codes, g, vals, [3][]uint32{}); err != nil {
			t.Fatal(err)
		}
		return vals[tokV][0]
	}
	rng := rand.New(rand.NewSource(29))
	decisive, cases := 0, 0
	for c := range 600 {
		l := make([]float32, n)
		for i := range l {
			l[i] = float32(rng.NormFloat64() * 3)
		}
		k := 1 + rng.Intn(K)
		invT := 1 / (0.3 + 1.5*rng.Float64())
		topp, minp := math.Inf(1), 0.0
		if c%3 != 0 {
			topp = 0.3 + 0.7*rng.Float64()
		}
		if c%2 == 0 {
			minp = 0.1 * rng.Float64()
		}
		u := float64(float32(rng.Float64()))
		switch c % 50 {
		case 0:
			u = 0
		case 1:
			u = float64(math.Nextafter32(1, 0))
		}
		want, margin := SampleWalkRef(l, K, k, invT, topp, minp, u)
		got := walk(l, k, float32(invT), float32(topp), float32(minp), float32(u))
		cases++
		if margin < 1e-4 && c%50 != 0 {
			continue
		}
		decisive++
		if got != want {
			t.Fatalf("case %d (k %d, 1/T %.3f, top-p %.3f, min-p %.3f, u %.6f): drew %d, want %d (margin %.3g)", c, k, invT, topp, minp, u, got, want, margin)
		}
	}
	if decisive < cases*9/10 {
		t.Fatalf("%d of %d draws decisive: the comparison sees too little", decisive, cases)
	}
	t.Logf("%d of %d walks decisive and right", decisive, cases)
}
