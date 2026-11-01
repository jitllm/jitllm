package jit

import (
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// SampleDrawRef is graph.SampleDraw's definition with no penalty, in
// float64: the K best logits (the lower id on a tie), the first k of them,
// top-p and then min-p over their softmax at temperature 1 with the best
// always kept, then the Gumbel-max at 1/T over UniformRef's uniforms keyed
// by each candidate's id. It returns the token, the kept candidates' ids
// and the margin of the best score over the next.
func SampleDrawRef(logits []float32, K, k int, invT, topp, minp float64, seed, pos uint32) (tok uint32, kept []uint32, margin float64) {
	order := make([]int, len(logits))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return logits[order[a]] > logits[order[b]] })
	K = min(K, len(logits))
	ids := make([]uint32, K)
	for i := range ids {
		ids[i] = uint32(order[i])
	}
	v0 := float64(logits[ids[0]])
	p := make([]float64, K)
	sum := 0.0
	for i := range min(k, K) {
		p[i] = math.Exp(float64(logits[ids[i]]) - v0)
		sum += p[i]
	}
	u := UniformRef(ids, seed, pos, 10)
	best, second := math.Inf(-1), math.Inf(-1)
	before := 0.0
	for i := range K {
		p[i] /= sum
		keep := i < k && before < topp && p[i] >= minp*p[0]
		before += p[i]
		if i == 0 {
			keep = true
		}
		if !keep {
			continue
		}
		kept = append(kept, ids[i])
		s := (float64(logits[ids[i]])-v0)*invT - math.Log(-math.Log(float64(u[i])))
		switch {
		case s > best:
			best, second, tok = s, best, ids[i]
		case s > second:
			second = s
		}
	}
	return tok, kept, best - second
}

// SampleDrawGraph is graph.SampleDraw over n logits, k candidates and a
// history of h, with its input values by name.
func SampleDrawGraph(n, k, h int) (*graph.Graph, map[string]graph.ValueID) {
	g := (&graph.SampleDraw{SampleTopK: graph.SampleTopK{NVocab: n, K: k, Hist: h}}).Build()
	names := map[string]graph.ValueID{}
	for _, v := range g.In {
		names[g.Values[v].Name] = v
	}
	names["token"] = g.Out[0]
	return g, names
}

// TestComposedCPUSampleDraw holds graph.SampleDraw on the CPU device to its
// definition: the token of every draw whose best score clears the next by
// a margin float32 cannot cross, over random logits, k, temperatures and
// cuts; and, as a sampler, each kept token drawn over many positions in the
// proportion its tempered softmax over the survivors gives it.
func TestComposedCPUSampleDraw(t *testing.T) {
	const n, K, H = 300, 16, 8
	g, in := SampleDrawGraph(n, K, H)
	if err := graph.Verify(g, graph.Facts{}); err != nil {
		t.Fatal(err)
	}
	_, codes, err := Default().Form(Request{Graph: g, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: 1, Page: 64, Keys: 1}, Device: cpuDevice()})
	if err != nil {
		t.Fatal(err)
	}
	args := make([]uint32, 2+H)
	args[1] = math.Float32bits(1)
	for i := 2; i < len(args); i++ {
		args[i] = n
	}
	draw := func(logits []float32, k int, invT, topp, minp float32, seed, pos uint32) uint32 {
		lw := make([]uint32, n)
		for i, v := range logits {
			lw[i] = math.Float32bits(v)
		}
		vals := map[graph.ValueID][]uint32{
			in["logits"]: lw, in["sample.args"]: append([]uint32(nil), args...), in["sample.penalty"]: {math.Float32bits(1)},
			in["sample.draw"]:    {math.Float32bits(float32(k)), math.Float32bits(invT), math.Float32bits(topp), math.Float32bits(minp)},
			in["sample.counter"]: {seed, pos}, in["token"]: {0},
		}
		if err := runCPUValues(codes, g, vals, [3][]uint32{}); err != nil {
			t.Fatal(err)
		}
		return vals[in["token"]][0]
	}
	rng := rand.New(rand.NewSource(23))
	logits := func() []float32 {
		l := make([]float32, n)
		for i := range l {
			l[i] = float32(rng.NormFloat64() * 3)
		}
		return l
	}
	decisive, cases := 0, 0
	for c := range 400 {
		l := logits()
		k := 1 + rng.Intn(K)
		invT := 1 / (0.3 + 1.5*rng.Float64())
		topp, minp := math.Inf(1), 0.0
		if c%3 != 0 {
			topp = 0.3 + 0.7*rng.Float64()
		}
		if c%2 == 0 {
			minp = 0.1 * rng.Float64()
		}
		seed, pos := rng.Uint32(), rng.Uint32()
		want, _, margin := SampleDrawRef(l, K, k, invT, topp, minp, seed, pos)
		got := draw(l, k, float32(invT), float32(topp), float32(minp), seed, pos)
		cases++
		if margin < 1e-3 {
			continue
		}
		decisive++
		if got != want {
			t.Fatalf("case %d (k %d, 1/T %.3f, top-p %.3f, min-p %.3f): drew %d, want %d (margin %.3g)", c, k, invT, topp, minp, got, want, margin)
		}
	}
	if decisive < cases*9/10 {
		t.Fatalf("%d of %d draws decisive: the comparison sees too little", decisive, cases)
	}

	// As a sampler: one set of logits drawn at many positions.
	l := logits()
	const k, invT, topp, minp, draws = 12, 1 / 1.3, 0.9, 0.01, 4000
	_, kept, _ := SampleDrawRef(l, K, k, invT, topp, minp, 0, 0)
	v0 := float64(l[kept[0]])
	want, sum := map[uint32]float64{}, 0.0
	for _, id := range kept {
		want[id] = math.Exp((float64(l[id]) - v0) * invT)
		sum += want[id]
	}
	got := map[uint32]int{}
	for pos := range uint32(draws) {
		got[draw(l, k, float32(invT), float32(topp), float32(minp), 99, pos)]++
	}
	for id, c := range got {
		if _, ok := want[id]; !ok {
			t.Fatalf("drew %d %d times, outside the survivors %v", id, c, kept)
		}
	}
	for id, w := range want {
		f, e := float64(got[id])/draws, w/sum
		if math.Abs(f-e) > 4*math.Sqrt(e*(1-e)/draws)+1e-3 {
			t.Errorf("token %d drawn %.4f of the time, its tempered share %.4f", id, f, e)
		}
	}
	t.Logf("%d of %d draws decisive and right; %d survivors drawn in their tempered shares over %d positions", decisive, cases, len(kept), draws)
}
