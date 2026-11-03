package composer

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/oracle"
)

// TestComposedCPUSampleCut holds graph.SampleCut on the CPU device to its
// definition, the host sampler's draw with no top-k and a cut: over random
// logits (some with values repeated), temperatures, penalties, top-p (off
// among them), min-p (off, and past 1, where only the best survives) and
// uniforms, the token of every draw no float32 rounding can flip.
func TestComposedCPUSampleCut(t *testing.T) {
	const n, H = 300, 16
	g := (&graph.SampleCut{NVocab: n}).Build()
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
	ids := make([]uint32, n)
	for i := range ids {
		ids[i] = uint32(i)
	}
	draw := func(logits []float32, hist []int, pen, invT, topp, minp, u float32) uint32 {
		lw := make([]uint32, n)
		for i, v := range logits {
			lw[i] = math.Float32bits(v)
		}
		win := windowWords(hist, n)
		vals := map[graph.ValueID][]uint32{
			in["logits"]: lw, in["sample.window"]: win, in["sample.penalty"]: {math.Float32bits(pen)},
			in["sample.ids"]: append([]uint32(nil), ids...),
			in["sample.cut"]: {math.Float32bits(invT), math.Float32bits(topp), math.Float32bits(minp), math.Float32bits(u)},
			tokV:             {0},
		}
		if err := runCPUValues(codes, g, vals, [3][]uint32{}); err != nil {
			t.Fatal(err)
		}
		return vals[tokV][0]
	}
	rng := rand.New(rand.NewSource(37))
	decisive, cases := 0, 0
	for c := range 500 {
		l := make([]float32, n)
		for i := range l {
			l[i] = float32(rng.NormFloat64() * 3)
		}
		if c%4 == 0 {
			for i := 0; i < n; i += 3 {
				l[i] = l[(i*7)%n]
			}
		}
		var hist []int
		pen := float32(1)
		if c%2 == 0 {
			pen = 1.1 + 0.5*rng.Float32()
			for range rng.Intn(H + 1) {
				hist = append(hist, rng.Intn(n/10))
			}
		}
		invT := float32(1 / (0.4 + 1.4*rng.Float64()))
		topp, minp := math.Inf(1), 0.0
		switch c % 5 {
		case 0, 1:
			topp = float64(float32(0.3 + 0.69*rng.Float64()))
		case 2:
			minp = float64(float32(0.2 * rng.Float64()))
		case 3:
			topp, minp = float64(float32(0.5+0.49*rng.Float64())), float64(float32(0.1*rng.Float64()))
		}
		if c%37 == 0 {
			minp = 1.5
		}
		u := float64(float32(rng.Float64()))
		if c%50 == 0 {
			u = 0
		}
		want, margin := oracle.SampleCut(l, hist, pen, invT, topp, minp, u)
		got := draw(l, hist, pen, invT, float32(topp), float32(minp), float32(u))
		cases++
		if margin < 1e-4 {
			continue
		}
		decisive++
		if got != want {
			t.Fatalf("case %d (history %v at %.3f, 1/T %.3f, top-p %.4f, min-p %.4f, u %.6f): drew %d, want %d (margin %.3g)",
				c, hist, pen, invT, topp, minp, u, got, want, margin)
		}
	}
	if decisive < cases*8/10 {
		t.Fatalf("%d of %d draws decisive: the comparison sees too little", decisive, cases)
	}
	t.Logf("%d of %d cut draws decisive and right", decisive, cases)
}

// TestComposedCPUSampleCutTopK holds graph.SampleCut with TopK on the CPU
// device to the host sampler's top-k draw at any k (SampleWalkRef over the
// whole row, the penalty applied first): the k best, top-p and min-p over
// their softmax at temperature 1, the walk over the survivors at 1/T, the
// token of every draw no float32 rounding can flip.
func TestComposedCPUSampleCutTopK(t *testing.T) {
	const n, H = 300, 16
	g := (&graph.SampleCut{NVocab: n, TopK: true}).Build()
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
	ids := make([]uint32, n)
	for i := range ids {
		ids[i] = uint32(i)
	}
	f := math.Float32bits
	rng := rand.New(rand.NewSource(43))
	decisive, cases := 0, 0
	for c := range 300 {
		l := make([]float32, n)
		for i := range l {
			l[i] = float32(rng.NormFloat64() * 3)
		}
		var hist []int
		pen := float32(1)
		if c%2 == 0 {
			pen = 1.1 + 0.5*rng.Float32()
			for range rng.Intn(H + 1) {
				hist = append(hist, rng.Intn(n/10))
			}
		}
		v := append([]float32(nil), l...)
		oracle.PenalizeOnce(v, hist, pen)
		k := 2 + rng.Intn(n-2)
		invT := float32(1 / (0.4 + 1.4*rng.Float64()))
		topp, minp := math.Inf(1), 0.0
		switch c % 4 {
		case 0:
			topp = float64(float32(0.3 + 0.69*rng.Float64()))
		case 1:
			minp = float64(float32(0.05 * rng.Float64()))
		case 2:
			topp, minp = float64(float32(0.5+0.49*rng.Float64())), float64(float32(0.02*rng.Float64()))
		}
		u := float64(float32(rng.Float64()))
		want, margin := oracle.SampleWalk(v, n, k, float64(invT), topp, minp, u)
		lw := make([]uint32, n)
		for i, x := range l {
			lw[i] = f(x)
		}
		win := windowWords(hist, n)
		vals := map[graph.ValueID][]uint32{
			in["logits"]: lw, in["sample.window"]: win, in["sample.penalty"]: {f(pen)},
			in["sample.ids"]: append([]uint32(nil), ids...),
			in["sample.cut"]: {f(invT), f(float32(topp)), f(float32(minp)), f(float32(u)), f(float32(k))},
			tokV:             {0},
		}
		if err := runCPUValues(codes, g, vals, [3][]uint32{}); err != nil {
			t.Fatal(err)
		}
		cases++
		if margin < 1e-4 {
			continue
		}
		decisive++
		if got := vals[tokV][0]; got != want {
			t.Fatalf("case %d (k %d, history %v at %.3f, 1/T %.3f, top-p %.4f, min-p %.4f, u %.6f): drew %d, want %d (margin %.3g)",
				c, k, hist, pen, invT, topp, minp, u, got, want, margin)
		}
	}
	if decisive < cases*8/10 {
		t.Fatalf("%d of %d draws decisive: the comparison sees too little", decisive, cases)
	}
	t.Logf("%d of %d top-k draws decisive and right", decisive, cases)
}
