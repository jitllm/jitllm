package composer

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// windowWords is hist as the forms read the repeat window (sample.window):
// each of n tokens' count in it, a float's bits.
func windowWords(hist []int, n int) []uint32 {
	cnt := make([]float32, n)
	for _, h := range hist {
		cnt[h]++
	}
	w := make([]uint32, n)
	for i, x := range cnt {
		w[i] = math.Float32bits(x)
	}
	return w
}

// TestComposedCPUSampleRow holds graph.SampleRow on the CPU device to its
// definition, the host sampler's draw with no top-k and no cut: over random
// logits (some with values repeated, so the order's ties fall to the id),
// temperatures, repeat penalties over a history with its ids repeated, and
// uniforms (0 among them), the token of every draw whose point clears each
// edge of the cumulative mass by a margin float32 cannot cross.
func TestComposedCPUSampleRow(t *testing.T) {
	const n, H = 300, 16
	g := (&graph.SampleRow{NVocab: n}).Build()
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
	draw := func(logits []float32, hist []int, pen, invT, u float32) uint32 {
		lw := make([]uint32, n)
		for i, v := range logits {
			lw[i] = math.Float32bits(v)
		}
		win := windowWords(hist, n)
		vals := map[graph.ValueID][]uint32{
			in["logits"]: lw, in["sample.window"]: win, in["sample.penalty"]: {math.Float32bits(pen)},
			in["sample.ids"]: append([]uint32(nil), ids...), in["sample.row"]: {math.Float32bits(invT), math.Float32bits(u)},
			tokV: {0},
		}
		if err := runCPUValues(codes, g, vals, [3][]uint32{}); err != nil {
			t.Fatal(err)
		}
		return vals[tokV][0]
	}
	rng := rand.New(rand.NewSource(31))
	decisive, cases := 0, 0
	for c := range 400 {
		l := make([]float32, n)
		for i := range l {
			l[i] = float32(rng.NormFloat64() * 3)
		}
		if c%4 == 0 {
			// Values repeated: ties the order breaks by id.
			for i := 0; i < n; i += 3 {
				l[i] = l[(i*7)%n]
			}
		}
		masked := c%5 == 1
		if masked {
			// A grammar's mask: some tokens at -inf, which no draw may land on.
			for i := range l {
				if rng.Intn(20) == 0 {
					l[i] = float32(math.Inf(-1))
				}
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
		u := float64(float32(rng.Float64()))
		if c%50 == 0 {
			u = 0
		}
		want, margin := oracle.SampleRow(l, hist, pen, invT, u)
		got := draw(l, hist, pen, invT, float32(u))
		if masked && math.IsInf(float64(l[got]), -1) {
			t.Fatalf("case %d: drew token %d, whose logit is -inf", c, got)
		}
		cases++
		if margin < 1e-4 && c%50 != 0 {
			continue
		}
		decisive++
		if got != want {
			t.Fatalf("case %d (history %v at %.3f, 1/T %.3f, u %.6f): drew %d, want %d (margin %.3g)", c, hist, pen, invT, u, got, want, margin)
		}
	}
	if decisive < cases*9/10 {
		t.Fatalf("%d of %d draws decisive: the comparison sees too little", decisive, cases)
	}
	t.Logf("%d of %d row draws decisive and right", decisive, cases)
}
