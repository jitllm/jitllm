package composer

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// TestComposedCPUPenaltiesFollowLlamaCpp holds the sampler's penalized row
// on the CPU device to llama.cpp's penalties (oracle.Penalize), bit for bit:
// over random logits and windows, the repeat penalty once a windowed token,
// then frequency times its count plus presence subtracted, each alone and
// together. Its violation, the two subtracted penalties swapped, is seen.
func TestComposedCPUPenaltiesFollowLlamaCpp(t *testing.T) {
	const n = 64
	g := (&graph.SampleTopK{NVocab: n, K: n}).Build()
	if err := graph.Verify(g, graph.Facts{}); err != nil {
		t.Fatal(err)
	}
	in := map[string]graph.ValueID{}
	for _, v := range g.In {
		in[g.Values[v].Name] = v
	}
	valsV, idsV := g.Out[0], g.Out[1]
	_, codes, err := Default().Form(Request{Graph: g, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: 1, Page: 64, Keys: 1}, Device: cpuDevice(hostArm())})
	if err != nil {
		t.Fatal(err)
	}
	f := math.Float32bits
	rng := rand.New(rand.NewSource(41))
	swapSeen := false
	for c := range 300 {
		l := make([]float32, n)
		lw := make([]uint32, n)
		for i := range l {
			l[i] = float32(rng.NormFloat64() * 4)
			lw[i] = f(l[i])
		}
		hist := make([]int, rng.Intn(24))
		win := make([]uint32, n)
		counts := make([]float32, n)
		for i := range hist {
			hist[i] = rng.Intn(n / 2)
			counts[hist[i]]++
		}
		for i, x := range counts {
			win[i] = f(x)
		}
		pen := [3]float32{1 + 0.6*rng.Float32(), 2*rng.Float32() - 0.5, 1.5 * rng.Float32()}
		switch c % 4 {
		case 0:
			pen[1] = 0
		case 1:
			pen[2] = 0
		case 2:
			pen[0] = 1
		}
		vals := map[graph.ValueID][]uint32{
			in["logits"]: lw, in["sample.window"]: win, in["sample.penalty"]: {f(pen[0]), f(pen[1]), f(pen[2])},
			valsV: make([]uint32, n), idsV: make([]uint32, n),
		}
		if err := runCPUValues(hostArm(), codes, g, vals, [3][]uint32{}); err != nil {
			t.Fatal(err)
		}
		want := append([]float32(nil), l...)
		oracle.Penalize(want, hist, pen[0], pen[1], pen[2])
		swapped := append([]float32(nil), l...)
		oracle.Penalize(swapped, hist, pen[0], pen[2], pen[1])
		for i := range n {
			id := vals[idsV][i]
			if vals[valsV][i] != f(want[id]) {
				t.Fatalf("case %d (penalties %v): token %d is %v, want %v", c, pen, id, math.Float32frombits(vals[valsV][i]), want[id])
			}
			swapSeen = swapSeen || vals[valsV][i] != f(swapped[id])
		}
	}
	if !swapSeen {
		t.Fatal("presence and frequency swapped give the same rows: the gate cannot tell them apart")
	}
}
