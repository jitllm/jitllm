package composer

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUExpAtItsEdges runs exp (graph.MapExp) on the CPU over a
// ragged row from -Inf to 88: at or below -88, -Inf among them, it is 0
// exactly, so a token a cut sends to -Inf keeps no probability; between -88
// and the smallest normal's logarithm at most that normal; above it within
// 2e-6 of math.Exp. The clamp at -87.3 it had, which made exp(-Inf) 2^-126,
// is the violation.
func TestComposedCPUExpAtItsEdges(t *testing.T) {
	x := []float32{float32(math.Inf(-1)), -1e30, -1000, -104, -88.8, -88.4, -88, -87.9, -87.6, -87.34, -87.3, -87,
		-60, -20.5, -1, -0.3, 0, 0.7, 1, 13, 44.25, 87.9, 88}
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	b := graph.NewBlock(-1)
	in := b.In("x", len(x))
	b.Out(b.Emit(graph.OpMap, []graph.ValueID{in}, []graph.Result{{Name: "exp", Width: len(x)}}, nil,
		graph.Attrs{Fn: graph.MapExp}, nil, nil)[0])
	_, codes, err := r.Form(Request{Graph: b.G, Step: Step{Rows: 1, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
	if err != nil {
		t.Fatal(err)
	}
	got, err := runCPUBound(hostArm(), codes, x, len(x), [3][]uint32{}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	const minNormal = 1.17549435e-38
	zeros := 0
	for i, v := range x {
		g, w := got[i], math.Exp(float64(v))
		switch {
		case v <= -88:
			if g != 0 {
				t.Fatalf("exp(%g) = %g, not 0", v, g)
			}
			zeros++
		case w < minNormal:
			if !(g >= 0 && g <= minNormal) {
				t.Fatalf("exp(%g) = %g, past the smallest normal (%g)", v, g, w)
			}
		default:
			if e := math.Abs(float64(g)-w) / w; !(e <= 2e-6) {
				t.Fatalf("exp(%g) = %g, %g from %g", v, g, e, w)
			}
		}
	}
	t.Logf("%d values, %d at or below -88 exactly 0", len(x), zeros)
}
