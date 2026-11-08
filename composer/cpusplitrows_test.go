package composer

import (
	"testing"

	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUSplitRows holds graph.OpSplitRows to its definition on the
// CPU: five rows of twelve split into rows of four (fifteen), then of one
// (sixty), each element where OpFold's inverse puts it. A split that read
// the input's rows transposed is the violation.
func TestComposedCPUSplitRows(t *testing.T) {
	const R, W = 5, 12
	x := make([]float32, R*W)
	for i := range x {
		x[i] = float32(i) + 0.5
	}
	for _, w := range []int{4, 1} {
		b := graph.NewBlock(-1)
		in := b.In("x", W)
		b.Out(b.Emit(graph.OpSplitRows, []graph.ValueID{in}, []graph.Result{{Name: "cells", Width: w, Rows: R * (W / w)}}, nil,
			graph.Attrs{Width: w}, nil, nil)[0])
		got, err := runCPUBound(hostArm(), slotForm(t, b.G, R), x, R*W, [3][]uint32{}, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		for r := range R * (W / w) {
			for e := range w {
				want := x[(r/(W/w))*W+(r%(W/w))*w+e]
				if g := got[r*w+e]; g != want {
					t.Fatalf("width %d: row %d element %d: %g, want %g", w, r, e, g, want)
				}
			}
		}
	}
}
