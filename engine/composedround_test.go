package engine

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPURoundings holds the composer's floor, ceil, truncation and
// rounding half to even to math's, exactly, over halves, near-integers,
// negatives, zero's signs and floats at 2^23 and past (every one an integer
// already); and the binary32 rounding to the value itself. Each is held
// against another of them: an input set on which floor and ceil, or the two
// roundings, agree proves neither.
func TestComposedCPURoundings(t *testing.T) {
	xs := []float32{0, float32(math.Copysign(0, -1)), 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.49999997, 0.50000006,
		1, -1, 3.7, -3.7, 2.2, -2.2, 0x1p23, -0x1p23, 0x1p23 + 1, 0x1p24 + 2, 0x1p30, -0x1p30, 1e-30, -1e-30,
		1 - 0x1p-24, -1 + 0x1p-24, 1e9, -1e9}
	for _, x := range sweep(5, 40) {
		xs = append(xs, x, -x, x+0.5, -x-0.5)
	}
	for _, c := range []struct {
		fn    graph.MapFn
		ref   func(float64) float64
		other func(float64) float64
	}{
		{graph.MapFloor, math.Floor, math.Ceil},
		{graph.MapCeil, math.Ceil, math.Floor},
		{graph.MapTrunc, math.Trunc, math.Floor},
		{graph.MapRoundEven, math.RoundToEven, math.Round},
		{graph.MapF32, func(x float64) float64 { return x }, math.Floor},
	} {
		got := unary(t, c.fn, xs)
		parts := 0
		for i, x := range xs {
			want := float32(c.ref(float64(x)))
			if math.Float32bits(got[i]) != math.Float32bits(want) && !(got[i] == 0 && want == 0) {
				t.Errorf("%v(%v) is %v, want %v", c.fn, x, got[i], want)
			}
			if float32(c.other(float64(x))) != want {
				parts++
			}
		}
		if parts == 0 {
			t.Errorf("%v: the inputs never part it from its neighbour: this proves nothing", c.fn)
		}
		t.Logf("%v: %d inputs exact, %d of them where its neighbour differs", c.fn, len(xs), parts)
	}
}
