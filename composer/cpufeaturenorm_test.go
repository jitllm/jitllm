package composer

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUFeatureNorm holds graph.OpFeatureNorm to its definition on
// the CPU: nine live rows of twelve, three columns, each column over its
// first eight rows (Offset one) -- (x - mean) / (std + 1e-5), std over n-1
// -- and the rows from the eighth on zero. Statistics over every live row
// are the violation.
func TestComposedCPUFeatureNorm(t *testing.T) {
	const R, W, live, drop, eps = 12, 3, 9, 1, 1e-5
	x := make([]float32, R*W)
	for i := range x {
		x[i] = float32(math.Sin(0.7*float64(i)) * float64(1+i%5))
	}
	b := graph.NewBlock(-1)
	in := b.In("x", W)
	b.Out(b.Emit(graph.OpFeatureNorm, []graph.ValueID{in}, []graph.Result{{Name: "norm", Width: W}}, nil,
		graph.Attrs{Offset: drop, Eps: eps}, nil, nil)[0])
	codes := slotForm(t, b.G, R)
	pos := make([]uint32, R+1)
	pos[0] = live
	got, err := runCPUBound(hostArm(), codes, x, R*W, [3][]uint32{}, nil, func(o Operand) []uint32 {
		if o.Name == "rope.pos" {
			return pos
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	n := live - drop
	for col := range W {
		mean, sq := 0.0, 0.0
		for r := range n {
			mean += float64(x[r*W+col])
		}
		mean /= float64(n)
		for r := range n {
			d := float64(x[r*W+col]) - mean
			sq += d * d
		}
		std := math.Sqrt(sq / float64(n-1))
		for r := range R {
			want := 0.0
			if r < n {
				want = (float64(x[r*W+col]) - mean) / (std + eps)
			}
			if d := math.Abs(float64(got[r*W+col]) - want); math.IsNaN(d) || d > 1e-5*max(1, math.Abs(want)) {
				t.Fatalf("row %d column %d: %g, want %g", r, col, got[r*W+col], want)
			}
		}
	}
}
