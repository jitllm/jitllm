package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
)

// TestComposedCPULog holds the composer's logarithm to math.Log within four
// units of the result's last place over the sweep's magnitudes, subnormals
// and the largest float among them, and to its edges: zero is -Inf, +Inf
// itself, a negative number and NaN NaN. Held to log10 it is wrong by its
// whole value.
func TestComposedCPULog(t *testing.T) {
	var xs []float32
	for _, x := range sweep(17, 80) {
		if x > 0 {
			xs = append(xs, x)
		}
	}
	xs = append(xs, 1, 2, 1.5, 0.75, math.Sqrt2, 1e-40, 1e-45, math.MaxFloat32, 0x1p-126, 1+0x1p-23, 1-0x1p-24)
	edges := []float32{0, float32(math.Inf(1)), -1, float32(math.NaN())}
	got := unary(t, graph.MapLog, append(append([]float32(nil), xs...), edges...))
	worst, wrong := 0.0, 0.0
	for i, x := range xs {
		ref := math.Log(float64(x))
		e := math.Abs(float64(got[i])-ref) / max(math.Abs(ref), 0x1p-24)
		if math.IsNaN(e) {
			e = math.Inf(1)
		}
		worst = max(worst, e)
		wrong = max(wrong, math.Abs(float64(got[i])-math.Log10(float64(x))))
	}
	const bound = 4 * 0x1p-23
	t.Logf("log: worst relative error %.3g over %d inputs (bound %g); %.3g from log10", worst, len(xs), bound, wrong)
	if !(worst <= bound) {
		t.Errorf("log is %.3g from its definition, bound %g", worst, bound)
	}
	if !(wrong > 1e3*bound) {
		t.Errorf("log is only %.3g from log10: the comparison cannot tell them apart", wrong)
	}
	e := got[len(xs):]
	if !math.IsInf(float64(e[0]), -1) || !math.IsInf(float64(e[1]), 1) || !math.IsNaN(float64(e[2])) || !math.IsNaN(float64(e[3])) {
		t.Errorf("log of 0, +Inf, -1 and NaN is %v, want -Inf, +Inf, NaN, NaN", e)
	}
}
