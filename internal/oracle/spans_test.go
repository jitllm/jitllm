package oracle

import (
	"math"
	"strings"
	"testing"
)

// TestSpanSumsSumEachRun holds SpanSums to each run's sum in float64,
// rounded once, at every run length dividing a row of 24, every element of
// a run alike; a row a run does not divide is refused.
func TestSpanSumsSumEachRun(t *testing.T) {
	x := make([]float32, 24)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*1.7)) * float32(1+i%5) * 1e3
	}
	for _, span := range []int{1, 2, 3, 4, 6, 8, 12, 24} {
		out := make([]float32, len(x))
		if err := SpanSums(out, x, span); err != nil {
			t.Fatal(err)
		}
		for i := range x {
			lo := i - i%span
			var want float64
			for _, v := range x[lo : lo+span] {
				want += float64(v)
			}
			if got := float64(out[i]); math.Abs(got-want) > 1e-6*math.Abs(want)+1e-3 {
				t.Fatalf("runs of %d: element %d sums to %v, want %v", span, i, got, want)
			}
		}
	}
	if err := SpanSums(make([]float32, 24), x, 5); err == nil || !strings.Contains(err.Error(), "runs of 5") {
		t.Fatalf("runs of 5 in a row of 24: %v", err)
	}
}
