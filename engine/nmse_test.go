package engine

import (
	"math"

	"github.com/jitllm/jitllm/composer"
)

// nmseOf is the NMSE of got against want and the largest absolute difference.
func nmseOf(got, want []float32) (nmse, worst float64) {
	var num, den float64
	for i := range want {
		d := float64(got[i] - want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
		worst = math.Max(worst, math.Abs(d))
	}
	return num / den, worst
}

// nmse32 is nmseOf with its arguments the other way round.
func nmse32(want, got []float32) (float64, float64) { return nmseOf(got, want) }

// nonZero is the declines a census counted, by name.
func nonZero(d []int) map[string]int {
	out := map[string]int{}
	for i, n := range d {
		if n != 0 {
			out[composer.DeclineName(i)] = n
		}
	}
	return out
}
