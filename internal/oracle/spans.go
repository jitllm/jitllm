package oracle

import "fmt"

// SpanSums is each element of x as the sum of the run of span elements of
// x it is in (elements i-i%span onward), into out: an iteration's term
// summed over runs of its row. Each run's sum is Neumaier-compensated, as
// MatVec's are, so the reference is a different method from a kernel's
// butterfly of lanes.
func SpanSums(out, x []float32, span int) error {
	if span < 1 || len(x)%span != 0 {
		return fmt.Errorf("oracle: SpanSums: runs of %d in a row of %d", span, len(x))
	}
	if len(out) < len(x) {
		return fmt.Errorf("oracle: SpanSums: out has %d elements, need %d", len(out), len(x))
	}
	for lo := 0; lo < len(x); lo += span {
		var sum, comp float32
		for _, v := range x[lo : lo+span] {
			t := sum + v
			if abs32(sum) >= abs32(v) {
				comp += (sum - t) + v
			} else {
				comp += (v - t) + sum
			}
			sum = t
		}
		for i := lo; i < lo+span; i++ {
			out[i] = sum + comp
		}
	}
	return nil
}
