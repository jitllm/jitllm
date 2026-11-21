package bench

import (
	"github.com/jitllm/jitllm/internal/oracle"
	"math/rand"
	"runtime"
	"testing"
)

// TestSoftmaxOnline measures the three-pass softmax against the one-pass
// online form, paired and interleaved, at attention row lengths.
//
// The physics guard is off: the rows live in L1 by design, so an implied
// bandwidth above the DRAM wall is correct here.
func TestSoftmaxOnline(t *testing.T) {
	for _, n := range []int{128, 821, 4096} {
		for _, shape := range []struct {
			name string
			gen  func(rng *rand.Rand, i, n int) float64
		}{
			// Attention scores are unsorted, which is the case the three-pass
			// form is claimed to win.
			{"unsorted", func(rng *rand.Rand, i, n int) float64 { return rng.NormFloat64() * 4 }},
			// Ascending is the online form's worst case: every element is a new
			// maximum, so the rescale runs n times.
			{"ascending", func(rng *rand.Rand, i, n int) float64 { return float64(i) * 8 / float64(n) }},
		} {
			t.Run(shape.name+"/"+itoa(n), func(t *testing.T) {
				rng := rand.New(rand.NewSource(int64(n)))
				src := make([]float64, n)
				for i := range src {
					src[i] = shape.gen(rng, i, n)
				}
				buf := make([]float64, n)
				run := func(f func([]float64)) func(int) {
					return func(iters int) {
						for it := 0; it < iters; it++ {
							copy(buf, src)
							f(buf)
							runtime.KeepAlive(buf[0])
						}
					}
				}
				r := AB(
					Case{Name: "3-pass", Fn: run(oracle.Softmax)},
					Case{Name: "online", Fn: run(oracle.SoftmaxOnline)},
					21, 200)
				t.Log(r)
				if !r.Stable() {
					t.Logf("  UNSTABLE (IQR/median %.1f%%) -- reporting, not concluding",
						100*r.IQR/r.Median)
				}
			})
		}
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}
