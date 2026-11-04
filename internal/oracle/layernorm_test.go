package oracle_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"
)

// LayerNorm is gated against an independent float64 evaluation of its own
// definition, at the widths a real tower uses, with and without a bias.
//
// It also asserts LayerNorm and RMSNorm disagree on non-zero-mean input: they
// differ only by the mean subtraction, so an implementation that forgot it
// would still look plausible.
func TestLayerNorm(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, n := range []int{8, 64, 768, 1152} {
		x := make([]float64, n)
		w := make([]float64, n)
		b := make([]float64, n)
		for i := range x {
			x[i] = rng.NormFloat64()*3 + 1.5 // deliberately non-zero mean
			w[i] = rng.NormFloat64()
			b[i] = rng.NormFloat64()
		}
		const eps = 1e-5

		for _, withBias := range []bool{false, true} {
			bb := b
			if !withBias {
				bb = nil
			}
			got := make([]float64, n)
			oracle.LayerNorm(got, x, w, bb, eps)

			var mean float64
			for _, v := range x {
				mean += v
			}
			mean /= float64(n)
			var vs float64
			for _, v := range x {
				vs += (v - mean) * (v - mean)
			}
			inv := 1 / math.Sqrt(vs/float64(n)+eps)
			var sse, sy2 float64
			for i := range x {
				want := (x[i] - mean) * inv * w[i]
				if withBias {
					want += b[i]
				}
				d := got[i] - want
				sse += d * d
				sy2 += want * want
			}
			if nmse := sse / sy2; nmse > 1e-24 || math.IsNaN(nmse) {
				t.Errorf("n=%d bias=%v: NMSE %.3e", n, withBias, nmse)
			}
		}

		// The mean subtraction must actually happen.
		ln := make([]float64, n)
		rms := make([]float64, n)
		oracle.LayerNorm(ln, x, w, nil, eps)
		oracle.RMSNorm(rms, x, w, eps)
		same := true
		for i := range ln {
			if math.Abs(ln[i]-rms[i]) > 1e-9 {
				same = false
				break
			}
		}
		if same {
			t.Errorf("n=%d: LayerNorm agrees with RMSNorm on non-zero-mean input -- the mean is not being subtracted", n)
		}
	}
}
