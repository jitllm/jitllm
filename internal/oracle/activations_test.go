package oracle

import (
	"math"
	"math/rand"
	"testing"
)

// TestPackActivationsNeverEmitsInt8Min holds the reference int8 form to the
// invariant the MSL dot depends on: no operand byte is -128.
// unpack_snorm4x8_to_float is clamp(v/127, -1, 1), so a -128 would come back
// as -127, silently wrong by 1/127. The quantizer divides by the block's amax
// and scales by 127, which cannot reach -128; this holds it to it.
func TestPackActivationsNeverEmitsInt8Min(t *testing.T) {
	// Activations: the adversarial input is one whose extreme lands exactly on
	// the negative rail, since that is the only value that can round past it.
	rng := rand.New(rand.NewSource(7))
	for _, tc := range []struct {
		name string
		gen  func(i, n int) float64
	}{
		{"all equal negative", func(i, n int) float64 { return -1.5 }},
		{"one deep negative", func(i, n int) float64 {
			if i == n/2 {
				return -8192
			}
			return rng.NormFloat64()
		}},
		{"negative rail", func(i, n int) float64 {
			if i%3 == 0 {
				return -math.MaxFloat32
			}
			return rng.NormFloat64() * math.MaxFloat32
		}},
		{"denormal", func(i, n int) float64 { return -math.SmallestNonzeroFloat64 * float64(i+1) }},
		{"zeros", func(i, n int) float64 { return 0 }},
	} {
		for _, window := range []int{32, 256} {
			const n = 512
			x := make([]float32, n)
			for i := range x {
				x[i] = float32(tc.gen(i, n))
			}
			a := make([]uint32, n/4)
			as := make([]float32, n/32)
			sum := make([]float32, 2*(n/32))
			if err := PackActivationsInto(a, as, sum, x, window); err != nil {
				t.Fatalf("%s window=%d: %v", tc.name, window, err)
			}
			for wi, w := range a {
				for b := 0; b < 4; b++ {
					if v := int8(w >> (8 * b)); v == -128 {
						t.Fatalf("%s window=%d: activation word %d byte %d is -128",
							tc.name, window, wi, b)
					}
				}
			}
		}
	}

}
