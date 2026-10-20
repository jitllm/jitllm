package oldkernels

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestPackNeverEmitsInt8Min holds both packers to the invariant the MSL dot
// depends on: no operand byte is -128.
//
// unpack_snorm4x8_to_float is clamp(v/127, -1, 1), so a -128 would come back as
// -127, silently wrong by 1/127. Both quantizers divide by the block's amax and
// scale by 127, which cannot reach -128; this holds them to it.
func TestPackNeverEmitsInt8Min(t *testing.T) {
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

	// Every format the matvec sends through ir.Dot4Bounded must mask its weights
	// to a small unsigned range before the dot sees them. Q8_0 is the exception
	// and is declared so (signedQ); it takes the unbounded path.
	const nrows, k = 2, 256
	for q := Quant(0); int(q) < kernels.Quants(); q++ {
		qi := infoOf(q)
		if qi.blockB == 0 {
			continue
		}
		// Only the 8-bit formats hand their packed byte straight to the dot. A
		// 4-bit format masks with 0x0F0F0F0F first, so its packed bytes (0x80
		// among them) are never operands.
		if qi.bits != 8 {
			continue
		}
		// The fill matters: as a signed byte 0xFF is -1, not -128, so a fill of
		// 0xFF alone let a mislabelled Q8_0 pass. 0x80 is the byte under test;
		// the others cover the k-quant field extremes.
		var sawMin bool
		for _, fill := range []func(i int) byte{
			func(i int) byte { return 0x80 },
			func(i int) byte { return 0xFF },
			func(i int) byte { return 0x00 },
			func(i int) byte { return byte(i) },
			func(i int) byte { return byte(i * 31) },
		} {
			src := make([]byte, nrows*(k/qi.blockE)*qi.blockB)
			for i := range src {
				src[i] = fill(i)
			}
			qs, _, _, err := PackWeights(q, src, nrows, k)
			if err != nil {
				t.Fatalf("%v: PackWeights: %v", q, err)
			}
			for _, w := range qs {
				for b := 0; b < 4; b++ {
					if int8(w>>(8*b)) == -128 {
						sawMin = true
					}
				}
			}
		}
		if sawMin && !qi.signedQ {
			t.Errorf("%v: packs -128 but is not marked signedQ, so the matvec sends it "+
				"through Dot4Bounded and MSL will widen it as -127", q)
		}
	}
}
