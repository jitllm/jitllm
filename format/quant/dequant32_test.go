package quant

import (
	"math"
	"testing"
)

// TestDequant32IsBitIdenticalToNarrowedDequant asserts zero ULP between the two
// destinations, not a tolerance; Dequant32's doc gives the arithmetic that
// makes exact equality hold. A decoder that reads the wrong nibble still lands
// "close", so a tolerance would hide it.
func TestDequant32IsBitIdenticalToNarrowedDequant(t *testing.T) {
	// PackedTypes rather than a hand-written list, so no format is missed.
	for _, ty := range append([]Type{F32, F16}, PackedTypes()...) {
		be, bb := int(ty.BlockElems()), int(ty.BlockBytes())
		const blocks = 7
		src := make([]byte, blocks*bb)
		// Deterministic bytes that exercise every nibble, sign and scale slot.
		for i := range src {
			src[i] = byte(i*181 + i/13*7)
		}
		// Finite scales rather than random bits, which can land on NaN/Inf
		// and make the comparison vacuous.
		if ty != F32 && ty != F16 && !PlantScales(ty, src, 0) {
			t.Fatalf("%v: PlantScales does not know it", ty)
		}
		n := blocks * be
		d64 := make([]float64, n)
		d32 := make([]float32, n)
		if err := Dequant(ty, src, d64); err != nil {
			t.Fatalf("%v: %v", ty, err)
		}
		if err := Dequant32(ty, src, d32); err != nil {
			t.Fatalf("%v: %v", ty, err)
		}
		bad := 0
		for i := 0; i < n; i++ {
			want, got := float32(d64[i]), d32[i]
			// Bit comparison, so a signed zero mismatch is a failure too.
			if math.Float32bits(got) != math.Float32bits(want) {
				if bad < 4 {
					t.Errorf("%v[%d] = %v (%#08x), narrowed f64 = %v (%#08x)",
						ty, i, got, math.Float32bits(got), want, math.Float32bits(want))
				}
				bad++
			}
			if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) {
				t.Fatalf("%v[%d] is not finite (%v); the fixture is degenerate and "+
					"this comparison would prove nothing", ty, i, got)
			}
		}
		if bad != 0 {
			t.Errorf("%v: %d/%d elements differ", ty, bad, n)
		} else {
			t.Logf("%v: %d elements, zero ULP", ty, n)
		}
	}
}
