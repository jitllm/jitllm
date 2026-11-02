package kernels

import "testing"

// TestPackNeverEmitsInt8Min holds the weight packer to the invariant the MSL
// dot depends on: no operand byte is -128 where the dot reads it signed.
// unpack_snorm4x8_to_float is clamp(v/127, -1, 1), so a -128 would come back
// as -127, silently wrong by 1/127 (the activations' half is the oracle's,
// TestPackActivationsNeverEmitsInt8Min).
func TestPackNeverEmitsInt8Min(t *testing.T) {
	// Every format the matvec sends through ir.Dot4Bounded must mask its weights
	// to a small unsigned range before the dot sees them. Q8_0 is the exception
	// and is declared so (signedQ); it takes the unbounded path.
	const nrows, k = 2, 256
	for q := Quant(0); int(q) < len(qtab); q++ {
		qi := q.info()
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
