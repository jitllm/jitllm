package oracle

import "math"

// E4M3 is an FP8-E4M3 (fn) byte's value: a sign, four exponent bits biased
// by 7 and three mantissa bits, subnormal at a zero exponent. The fn variant
// has no infinity; its one NaN pattern a magnitude of 0x7F.
func E4M3(b byte) float32 {
	e, m := int(b>>3&15), float64(b&7)
	var v float64
	switch {
	case b&0x7F == 0x7F:
		v = math.NaN()
	case e == 0:
		v = math.Ldexp(m/8, -6)
	default:
		v = math.Ldexp(1+m/8, e-7)
	}
	if b&0x80 != 0 {
		v = -v
	}
	return float32(v)
}

// E2M1 is an FP4-E2M1 nibble's value: a sign, two exponent bits biased by 1
// and one mantissa bit.
func E2M1(n byte) float32 {
	e, m := int(n>>1&3), float64(n&1)
	v := m / 2
	if e > 0 {
		v = math.Ldexp(1+m/2, e-1)
	}
	if n&8 != 0 {
		v = -v
	}
	return float32(v)
}

// E8M0 is an unsigned power-of-two exponent byte, 2^(b-127); 255 is NaN.
func E8M0(b byte) float32 {
	if b == 255 {
		return float32(math.NaN())
	}
	return float32(math.Ldexp(1, int(b)-127))
}

// DequantNVFP4 decodes rows rows of k NVFP4 weights as a checkpoint stores
// them: codes two to a byte, element 2j in the low nibble, an E4M3 scale per
// 16 elements of a row, and global, the sheet's amax over its denominator.
// A weight is code times its block's scale, then times global.
func DequantNVFP4(dst []float32, codes, scales []byte, global float32, rows, k int) {
	for r := range rows {
		for c := range k {
			n := codes[(r*k+c)/2] >> (4 * (c % 2)) & 15
			dst[r*k+c] = E2M1(n) * E4M3(scales[r*(k/16)+c/16]) * global
		}
	}
}

// DequantMXFP8 decodes rows rows of k MXFP8 weights as a checkpoint stores
// them: an E4M3 byte an element and an E8M0 exponent per 32 elements of a
// row.
func DequantMXFP8(dst []float32, vals, scales []byte, rows, k int) {
	for r := range rows {
		for c := range k {
			dst[r*k+c] = E4M3(vals[r*k+c]) * E8M0(scales[r*(k/32)+c/32])
		}
	}
}
