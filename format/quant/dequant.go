package quant

import (
	"encoding/binary"
	"fmt"
	"math"
)

// DecodeHalf decodes an IEEE binary16. Exported so other packages share this
// one decoder rather than a copy (a copy once got every subnormal wrong).
// Not named F16: that is already a Type constant in this package.
func DecodeHalf(u uint16) float64 { return f16(u) }

// f16 converts an IEEE binary16 (ggml_half) to float64, exactly.
func f16(u uint16) float64 {
	sign := 1.0
	if u&0x8000 != 0 {
		sign = -1
	}
	exp := int(u>>10) & 0x1f
	mant := int(u & 0x3ff)
	switch exp {
	case 0: // subnormal (and zero)
		return sign * float64(mant) * math.Exp2(-24)
	case 0x1f:
		if mant != 0 {
			return math.NaN()
		}
		return sign * math.Inf(1)
	}
	return sign * float64(1024+mant) * math.Exp2(float64(exp-25))
}

// Dequant expands src into dst, which must be exactly the elements of a whole
// number of blocks. It is the reference implementation every generated kernel
// is held to: float64, no SIMD, no cleverness.
//
// The caller owns dst; there is deliberately no API that allocates a whole
// tensor.
func Dequant(t Type, src []byte, dst []float64) error { return dequant(t, src, dst) }

// Dequant32 is Dequant writing float32, which is what every caller outside the
// oracle wants.
//
// Only the destination type is generic; the arithmetic is one implementation.
// It is numerically free: an f16 scale times a small integer quant is exact in
// f32 for every format (coefficients fit 24 bits), and the one rounding step
// (Q4_K/Q5_K's final subtraction) is exact in f64, so both destinations round
// the same value once. dequant32_test.go asserts zero ULP.
func Dequant32(t Type, src []byte, dst []float32) error { return dequant(t, src, dst) }

// destination is the only thing the decoders are generic over.
type destination interface{ ~float32 | ~float64 }

func dequant[T destination](t Type, src []byte, dst []T) error {
	if !t.Known() {
		return fmt.Errorf("gguf: dequant: unknown type %d", uint32(t))
	}
	be, bb := t.BlockElems(), t.BlockBytes()
	n := uint64(len(dst))
	if n%be != 0 {
		return fmt.Errorf("gguf: dequant %s: %d elements is not a multiple of %d per block", t, n, be)
	}
	blocks := n / be
	if need := blocks * bb; uint64(len(src)) < need {
		return fmt.Errorf("gguf: dequant %s: need %d bytes for %d elements, have %d", t, need, n, len(src))
	}

	switch t {
	case F32:
		for i := range dst {
			dst[i] = T(math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:])))
		}
	case F16:
		for i := range dst {
			dst[i] = T(f16(binary.LittleEndian.Uint16(src[i*2:])))
		}
	case BF16:
		// bfloat16 is the top half of a float32, so widening is an exact shift:
		// no rebias, no subnormal case, no rounding.
		for i := range dst {
			dst[i] = T(math.Float32frombits(uint32(binary.LittleEndian.Uint16(src[i*2:])) << 16))
		}

	case Q4_0:
		// block_q4_0 { ggml_half d; uint8_t qs[16]; } — 32 elements in 18 bytes.
		//
		// The nibbles are not interleaved: low nibbles are elements 0..15 and
		// high nibbles elements 16..31.
		for b := uint64(0); b < blocks; b++ {
			blk := src[b*18:]
			d := f16(binary.LittleEndian.Uint16(blk))
			out := dst[b*32:]
			for j := 0; j < 16; j++ {
				q := blk[2+j]
				out[j] = T(float64(int(q&0x0f)-8) * d)
				out[j+16] = T(float64(int(q>>4)-8) * d)
			}
		}

	case Q5_0:
		// block_q5_0 { ggml_half d; uint8_t qh[4]; uint8_t qs[16]; } — 32
		// elements in 22 bytes. Q4_0's nibbles plus a 32-bit plane holding each
		// weight's fifth bit, so the quant is 0..31 and the offset is -16.
		// Element l takes bit l of qh, for all 32 (ggml's (qh >> (j+12)) & 0x10
		// is bit j+16 written differently).
		for b := uint64(0); b < blocks; b++ {
			blk := src[b*22:]
			d := f16(binary.LittleEndian.Uint16(blk))
			qh := binary.LittleEndian.Uint32(blk[2:])
			out := dst[b*32:]
			for j := 0; j < 16; j++ {
				lo, hi := int(blk[6+j]&0x0f), int(blk[6+j]>>4)
				if qh&(1<<uint(j)) != 0 {
					lo += 16
				}
				if qh&(1<<uint(j+16)) != 0 {
					hi += 16
				}
				out[j] = T(float64(lo-16) * d)
				out[j+16] = T(float64(hi-16) * d)
			}
		}

	case Q5_1:
		// block_q5_1 { ggml_half d; ggml_half m; uint8_t qh[4]; uint8_t qs[16]; }
		// -- 32 elements in 24 bytes. Q5_0's planes with an f16 minimum in place
		// of the fixed offset: the quant is 0..31 and the weight is d*q + m.
		for b := uint64(0); b < blocks; b++ {
			blk := src[b*24:]
			d := f16(binary.LittleEndian.Uint16(blk))
			m := f16(binary.LittleEndian.Uint16(blk[2:]))
			qh := binary.LittleEndian.Uint32(blk[4:])
			out := dst[b*32:]
			for j := 0; j < 16; j++ {
				lo, hi := int(blk[8+j]&0x0f), int(blk[8+j]>>4)
				if qh&(1<<uint(j)) != 0 {
					lo += 16
				}
				if qh&(1<<uint(j+16)) != 0 {
					hi += 16
				}
				out[j] = T(float64(lo)*d + m)
				out[j+16] = T(float64(hi)*d + m)
			}
		}

	case Q8_0:
		// block_q8_0 { ggml_half d; int8_t qs[32]; } — 32 elements in 34 bytes.
		for b := uint64(0); b < blocks; b++ {
			blk := src[b*34:]
			d := f16(binary.LittleEndian.Uint16(blk))
			out := dst[b*32:]
			for j := 0; j < 32; j++ {
				out[j] = T(float64(int8(blk[2+j])) * d)
			}
		}

	case MXFP4:
		// block_mxfp4 { uint8_t e; uint8_t qs[16]; } -- 32 elements in 17 bytes,
		// nibbles laid out as Q4_0's (low = element j, high = element j+16).
		//
		// The code is an e2m1 float, not an affine integer: bit 3 is the sign and
		// the low three bits index {0, 0.5, 1, 1.5, 2, 3, 4, 6}. MXFP4Values holds
		// those doubled, so the scale is 2^(e-128) (ggml's e8m0_to_fp32_half).
		for b := uint64(0); b < blocks; b++ {
			blk := src[b*17:]
			d := math.Ldexp(1, int(blk[0])-128)
			out := dst[b*32:]
			for j := 0; j < 16; j++ {
				out[j] = T(float64(MXFP4Values[blk[1+j]&0xF]) * d)
				out[j+16] = T(float64(MXFP4Values[blk[1+j]>>4]) * d)
			}
		}

	case IQ4_NL:
		// block_iq4_nl { ggml_half d; uint8_t qs[16]; } -- 32 elements in 18
		// bytes, Q4_0's layout (low nibble element j, high nibble j+16), the
		// nibble an index into a fixed non-linear int8 table rather than q-8.
		// d times an int8 is exact in f32, so both destinations agree.
		for b := uint64(0); b < blocks; b++ {
			blk := src[b*18:]
			d := f16(binary.LittleEndian.Uint16(blk))
			out := dst[b*32:]
			for j := 0; j < 16; j++ {
				out[j] = T(float64(IQ4NLValues[blk[2+j]&0xF]) * d)
				out[j+16] = T(float64(IQ4NLValues[blk[2+j]>>4]) * d)
			}
		}

	case Q3_K:
		dequantQ3K(src, dst, blocks)
	case Q4_K:
		dequantQ4K(src, dst, blocks)
	case Q5_K:
		dequantQ5K(src, dst, blocks)
	case Q6_K:
		dequantQ6K(src, dst, blocks)

	default:
		return fmt.Errorf("gguf: dequant %s: not implemented yet", t)
	}
	return nil
}

// MXFP4Values is the e2m1 code table, doubled so every entry is an integer --
// ggml's kvalues_mxfp4. The scale that goes with it is 2^(e-128), not 2^(e-127).
var MXFP4Values = [16]int8{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}

// IQ4NLValues is IQ4_NL's code table, ggml's kvalues_iq4nl.
var IQ4NLValues = [16]int8{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}

// Dequantable reports whether Dequant handles type t.
func Dequantable(t Type) bool {
	switch t {
	case F32, F16, BF16, Q4_0, Q5_0, Q5_1, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, MXFP4, IQ4_NL:
		return true
	}
	return false
}

// ---------------------------------------------------------------------------
// K-quants. QK_K is 256: a "super-block" of 256 elements carries one or two f16
// super-scales plus a packed table of per-64 or per-16 sub-scales.
//
// What is load-bearing is the format (where the super-scales sit, how the
// 6-bit scale/min pairs are stitched out of twelve bytes, which nibble is
// element j). The float32 intermediates are the format's natural width; as
// Dequant32 explains, f64 would round to the same float32 output.
//
// The strongest test is raw uint32 comparison against libggml with no
// tolerance: a tolerance wide enough to absorb f32 rounding would hide a
// sub-scale read from the next nibble.

const qkK = 256

func dequantQ3K[T destination](src []byte, dst []T, blocks uint64) {
	// block_q3_K { uint8 hmask[32]; uint8 qs[64]; uint8 scales[12]; f16 d; }
	for b := uint64(0); b < blocks; b++ {
		blk := src[b*110:]
		hm := blk[0:32]
		qs := blk[32:96]
		dAll := float32(f16(binary.LittleEndian.Uint16(blk[108:])))

		// The 16 six-bit sub-scales are packed across 12 bytes: four low bits in
		// place, two high bits gathered from the last four bytes.
		var aux [4]uint32
		aux[0] = binary.LittleEndian.Uint32(blk[96:])
		aux[1] = binary.LittleEndian.Uint32(blk[100:])
		aux[2] = binary.LittleEndian.Uint32(blk[104:])
		const kmask1, kmask2 = 0x03030303, 0x0f0f0f0f
		tmp := aux[2]
		// aux[2] and aux[3] must be computed from the original aux[0]/aux[1],
		// before either is overwritten below.
		aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
		aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
		aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
		aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
		var sc [16]int8
		for i := 0; i < 4; i++ {
			sc[i*4+0] = int8(aux[i])
			sc[i*4+1] = int8(aux[i] >> 8)
			sc[i*4+2] = int8(aux[i] >> 16)
			sc[i*4+3] = int8(aux[i] >> 24)
		}

		out := dst[b*qkK:]
		o, is, m := 0, 0, uint8(1)
		for n := 0; n < qkK; n += 128 {
			q := qs[n/4:]
			shift := uint(0)
			for j := 0; j < 4; j++ {
				for _, base := range [2]int{0, 16} {
					dl := dAll * float32(int32(sc[is])-32)
					is++
					for l := 0; l < 16; l++ {
						// A set high-mask bit means do not subtract 4; inverting
						// it is the classic q3_K bug and yields plausible text.
						v := int32((q[l+base] >> shift) & 3)
						if hm[l+base]&m == 0 {
							v -= 4
						}
						out[o] = T(float64(dl * float32(v)))
						o++
					}
				}
				shift += 2
				m <<= 1
			}
		}
	}
}

// scaleMinK4 unpacks the j-th 6-bit scale/min pair from a q4_K or q5_K block's
// 12 packed bytes. j < 4 reads them in place; j >= 4 stitches four low bits from
// one byte together with two high bits borrowed from another.
func scaleMinK4(j int, q []byte) (sc, m uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4), (q[j+4] >> 4) | ((q[j] >> 6) << 4)
}

func dequantQ4K[T destination](src []byte, dst []T, blocks uint64) {
	// block_q4_K { f16 d; f16 dmin; uint8 scales[12]; uint8 qs[128]; }
	for b := uint64(0); b < blocks; b++ {
		blk := src[b*144:]
		d := float32(f16(binary.LittleEndian.Uint16(blk[0:])))
		dmin := float32(f16(binary.LittleEndian.Uint16(blk[2:])))
		scales := blk[4:16]
		qs := blk[16:144]

		out := dst[b*qkK:]
		o, is := 0, 0
		for j := 0; j < qkK; j += 64 {
			q := qs[j/2:]
			sc1, m1b := scaleMinK4(is+0, scales)
			d1, min1 := d*float32(sc1), dmin*float32(m1b)
			sc2, m2b := scaleMinK4(is+1, scales)
			d2, min2 := d*float32(sc2), dmin*float32(m2b)
			// Low nibbles are the first 32 outputs, high nibbles the next 32 —
			// not interleaved 2l/2l+1.
			for l := 0; l < 32; l++ {
				out[o] = T(float64(d1*float32(q[l]&0xF) - min1))
				o++
			}
			for l := 0; l < 32; l++ {
				out[o] = T(float64(d2*float32(q[l]>>4) - min2))
				o++
			}
			is += 2
		}
	}
}

func dequantQ5K[T destination](src []byte, dst []T, blocks uint64) {
	// block_q5_K { f16 d; f16 dmin; uint8 scales[12]; uint8 qh[32]; uint8 qs[128]; }
	for b := uint64(0); b < blocks; b++ {
		blk := src[b*176:]
		d := float32(f16(binary.LittleEndian.Uint16(blk[0:])))
		dmin := float32(f16(binary.LittleEndian.Uint16(blk[2:])))
		scales := blk[4:16]
		qh := blk[16:48]
		qs := blk[48:176]

		out := dst[b*qkK:]
		o, is := 0, 0
		u1, u2 := uint8(1), uint8(2)
		for j := 0; j < qkK; j += 64 {
			// ql advances 32 bytes per group, qh does not: the same 32 high-bit
			// bytes serve all four groups, two bits each, selected by u1/u2.
			ql := qs[j/2:]
			sc1, m1b := scaleMinK4(is+0, scales)
			d1, min1 := d*float32(sc1), dmin*float32(m1b)
			sc2, m2b := scaleMinK4(is+1, scales)
			d2, min2 := d*float32(sc2), dmin*float32(m2b)
			for l := 0; l < 32; l++ {
				v := int32(ql[l] & 0xF)
				if qh[l]&u1 != 0 {
					v += 16
				}
				out[o] = T(float64(d1*float32(v) - min1))
				o++
			}
			for l := 0; l < 32; l++ {
				v := int32(ql[l] >> 4)
				if qh[l]&u2 != 0 {
					v += 16
				}
				out[o] = T(float64(d2*float32(v) - min2))
				o++
			}
			is += 2
			u1 <<= 2
			u2 <<= 2
		}
	}
}

func dequantQ6K[T destination](src []byte, dst []T, blocks uint64) {
	// block_q6_K { uint8 ql[128]; uint8 qh[64]; int8 scales[16]; f16 d; }
	for b := uint64(0); b < blocks; b++ {
		blk := src[b*210:]
		d := float32(f16(binary.LittleEndian.Uint16(blk[208:])))

		out := dst[b*qkK:]
		for n := 0; n < qkK; n += 128 {
			ql := blk[n/2:]
			qh := blk[128+n/4:]
			sc := blk[192+n/16:]
			y := out[n:]
			for l := 0; l < 32; l++ {
				// The scale index strides by 2, not by 1 —
				// sc[is+0], sc[is+2], sc[is+4], sc[is+6] with is = l/16.
				is := l / 16
				q1 := int32(int8((ql[l+0]&0xF)|(((qh[l]>>0)&3)<<4))) - 32
				q2 := int32(int8((ql[l+32]&0xF)|(((qh[l]>>2)&3)<<4))) - 32
				q3 := int32(int8((ql[l+0]>>4)|(((qh[l]>>4)&3)<<4))) - 32
				q4 := int32(int8((ql[l+32]>>4)|(((qh[l]>>6)&3)<<4))) - 32
				y[l+0] = T(float64(d * float32(int8(sc[is+0])) * float32(q1)))
				y[l+32] = T(float64(d * float32(int8(sc[is+2])) * float32(q2)))
				y[l+64] = T(float64(d * float32(int8(sc[is+4])) * float32(q3)))
				y[l+96] = T(float64(d * float32(int8(sc[is+6])) * float32(q4)))
			}
		}
	}
}

// EncodeHalf rounds an f32 to binary16, round-to-nearest-even, and is the exact
// inverse of DecodeHalf on every representable half.
//
// It handles subnormals and overflow (to Inf) rather than assuming the normal
// range: a KV cache holds activations, and outlier channels leave it.
// TestEncodeHalfRoundTrips checks all 65536 patterns.
func EncodeHalf(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b >> 16 & 0x8000)
	exp := int32(b>>23) & 0xFF
	man := b & 0x7FFFFF

	if exp == 0xFF { // Inf, or NaN which must stay NaN
		if man != 0 {
			return sign | 0x7E00
		}
		return sign | 0x7C00
	}
	e := exp - 127 + 15
	if e >= 0x1F {
		return sign | 0x7C00 // overflows binary16
	}
	if e <= 0 {
		if e < -10 {
			return sign // underflows even the subnormals
		}
		// Subnormal: the implicit leading 1 becomes explicit and the whole
		// significand shifts right, so the rounding point moves with e.
		man |= 0x800000
		shift := uint32(14 - e)
		h := man >> shift
		rem := man & (1<<shift - 1)
		if half := uint32(1) << (shift - 1); rem > half || (rem == half && h&1 == 1) {
			h++
		}
		return sign | uint16(h)
	}
	h := uint16(e)<<10 | uint16(man>>13)
	// Round to nearest even. A carry out of the mantissa lands in the exponent
	// by construction, which is what makes 0x3FF -> next exponent correct
	// without a special case.
	if rem := man & 0x1FFF; rem > 0x1000 || (rem == 0x1000 && h&1 == 1) {
		h++
	}
	return sign | h
}
