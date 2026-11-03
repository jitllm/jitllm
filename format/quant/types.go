// Package quant is the quantization formats themselves: the type enum, each
// format's block geometry, and the reference decoders.
//
// It is not the container: the engine names weight types without importing a
// file-format package. gguf keeps the header, KV block, tensor table and
// reader.
package quant

import "math"

// Type is a ggml_type. The values are ABI: they appear in every GGUF file.
type Type uint32

const (
	F32  Type = 0
	F16  Type = 1
	Q4_0 Type = 2
	Q4_1 Type = 3
	Q5_0 Type = 6
	Q5_1 Type = 7
	Q8_0 Type = 8
	Q8_1 Type = 9
	Q2_K Type = 10
	Q3_K Type = 11
	Q4_K Type = 12
	Q5_K Type = 13
	Q6_K Type = 14
	Q8_K Type = 15
	// IQ4_NL is Q4_0's block with the nibble an index into a fixed
	// non-linear table (IQ4NLValues). Qwen3.8-Flash-Next's n-gram table ships
	// in it, read only by a row gather.
	IQ4_NL Type = 20
	BF16   Type = 30
	// TQ1_0 and TQ2_0 are llama.cpp's ternary formats: 256 weights of {-1, 0,
	// 1} times one f16 scale, base-3 packed five to a byte (TQ1_0) or as 2-bit
	// codes q+1 (TQ2_0). TQ2_0 is the engine's ternary weight type; TQ1_0
	// converts into it exactly.
	TQ1_0 Type = 34
	TQ2_0 Type = 35
	// I2_S is bitnet.cpp's ternary type, the number its fork of ggml gives it
	// (ggml itself retired 36). It has no block: a tensor is n/4 bytes of
	// 2-bit codes and then one f32 scale, padded to 32 bytes, so it is sized
	// by I2SBytes rather than by blockInfo and decoded only by the converter.
	I2_S Type = 36
	// TQ2_F32 is TQ2_0's block with its scale an f32: 64 bytes of codes and
	// then the scale, 68 bytes a 256-block. It is the engine's own, not a
	// ggml type -- no GGUF carries it (gguf refuses an engine type) -- and it
	// exists so a ternary weight whose scale is not an f16 (bitnet.cpp's
	// I2_S, a latent checkpoint ternarized in f32) converts without rounding
	// its scale: the precision of the scale is the type's, as its grouping is.
	TQ2_F32 Type = EngineTypes + 35
	// MXFP4 is the OCP microscaling format gpt-oss ships its expert banks in:
	// 32 e2m1 codes and one E8M0 exponent per block.
	MXFP4 Type = 39
)

// blockInfoOf is elements-per-block and bytes-per-block. A type absent from this
// table is either removed from ggml (4, 5, 31-33, 36-38) or newer than this
// table; either way we cannot compute a tensor's size, so Open rejects it. That
// is a parse-time refusal, not a "we have no fast kernel" refusal — see nn.Ref.
func blockInfoOf(t Type) ([2]uint64, bool) {
	switch t {
	case F32:
		return [2]uint64{1, 4}, true
	case F16:
		return [2]uint64{1, 2}, true
	case BF16:
		return [2]uint64{1, 2}, true
	case Q4_0:
		return [2]uint64{32, 18}, true
	case Q4_1:
		return [2]uint64{32, 20}, true
	case Q5_0:
		return [2]uint64{32, 22}, true
	case Q5_1:
		return [2]uint64{32, 24}, true
	case Q8_0:
		return [2]uint64{32, 34}, true
	case Q8_1:
		return [2]uint64{32, 36}, true
	case Q2_K:
		return [2]uint64{256, 84}, true
	case Q3_K:
		return [2]uint64{256, 110}, true
	case Q4_K:
		return [2]uint64{256, 144}, true
	case Q5_K:
		return [2]uint64{256, 176}, true
	case Q6_K:
		return [2]uint64{256, 210}, true
	case Q8_K:
		return [2]uint64{256, 292}, true
	case 16:
		return [2]uint64{256, 66}, true
	case 17:
		return [2]uint64{256, 74}, true
	case 18:
		return [2]uint64{256, 98}, true
	case 19:
		return [2]uint64{256, 50}, true
	case 20:
		return [2]uint64{32, 18}, true
	case 21:
		return [2]uint64{256, 110}, true
	case 22:
		return [2]uint64{256, 82}, true
	case 23:
		return [2]uint64{256, 136}, true
	case 24:
		return [2]uint64{1, 1}, true
	case 25:
		return [2]uint64{1, 2}, true
	case 26:
		return [2]uint64{1, 4}, true
	case 27:
		return [2]uint64{1, 8}, true
	case 28:
		return [2]uint64{1, 8}, true
	case 29:
		return [2]uint64{256, 56}, true
	case TQ1_0:
		return [2]uint64{256, 54}, true
	case TQ2_0:
		return [2]uint64{256, 66}, true
	case TQ2_F32:
		return [2]uint64{256, 68}, true
	case MXFP4:
		return [2]uint64{32, 17}, true
	}
	return [2]uint64{}, false
}

func typeName(t Type) (string, bool) {
	switch t {
	case F32:
		return "F32", true
	case F16:
		return "F16", true
	case Q4_0:
		return "Q4_0", true
	case Q4_1:
		return "Q4_1", true
	case Q5_0:
		return "Q5_0", true
	case Q5_1:
		return "Q5_1", true
	case Q8_0:
		return "Q8_0", true
	case Q8_1:
		return "Q8_1", true
	case Q2_K:
		return "Q2_K", true
	case Q3_K:
		return "Q3_K", true
	case Q4_K:
		return "Q4_K", true
	case Q5_K:
		return "Q5_K", true
	case Q6_K:
		return "Q6_K", true
	case Q8_K:
		return "Q8_K", true
	case 16:
		return "IQ2_XXS", true
	case 17:
		return "IQ2_XS", true
	case 18:
		return "IQ3_XXS", true
	case 19:
		return "IQ1_S", true
	case 20:
		return "IQ4_NL", true
	case 21:
		return "IQ3_S", true
	case 22:
		return "IQ2_S", true
	case 23:
		return "IQ4_XS", true
	case 24:
		return "I8", true
	case 25:
		return "I16", true
	case 26:
		return "I32", true
	case 27:
		return "I64", true
	case 28:
		return "F64", true
	case 29:
		return "IQ1_M", true
	case BF16:
		return "BF16", true
	case TQ1_0:
		return "TQ1_0", true
	case TQ2_0:
		return "TQ2_0", true
	case I2_S:
		return "I2_S", true
	case TQ2_F32:
		return "TQ2_F32", true
	case MXFP4:
		return "MXFP4", true
	}
	return "", false
}

func (t Type) String() string {
	if s, ok := typeName(t); ok {
		return s
	}
	return "type(" + itoa(uint64(t)) + ")"
}

// EngineTypes is where the engine's own block formats start: past every ggml
// type, so a GGUF naming one is refused (Engine) rather than read.
const EngineTypes Type = 1 << 16

// Engine reports whether t is one of the engine's own block formats, which
// no GGUF may carry.
func (t Type) Engine() bool { return t >= EngineTypes }

// I2SBytes is an I2_S tensor's size for n elements: the codes, then the
// 32-byte tail that holds the scale.
func I2SBytes(n uint64) uint64 { return n/4 + 32 }

// BlockElems is how many logical elements one block encodes.
func (t Type) BlockElems() uint64 { b, _ := blockInfoOf(t); return b[0] }

// BlockBytes is the on-disk size of one block.
func (t Type) BlockBytes() uint64 { b, _ := blockInfoOf(t); return b[1] }

// Known reports whether we can compute this type's size. Dispatch on the type,
// never on the block size: 18 bytes per 32 elements is Q4_0 and is also IQ4_NL.
func (t Type) Known() bool { _, ok := blockInfoOf(t); return ok }

func itoa(v uint64) string {
	if v == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	return string(b[i:])
}

// PackedTypes is every quantized type this engine packs into a container and
// generates a kernel for. It is the list a gate should sweep.
//
// A gate that sweeps this cannot miss a newly added format, where hand-written
// lists in each test once did.
func PackedTypes() []Type {
	return []Type{Q4_0, Q5_0, Q5_1, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, MXFP4, TQ2_0, TQ2_F32}
}

// PlantScales overwrites every block's scale fields in src -- a buffer of whole
// blocks of t, typically random bytes -- with finite values, drawn in turn from
// a sweep that reaches the extremes, and reports false for a type it does not
// know. seed shifts where in the sweep each block starts.
//
// Random bytes need planted finite scales: a random f16 is Inf or NaN about
// one time in 32, and a NaN in both arms reads as a kernel failure. The range
// matters too, since only extremes expose a subnormal decode bug: the f16
// list carries 0x0001 and 0x03FF (subnormal), 0x0400 (smallest normal),
// 0x7BFF (largest finite), and negatives for the formats whose quantizer
// writes them. MXFP4's scale is an E8M0 byte with its own sweep.
//
// A caller sweeping PackedTypes should fail on false rather than test noise.
func PlantScales(t Type, src []byte, seed int) bool {
	if t == MXFP4 {
		e8m0 := []byte{127, 128, 126, 129, 104, 108, 113, 114, 120, 135, 143, 131}
		for b := 0; b+17 <= len(src); b += 17 {
			src[b] = e8m0[(b/17+seed)%len(e8m0)]
		}
		return true
	}
	if t == TQ2_F32 {
		// An f32 scale: the f16 sweep's values widened, and a few no f16
		// holds, at the 68-byte block's tail.
		f32 := []float32{1, 0.5, 2, 0.25, 3, 0.1171875, 1.5, 4, 2.0796757, 0.0004213, 0.7310586, 61.47}
		for b := 0; b+68 <= len(src); b += 68 {
			u := math.Float32bits(f32[(b/68+seed)%len(f32)])
			src[b+64], src[b+65], src[b+66], src[b+67] = byte(u), byte(u>>8), byte(u>>16), byte(u>>24)
		}
		return true
	}
	offs, ok := scaleOffsets(t)
	if !ok {
		return false
	}
	finite := []uint16{
		0x3C00, 0x3800, 0x4000, 0x3400, 0x4200, 0x2E00, 0x3E00, 0x4400,
		0x0001, 0x0200, 0x03FF, 0x0400, 0x1000, 0x7BFF, 0x5000, 0x0801,
	}
	if len(offs) == 1 && offs[0] == 0 {
		// Q4_0's quantizer writes d = max/-8, so a negative scale is what real
		// files hold; a kernel that drops the sign must fail. The k-quants'
		// scales are positive, and a negative dmin only manufactures
		// cancellation an NMSE bound was never sized for.
		finite = append(finite, 0xBC00, 0xB800)
	}
	bb := int(t.BlockBytes())
	if t == Q5_1 {
		// Q5_1's minimum is planted from its scale: the quantizer writes
		// d = (max-min)/31 and m = min, so |m| <= 31*d in a real block. An
		// independent m would make the gate measure int8 resolution rather
		// than the kernel. m still spans both signs and zero.
		mult := []float32{-16, -31, -8.5, 0, 3, -1, -24, 12}
		for b := 0; b+bb <= len(src); b += bb {
			i := b/bb + seed
			h := finite[i%len(finite)]
			v := float32(DecodeHalf(h)) * mult[i%len(mult)]
			v = max(-65504, min(65504, v)) // finite: 65504*31 is not a half
			m := EncodeHalf(v)
			src[b], src[b+1] = byte(h), byte(h>>8)
			src[b+2], src[b+3] = byte(m), byte(m>>8)
		}
		return true
	}
	for b := 0; b+bb <= len(src); b += bb {
		for oi, o := range offs {
			h := finite[(b/bb*len(offs)+oi+seed)%len(finite)]
			src[b+o], src[b+o+1] = byte(h), byte(h>>8)
		}
	}
	return true
}

// scaleOffsets is where a block's f16 super-scale (and its minimum, where the
// format has one) sits inside one on-disk block.
func scaleOffsets(t Type) ([]int, bool) {
	switch t {
	case Q4_0, Q5_0, Q8_0:
		return []int{0}, true // d
	case Q4_K, Q5_K:
		return []int{0, 2}, true // d, dmin
	case Q5_1:
		return []int{0, 2}, true // d, m
	case Q6_K:
		return []int{208}, true // d, after the quants and scales
	case Q3_K:
		return []int{108}, true
	case TQ2_0:
		return []int{64}, true // d, after the 64 code bytes
	}
	return nil, false
}
