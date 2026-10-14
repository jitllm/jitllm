// Package quant is the quantization formats themselves: the type enum, each
// format's block geometry, and the reference decoders.
//
// It is not the container: the engine names weight types without importing a
// file-format package. gguf keeps the header, KV block, tensor table and
// reader.
package quant

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
	// MXFP4 is the OCP microscaling format gpt-oss ships its expert banks in:
	// 32 e2m1 codes and one E8M0 exponent per block.
	MXFP4 Type = 39
)

// blockInfo is elements-per-block and bytes-per-block. A type absent from this
// table is either removed from ggml (4, 5, 31-33, 36-38) or newer than this
// table; either way we cannot compute a tensor's size, so Open rejects it. That
// is a parse-time refusal, not a "we have no fast kernel" refusal — see nn.Ref.
var blockInfo = map[Type][2]uint64{
	F32: {1, 4}, F16: {1, 2}, BF16: {1, 2},
	Q4_0: {32, 18}, Q4_1: {32, 20}, Q5_0: {32, 22}, Q5_1: {32, 24},
	Q8_0: {32, 34}, Q8_1: {32, 36},
	Q2_K: {256, 84}, Q3_K: {256, 110}, Q4_K: {256, 144},
	Q5_K: {256, 176}, Q6_K: {256, 210}, Q8_K: {256, 292},
	16:    {256, 66},  // IQ2_XXS
	17:    {256, 74},  // IQ2_XS
	18:    {256, 98},  // IQ3_XXS
	19:    {256, 50},  // IQ1_S
	20:    {32, 18},   // IQ4_NL
	21:    {256, 110}, // IQ3_S
	22:    {256, 82},  // IQ2_S
	23:    {256, 136}, // IQ4_XS
	24:    {1, 1},     // I8
	25:    {1, 2},     // I16
	26:    {1, 4},     // I32
	27:    {1, 8},     // I64
	28:    {1, 8},     // F64
	29:    {256, 56},  // IQ1_M
	34:    {256, 54},  // TQ1_0
	35:    {256, 66},  // TQ2_0
	MXFP4: {32, 17},
}

var typeNames = map[Type]string{
	F32: "F32", F16: "F16", Q4_0: "Q4_0", Q4_1: "Q4_1", Q5_0: "Q5_0", Q5_1: "Q5_1",
	Q8_0: "Q8_0", Q8_1: "Q8_1", Q2_K: "Q2_K", Q3_K: "Q3_K", Q4_K: "Q4_K", Q5_K: "Q5_K",
	Q6_K: "Q6_K", Q8_K: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
	20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
	27: "I64", 28: "F64", 29: "IQ1_M", BF16: "BF16", 34: "TQ1_0", 35: "TQ2_0", MXFP4: "MXFP4",
}

func (t Type) String() string {
	if s, ok := typeNames[t]; ok {
		return s
	}
	return "type(" + itoa(uint64(t)) + ")"
}

// BlockElems is how many logical elements one block encodes.
func (t Type) BlockElems() uint64 { return blockInfo[t][0] }

// BlockBytes is the on-disk size of one block.
func (t Type) BlockBytes() uint64 { return blockInfo[t][1] }

// Known reports whether we can compute this type's size. Dispatch on the type,
// never on the block size: 18 bytes per 32 elements is Q4_0 and is also IQ4_NL.
func (t Type) Known() bool { _, ok := blockInfo[t]; return ok }

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
var PackedTypes = []Type{Q4_0, Q5_0, Q5_1, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, MXFP4}

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
	}
	return nil, false
}
