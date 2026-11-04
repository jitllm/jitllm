package layout

import (
	"github.com/jitllm/jitllm/format/quant"
)

// Quant is the weight format the device kernels understand.
//
// It is a local enum rather than quant.Type because it lists what has a GPU
// kernel, not what jitllm can read; tier.quantOf is the one conversion.
type Quant uint8

const (
	Q4_0 Quant = iota
	Q8_0
	Q4_K
	Q3_K
	Q5_K
	Q6_K
	// Appended, not inserted: the value indexes the string table and qtab, and
	// renumbering would repoint every lowertest golden to another format.
	Q5_0
	MXFP4
	// The float formats are the device's own: a weight the container keeps
	// verbatim. Their "pack" is a copy of the raw rows; the matvec reads them
	// row-major against the float activation, passed where a quant's scale plane
	// goes. QuantOf never returns them; only the device tier asks, via FloatOf.
	Float32
	Float16
	BFloat16
	// Q5_1 is Q5_0's planes with an f16 minimum: weight = d*q + m. Its d word
	// carries both halves the way a k-quant's does, the minimum the high half
	// alone, with no per-sub-block integer to multiply it by.
	Q5_1
	// TQ2 is a ternary weight, llama.cpp's TQ2_0: codes q+1 in {0, 1, 2} for
	// weights {-1, 0, 1} in a 2-bit plane (four fields a byte, Fields), one f16
	// scale per 256 weights repeated on each 32-wide sub-block, and a bias of
	// one. Ternary x int8 is an unsigned dot of the codes with the activation
	// sum subtracted, the activation sum taken once per token by the
	// quantizer. Its activations are always int8 (see qinfo.intAct).
	TQ2
	// TQ2F is TQ2 with its scale an f32 (quant.TQ2_F32): the d word is the
	// scale's bits, one row a word (DSlots 1). See qinfo.d32.
	TQ2F
)

func (q Quant) String() string {
	return [...]string{"Q4_0", "Q8_0", "Q4_K", "Q3_K", "Q5_K", "Q6_K", "Q5_0", "MXFP4",
		"F32", "F16", "BF16", "Q5_1", "TQ2", "TQ2F"}[q]
}

// FloatOf maps an unquantized weight type to its device format.
func FloatOf(t quant.Type) (Quant, bool) {
	switch t {
	case quant.F32:
		return Float32, true
	case quant.F16:
		return Float16, true
	case quant.BF16:
		return BFloat16, true
	}
	return 0, false
}

// IsFloat reports whether q is a float format: raw rows, no scales, and no
// tensor-core kernel.
func IsFloat(q Quant) bool { return q.info().float > 0 }

// info is the whole format table.
//
// Every format reduces to value = scale*q - bias, which is what lets one
// kernel serve all of them. The repack at upload extracts the integer q and the
// per-sub-block scale; the zero point becomes a bias, either a constant times
// the scale or, for Q4_K/Q5_K, a separate per-sub-block minimum:
//
//	Q4_0  4 bits, /32   q 0..15   scale d          bias 8*scale
//	Q8_0  8 bits, /32   q signed  scale d          bias 0
//	Q4_K  4 bits, /32   q 0..15   scale d*sc[s]    bias dmin*m[s]   (array)
//	Q3_K  4 bits, /16   q 0..7    scale d*sc[g]    bias 4*scale
//	Q5_K  4+1,    /32   q 0..31   scale d*sc[s]    bias dmin*m[s]   (array)
//	Q6_K  4+2,    /16   q 0..63   scale d*sc[g]    bias 32*scale
type qinfo struct {
	bits int // PRIMARY plane width per weight: 2, 4 or 8 (Fields codes a byte)
	// hi is the secondary plane's width: 0 when the quants fit the primary
	// plane, 2 for Q6_K and 1 for Q5_K. The planes keep the payload at its
	// native width; decode is bandwidth-bound, so a byte per weight was a pure
	// loss. A weight is primary | hi<<bits (see packSub).
	hi        int
	sub       int     // elements sharing one scale: 16 or 32
	biasK     float32 // bias = biasK * scale, when biasArray is false
	biasArray bool    // the format stores its own per-sub-block minimum
	blockB    int     // GGUF bytes per source block
	blockE    int     // GGUF elements per source block
	// signedQ marks the one format (Q8_0) whose payload reaches a full signed
	// byte. Every other format's quants are small and unsigned, so with the
	// activations scaled to [-127,127] they can use ir.Dot4Bounded.
	signedQ bool
	// perSuper is sub-blocks sharing one super-scale d, and scOff the integer
	// added to the stored per-sub-block scale. perSuper == 1 means the format
	// has no per-sub-block scale at all and d is the scale.
	perSuper int
	scOff    int
	// codes, when set, is what a primary code means: the weight is
	// scale*codes[q] - bias. It exists for MXFP4, whose e2m1 code is a float
	// and not affine in any offset. The entries are biased to be unsigned, so
	// the dot and bias correction are the ordinary 4-bit ones.
	codes bool
	// float is bytes per weight for a float format (4 or 2), 0 otherwise;
	// bf16 picks bfloat16 over binary16 at 2.
	float int
	bf16  bool
	// e8m0 says the super-scale is a power of two stored as one byte rather
	// than an f16 (MXFP4's E8M0 exponent). See DSlots.
	e8m0 bool
	// intAct says the format's arithmetic is defined over int8 activations:
	// a ternary model is trained with every input of its projections
	// quantized per token, so a kernel that takes binary16 activations
	// (the Volta tile, GemmVolta) computes a different function, not a
	// rounding of the same one. Such kernels decline the format by name.
	intAct bool
	// d32 says the d word is the scale as an f32, whole -- a precision of
	// the scale, as perSuper is its grouping. A reader takes the word's bits
	// where every other format converts an f16 half.
	d32 bool
}

// Scales are stored unfolded: one f16 d per super-block plus one integer byte
// per sub-block. Folding d*sc[s] into one f16 would lose precision (the
// product needs about sixteen mantissa bits), and a folded f32 is larger.
//
// Named fields rather than positional: a bool was once added mid-struct and
// every positional literal silently changed meaning.
func qtab(q Quant) qinfo {
	switch q {
	case Q4_0:
		return qinfo{bits: 4, sub: 32, biasK: 8, blockB: 18, blockE: 32, perSuper: 1}
	case Q8_0:
		return qinfo{bits: 8, sub: 32, biasK: 0, blockB: 34, blockE: 32, perSuper: 1, signedQ: true}
	case Q4_K:
		return qinfo{bits: 4, sub: 32, biasArray: true, blockB: 144, blockE: 256, perSuper: 8}
	case Q3_K:
		return qinfo{bits: 4, sub: 16, biasK: 4, blockB: 110, blockE: 256, perSuper: 16, scOff: -32}
	// Q5_K is a 4-bit primary plane plus a one-bit secondary plane, the two-plane
	// form Q6_K uses; readers derive lanes and masks from qi.hi.
	case Q5_K:
		return qinfo{bits: 4, hi: 1, sub: 32, biasArray: true, blockB: 176, blockE: 256, perSuper: 8}
	case Q6_K:
		return qinfo{bits: 4, hi: 2, sub: 16, biasK: 32, blockB: 210, blockE: 256, perSuper: 16, scOff: -128}
	// Q5_0 is Q4_0's scales (perSuper 1: d is the scale) with Q5_K's one-bit
	// secondary plane.
	case Q5_0:
		return qinfo{bits: 4, hi: 1, sub: 32, biasK: 16, blockB: 22, blockE: 32, perSuper: 1}
	// MXFP4 is Q4_0 with a code table: the payload is the raw e2m1 code and the
	// kernels translate it; its E8M0 exponent is the one-byte d (see e8m0).
	case MXFP4:
		return qinfo{bits: 4, sub: 32, biasK: mxfp4Bias, blockB: 17, blockE: 32, perSuper: 1, codes: true, e8m0: true}
	// Q5_1 is Q5_0 with a minimum. Readers compute scale*q - dmin*m; with no sc
	// plane m is one, and the packer stores the minimum negated as dmin, so
	// d*q + m = d*q - (-m)*1: the k-quant formula at m = 1.
	case Q5_1:
		return qinfo{bits: 4, hi: 1, sub: 32, biasArray: true, blockB: 24, blockE: 32, perSuper: 1}
	// TQ2 is a 2-bit plane with a bias of one: q in {0, 1, 2} (3 is the
	// format's own unused code, read as 2), and the source block's one d
	// written on each of its eight sub-blocks.
	case TQ2:
		return qinfo{bits: 2, sub: 32, biasK: 1, blockB: 66, blockE: 256, perSuper: 1, intAct: true}
	// TQ2F is TQ2 with an f32 scale, a row to each d word.
	case TQ2F:
		return qinfo{bits: 2, sub: 32, biasK: 1, blockB: 68, blockE: 256, perSuper: 1, intAct: true, d32: true}
	// A "block" of a float format is one 32-element activation block.
	case Float32:
		return qinfo{sub: 32, blockB: 128, blockE: 32, perSuper: 1, float: 4}
	case Float16:
		return qinfo{sub: 32, blockB: 64, blockE: 32, perSuper: 1, float: 2}
	case BFloat16:
		return qinfo{sub: 32, blockB: 64, blockE: 32, perSuper: 1, float: 2, bf16: true}
	}
	return qinfo{}
}

// mxfp4Codes is quant.MXFP4Values() (the doubled e2m1 values, -12..12) plus
// mxfp4Bias, so every entry is an unsigned byte a VPDPBUSD can take.
func mxfp4Codes() (c [16]uint8) {
	for i, v := range quant.MXFP4Values() {
		c[i] = uint8(int(v) + mxfp4Bias)
	}
	return c
}

const mxfp4Bias = 12

// Codes is the format's code table, nil for a format whose code is its
// integer. A reader applies it to the primary code first.
func Codes(q Quant) *[16]uint8 {
	if !q.info().codes {
		return nil
	}
	c := mxfp4Codes()
	return &c
}

func (q Quant) info() qinfo { return qtab(q) }

// Layout exposes the device layout for tests: elements per scale, payload bits
// per weight, the constant bias multiplier, and whether the format stores its
// own per-sub-block minimum instead.
func Layout(q Quant) (sub, bits int, biasK float32, biasArray bool) {
	i := q.info()
	return i.sub, i.bits, i.biasK, i.biasArray
}

// D32 reports whether q's d word is the scale as an f32 (TQ2F): a reader
// takes its bits rather than converting an f16 half.
func D32(q Quant) bool { return q.info().d32 }

// E8M0 reports whether q's d word holds E8M0 exponents, one byte a block
// (MXFP4).
func E8M0(q Quant) bool { return q.info().e8m0 }

// HiPlane is the secondary payload plane's width, or 0 when the quants fit the
// primary one. A weight is primary | hi<<bits; see qinfo.hi and packSub.
func HiPlane(q Quant) int { return q.info().hi }

// ScaleLayout exposes how the scale is stored: sub-blocks per super-scale, and
// the integer offset added to the stored per-sub-block scale.
func ScaleLayout(q Quant) (perSuper, scOff int) {
	i := q.info()
	return i.perSuper, i.scOff
}

// SignedPayload reports whether the payload bytes are signed. Only Q8_0 is;
// a host kernel using an unsigned-times-signed multiply needs a bias of 128
// for it.
func SignedPayload(q Quant) bool { return q.info().signedQ }
