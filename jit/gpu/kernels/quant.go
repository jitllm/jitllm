// Package kernels is the device weight formats' vocabulary: the formats
// a device reads (Quant) and how each is laid out and packed, the activation
// kinds, the arena weights live in and the rotary table's constants. It
// emits no kernel; the composer forms them (jit/gpu/compose).
package kernels

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
	// carries both halves the way a k-quant's does; see MinInD.
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
	codes *[16]uint8
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
var qtab = [...]qinfo{
	Q4_0: {bits: 4, sub: 32, biasK: 8, blockB: 18, blockE: 32, perSuper: 1},
	Q8_0: {bits: 8, sub: 32, biasK: 0, blockB: 34, blockE: 32, perSuper: 1, signedQ: true},
	Q4_K: {bits: 4, sub: 32, biasArray: true, blockB: 144, blockE: 256, perSuper: 8},
	Q3_K: {bits: 4, sub: 16, biasK: 4, blockB: 110, blockE: 256, perSuper: 16, scOff: -32},
	// Q5_K is a 4-bit primary plane plus a one-bit secondary plane, the two-plane
	// form Q6_K uses; readers derive lanes and masks from qi.hi.
	Q5_K: {bits: 4, hi: 1, sub: 32, biasArray: true, blockB: 176, blockE: 256, perSuper: 8},
	Q6_K: {bits: 4, hi: 2, sub: 16, biasK: 32, blockB: 210, blockE: 256, perSuper: 16, scOff: -128},
	// Q5_0 is Q4_0's scales (perSuper 1: d is the scale) with Q5_K's one-bit
	// secondary plane.
	Q5_0: {bits: 4, hi: 1, sub: 32, biasK: 16, blockB: 22, blockE: 32, perSuper: 1},
	// MXFP4 is Q4_0 with a code table: the payload is the raw e2m1 code and the
	// kernels translate it; its E8M0 exponent is the one-byte d (see e8m0).
	MXFP4: {bits: 4, sub: 32, biasK: mxfp4Bias, blockB: 17, blockE: 32, perSuper: 1, codes: &mxfp4Codes, e8m0: true},
	// Q5_1 is Q5_0 with a minimum. Readers compute scale*q - dmin*m; with no sc
	// plane m is one, and the packer stores the minimum negated as dmin, so
	// d*q + m = d*q - (-m)*1. See MinInD.
	Q5_1: {bits: 4, hi: 1, sub: 32, biasArray: true, blockB: 24, blockE: 32, perSuper: 1},
	// TQ2 is a 2-bit plane with a bias of one: q in {0, 1, 2} (3 is the
	// format's own unused code, read as 2), and the source block's one d
	// written on each of its eight sub-blocks.
	TQ2: {bits: 2, sub: 32, biasK: 1, blockB: 66, blockE: 256, perSuper: 1, intAct: true},
	// TQ2F is TQ2 with an f32 scale, a row to each d word.
	TQ2F: {bits: 2, sub: 32, biasK: 1, blockB: 68, blockE: 256, perSuper: 1, intAct: true, d32: true},
	// A "block" of a float format is one 32-element activation block.
	Float32:  {sub: 32, blockB: 128, blockE: 32, perSuper: 1, float: 4},
	Float16:  {sub: 32, blockB: 64, blockE: 32, perSuper: 1, float: 2},
	BFloat16: {sub: 32, blockB: 64, blockE: 32, perSuper: 1, float: 2, bf16: true},
}

// mxfp4Codes is quant.MXFP4Values (the doubled e2m1 values, -12..12) plus
// mxfp4Bias, so every entry is an unsigned byte a VPDPBUSD can take.
var mxfp4Codes = func() (c [16]uint8) {
	for i, v := range quant.MXFP4Values {
		c[i] = uint8(int(v) + mxfp4Bias)
	}
	return c
}()

const mxfp4Bias = 12

// DeviceWhyNot says why the device kernels cannot run a format, or "" when
// they can. A format the host packs and the device cannot decode is declined
// by name, never lowered as though its code were an integer.
func DeviceWhyNot(q Quant) string {
	if c := q.info().codes; c != nil && c != &mxfp4Codes {
		return q.String() + "'s code table has no device decoder; only MXFP4's e2m1 has one (decodeE2M1)"
	}
	return ""
}

// Codes is the format's code table, nil for a format whose code is its
// integer. A reader applies it to the primary code first.
func Codes(q Quant) *[16]uint8 { return q.info().codes }

func (q Quant) info() qinfo { return qtab[q] }

// QInfo is a format's layout, every field qinfo's: what a kernel emitter
// reads to address and decode the format.
type QInfo struct {
	Bits, Hi, Sub   int
	BiasK           float32
	BiasArray       bool
	BlockB, BlockE  int
	SignedQ         bool
	PerSuper, ScOff int
	Codes           *[16]uint8
	Float           int
	BF16, E8M0      bool
	IntAct, D32     bool
}

// Info is q's layout.
func (q Quant) Info() QInfo {
	i := qtab[q]
	return QInfo{Bits: i.bits, Hi: i.hi, Sub: i.sub, BiasK: i.biasK, BiasArray: i.biasArray, BlockB: i.blockB,
		BlockE: i.blockE, SignedQ: i.signedQ, PerSuper: i.perSuper, ScOff: i.scOff, Codes: i.codes, Float: i.float,
		BF16: i.bf16, E8M0: i.e8m0, IntAct: i.intAct, D32: i.d32}
}

// Quants is how many formats there are: every Quant below it is one.
func Quants() int { return len(qtab) }

// MXFP4Bias is the offset MXFP4's biased e2m1 codes carry.
const MXFP4Bias = mxfp4Bias

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

// MinInD reports a format whose minimum is the d word's high half alone, with
// no per-sub-block integer to multiply it by: Q5_1. Its bias is dmin itself,
// the k-quant formula at m = 1.
func MinInD(q Quant) bool { i := q.info(); return i.biasArray && i.perSuper == 1 }

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

// MatVecShape is a decode matvec: out[row] = dot(quantized row, activations).
//
// K and Rows are baked; shapes come from the container, so nothing is
// speculative.
