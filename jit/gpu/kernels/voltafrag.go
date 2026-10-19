package kernels

import "github.com/jitllm/jitllm/jit/gpu/ir"

// The binary16 GEMM's fragments, for the GEMM former (compose.Gemm) that owns
// the loop nest: a weight format's dequantization to binary16 fragments
// (VoltaDeq), the format's numbers the staging is laid out by (GemmFormat),
// the shared-memory swizzles, and a warp lane's place in the m8n8k4
// instruction's fragments (VoltaLane). They are the arithmetic GemmVolta runs,
// so a formed GEMM and GemmVolta compute the same bits.

// GemmFormat is format q's numbers a GEMM stages by: the sub-block, the
// payload's bits and secondary plane's bits an element, a float format's
// bytes an element (0 for a quantized one) and the block a row's k is a
// multiple of.
func GemmFormat(q Quant) (sub, bits, hi, floatBytes, blockE int) {
	qi := q.info()
	return qi.sub, qi.bits, qi.hi, qi.float, qi.blockE
}

// VoltaDeq is a weight format's dequantization to binary16 fragments over
// rows weight rows: the packed words of a sub-block, its scale words, the
// scales they hold, and the fragments the instruction reads.
type VoltaDeq struct{ d *voltaDeq }

// NewVoltaDeq is the dequantization of format t's planes pQS, pD and pSC.
func NewVoltaDeq(b *ir.Builder, t Quant, pQS, pD, pSC ir.Value, rows int) *VoltaDeq {
	return &VoltaDeq{newVoltaDeq(b, t, pQS, pD, pSC, rows)}
}

// SetCarry takes the scale stream's carried form for k and kb sub-blocks a
// trip, where the format has one; it reports whether it did.
func (v *VoltaDeq) SetCarry(k, kb int) bool { return v.d.SetCarry(k, kb) }

// Words is a sub-block's packed words at index idx plus im.
func (v *VoltaDeq) Words(idx ir.Value, im int64) []ir.Value { return v.d.words(idx, im) }

// ScaleWords is a sub-block's scale words, for sub-block subIdx of row.
func (v *VoltaDeq) ScaleWords(subIdx, row ir.Value) []ir.Value { return v.d.scaleWords(subIdx, row) }

// ScalesOfX is the scale and minimum the scale words w hold, x the carried
// first word where the stream is carried.
func (v *VoltaDeq) ScalesOfX(w []ir.Value, x, subIdx, row ir.Value) (s2, m2 ir.Value) {
	return v.d.scalesOfX(w, x, subIdx, row)
}

// Frags is the sub-block's binary16 fragments from its packed words raw.
func (v *VoltaDeq) Frags(raw []ir.Value, s2, m2 ir.Value) [][]ir.Value { return v.d.frags(raw, s2, m2) }

// ScFirst is the carried word of the first trip.
func (v *VoltaDeq) ScFirst(subIdx, row ir.Value) ir.Value { return v.d.scFirst(subIdx, row) }

// ScNext is the carried word of the trip after cur, from this trip's x and
// its second scale word w1.
func (v *VoltaDeq) ScNext(x, w1, cur ir.Value, kb int) ir.Value { return v.d.scNext(x, w1, cur, kb) }

// F16Swizzle is an m16n8 tile's chunk permutation of a row (f16Swizzle).
func F16Swizzle(b *ir.Builder, r ir.Value) ir.Value { return f16Swizzle(b, r) }

// VoltaSwizzle is the m8n8k4 tile's chunk permutation of a row of c chunks
// (voltaSwizzle).
func VoltaSwizzle(b *ir.Builder, r ir.Value, c int, weights bool) ir.Value {
	return voltaSwizzle(b, r, c, weights)
}

// VoltaLane is a warp lane's place in the m8n8k4 instruction's fragments.
type VoltaLane struct{ v voltaLane }

// NewVoltaLane is lane's place.
func NewVoltaLane(b *ir.Builder, lane ir.Value) VoltaLane { return VoltaLane{voltaLane{b, lane}} }

// Quad is the lane's quad-pair; Row its A row (and B column); DRow and DCol
// accumulator i's output row and column.
func (l VoltaLane) Quad() ir.Value      { return l.v.quad() }
func (l VoltaLane) Row() ir.Value       { return l.v.row() }
func (l VoltaLane) DRow(i int) ir.Value { return l.v.dRow(i) }
func (l VoltaLane) DCol(i int) ir.Value { return l.v.dCol(i) }
