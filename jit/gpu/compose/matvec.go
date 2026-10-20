package compose

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// MatVecOperand is a matvec's int8 operand as the row group reads it: the
// codes (four to a word, K/4 words a token) and, at AX, each 32-element
// block's scale (K/32 a token), then each 16 elements' code sum after every
// token's scales.
type MatVecOperand struct {
	A, AX int // the group's buffers: BufU32 and BufRow (read by index)
	Toks  int // tokens the operand holds, so where the sums start
}

// MatVecWeight is a packed weight as the row group reads it: its payload,
// its super-scale plane and, for a format with per-sub-block scales, that
// plane (kernels.PackWeights' layout).
type MatVecWeight struct {
	T             kernels.Quant
	K, Rows       int
	QS, D, SC     int // the group's BufU32 buffers; SC -1 where the format has none
}

// MatVec is node: the product of the weight's row e (the element) with the
// operand's token t (the group's row), sum over k of w[e][k] a[t][k], by the
// definitions of the packed layout: a fold over the row's sub-blocks, each
// the integer dot of its codes with the operand's, scaled by the weight's
// and the operand's scales.
//
// The weight's value is scale*q - bias (kernels.PackWeights): for a format
// with signed codes and no bias that is d*q, and a sub-block's contribution
// d_w * d_a * sum(q_w q_a). A format whose layout this does not state yet is
// refused by name.
func (g *Row) MatVec(w MatVecWeight, a MatVecOperand) (int, error) {
	info := w.T.Info()
	switch {
	case info.Float > 0:
		return 0, fmt.Errorf("compose: a matvec over float weights (%v) has no node form yet", w.T)
	case info.Bits != 8 || info.Hi != 0 || info.PerSuper != 1 || info.BiasK != 0 || info.BiasArray ||
		info.Codes != nil || info.E8M0 || info.D32:
		return 0, fmt.Errorf("compose: a matvec over %v has no node form yet", w.T)
	case w.K%info.Sub != 0 || info.Sub%32 != 0:
		return 0, fmt.Errorf("compose: a matvec over %v at K=%d", w.T, w.K)
	}
	sub := info.Sub
	nsub := w.K / sub
	words := sub * info.Bits / 32 // payload words a sub-block
	slots := kernels.DSlots(w.T)
	e, t := g.Column(), g.RowIdx()
	u := g.UConst
	rows := u(w.Rows)

	c := g.Loop(nsub)
	s := g.Iter(c)
	acc := g.Fold(c, g.Const(0))
	// The sub-block's integer dot: payload word (s*words + j)*Rows + e, the
	// operand's word t*K/4 + s*words + j.
	isum := g.Add(Node{Kind: NIConst})
	for j := 0; j < words; j++ {
		wi := g.UBin(OpAdd, g.UBin(OpMul, g.UBin(OpAdd, g.UBin(OpMul, s, u(words)), u(j)), rows), e)
		ai := g.UBin(OpAdd, g.UBin(OpAdd, g.UBin(OpMul, t, u(w.K/4)), g.UBin(OpMul, s, u(sub/4))), u(j))
		isum = g.Add(Node{Kind: NDot4, A: g.LoadAt(w.QS, wi), B: g.LoadAt(a.A, ai), C: isum})
	}
	// The weight's scale: the binary16 of super-block s (one a sub-block
	// here), slots rows to a word, row e's half (kernels.DIndex).
	var d int
	if slots == 2 {
		half := u((w.Rows + 1) / 2)
		di := g.UBin(OpAdd, g.UBin(OpMul, s, half), g.UBin(OpDiv, e, u(2)))
		shift := g.UBin(OpMul, g.UBin(OpRem, e, u(2)), u(16))
		d = g.Add(Node{Kind: NUnary, Op: OpHalf, A: g.UBin(OpShr, g.LoadAt(w.D, di), shift)})
	} else {
		di := g.UBin(OpAdd, g.UBin(OpMul, s, rows), e)
		d = g.Add(Node{Kind: NUnary, Op: OpHalf, A: g.LoadAt(w.D, di)})
	}
	// The operand's scale of block t*K/32 + s (sub is 32 here).
	ax := g.LoadAt(a.AX, g.UBin(OpAdd, g.UBin(OpMul, t, u(w.K/32)), g.UBin(OpMul, s, u(sub/32))))
	q := g.Add(Node{Kind: NUnary, Op: OpFloat, A: isum})
	g.SetFold(acc, g.Add(Node{Kind: NBinary, Op: OpFma, A: q, B: g.Bin(OpMul, d, ax), C: acc}))
	return acc, nil
}
