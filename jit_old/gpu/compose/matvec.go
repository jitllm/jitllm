package compose

import (
	"fmt"

	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// MatVecOperand is a matvec's operand as the row group reads it. For a
// quantized weight, the int8 operand: the codes (four to a word, K/4 words
// a token) and, at AX, each 32-element block's scale (K/32 a token), then
// each 16 elements' code sum after every token's scales. For a float weight,
// the float rows at X, K a token.
type MatVecOperand struct {
	A, AX int // the group's buffers: BufU32 and BufRow (read by index)
	X     int // the float rows (BufRow, read by index)
	Toks  int // tokens the operand holds, so where the sums start
	// One is every group row reading the operand's first token: the slots of
	// a mixture's gate and up, which share one activation.
	One bool
}

// MatVecWeight is a packed weight as the row group reads it: its payload,
// its super-scale plane and, for a format with per-sub-block scales, that
// plane (kernels.PackWeights' layout).
type MatVecWeight struct {
	T         kernels.Quant
	K, Rows   int
	QS, D, SC int // the group's BufU32 buffers; SC -1 where the format has none
	// Experts, above one, is a bank of that many sheets back to back, each
	// row of the group reading the sheet its expert id at Sel (BufU32, a row
	// an id) names.
	Experts, Sel int
	// SelPer, above one, is the rows one id at Sel serves: a sorted run's
	// group of tokens.
	SelPer int
}

// MatVec is node: the product of the weight's row e (the element) with the
// operand's token t (the group's row), sum over k of w[e][k] a[t][k], by the
// definitions of the packed layout: a fold over the row's sub-blocks, each
// the integer dot of its codes with the operand's (a fold over its words),
// scaled by the weight's scale and then the operand's.
//
// The weight's value is scale*q - bias (kernels.PackWeights): for a format
// with signed codes and no bias that is d*q, and a sub-block's contribution
// d_w * d_a * sum(q_w q_a). A format whose layout this does not state yet is
// refused by name.
func (g *Row) MatVec(w MatVecWeight, a MatVecOperand) (int, error) {
	info := w.T.Info()
	if info.Float > 0 {
		return g.floatMatVec(w, a, info)
	}
	if info.Bits == 2 && info.Hi == 0 && info.PerSuper == 1 && !info.BiasArray && info.Codes == nil && !info.E8M0 {
		return g.planeMatVec(w, a, info)
	}
	if info.Bits == 4 && !info.D32 && (info.Codes == nil || kernels.DeviceWhyNot(w.T) == "") {
		return g.nibbleMatVec(w, a, info)
	}
	switch {
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
	sheetQ, sheetD, err := g.sheets(w, t)
	if err != nil {
		return 0, err
	}
	at := a.token(g, t)

	c := g.Loop(nsub)
	s := g.Iter(c)
	acc := g.Fold(c, g.Const(0))
	// The sub-block's integer dot, a fold over its words j: payload word
	// (s*words + j)*Rows + e, the operand's word t*K/4 + s*words + j.
	cj := g.Loop(words)
	j := g.Iter(cj)
	isum := g.Fold(cj, g.Add(Node{Kind: NIConst}))
	wi := sheetQ(g.UBin(OpAdd, g.UBin(OpMul, g.UBin(OpAdd, g.UBin(OpMul, s, u(words)), j), rows), e))
	ai := g.UBin(OpAdd, g.UBin(OpAdd, g.UBin(OpMul, at, u(w.K/4)), g.UBin(OpMul, s, u(sub/4))), j)
	g.SetFold(isum, g.Add(Node{Kind: NDot4, A: g.LoadAt(w.QS, wi), B: g.LoadAt(a.A, ai), C: isum}))
	// The weight's scale: the binary16 of super-block s (one a sub-block
	// here), slots rows to a word, row e's half (kernels.DIndex).
	var d int
	if slots == 2 {
		half := u((w.Rows + 1) / 2)
		di := sheetD(g.UBin(OpAdd, g.UBin(OpMul, s, half), g.UBin(OpDiv, e, u(2))))
		shift := g.UBin(OpMul, g.UBin(OpRem, e, u(2)), u(16))
		d = g.Add(Node{Kind: NUnary, Op: OpHalf, A: g.UBin(OpShr, g.LoadAt(w.D, di), shift)})
	} else {
		di := sheetD(g.UBin(OpAdd, g.UBin(OpMul, s, rows), e))
		d = g.Add(Node{Kind: NUnary, Op: OpHalf, A: g.LoadAt(w.D, di)})
	}
	// The operand's scale of block t*K/32 + s (sub is 32 here).
	ax := g.LoadAt(a.AX, g.UBin(OpAdd, g.UBin(OpMul, at, u(w.K/32)), g.UBin(OpMul, s, u(sub/32))))
	// The weight's scale on the integer dot, the operand's by the fused add.
	q := g.Add(Node{Kind: NUnary, Op: OpFloat, A: isum})
	g.SetFold(acc, g.Add(Node{Kind: NBinary, Op: OpFma, A: g.Bin(OpMul, q, d), B: ax, C: acc}))
	return acc, nil
}

// floatMatVec is MatVec over a float weight, stored row-major (row e's K
// values at e*K, info.Float bytes each, a word holding 4/Float of them low
// first): a fold over the row's sub-blocks (32 elements, or the widest of
// 16, 8, 4 dividing K where 32 does not), each the sum of
// its products in element order from zero, added into the accumulator. A
// binary16 element is its half of the word; a bfloat16 one the half moved
// to the top of an f32.
func (g *Row) floatMatVec(w MatVecWeight, a MatVecOperand, info kernels.QInfo) (int, error) {
	sub := 32
	for _, n := range []int{32, 16, 8, 4} {
		if w.K%n == 0 {
			sub = n
			break
		}
	}
	if w.K%sub != 0 || (info.Float != 4 && info.Float != 2) {
		return 0, fmt.Errorf("compose: a float matvec over %v at K=%d", w.T, w.K)
	}
	per := 4 / info.Float // elements a word
	e, t := g.Column(), g.RowIdx()
	u := g.UConst
	sheetQ, _, err := g.sheets(w, t)
	if err != nil {
		return 0, err
	}
	at := a.token(g, t)
	c := g.Loop(w.K / sub)
	s := g.Iter(c)
	acc := g.Fold(c, g.Const(0))
	wRow := sheetQ(g.UBin(OpAdd, g.UBin(OpMul, e, u(w.K/per)), g.UBin(OpMul, s, u(sub/per))))
	// The operand's element is its int8 word's, four elements a word: what the
	// quantized formats read the operand by.
	aw := g.UBin(OpAdd, g.UBin(OpMul, at, u(w.K/4)), g.UBin(OpMul, s, u(sub/4)))
	xRow := g.UBin(OpMul, aw, u(4))
	dot := g.Const(0)
	for i := 0; i < sub; i++ {
		word := g.LoadAt(w.QS, g.UBin(OpAdd, wRow, u(i/per)))
		var wv int
		switch {
		case per == 1:
			wv = g.Add(Node{Kind: NUnary, Op: OpBits, A: word})
		case info.BF16 && i%2 == 0:
			wv = g.Add(Node{Kind: NUnary, Op: OpBits, A: g.UBin(OpShl, word, u(16))})
		case info.BF16:
			wv = g.Add(Node{Kind: NUnary, Op: OpBits, A: g.UBin(OpAnd, word, u(0xFFFF0000))})
		case i%2 == 0:
			wv = g.Add(Node{Kind: NUnary, Op: OpHalf, A: word})
		default:
			wv = g.Add(Node{Kind: NUnary, Op: OpHalf, A: g.UBin(OpShr, word, u(16))})
		}
		xv := g.LoadAt(a.X, g.UBin(OpAdd, xRow, u(i)))
		dot = g.Add(Node{Kind: NBinary, Op: OpFma, A: wv, B: xv, C: dot})
	}
	g.SetFold(acc, g.Bin(OpAdd, acc, dot))
	return acc, nil
}

// SumPlanes is the element's sum over planes partial planes of buffer part
// (BufRow, read by index), plane p at p*Rows*Width: what a split's segments
// wrote, summed in segment order from zero.
func (g *Row) SumPlanes(part, planes int) int {
	u := g.UConst
	at := g.UBin(OpAdd, g.UBin(OpMul, g.RowIdx(), u(g.Width)), g.Column())
	return g.FoldSum(planes, func(p int) int {
		return g.LoadAt(part, g.UBin(OpAdd, g.UBin(OpMul, p, u(g.Rows*g.Width)), at))
	})
}

// sheets are the offsets a bank's row t adds into the payload and the scale
// plane: its expert's sheet, the id at Sel times a sheet's words; the index
// itself for a dense weight.
func (g *Row) sheets(w MatVecWeight, t int) (q, d func(int) int, err error) {
	id := func(i int) int { return i }
	if w.Experts <= 1 {
		return id, id, nil
	}
	nq, nd, _, err := kernels.PackedWords(w.T, w.Rows, w.K)
	if err != nil {
		return nil, nil, err
	}
	at := t
	if w.SelPer > 1 {
		at = g.UBin(OpDiv, t, g.UConst(w.SelPer))
	}
	e := g.LoadAt(w.Sel, at)
	off := func(words int) func(int) int {
		return func(i int) int { return g.UBin(OpAdd, g.UBin(OpMul, e, g.UConst(words)), i) }
	}
	return off(nq), off(nd), nil
}

// token is the operand's token group row t reads: t, or the first where
// every row shares it.
func (a MatVecOperand) token(g *Row, t int) int {
	if a.One {
		return g.UConst(0)
	}
	return t
}

// planeMatVec is a matvec over a 2-bit plane with a constant zero point
// (TQ2): each word four fields of four codes, field f's codes against the
// operand's words f*words on; the integer dot corrected by the zero point
// times the operand's sum, then scaled -- an f16 scale, or the d word itself
// as an f32 (D32).
func (g *Row) planeMatVec(w MatVecWeight, a MatVecOperand, info kernels.QInfo) (int, error) {
	sub := info.Sub
	if w.K%sub != 0 || sub%32 != 0 {
		return 0, fmt.Errorf("compose: a matvec over %v at K=%d", w.T, w.K)
	}
	nsub := w.K / sub
	words := sub * info.Bits / 32
	slots := kernels.DSlots(w.T)
	e, t := g.Column(), g.RowIdx()
	u, ub := g.UConst, g.UBin
	rows := u(w.Rows)
	sheetQ, sheetD, err := g.sheets(w, t)
	if err != nil {
		return 0, err
	}
	at := a.token(g, t)
	c := g.Loop(nsub)
	s := g.Iter(c)
	acc := g.Fold(c, g.Const(0))
	cj := g.Loop(words)
	j := g.Iter(cj)
	isum := g.Fold(cj, g.Add(Node{Kind: NIConst}))
	x := g.LoadAt(w.QS, sheetQ(ub(OpAdd, ub(OpMul, ub(OpAdd, ub(OpMul, s, u(words)), j), rows), e)))
	aw := ub(OpAdd, ub(OpAdd, ub(OpMul, at, u(w.K/4)), ub(OpMul, s, u(sub/4))), j)
	dot := isum
	for f := 0; f < 4; f++ {
		q := x
		if f > 0 {
			q = ub(OpShr, x, u(2*f))
		}
		q = g.zeroPoint(ub(OpAnd, q, u(0x03030303)), info.BiasK)
		dot = g.Add(Node{Kind: NDot4, A: q, B: g.LoadAt(a.A, ub(OpAdd, aw, u(f*words))), C: dot, At: 1})
	}
	g.SetFold(isum, dot)
	var d int
	switch {
	case info.D32:
		d = g.Add(Node{Kind: NUnary, Op: OpBits, A: g.LoadAt(w.D, sheetD(ub(OpAdd, ub(OpMul, s, rows), e)))})
	case slots == 2:
		half := u((w.Rows + 1) / 2)
		di := sheetD(ub(OpAdd, ub(OpMul, s, half), ub(OpDiv, e, u(2))))
		shift := ub(OpMul, ub(OpRem, e, u(2)), u(16))
		d = g.Add(Node{Kind: NUnary, Op: OpHalf, A: ub(OpShr, g.LoadAt(w.D, di), shift)})
	default:
		d = g.Add(Node{Kind: NUnary, Op: OpHalf, A: g.LoadAt(w.D, sheetD(ub(OpAdd, ub(OpMul, s, rows), e)))})
	}
	nb := a.Toks * w.K / 32
	si := ub(OpAdd, u(nb), ub(OpAdd, ub(OpMul, at, u(w.K/16)), ub(OpMul, s, u(sub/16))))
	sum := g.LoadAt(a.AX, si)
	for k := 1; k < sub/16; k++ {
		sum = g.Bin(OpAdd, sum, g.LoadAt(a.AX, ub(OpAdd, si, u(k))))
	}
	q := g.Add(Node{Kind: NUnary, Op: OpFloat, A: isum})
	v := g.corrected(g.Bin(OpMul, d, g.Add(Node{Kind: NBinary, Op: OpFma, A: g.Const(-info.BiasK), B: sum, C: q})), g.Bin(OpMul, q, d))
	ax := g.LoadAt(a.AX, ub(OpAdd, ub(OpMul, at, u(w.K/32)), ub(OpMul, s, u(sub/32))))
	g.SetFold(acc, g.Add(Node{Kind: NBinary, Op: OpFma, A: v, B: ax, C: acc}))
	return acc, nil
}

// nibbleMatVec is MatVec over a 4-bit plane (Q4_0, Q5_0, Q5_1, Q3_K,
// Q4_K, Q5_K, Q6_K, MXFP4): a sub-block's words, each word's low nibbles
// the sub-block's first half and its high nibbles the second, a byte lane
// of four codes against the operand's word; a secondary plane's codes
// (Hi bits a weight, read first) joined above each nibble; a coded
// format's codes mapped to their values (e2m1). The integer dot is scaled
// by the super-block's scale (an f16, or an E8M0 exponent byte, several
// rows to a word: kernels.DIndex) times the sub-block's integer scale
// (a byte of the sc plane, offset by ScOff, or a six-bit field of the
// k-quants' stream), less the operand's sum times the zero point or the
// sub-block's minimum.
func (g *Row) nibbleMatVec(w MatVecWeight, a MatVecOperand, info kernels.QInfo) (int, error) {
	sub := info.Sub
	if w.K%sub != 0 || sub%16 != 0 {
		return 0, fmt.Errorf("compose: a matvec over %v at K=%d", w.T, w.K)
	}
	if info.PerSuper > 1 && w.SC < 0 {
		return 0, fmt.Errorf("compose: a matvec over %v with no sub-scale plane", w.T)
	}
	nsub := w.K / sub
	pw, hw := sub*4/32, sub*info.Hi/32
	words := pw + hw
	slots := kernels.DSlots(w.T)
	e, t := g.Column(), g.RowIdx()
	u, ub := g.UConst, g.UBin
	rows := u(w.Rows)
	sheetQ, sheetD, err := g.sheets(w, t)
	if err != nil {
		return 0, err
	}
	sheetSC := sheetD
	if w.Experts > 1 && info.PerSuper > 1 {
		_, _, nsc, err := kernels.PackedWords(w.T, w.Rows, w.K)
		if err != nil {
			return 0, err
		}
		at := t
		if w.SelPer > 1 {
			at = ub(OpDiv, t, u(w.SelPer))
		}
		ex := g.LoadAt(w.Sel, at)
		sheetSC = func(i int) int { return ub(OpAdd, ub(OpMul, ex, u(nsc)), i) }
	}
	at := a.token(g, t)
	c := g.Loop(nsub)
	s := g.Iter(c)
	acc := g.Fold(c, g.Const(0))
	word := func(k int) int {
		return g.LoadAt(w.QS, sheetQ(ub(OpAdd, ub(OpMul, ub(OpAdd, ub(OpMul, s, u(words)), u(k)), rows), e)))
	}
	var his []int
	for k := 0; k < hw; k++ {
		his = append(his, word(pw+k))
	}
	hiOf := func(gi int) int {
		lanes := 8 / info.Hi
		v := his[gi/lanes]
		if sh := info.Hi * (gi % lanes); sh > 0 {
			v = ub(OpShr, v, u(sh))
		}
		m := uint32(1)<<uint(info.Hi) - 1
		return ub(OpShl, ub(OpAnd, v, u(int(m|m<<8|m<<16|m<<24))), u(4))
	}
	bounded := 1
	if info.SignedQ {
		bounded = 0
	}
	mask := u(0x0F0F0F0F)
	aw0 := ub(OpAdd, ub(OpMul, at, u(w.K/4)), ub(OpMul, s, u(sub/4)))
	dot := g.Add(Node{Kind: NIConst})
	for k := 0; k < pw; k++ {
		x := word(k)
		lo, hi := ub(OpAnd, x, mask), ub(OpAnd, ub(OpShr, x, u(4)), mask)
		if info.Codes != nil {
			lo, hi = g.e2m1(lo), g.e2m1(hi)
		}
		if hw > 0 {
			lo, hi = ub(OpXor, lo, hiOf(k)), ub(OpXor, hi, hiOf(k+pw))
		}
		if info.BiasK != 0 && !info.BiasArray {
			lo, hi = g.zeroPoint(lo, info.BiasK), g.zeroPoint(hi, info.BiasK)
		}
		dot = g.Add(Node{Kind: NDot4, A: lo, B: g.LoadAt(a.A, ub(OpAdd, aw0, u(k))), C: dot, At: bounded})
		dot = g.Add(Node{Kind: NDot4, A: hi, B: g.LoadAt(a.A, ub(OpAdd, aw0, u(k+sub/8))), C: dot, At: bounded})
	}
	super := s
	if info.PerSuper > 1 {
		super = ub(OpShr, s, u(log2(info.PerSuper)))
	}
	var dw int
	if slots > 1 {
		di := sheetD(ub(OpAdd, ub(OpMul, super, u((w.Rows+slots-1)/slots)), ub(OpDiv, e, u(slots))))
		dw = ub(OpShr, g.LoadAt(w.D, di), ub(OpMul, ub(OpRem, e, u(slots)), u(32/slots)))
	} else {
		dw = g.LoadAt(w.D, sheetD(ub(OpAdd, ub(OpMul, super, rows), e)))
	}
	var d int
	if info.E8M0 {
		d = g.Add(Node{Kind: NUnary, Op: OpBits, A: ub(OpShl, ub(OpAnd, dw, u(0xFF)), u(23))})
	} else {
		d = g.Add(Node{Kind: NUnary, Op: OpHalf, A: dw})
	}
	var dmin int
	if info.BiasArray {
		dmin = g.Add(Node{Kind: NUnary, Op: OpHalf, A: ub(OpShr, dw, u(16))})
	}
	scale, mInt := d, -1
	float := func(n int) int { return g.Add(Node{Kind: NUnary, Op: OpFloat, A: n}) }
	scAt := func(wi int) int { return g.LoadAt(w.SC, sheetSC(ub(OpAdd, ub(OpMul, wi, rows), e))) }
	switch {
	case kernels.ScStream(w.T):
		pos := ub(OpMul, s, u(12))
		lo, bit := ub(OpShr, pos, u(5)), ub(OpAnd, pos, u(31))
		hi := ub(OpAdd, lo, g.Select(g.Less(u(20), bit), u(1), u(0)))
		pair := g.Add(Node{Kind: NUBinary, Op: OpFunnel, A: scAt(lo), B: scAt(hi), C: bit})
		scale = g.Bin(OpMul, d, float(ub(OpAnd, pair, u(63))))
		mInt = ub(OpAnd, ub(OpShr, pair, u(6)), u(63))
	case info.PerSuper > 1:
		per, stride := 4, 8
		if info.BiasArray {
			per, stride = 2, 16
		}
		sw := scAt(ub(OpShr, s, u(log2(per))))
		base := ub(OpMul, ub(OpAnd, s, u(per-1)), u(stride))
		sc := ub(OpAnd, ub(OpShr, sw, base), u(0xFF))
		if info.ScOff != 0 {
			sc = ub(OpSub, sc, g.Add(Node{Kind: NIConst, At: -info.ScOff}))
		}
		scale = g.Bin(OpMul, d, float(sc))
		if info.BiasArray {
			mInt = ub(OpAnd, ub(OpShr, sw, ub(OpAdd, base, u(8))), u(0xFF))
		}
	}
	ablk := s
	if sub == 16 {
		ablk = ub(OpShr, s, u(1))
	}
	da := g.LoadAt(a.AX, ub(OpAdd, ub(OpMul, at, u(w.K/32)), ablk))
	sumA := func() int {
		base := ub(OpAdd, u(a.Toks*w.K/32), ub(OpMul, at, u(w.K/16)))
		if sub == 16 {
			return g.LoadAt(a.AX, ub(OpAdd, base, s))
		}
		i0 := ub(OpAdd, base, ub(OpMul, s, u(2)))
		return g.Bin(OpAdd, g.LoadAt(a.AX, i0), g.LoadAt(a.AX, ub(OpAdd, i0, u(1))))
	}
	q := float(dot)
	var v int
	switch {
	case info.BiasArray:
		bias := dmin
		if mInt >= 0 {
			bias = g.Bin(OpMul, dmin, float(mInt))
		}
		v = g.Bin(OpSub, g.Bin(OpMul, q, scale), g.Bin(OpMul, sumA(), bias))
	case info.BiasK == 0:
		v = g.Bin(OpMul, q, scale)
	default:
		v = g.corrected(g.Bin(OpMul, scale, g.Add(Node{Kind: NBinary, Op: OpFma, A: g.Const(-info.BiasK), B: sumA(), C: q})), g.Bin(OpMul, q, scale))
	}
	g.SetFold(acc, g.Add(Node{Kind: NBinary, Op: OpFma, A: v, B: da, C: acc}))
	return acc, nil
}

// e2m1 maps four e2m1 codes, a nibble in each byte lane of x, to their
// biased values (the code table kernels.Codes holds) by arithmetic: with q
// = s|e1|e0|m, the doubled magnitude l + (l if b2) + 4*b2 + 2*(b0 b1 b2), l
// = q&3, and the biased code (v+bias) - (2v if s).
func (g *Row) e2m1(x int) int {
	u, ub := g.UConst, g.UBin
	ones, ff := u(0x01010101), u(0xFF)
	l := ub(OpAnd, x, u(0x03030303))
	b2 := ub(OpAnd, ub(OpShr, x, u(2)), ones)
	v := ub(OpAdd, l, ub(OpAnd, l, ub(OpMul, b2, ff)))
	v = ub(OpAdd, v, ub(OpShl, b2, u(2)))
	t := ub(OpAnd, ub(OpAnd, x, ub(OpShr, x, u(1))), b2)
	v = ub(OpAdd, v, ub(OpShl, t, u(1)))
	neg := ub(OpMul, ub(OpAnd, ub(OpShr, x, u(3)), ones), ff)
	b := uint32(kernels.MXFP4Bias)
	return ub(OpSub, ub(OpAdd, v, u(int(b|b<<8|b<<16|b<<24))), ub(OpAnd, ub(OpShl, v, u(1)), neg))
}

// centreTok is the tokens a thread from which a constant zero point rides
// the weights: a fixed cost per weight word against a correction per token.
const centreTok = 8

// zeroPoint is codes x against zero point k, for the centring pass.
func (g *Row) zeroPoint(x int, k float32) int {
	return g.Add(Node{Kind: NHint, Hint: HintZeroPoint, A: x, At: int(k)})
}

// corrected is v, the corrected value, or c where the codes are centred.
func (g *Row) corrected(v, c int) int {
	return g.Add(Node{Kind: NHint, Hint: HintCorrected, A: v, B: c})
}

// centred takes each zero point's hints under schedule s: at centreTok
// tokens a thread the codes carry the zero point, each byte moved by
// 0x80 - k and its sign bit flipped (an add and an xor: the bytes stay
// above -128), and the corrected values are the centred ones; else the
// codes and the corrections stand.
func (g *Row) centred(s Schedule) *Row {
	var r *Row
	on := max(s.Tile, 1) >= centreTok && !s.MMA
	for i, nd := range g.Nodes {
		if nd.Kind != NHint || nd.Hint != HintZeroPoint && nd.Hint != HintCorrected {
			continue
		}
		if r == nil {
			r = g.clone()
		}
		n := nd.A
		switch {
		case on && nd.Hint == HintCorrected:
			n = nd.B
		case on:
			c := uint32(0x80-nd.At) & 0xFF
			n = r.UBin(OpXor, r.UBin(OpAdd, nd.A, r.UConst(int(c|c<<8|c<<16|c<<24))), r.UConst(0x80808080))
		}
		r.Nodes[i] = r.Nodes[n]
	}
	if r == nil {
		return g
	}
	return r
}
