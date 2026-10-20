package compose

import "github.com/jitllm/jitllm/jit/gpu/kernels"

// RopeTable is the rotary table of g's rows, a (cos, sin) pair an element:
// the position (word 1+row of pos, word 0 the live rows) in base-128
// digits against the folded planes, reduced mod 4 on the planes' grid and
// evaluated by the host's polynomials (kernels.RopeTabConsts). g is Width
// pairs by Rows rows; out, planes, consts and pos are its buffers.
func (g *Row) RopeTable(out, planes, consts, pos int) {
	u, ub := g.UConst, g.UBin
	stride := kernels.RopeTabStride(g.Width)
	g.HasLive, g.LiveBuf, g.LiveAt = true, pos, 0
	cw := func(w int) int { return g.LoadAt(consts, u(w)) }
	cf := func(w int) int { return g.bits(cw(w)) }
	fl := func(x int) int { return g.Add(Node{Kind: NUnary, Op: OpFloat, A: x}) }
	fma := func(a, b, c int) int { return g.Add(Node{Kind: NBinary, Op: OpFma, A: a, B: b, C: c}) }
	p, row := g.Column(), g.RowIdx()
	at := g.LoadAt(pos, ub(OpAdd, row, u(1)))
	var d [kernels.RopeTabDigits]int
	for i := range d {
		x := at
		if i > 0 {
			x = ub(OpShr, at, u(i*kernels.RopeTabDigitBits))
		}
		d[i] = fl(ub(OpAnd, x, u(1<<kernels.RopeTabDigitBits-1)))
	}
	plane := func(i int) int { return g.LoadAt(planes, ub(OpAdd, p, u(i*stride))) }
	sum := func(base int) int {
		acc := g.Bin(OpMul, d[0], plane(base))
		for i := 1; i < len(d); i++ {
			acc = g.Bin(OpAdd, acc, g.Bin(OpMul, d[i], plane(base+i)))
		}
		return acc
	}
	s, c := sum(0), sum(len(d))
	si := g.Add(Node{Kind: NUnary, Op: OpAsWord, A: g.Add(Node{Kind: NUnary, Op: OpInt, A: g.Bin(OpMul, s, g.Const(kernels.RopeTabGrid))})})
	rem := g.Bin(OpMul, fl(ub(OpAnd, si, cw(kernels.RopeTabMod4MaskWord))), g.Const(1.0/kernels.RopeTabGrid))
	n := g.Add(Node{Kind: NUnary, Op: OpInt, A: g.Bin(OpAdd, g.Bin(OpAdd, rem, c), cf(kernels.RopeTabHalfWord))})
	f := g.Bin(OpSub, rem, fl(n))
	nb := g.Add(Node{Kind: NUnary, Op: OpAsWord, A: n})
	hi, lo := cf(kernels.RopeTabPio2HiWord), cf(kernels.RopeTabPio2LoWord)
	r := fma(f, hi, fma(f, lo, g.Bin(OpMul, c, hi)))
	z := g.Bin(OpMul, r, r)
	sp := fma(z, cf(kernels.RopeTabSinC7Word), cf(kernels.RopeTabSinC5Word))
	sp = fma(z, sp, cf(kernels.RopeTabSinC3Word))
	sp = fma(g.Bin(OpMul, r, z), sp, r)
	cp := cf(kernels.RopeTabCosC8Word)
	for _, w := range []int{kernels.RopeTabCosC6Word, kernels.RopeTabCosC4Word, kernels.RopeTabNegHalfWord, kernels.RopeTabOneWord} {
		cp = fma(z, cp, cf(w))
	}
	swap := g.Less(u(0), ub(OpAnd, nb, u(1)))
	sv, cv := g.Select(swap, cp, sp), g.Select(swap, sp, cp)
	sign := cw(kernels.RopeTabSignVecWord)
	neg := func(x, m int) int {
		return g.bits(ub(OpXor, g.word(x), ub(OpAnd, ub(OpShl, m, u(30)), sign)))
	}
	sv = neg(sv, nb)
	cv = neg(cv, ub(OpAdd, nb, cw(kernels.RopeTabOneVecWord)))
	ms := g.LoadAt(planes, u(8*stride))
	o := ub(OpMul, ub(OpAdd, ub(OpMul, row, u(g.Width)), p), u(2))
	g.StoreAt(out, g.Bin(OpMul, cv, ms), o)
	g.StoreAt(out, g.Bin(OpMul, sv, ms), ub(OpAdd, o, u(1)))
}
