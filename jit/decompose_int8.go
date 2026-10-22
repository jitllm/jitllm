package jit

import (
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/ir"
)

// abs is |a| by its sign bit.
func (x *exprs) abs(a int) int {
	return x.un(OpBitcast, ir.F32, x.op(OpAnd, ir.U32, x.un(OpBitcast, ir.U32, a), x.u(0x7fffffff)))
}

// scale is amax/127 correctly rounded: the mantissa's quotient in integers
// where the exponent allows it, the float division where it does not.
func (x *exprs) scale(amax int) int {
	U := ir.U32
	b := x.un(OpBitcast, U, amax)
	e := x.op(OpShr, U, b, x.u(23))
	m := x.op(OpAdd, U, x.op(OpAnd, U, b, x.u(0x7fffff)), x.u(0x800000))
	q := x.op(OpDiv, U, x.op(OpShl, U, m, x.u(8)), x.u(127))
	sh := x.op(OpAdd, U, x.op(OpShr, U, q, x.u(25)), x.u(1))
	half := x.op(OpShl, U, x.u(1), x.op(OpSub, U, sh, x.u(1)))
	mask := x.op(OpSub, U, x.op(OpShl, U, half, x.u(1)), x.u(1))
	round := x.sel(U, x.op(OpLess, ir.Pred, x.op(OpAnd, U, q, mask), half), x.u(0), x.u(1))
	q = x.op(OpAdd, U, x.op(OpShr, U, q, sh), round)
	over := x.op(OpShr, U, q, x.u(24))
	exp := x.op(OpAdd, U, x.op(OpSub, U, x.op(OpAdd, U, e, sh), x.u(8)), over)
	bits := x.op(OpAdd, U, x.op(OpShl, U, exp, x.u(23)), x.op(OpAnd, U, x.op(OpShr, U, q, over), x.u(0x7fffff)))
	exact := x.op(OpLess, ir.Pred, x.op(OpSub, U, e, x.u(8)), x.u(247))
	return x.sel(ir.F32, exact, x.un(OpBitcast, ir.F32, bits), x.op(OpDiv, ir.F32, amax, x.f(127)))
}

// inverse is 1/d for d = scale(amax): 127/amax refined once by Newton, then
// corrected to the float nearest 1/d by testing the four candidates below
// it in integers where every exponent allows, zero where d is not positive.
func (x *exprs) inverse(amax, d int) int {
	U, F := ir.U32, ir.F32
	r := x.op(OpDiv, F, x.f(127), amax)
	r = x.op(OpAdd, F, r, x.op(OpMul, F, r, x.op(OpSub, F, x.f(1), x.op(OpMul, F, d, r))))
	pos := x.op(OpLess, ir.Pred, x.f(0), d)
	ob, db := x.un(OpBitcast, U, x.f(1)), x.un(OpBitcast, U, d)
	oe, de := x.op(OpShr, U, ob, x.u(23)), x.op(OpShr, U, db, x.u(23))
	om := x.op(OpAdd, U, x.op(OpAnd, U, ob, x.u(0x7fffff)), x.u(0x800000))
	dm := x.op(OpAdd, U, x.op(OpAnd, U, db, x.u(0x7fffff)), x.u(0x800000))
	rb := x.un(OpBitcast, U, r)
	base := x.op(OpSub, U, rb, x.u(2))
	acc := base
	for k := uint32(0); k < 4; k++ {
		c := x.op(OpAdd, U, base, x.u(k))
		sh := x.op(OpSub, U, x.op(OpAdd, U, oe, x.u(151)), x.op(OpAdd, U, x.op(OpShr, U, c, x.u(23)), de))
		cm := x.op(OpAdd, U, x.op(OpShl, U, x.op(OpAdd, U, x.op(OpAnd, U, c, x.u(0x7fffff)), x.u(0x800000)), x.u(1)), x.u(1))
		diff := x.op(OpSub, U, x.op(OpSub, U, x.op(OpShl, U, om, sh), x.op(OpMul, U, cm, dm)), x.u(1))
		acc = x.op(OpAdd, U, acc, x.sel(U, x.op(OpLess, ir.Pred, diff, x.u(0x7fffffff)), x.u(1), x.u(0)))
	}
	okO := x.op(OpLess, ir.Pred, x.op(OpSub, U, oe, x.u(1)), x.u(254))
	okD := x.op(OpLess, ir.Pred, x.op(OpSub, U, de, x.u(1)), x.u(254))
	okR := x.op(OpLess, ir.Pred, x.op(OpSub, U, x.op(OpShr, U, rb, x.u(23)), x.u(2)), x.u(252))
	v := x.sel(F, okR, x.un(OpBitcast, F, acc), r)
	v = x.sel(F, okD, v, r)
	v = x.sel(F, okO, v, r)
	return x.sel(F, pos, v, x.f(0))
}

// int8 is value v's int8 form, rows of w, as primitives: each window's
// amax, its scale, then each word -- its window's inverse scale, its
// elements' codes packed, and the codes' sum -- and the sums a Sums. The
// words and the scales -- the windows' first, the sums' after -- are what it
// keeps.
func (c *Context) int8(v graph.ValueID) []Prim {
	q := c.Int8
	U, F, I := ir.U32, ir.F32, ir.I32
	sh := c.rows(v)
	R, W := sh[0], sh[1]
	nw, nwd, ns := W/q.Window, W/q.Word, W/q.Sums
	x := c.value(v)
	temp := func(name string, t ir.Type) Operand {
		return Operand{Kind: OperandScratch, Name: x.Name + "." + name, Elem: t, Temp: true}
	}
	amax, wsum := temp("amax", F), temp("wsum", I)
	words := Operand{Kind: OperandScratch, Name: x.Name + ".q8a", Elem: U}
	scales := Operand{Kind: OperandScratch, Name: x.Name + ".q8x"}
	win := []int{R, nw}
	flat := func(e *exprs, r, i, n int) int { return e.op(OpAdd, U, e.op(OpMul, U, r, e.u(uint32(n))), i) }

	var a exprs
	a.abs(a.at(0, sh, a.coord(0), a.op(OpAdd, U, a.op(OpMul, U, a.coord(1), a.u(uint32(q.Window))), a.coord(2))))

	var d exprs
	ds := d.scale(d.at(0, win, d.coord(0), d.coord(1)))
	dix := flat(&d, d.coord(0), d.coord(1), nw)

	// A word: its window's inverse, each element's code rounded half away
	// from zero, the codes packed a byte each, and their sum.
	var p exprs
	pr, pw := p.coord(0), p.coord(1)
	pwin := p.op(OpDiv, U, p.op(OpMul, U, pw, p.u(uint32(q.Word))), p.u(uint32(q.Window)))
	inv := p.inverse(p.at(1, win, pr, pwin), p.load(2, flat(&p, pr, pwin, nw)))
	packed, sum := -1, -1
	for j := 0; j < q.Word; j++ {
		e := p.op(OpAdd, U, p.op(OpMul, U, pw, p.u(uint32(q.Word))), p.u(uint32(j)))
		t := p.op(OpMul, F, p.at(0, sh, pr, e), inv)
		half := p.un(OpBitcast, F, p.op(OpAdd, U, p.op(OpAnd, U, p.un(OpBitcast, U, t), p.u(0x80000000)), p.u(0x3f000000)))
		code := p.un(OpCvtI32, I, p.op(OpAdd, F, t, half))
		b := p.op(OpAnd, U, p.un(OpBitcast, U, code), p.u(255))
		if j == 0 {
			packed, sum = b, code
			continue
		}
		packed = p.op(OpAdd, U, packed, p.op(OpShl, U, b, p.u(uint32(8*j))))
		sum = p.op(OpAdd, I, sum, code)
	}
	wix := flat(&p, pr, pw, nwd)

	per := q.Sums / q.Word
	var f exprs
	f.un(OpCvtF32, F, f.at(0, []int{R, nwd}, f.coord(0), f.op(OpAdd, U, f.op(OpMul, U, f.coord(1), f.u(uint32(per))), f.coord(2))))
	var fi exprs
	fi.op(OpAdd, U, flat(&fi, fi.coord(0), fi.coord(1), ns), fi.u(uint32(R*nw)))

	ext := R * (nw + ns)
	return []Prim{
		{Kind: PrimReduce, Shape: []int{R, nw, q.Window}, In: []Operand{x}, Out: amax, Combine: OpMax, Expr: a.e},
		{Kind: PrimMap, Shape: win, In: []Operand{amax}, Expr: d.e, Stores: []PrimStore{{Out: scales, Index: dix, Value: ds, Extent: ext}}},
		{Kind: PrimMap, Shape: []int{R, nwd}, In: []Operand{x, amax, scales}, Expr: p.e, Facts: Facts{Extents: map[int]int{2: ext}},
			Stores: []PrimStore{{Out: words, Index: wix, Value: packed, Extent: R * nwd},
				{Out: wsum, Ax: []int{pr, pw}, Ext: []int{R, nwd}, Value: sum}}},
		{Kind: PrimReduce, Shape: []int{R, ns, per}, In: []Operand{wsum}, Out: scales, Combine: OpAdd, Start: StartFirst,
			Expr: f.e, Index: fi.e, Facts: Facts{OutExtent: ext}},
	}
}
