package composer

import (
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/graph"
)

// abs is |a| by its sign bit.
func (x *exprs) abs(a int) int {
	return x.un(OpBitcast, ir.F32, x.op(OpAnd, ir.U32, x.un(OpBitcast, ir.U32, a), x.u(0x7fffffff)))
}

// mant256Over127 is m*256/127 floored, for a mantissa m of 24 bits: 2m plus
// floor(2m/127), the quotient taken through a float estimate (2m is exact in
// one) one under at worst and corrected by its remainder, so no lane divides
// integers.
func (x *exprs) mant256Over127(m int) int {
	U, F := ir.U32, ir.F32
	m2 := x.op(OpShl, U, m, x.u(1))
	est := x.op(OpSub, F, x.op(OpMul, F, x.un(OpCvtF32, F, x.un(OpBitcast, ir.I32, m2)), x.f(1.0/127)), x.f(0.5))
	t := x.un(OpBitcast, U, x.un(OpCvtI32, ir.I32, est))
	r := x.op(OpSub, U, m2, x.op(OpMul, U, t, x.u(127)))
	t = x.op(OpAdd, U, t, x.sel(U, x.op(OpLess, ir.Pred, x.u(126), r), x.u(1), x.u(0)))
	return x.op(OpAdd, U, m2, t)
}

// scale is amax/127 correctly rounded: the mantissa's quotient in integers
// where the exponent allows it, the float division where it does not.
func (x *exprs) scale(amax int) int {
	U := ir.U32
	b := x.un(OpBitcast, U, amax)
	e := x.op(OpShr, U, b, x.u(23))
	m := x.op(OpAdd, U, x.op(OpAnd, U, b, x.u(0x7fffff)), x.u(0x800000))
	q := x.mant256Over127(m)
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
	v, pos := x.nearestRecip(d, x.op(OpDiv, ir.F32, x.f(127), amax))
	return x.sel(ir.F32, pos, v, x.f(0))
}

// inverseOf is inverse from the refined estimate r of 127/amax: the
// candidates' test alone, for a device that holds it a map apart.
func (x *exprs) inverseOf(d, r, rank int) int {
	v, pos := x.recipPick(d, r, rank)
	return x.sel(ir.F32, pos, v, x.f(0))
}

// recip is 1/a correctly rounded: the nearest float to the reciprocal of |a|
// from the device's 1/|a|, with a's sign; the device's own where an exponent
// leaves the integer test's range.
func (x *exprs) recip(a int) int {
	U, F := ir.U32, ir.F32
	m := x.abs(a)
	r := x.op(OpDiv, F, x.f(1), m)
	v, _ := x.nearestRecip(m, r)
	normal := x.op(OpLess, ir.Pred, x.op(OpSub, U, x.op(OpShr, U, x.un(OpBitcast, U, m), x.u(23)), x.u(1)), x.u(254))
	v = x.sel(F, normal, v, r)
	return x.un(OpBitcast, F, x.op(OpAdd, U, x.un(OpBitcast, U, v), x.op(OpAnd, U, x.un(OpBitcast, U, a), x.u(0x80000000))))
}

// round is a to the nearest integer, halves away from zero: |a|'s integer
// part, one more where the fraction is a half or over, with a's sign; a
// itself from 2^23 on, where every float is an integer.
func (x *exprs) round(a int) int {
	U, F := ir.U32, ir.F32
	m := x.abs(a)
	f := x.un(OpCvtF32, F, x.un(OpCvtI32, ir.I32, m))
	r := x.sel(F, x.op(OpLess, ir.Pred, x.op(OpSub, F, m, f), x.f(0.5)), f, x.op(OpAdd, F, f, x.f(1)))
	r = x.un(OpBitcast, F, x.op(OpAdd, U, x.un(OpBitcast, U, r), x.op(OpAnd, U, x.un(OpBitcast, U, a), x.u(0x80000000))))
	return x.sel(F, x.op(OpLess, ir.Pred, m, x.f(1<<23)), r, a)
}

// trunc is a's integer part: a itself at 2^23 and past, where every float is
// an integer (and the conversion's range would not hold it).
func (x *exprs) trunc(a int) int {
	F := ir.F32
	t := x.un(OpCvtF32, F, x.un(OpCvtI32, ir.I32, a))
	t = x.un(OpBitcast, F, x.op(OpOr, ir.U32, x.un(OpBitcast, ir.U32, t), x.op(OpAnd, ir.U32, x.un(OpBitcast, ir.U32, a), x.u(0x80000000))))
	return x.sel(F, x.op(OpLess, ir.Pred, x.abs(a), x.f(1<<23)), t, a)
}

// floor is the greatest integer not above a: its integer part, less one
// where that is above it.
func (x *exprs) floor(a int) int {
	F := ir.F32
	t := x.trunc(a)
	return x.sel(F, x.op(OpLess, ir.Pred, a, t), x.op(OpSub, F, t, x.f(1)), t)
}

// roundEven is a to the nearest integer, a half to the even one: its integer
// part, a step away from zero where the fraction (exact below 2^23) is past
// a half, or a half and the integer part odd. a + 1/2 floored would round a
// float just below a half up where the sum rounds to the integer.
func (x *exprs) roundEven(a int) int {
	F := ir.F32
	t := x.trunc(a)
	r := x.abs(x.op(OpSub, F, a, t))
	step := x.op(OpAdd, F, t, x.sel(F, x.op(OpLess, ir.Pred, a, x.f(0)), x.f(-1), x.f(1)))
	odd := x.equal(x.abs(x.op(OpSub, F, t, x.op(OpMul, F, x.trunc(x.op(OpMul, F, t, x.f(0.5))), x.f(2)))), x.f(1), false)
	tie := x.op(OpLess, ir.Pred, x.f(0.5), x.op(OpMul, F, x.equal(r, x.f(0.5), false), odd))
	return x.sel(F, x.op(OpLess, ir.Pred, x.f(0.5), r), step, x.sel(F, tie, step, t))
}

// nearestRecip is the float nearest 1/d from r within a few units of it:
// r refined once by Newton, then the four candidates below it tested in
// integers where every exponent allows, r where one does not; pos is d > 0.
func (x *exprs) nearestRecip(d, r int) (v, pos int) {
	return x.nearestRecipRefined(d, x.refine(d, r))
}

// refine is r, an estimate of 1/d, refined once by Newton.
func (x *exprs) refine(d, r int) int {
	F := ir.F32
	return x.op(OpAdd, F, r, x.op(OpMul, F, r, x.op(OpSub, F, x.f(1), x.op(OpMul, F, d, r))))
}

// nearestRecipRefined is nearestRecip from an estimate already refined.
func (x *exprs) nearestRecipRefined(d, r int) (v, pos int) {
	return x.recipPick(d, r, x.recipRank(d, r))
}

// recipRank is the bits of the candidate below r's, plus the number of the
// four candidates from it up that lie below the true reciprocal of d, tested
// in integers.
func (x *exprs) recipRank(d, r int) int {
	U := ir.U32
	ob, db := x.un(OpBitcast, U, x.f(1)), x.un(OpBitcast, U, d)
	oe, de := x.op(OpShr, U, ob, x.u(23)), x.op(OpShr, U, db, x.u(23))
	om := x.op(OpAdd, U, x.op(OpAnd, U, ob, x.u(0x7fffff)), x.u(0x800000))
	dm := x.op(OpAdd, U, x.op(OpAnd, U, db, x.u(0x7fffff)), x.u(0x800000))
	base := x.op(OpSub, U, x.un(OpBitcast, U, r), x.u(2))
	acc := base
	for k := uint32(0); k < 4; k++ {
		c := x.op(OpAdd, U, base, x.u(k))
		sh := x.op(OpSub, U, x.op(OpAdd, U, oe, x.u(151)), x.op(OpAdd, U, x.op(OpShr, U, c, x.u(23)), de))
		cm := x.op(OpAdd, U, x.op(OpShl, U, x.op(OpAdd, U, x.op(OpAnd, U, c, x.u(0x7fffff)), x.u(0x800000)), x.u(1)), x.u(1))
		diff := x.op(OpSub, U, x.op(OpSub, U, x.op(OpShl, U, om, sh), x.op(OpMul, U, cm, dm)), x.u(1))
		acc = x.op(OpAdd, U, acc, x.sel(U, x.op(OpLess, ir.Pred, diff, x.u(0x7fffffff)), x.u(1), x.u(0)))
	}
	return acc
}

// recipPick is the reciprocal recipRank's acc names where every exponent
// allows, r where one does not; pos is d > 0.
func (x *exprs) recipPick(d, r, acc int) (v, pos int) {
	U, F := ir.U32, ir.F32
	pos = x.op(OpLess, ir.Pred, x.f(0), d)
	oe := x.op(OpShr, U, x.un(OpBitcast, U, x.f(1)), x.u(23))
	de := x.op(OpShr, U, x.un(OpBitcast, U, d), x.u(23))
	rb := x.un(OpBitcast, U, r)
	okO := x.op(OpLess, ir.Pred, x.op(OpSub, U, oe, x.u(1)), x.u(254))
	okD := x.op(OpLess, ir.Pred, x.op(OpSub, U, de, x.u(1)), x.u(254))
	okR := x.op(OpLess, ir.Pred, x.op(OpSub, U, x.op(OpShr, U, rb, x.u(23)), x.u(2)), x.u(252))
	v = x.sel(F, okR, x.un(OpBitcast, F, acc), r)
	v = x.sel(F, okD, v, r)
	v = x.sel(F, okO, v, r)
	return v, pos
}

// window is how many of value v's elements share a scale: the form's
// window, the whole row where the row is narrower (a window stated as the
// model's widest row).
func (c *Context) window(v graph.ValueID) int { return min(c.Int8.Window, c.width(v)) }

// slot is how many elements each of value v's scale words covers: its
// window, or the form's Slot where the window is wider, the window's scale
// then stored once a slot so that a sub-block's index scales to it.
func (c *Context) slot(v graph.ValueID) int {
	if s, ww := c.Int8.Slot, c.window(v); s > 0 && ww > s && ww%s == 0 {
		return s
	}
	return c.window(v)
}

// int8 is value v's int8 form, rows of w, as primitives: each window's
// amax, its scale, the scale's inverse, each element's code (a Sums of a
// window a row of the map), the codes' sums a Sums and the codes packed
// into words. The words and the
// scales are what it keeps, the scales a row's windows' and then its sums',
// row after row (scaleRow), so its layout holds any count of rows.
func (c *Context) int8(v graph.ValueID) []Prim {
	q := c.Int8
	U, F, I := ir.U32, ir.F32, ir.I32
	sh := c.rows(v)
	R, W := sh[0], sh[1]
	ww := c.window(v)
	nw, ns, nsl := W/ww, W/q.Sums, W/c.slot(v)
	rep := ww / c.slot(v)
	x := c.value(v)
	temp := func(name string, t ir.Type) Operand {
		return Operand{Kind: OperandScratch, Name: x.Name + "." + name, Elem: t, Temp: true}
	}
	amax, inv, codes := temp("amax", F), temp("inv", F), temp("q", I)
	scales := Operand{Kind: OperandScratch, Name: x.Name + ".q8x"}
	win := []int{R, nw}
	flat := func(e *exprs, r, i, n int) int { return e.op(OpAdd, U, e.op(OpMul, U, r, e.u(uint32(n))), i) }

	var a exprs
	a.abs(a.at(0, sh, a.coord(0), a.op(OpAdd, U, a.op(OpMul, U, a.coord(1), a.u(uint32(ww))), a.coord(2))))

	var d exprs
	var ds int
	dw := d.coord(1)
	if rep > 1 {
		dw = d.op(OpDiv, U, dw, d.u(uint32(rep)))
	}
	if c.exactDiv {
		ds = d.op(OpDiv, F, d.at(0, win, d.coord(0), dw), d.f(127))
	} else {
		ds = d.scale(d.at(0, win, d.coord(0), dw))
	}
	dix := flat(&d, d.coord(0), d.coord(1), nsl+ns)
	slots := []int{R, nsl}
	at := func(e *exprs) int {
		w := e.coord(1)
		if rep > 1 {
			w = e.op(OpMul, U, w, e.u(uint32(rep)))
		}
		return flat(e, e.coord(0), w, nsl+ns)
	}

	ext := R * (nsl + ns)
	facts := Facts{Extents: map[int]int{1: ext}}
	var r exprs
	var refined []Prim
	pick := []Operand{amax, scales}
	if dd := r.load(1, at(&r)); c.exactDiv {
		r.sel(F, r.op(OpLess, ir.Pred, r.f(0), dd), r.op(OpDiv, F, r.f(1), dd), r.f(0))
	} else {
		// The correctly rounded reciprocal in three maps, the estimate
		// refined, the candidates tested and the pick made, each within the
		// vector registers of a device that holds them.
		var n, m exprs
		est, rank := temp("est", F), temp("rank", U)
		n.refine(n.load(1, at(&n)), n.op(OpDiv, F, n.f(127), n.at(0, win, n.coord(0), n.coord(1))))
		m.recipRank(m.load(1, at(&m)), m.at(0, win, m.coord(0), m.coord(1)))
		refined = []Prim{
			{Kind: PrimMap, Shape: win, In: []Operand{amax, scales}, Out: est, Expr: n.e, Facts: facts},
			{Kind: PrimMap, Shape: win, In: []Operand{est, scales}, Out: rank, Expr: m.e, Facts: facts},
		}
		r.inverseOf(dd, r.at(0, win, r.coord(0), r.coord(1)), r.at(2, win, r.coord(0), r.coord(1)))
		pick = []Operand{est, scales, rank}
	}

	var k exprs
	cs, el, wi := sh, k.coord(1), -1
	if q.SumCodes && ww%q.Sums == 0 {
		cs = []int{R, ns, q.Sums}
		el = k.op(OpAdd, U, k.op(OpMul, U, k.coord(1), k.u(uint32(q.Sums))), k.coord(2))
		wi = k.op(OpDiv, U, k.coord(1), k.u(uint32(ww/q.Sums)))
	} else {
		wi = k.op(OpDiv, U, el, k.u(uint32(ww)))
	}
	t := k.op(OpMul, F, k.at(0, sh, k.coord(0), el), k.at(1, win, k.coord(0), wi))
	half := k.un(OpBitcast, F, k.op(OpAdd, U, k.op(OpAnd, U, k.un(OpBitcast, U, t), k.u(0x80000000)), k.u(0x3f000000)))
	k.un(OpCvtI32, I, k.op(OpAdd, F, t, half))

	ps := []Prim{
		{Kind: PrimReduce, Shape: []int{R, nw, ww}, In: []Operand{x}, Out: amax, Combine: OpMax, Expr: a.e},
		{Kind: PrimMap, Shape: slots, In: []Operand{amax}, Expr: d.e, Stores: []PrimStore{{Out: scales, Index: dix, Value: ds, Extent: ext}}},
	}
	ps = append(ps, refined...)
	ps = append(ps,
		Prim{Kind: PrimMap, Shape: win, In: pick, Out: inv, Expr: r.e, Facts: facts},
		Prim{Kind: PrimMap, Shape: cs, In: []Operand{x, inv}, Out: codes, Expr: k.e})
	return append(ps, c.int8Pack(v, codes, false, nsl)...)
}

// int8Codes is the int8 form of value v, a row of codes the graph states
// (graph.ElemI8, held as floats): its sums a Sums, the scales' only words,
// and its codes packed into words.
func (c *Context) int8Codes(v graph.ValueID) []Prim { return c.int8Pack(v, c.value(v), true, 0) }

// int8Pack is the sums of value v's codes a Sums into its scales after each
// row's nw windows' (a row nw+ns words) and the codes packed Word to a
// word: the codes i32, or floats where float is set.
func (c *Context) int8Pack(v graph.ValueID, codes Operand, float bool, nw int) []Prim {
	q := c.Int8
	U, F, I := ir.U32, ir.F32, ir.I32
	sh := c.rows(v)
	R, W := sh[0], sh[1]
	nwd, ns := W/q.Word, W/q.Sums
	x := c.value(v)
	words := Operand{Kind: OperandScratch, Name: x.Name + ".q8a", Elem: U}
	scales := Operand{Kind: OperandScratch, Name: x.Name + ".q8x"}
	code := func(e *exprs, ax ...int) int {
		a := e.at(0, sh, ax...)
		if float {
			a = e.un(OpCvtI32, I, a)
		}
		return a
	}

	var p exprs
	packed := -1
	for j := 0; j < q.Word; j++ {
		e := code(&p, p.coord(0), p.op(OpAdd, U, p.op(OpMul, U, p.coord(1), p.u(uint32(q.Word))), p.u(uint32(j))))
		b := p.op(OpAnd, U, e, p.u(255))
		if j > 0 {
			packed = p.op(OpAdd, U, packed, p.op(OpShl, U, b, p.u(uint32(8*j))))
			continue
		}
		packed = b
	}

	var fi exprs
	fi.op(OpAdd, U, fi.op(OpAdd, U, fi.op(OpMul, U, fi.coord(0), fi.u(uint32(nw+ns))), fi.coord(1)), fi.u(uint32(nw)))
	ext := R * (nw + ns)

	if !q.SumCodes {
		wsum := Operand{Kind: OperandScratch, Name: x.Name + ".wsum", Elem: I, Temp: true}
		var s exprs
		code(&s, s.coord(0), s.op(OpAdd, U, s.op(OpMul, U, s.coord(1), s.u(uint32(q.Word))), s.coord(2)))
		per := q.Sums / q.Word
		var f exprs
		f.un(OpCvtF32, F, f.at(0, []int{R, nwd}, f.coord(0), f.op(OpAdd, U, f.op(OpMul, U, f.coord(1), f.u(uint32(per))), f.coord(2))))
		return []Prim{
			{Kind: PrimMap, Shape: []int{R, nwd}, In: []Operand{codes}, Out: words, Expr: p.e},
			{Kind: PrimReduce, Shape: []int{R, nwd, q.Word}, In: []Operand{codes}, Out: wsum, Combine: OpAdd, Type: I, Expr: s.e},
			{Kind: PrimReduce, Shape: []int{R, ns, per}, In: []Operand{wsum}, Out: scales, Combine: OpAdd, Start: StartFirst,
				Expr: f.e, Index: fi.e, Facts: Facts{OutExtent: ext}},
		}
	}

	var f exprs
	if sum := f.at(0, []int{R, ns, q.Sums}, f.coord(0), f.coord(1), f.coord(2)); !float {
		f.un(OpCvtF32, F, sum)
	}
	return []Prim{
		{Kind: PrimReduce, Shape: []int{R, ns, q.Sums}, In: []Operand{codes}, Out: scales, Combine: OpAdd,
			Expr: f.e, Index: fi.e, Facts: Facts{OutExtent: ext}},
		{Kind: PrimMap, Shape: []int{R, nwd}, In: []Operand{codes}, Out: words, Expr: p.e},
	}
}
