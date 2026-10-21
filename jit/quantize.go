package jit

import (
	"math"

	"github.com/jitllm/jitllm/jit/ir"
)

// The int8 form of a row: windows of 32 by an exact scale amax/127, codes
// rounded half away from zero packed four to a word, and the codes' sums a
// sixteen.
const (
	quantWindow = 32
	quantWord   = 4
	quantSums   = 16
)

// exprs builds a primitive's function node by node.
type exprs struct{ e []Expr }

func (x *exprs) add(e Expr) int { x.e = append(x.e, e); return len(x.e) - 1 }

func (x *exprs) in(j int) int { return x.add(Expr{Op: ExprIn, In: j}) }

func (x *exprs) u(v uint32) int { return x.add(Expr{Op: OpConst, Type: ir.U32, Imm: int64(v)}) }

func (x *exprs) i(v int32) int { return x.add(Expr{Op: OpConst, Type: ir.I32, Imm: int64(v)}) }

func (x *exprs) f(v float32) int {
	return x.add(Expr{Op: OpConst, Type: ir.F32, Imm: int64(math.Float32bits(v))})
}

func (x *exprs) op(op Op, t ir.Type, a, b int) int { return x.add(Expr{Op: op, Type: t, A: a, B: b}) }

func (x *exprs) un(op Op, t ir.Type, a int) int { return x.add(Expr{Op: op, Type: t, A: a}) }

func (x *exprs) sel(t ir.Type, p, a, b int) int {
	return x.add(Expr{Op: OpSelect, Type: t, A: p, B: a, C: b})
}

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

// quantize is the int8 form of value v, w wide over rows rows: into codes
// (words) and scales (the windows' scales, then the sixteens' sums).
func quantize(v Operand, w, rows int, codes, scales Operand) []Prim {
	windows, words, sums := rows*w/quantWindow, rows*w/quantWord, rows*w/quantSums
	amax := Operand{Kind: OperandScratch, Name: v.Name + ".amax", Temp: true}
	inv := Operand{Kind: OperandScratch, Name: v.Name + ".inv", Temp: true}
	q := Operand{Kind: OperandScratch, Name: v.Name + ".q", Elem: ir.I32, Temp: true}
	wsum := Operand{Kind: OperandScratch, Name: v.Name + ".wsum", Elem: ir.I32, Temp: true}

	var a exprs
	a.abs(a.in(0))
	var d exprs
	d.scale(d.in(0))
	var r exprs
	r.inverse(r.in(0), r.in(1))
	var c exprs
	t := c.op(OpMul, ir.F32, c.in(0), c.in(1))
	half := c.un(OpBitcast, ir.F32, c.op(OpAdd, ir.U32, c.op(OpAnd, ir.U32, c.un(OpBitcast, ir.U32, t), c.u(0x80000000)), c.u(0x3f000000)))
	c.un(OpCvtI32, ir.I32, c.op(OpAdd, ir.F32, t, half))
	var p exprs
	packed := -1
	for j := 0; j < quantWord; j++ {
		b := p.op(OpAnd, ir.U32, p.in(j), p.u(255))
		if j > 0 {
			b = p.op(OpShl, ir.U32, b, p.u(uint32(8*j)))
			packed = p.op(OpAdd, ir.U32, packed, b)
			continue
		}
		packed = b
	}
	var s exprs
	s.in(0)
	var f exprs
	f.un(OpCvtF32, ir.F32, f.in(0))

	codes.Elem = ir.U32
	return []Prim{
		{Kind: PrimReduce, N: windows, K: quantWindow, In: []Operand{v}, At: []Access{atElement}, Out: amax,
			Combine: OpMax, Expr: a.e},
		{Kind: PrimMap, N: windows, In: []Operand{amax}, At: []Access{atElement}, Out: scales, Expr: d.e},
		{Kind: PrimMap, N: windows, In: []Operand{amax, scales}, At: []Access{atElement, atElement}, Out: inv, Expr: r.e},
		{Kind: PrimMap, N: rows * w, In: []Operand{v, inv}, At: []Access{atElement, {Div: quantWindow}}, Out: q, Expr: c.e},
		{Kind: PrimMap, N: words, In: []Operand{q, q, q, q}, Out: codes, Expr: p.e,
			At: []Access{{Mul: quantWord}, {Mul: quantWord, Off: 1}, {Mul: quantWord, Off: 2}, {Mul: quantWord, Off: 3}}},
		{Kind: PrimReduce, N: words, K: quantWord, In: []Operand{q}, At: []Access{atElement}, Out: wsum, Combine: OpAdd, Type: ir.I32, Expr: s.e},
		{Kind: PrimReduce, N: sums, K: quantSums / quantWord, In: []Operand{wsum}, At: []Access{atElement}, Out: scales,
			OutAt: Access{Off: windows}, Combine: OpAdd, Start: StartFirst, Expr: f.e},
	}
}
