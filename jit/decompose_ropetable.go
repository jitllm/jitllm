package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/ir"
)

// The rotary table's inputs the engine writes: the positions (the step's
// live row count, then a position a row), the model's planes (a position's
// four base-128 digits' angles a pair, for the sine's sum and the cosine's,
// then the table's scale) and the constant block of the range reduction
// and the two polynomials.
var (
	tablePos     = Operand{Kind: OperandTable, Name: "rope.pos", Elem: ir.U32}
	tablePlanes  = Operand{Kind: OperandTable, Name: "rope.planes"}
	tableRopeKon = Operand{Kind: OperandTable, Name: "rope.const", Elem: ir.U32}
)

const (
	ropeDigits    = 4
	ropeDigitBits = 7
	ropeGrid      = 1 << 12
	ropeLanes     = 8
)

// The constant block's words.
const (
	ropePio2Hi  = 1
	ropePio2Lo  = 2
	ropeSinC7   = 3
	ropeSinC5   = 4
	ropeSinC3   = 5
	ropeCosC8   = 6
	ropeCosC6   = 7
	ropeCosC4   = 8
	ropeNegHalf = 9
	ropeOne     = 10
	ropeSignVec = 12
	ropeOneVec  = 20
	ropeMod4    = 36
	ropeHalf    = 37
)

// decomposeRopeTable is a step's rotary table: for each row and pair, the
// pair's angle as the sum of the position's digits times the model's
// planes, a range reduction to a quadrant and a remainder, the remainder's
// sine and cosine by polynomials, swapped and signed by the quadrant, each
// times the table's scale.
func decomposeRopeTable(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.Out) != 1 || o.Attr.NRot <= 0 || o.Attr.NRot%2 != 0 {
		return nil, fmt.Errorf("a rotary table of %d", o.Attr.NRot)
	}
	U, F, I := ir.U32, ir.F32, ir.I32
	P := o.Attr.NRot / 2
	stride := (P + ropeLanes - 1) &^ (ropeLanes - 1)
	R := c.Step.Rows
	var x exprs
	r, p := x.coord(0), x.coord(1)
	cw := func(w int) int { return x.load(2, x.u(uint32(w))) }
	cf := func(w int) int { return x.un(OpBitcast, F, cw(w)) }
	pos := x.load(0, x.op(OpAdd, U, r, x.u(1)))
	var d [ropeDigits]int
	for i := range d {
		v := pos
		if i > 0 {
			v = x.op(OpShr, U, pos, x.u(uint32(i*ropeDigitBits)))
		}
		d[i] = x.un(OpCvtF32, F, x.op(OpAnd, U, v, x.u(1<<ropeDigitBits-1)))
	}
	plane := func(i int) int { return x.load(1, x.op(OpAdd, U, p, x.u(uint32(i*stride)))) }
	sum := func(base int) int {
		acc := x.op(OpMul, F, d[0], plane(base))
		for i := 1; i < ropeDigits; i++ {
			acc = x.op(OpAdd, F, acc, x.op(OpMul, F, d[i], plane(base+i)))
		}
		return acc
	}
	s, co := sum(0), sum(ropeDigits)
	si := x.un(OpBitcast, U, x.un(OpCvtI32, I, x.op(OpMul, F, s, x.f(ropeGrid))))
	rem := x.op(OpMul, F, x.un(OpCvtF32, F, x.op(OpAnd, U, si, cw(ropeMod4))), x.f(1.0/ropeGrid))
	n := x.un(OpCvtI32, I, x.op(OpAdd, F, x.op(OpAdd, F, rem, co), cf(ropeHalf)))
	f := x.op(OpSub, F, rem, x.un(OpCvtF32, F, n))
	nb := x.un(OpBitcast, U, n)
	hi, lo := cf(ropePio2Hi), cf(ropePio2Lo)
	fma := func(a, b, z int) int { return x.add(Expr{Op: OpFma, Type: F, A: a, B: b, C: z}) }
	rr := fma(f, hi, fma(f, lo, x.op(OpMul, F, co, hi)))
	z := x.op(OpMul, F, rr, rr)
	sp := fma(z, cf(ropeSinC7), cf(ropeSinC5))
	sp = fma(z, sp, cf(ropeSinC3))
	sp = fma(x.op(OpMul, F, rr, z), sp, rr)
	cp := cf(ropeCosC8)
	for _, w := range []int{ropeCosC6, ropeCosC4, ropeNegHalf, ropeOne} {
		cp = fma(z, cp, cf(w))
	}
	swap := x.op(OpLess, ir.Pred, x.u(0), x.op(OpAnd, U, nb, x.u(1)))
	sv, cv := x.sel(F, swap, cp, sp), x.sel(F, swap, sp, cp)
	neg := func(v, m int) int {
		bits := x.op(OpAnd, U, x.op(OpShl, U, m, x.u(30)), cw(ropeSignVec))
		return x.un(OpBitcast, F, x.op(OpXor, U, x.un(OpBitcast, U, v), bits))
	}
	sv = neg(sv, nb)
	cv = neg(cv, x.op(OpAdd, U, nb, cw(ropeOneVec)))
	ms := x.load(1, x.u(uint32(8*stride)))
	at := x.op(OpMul, U, x.op(OpAdd, U, x.op(OpMul, U, r, x.u(uint32(P))), p), x.u(2))
	cvs, svs := x.op(OpMul, F, cv, ms), x.op(OpMul, F, sv, ms)
	at1 := x.op(OpAdd, U, at, x.u(1))

	var live exprs
	live.load(0, live.u(0))
	out := Operand{Kind: OperandTable, Name: tableAngles.Name}
	return []Prim{{Kind: PrimMap, Shape: []int{R, P}, In: []Operand{tablePos, tablePlanes, tableRopeKon}, Expr: x.e,
		Stores: []PrimStore{{Out: out, Index: at, Value: cvs}, {Out: out, Index: at1, Value: svs}},
		Facts:  Facts{Rows: live.e}}}, nil
}
