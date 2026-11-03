package composer

import (
	"fmt"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/graph"
)

// decomposeQ8Rows is each row's floats as a q8 history's row: each block's
// amax, its scale and the scale's inverse, each element's code, the codes
// packed four to a word and each block's sum, and the pairs stored after the
// row's codes.
func decomposeQ8Rows(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 || c.width(o.Out[0]) != graph.Q8RowWords(c.width(o.In[0])) {
		return nil, fmt.Errorf("q8 rows of %d inputs into %d outputs", len(o.In), len(o.Out))
	}
	U, F, I := ir.U32, ir.F32, ir.I32
	sh := c.rows(o.In[0])
	R, W := sh[0], sh[1]
	nb := (W + q8Block - 1) / q8Block
	rs := graph.Q8RowWords(W)
	out := c.value(o.Out[0])
	out.Elem = U
	temp := func(name string, t ir.Type) Operand {
		return Operand{Kind: OperandScratch, Name: out.Name + "." + name, Elem: t, Temp: true}
	}
	amax, d, inv, codes, sums := temp("amax", F), temp("d", F), temp("inv", F), temp("q", I), temp("sum", I)
	blocks, padded := []int{R, nb}, []int{R, nb * q8Block}
	// The row's element i, zero past its width.
	elem := func(e *exprs, r, i int) int {
		if W%q8Block == 0 {
			return e.at(0, sh, r, i)
		}
		return e.sel(F, e.op(OpLess, ir.Pred, i, e.u(uint32(W))), e.at(0, sh, r, e.op(OpMin, U, i, e.u(uint32(W-1)))), e.f(0))
	}
	block := func(e *exprs) int { return e.op(OpAdd, U, e.op(OpMul, U, e.coord(1), e.u(q8Block)), e.coord(2)) }

	var a exprs
	a.abs(elem(&a, a.coord(0), block(&a)))

	var s exprs
	if m := s.at(0, blocks, s.coord(0), s.coord(1)); c.exactDiv {
		s.op(OpDiv, F, m, s.f(127))
	} else {
		s.scale(m)
	}

	var n exprs
	if m, dd := n.at(0, blocks, n.coord(0), n.coord(1)), n.at(1, blocks, n.coord(0), n.coord(1)); c.exactDiv {
		n.sel(F, n.op(OpLess, ir.Pred, n.f(0), dd), n.op(OpDiv, F, n.f(1), dd), n.f(0))
	} else {
		n.inverse(m, dd)
	}

	var k exprs
	r, i := k.coord(0), k.coord(1)
	t := k.op(OpMul, F, elem(&k, r, i), k.at(1, blocks, r, k.op(OpShr, U, i, k.u(5))))
	k.un(OpCvtI32, I, k.round(t))

	var p exprs
	packed := -1
	for j := 0; j < 4; j++ {
		b := p.op(OpAnd, U, p.at(0, padded, p.coord(0), p.op(OpAdd, U, p.op(OpMul, U, p.coord(1), p.u(4)), p.u(uint32(j)))), p.u(255))
		if j > 0 {
			b = p.op(OpShl, U, b, p.u(uint32(8*j)))
			packed = p.op(OpOr, U, packed, b)
			continue
		}
		packed = b
	}
	wix := p.op(OpAdd, U, p.op(OpMul, U, p.coord(0), p.u(uint32(rs))), p.coord(1))

	var u exprs
	u.at(0, padded, u.coord(0), block(&u))

	var q exprs
	pair := q.op(OpAdd, U, q.op(OpMul, U, q.coord(0), q.u(uint32(rs))), q.op(OpAdd, U, q.u(uint32(nb*q8Words)), q.op(OpMul, U, q.coord(1), q.u(2))))
	dv := q.un(OpBitcast, U, q.at(0, blocks, q.coord(0), q.coord(1)))
	sv := q.un(OpBitcast, U, q.op(OpMul, F, q.un(OpCvtF32, F, q.at(1, blocks, q.coord(0), q.coord(1))), q.f(-16)))
	pair1 := q.op(OpAdd, U, pair, q.u(1))

	ext := R * rs
	return []Prim{
		{Kind: PrimReduce, Shape: []int{R, nb, q8Block}, In: []Operand{c.value(o.In[0])}, Out: amax, Combine: OpMax, Expr: a.e},
		{Kind: PrimMap, Shape: blocks, In: []Operand{amax}, Out: d, Expr: s.e},
		{Kind: PrimMap, Shape: blocks, In: []Operand{amax, d}, Out: inv, Expr: n.e},
		{Kind: PrimMap, Shape: padded, In: []Operand{c.value(o.In[0]), inv}, Out: codes, Expr: k.e},
		{Kind: PrimMap, Shape: []int{R, nb * q8Words}, In: []Operand{codes}, Expr: p.e,
			Stores: []PrimStore{{Out: out, Index: wix, Value: packed, Extent: ext}}},
		{Kind: PrimReduce, Shape: []int{R, nb, q8Block}, In: []Operand{codes}, Out: sums, Combine: OpAdd, Type: I, Expr: u.e},
		{Kind: PrimMap, Shape: blocks, In: []Operand{d, sums}, Expr: q.e,
			Stores: []PrimStore{{Out: out, Index: pair, Value: dv, Extent: ext}, {Out: out, Index: pair1, Value: sv, Extent: ext}}},
	}, nil
}

// decomposeWidenQ8Rows is each q8 row's codes times their blocks' scales.
func decomposeWidenQ8Rows(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 || c.width(o.In[0]) != graph.Q8RowWords(c.width(o.Out[0])) {
		return nil, fmt.Errorf("floats of %d q8 rows into %d outputs", len(o.In), len(o.Out))
	}
	U, F, I := ir.U32, ir.F32, ir.I32
	W := c.width(o.Out[0])
	nb := (W + q8Block - 1) / q8Block
	sh := c.rows(o.In[0])
	in := c.value(o.In[0])
	in.Elem = U
	var x exprs
	r, i := x.coord(0), x.coord(1)
	w := x.at(0, sh, r, x.op(OpShr, U, i, x.u(2)))
	b := x.op(OpAnd, U, x.op(OpShr, U, w, x.op(OpMul, U, x.op(OpAnd, U, i, x.u(3)), x.u(8))), x.u(255))
	code := x.un(OpCvtF32, F, x.un(OpBitcast, I, x.op(OpSub, U, x.op(OpXor, U, b, x.u(128)), x.u(128))))
	scale := x.at(0, sh, r, x.op(OpAdd, U, x.u(uint32(nb*q8Words)), x.op(OpMul, U, x.op(OpShr, U, i, x.u(5)), x.u(2))))
	x.op(OpMul, F, x.un(OpBitcast, F, scale), code)
	return []Prim{{Kind: PrimMap, Shape: []int{sh[0], W}, In: []Operand{in}, Out: c.value(o.Out[0]), Expr: x.e}}, nil
}
