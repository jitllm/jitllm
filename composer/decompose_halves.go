package composer

import (
	"fmt"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/graph"
)

// decomposeHalves is each row's floats as binary16, two a word, packed as a
// binary16 history's pages hold them (decomposeKVAppend).
func decomposeHalves(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 || c.width(o.In[0]) != 2*c.width(o.Out[0]) {
		return nil, fmt.Errorf("binary16 halves of %d inputs into %d outputs", len(o.In), len(o.Out))
	}
	U := ir.U32
	sh := c.rows(o.In[0])
	out := c.value(o.Out[0])
	out.Elem = U
	var x exprs
	r, j := x.coord(0), x.coord(1)
	lo := x.at(0, sh, r, x.op(OpMul, U, j, x.u(2)))
	hi := x.at(0, sh, r, x.op(OpAdd, U, x.op(OpMul, U, j, x.u(2)), x.u(1)))
	x.op(OpOr, U, x.un(OpHalf, U, lo), x.op(OpShl, U, x.un(OpHalf, U, hi), x.u(16)))
	return []Prim{{Kind: PrimMap, Shape: []int{sh[0], sh[1] / 2}, In: []Operand{c.value(o.In[0])}, Out: out, Expr: x.e}}, nil
}

// decomposeWidenHalves is each row's binary16 halves, two a word, as floats,
// the low half first.
func decomposeWidenHalves(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 || c.width(o.Out[0]) != 2*c.width(o.In[0]) {
		return nil, fmt.Errorf("floats of %d binary16 inputs into %d outputs", len(o.In), len(o.Out))
	}
	U := ir.U32
	sh := c.rows(o.In[0])
	in := c.value(o.In[0])
	in.Elem = U
	var x exprs
	r, i := x.coord(0), x.coord(1)
	w := x.at(0, sh, r, x.op(OpShr, U, i, x.u(1)))
	x.un(OpWidenHalf, ir.F32, x.op(OpShr, U, w, x.op(OpMul, U, x.op(OpAnd, U, i, x.u(1)), x.u(16))))
	return []Prim{{Kind: PrimMap, Shape: []int{sh[0], 2 * sh[1]}, In: []Operand{in}, Out: c.value(o.Out[0]), Expr: x.e}}, nil
}
