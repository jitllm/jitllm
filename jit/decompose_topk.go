package jit

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/ir"
)

// decomposeTopK is the largest element of a row and where it is, the
// lower index of two equal: the row's maximum, then the least index whose
// element is not below it.
func decomposeTopK(o graph.Op, c *Context) ([]Prim, error) {
	if o.Attr.TopK != 1 || o.Attr.Span != 0 || len(o.In) != 1 || len(o.Out) != 2 {
		return nil, fmt.Errorf("a top %d of spans of %d", o.Attr.TopK, o.Attr.Span)
	}
	sh := c.rows(o.In[0])
	best := c.value(o.Out[1])
	tok := c.value(o.Out[0])
	tok.Elem = ir.I32

	var m exprs
	m.at(0, sh, m.coord(0), m.coord(1))

	var x exprs
	v := x.at(0, sh, x.coord(0), x.coord(1))
	b := x.at(1, []int{sh[0], 1}, x.coord(0), x.u(0))
	x.sel(ir.I32, x.op(OpLess, ir.Pred, v, b), x.add(Expr{Op: OpConst, Type: ir.I32, Imm: math.MaxInt32}), x.coord(1))

	return []Prim{
		{Kind: PrimReduce, Shape: sh, In: []Operand{c.value(o.In[0])}, Out: best, Combine: OpMax,
			Init: int64(math.Float32bits(float32(math.Inf(-1)))), Expr: m.e},
		{Kind: PrimReduce, Shape: sh, In: []Operand{c.value(o.In[0]), best}, Out: tok, Combine: OpMin, Type: ir.I32,
			Init: math.MaxInt32, Expr: x.e},
	}, nil
}
