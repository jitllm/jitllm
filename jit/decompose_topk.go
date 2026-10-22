package jit

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/ir"
)

// decomposeTopK is the largest element of a row and where it is, the
// lower index of two equal: the row's maximum carrying its index.
func decomposeTopK(o graph.Op, c *Context) ([]Prim, error) {
	if o.Attr.TopK > 1 || o.Attr.Span != 0 {
		return decomposeTopKRanked(o, c)
	}
	if o.Attr.TopK != 1 || o.Attr.Span != 0 || len(o.In) != 1 || len(o.Out) != 2 {
		return nil, fmt.Errorf("a top %d of spans of %d", o.Attr.TopK, o.Attr.Span)
	}
	sh := c.rows(o.In[0])
	tok := c.value(o.Out[0])
	tok.Elem = ir.I32

	var m exprs
	m.at(0, sh, m.coord(0), m.coord(1))
	var idx exprs
	idx.un(OpBitcast, ir.I32, idx.coord(1))

	return []Prim{{Kind: PrimReduce, Shape: sh, In: []Operand{c.value(o.In[0])}, Out: c.value(o.Out[1]), Combine: OpMax,
		Init: int64(math.Float32bits(float32(math.Inf(-1)))), Expr: m.e, Carry: idx.e, CarryOut: tok}}, nil
}
