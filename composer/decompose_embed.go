package composer

import (
	"fmt"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/graph"
)

// tableTokens is the step's tokens, a word a row, which the engine writes.

// tableRequest is a step's request, the caller's words: a decision's
// question type, its marker count, then a marker row a step row.

// Request words: the question type, the marker count, the first marker.
const (
	RequestType = iota
	RequestCount
	RequestRows
)

// decomposeEmbed is each row's token's row of the embedding (row 0 for an
// id past a table that stops short of the vocabulary), or its position's
// row of a position table (the position modulo Span where set), every
// element decoded from the weight's format: into an index row, each element
// an index. Over an index row In[0], each of a row's indices' rows of the
// table, side by side (decomposeEmbedRows).
func decomposeEmbed(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.Weight) != 1 || len(o.In) > 1 || len(o.Out) != 1 {
		return nil, fmt.Errorf("an embedding of %d weights and %d inputs", len(o.Weight), len(o.In))
	}
	w := o.Weight[0]
	wo, err := c.weight(w)
	if err != nil {
		return nil, err
	}
	if len(o.In) == 1 {
		return decomposeEmbedRows(o, c, w, wo)
	}
	R, W := c.Step.Rows, w.K
	var x exprs
	r, i := x.coord(0), x.coord(1)
	tok := x.load(0, r)
	in := []Operand{TableTokens.op()}
	facts := Facts{Extents: map[int]int{0: R}}
	if o.Attr.Positional {
		// The step's positions follow its live row count.
		tok = x.load(0, x.op(OpAdd, ir.U32, r, x.u(1)))
		if s := o.Attr.Span; s > 0 {
			tok = x.op(OpRem, ir.U32, tok, x.u(uint32(s)))
		}
		in[0] = TablePos.op()
		facts.Extents[0] = R + 1
	} else {
		// A table may stop short of the ids: an id past it reads row 0.
		tok = x.sel(ir.U32, x.op(OpLess, ir.Pred, tok, x.u(uint32(w.Rows))), tok, x.u(0))
	}
	v, err := weightElement(&x, wo, w.Rows, W, &in, &facts, tok, i)
	if err != nil {
		return nil, fmt.Errorf("an embedding of %s weights: %w", wo.Format, err)
	}
	if c.Graph.Values[o.Out[0]].Elem == graph.ElemU32 {
		x.un(OpBitcast, ir.U32, x.un(OpCvtI32, ir.I32, v))
	}
	return []Prim{{Kind: PrimMap, Shape: []int{R, W}, In: in, Out: c.value(o.Out[0]), Expr: x.e, Facts: facts}}, nil
}

// decomposeTakeRows is each row's request-named row of the input: row r
// the request's marker r.
func decomposeTakeRows(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 || c.width(o.In[0]) != c.width(o.Out[0]) {
		return nil, fmt.Errorf("a row take of %d operands", len(o.In))
	}
	U := ir.U32
	sh, in := c.rows(o.Out[0]), c.rows(o.In[0])
	R := sh[0]
	var x exprs
	at := x.op(OpMin, U, x.load(1, x.op(OpAdd, U, x.u(RequestRows), x.coord(0))), x.u(uint32(in[0]-1)))
	x.at(0, in, at, x.coord(1))
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0]), TableRequest.op()}, Out: c.value(o.Out[0]), Expr: x.e,
		Facts: Facts{Extents: map[int]int{1: RequestRows + R}}}}, nil
}

// decomposeEmbedRows is OpEmbed over an index row In[0] (u32, H indices a
// row): each index's row of the table, the H rows side by side, an index
// past the table's rows reading row 0 as a token past it does.
func decomposeEmbedRows(o graph.Op, c *Context, w graph.Binding, wo Operand) ([]Prim, error) {
	a := o.Attr
	H, W := c.width(o.In[0]), w.K
	if c.Graph.Values[o.In[0]].Elem != graph.ElemU32 || a.Positional || H < 1 || c.width(o.Out[0]) != H*W {
		return nil, fmt.Errorf("an embedding of %d rows of %d over %d indices into %d", w.Rows, W, H, c.width(o.Out[0]))
	}
	R := c.Step.Rows
	var x exprs
	r, h, i := x.coord(0), x.coord(1), x.coord(2)
	at := x.at(0, []int{R, H}, r, h)
	at = x.sel(ir.U32, x.op(OpLess, ir.Pred, at, x.u(uint32(w.Rows))), at, x.u(0))
	in := []Operand{c.value(o.In[0])}
	facts := Facts{Extents: map[int]int{}}
	if _, err := weightElement(&x, wo, w.Rows, W, &in, &facts, at, i); err != nil {
		return nil, fmt.Errorf("an embedding of %s weights: %w", wo.Format, err)
	}
	return []Prim{{Kind: PrimMap, Shape: []int{R, H, W}, In: in, Out: c.value(o.Out[0]), Expr: x.e, Facts: facts}}, nil
}

// decomposeToken is each row's token, read off the step's tokens table.
func decomposeToken(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 0 || len(o.Out) != 1 || c.width(o.Out[0]) != 1 {
		return nil, fmt.Errorf("a token of %d inputs into %d outputs", len(o.In), len(o.Out))
	}
	R := c.Step.Rows
	var x exprs
	x.load(0, x.coord(0))
	return []Prim{{Kind: PrimMap, Shape: []int{R, 1}, In: []Operand{TableTokens.op()}, Out: c.value(o.Out[0]), Expr: x.e,
		Facts: Facts{Extents: map[int]int{0: R}}}}, nil
}
