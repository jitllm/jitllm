package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/ir"
)

// RequestGrid is a picture's step's request word: its grid's width in
// patches.
const RequestGrid = 0

// decomposePatchify is each row's patch of the picture, gathered: a row
// past the Offset before it the patch its position less Offset names, of a
// grid RequestGrid patches wide, its element c channel c/(P*P), row (c/P)%P
// and column c%P of the patch; zero past 3*P*P and in the rows before
// Offset.
func decomposePatchify(o graph.Op, c *Context) ([]Prim, error) {
	P := o.Attr.Span
	if len(o.In) != 1 || len(o.Out) != 1 || P < 1 || c.width(o.In[0]) != 3*P*P || c.width(o.Out[0]) < 3*P*P {
		return nil, fmt.Errorf("a patch gather of %d operands, patches of %d", len(o.In), P)
	}
	U, F := ir.U32, ir.F32
	sh, ish := c.rows(o.Out[0]), c.rows(o.In[0])
	R, PW := sh[0], uint32(3*P*P)
	var x exprs
	row, col := x.coord(0), x.coord(1)
	u := func(v int) int { return x.u(uint32(v)) }
	lead := o.Attr.Offset
	r := x.op(OpSub, U, x.op(OpMax, U, x.load(2, x.op(OpAdd, U, row, u(1))), u(lead)), u(lead))
	gw := x.op(OpMax, U, x.load(1, x.u(RequestGrid)), u(1))
	py, pxi := x.op(OpDiv, U, r, gw), x.op(OpRem, U, r, gw)
	pp := uint32(P * P)
	ch, ky, kx := x.op(OpDiv, U, col, x.u(pp)), x.op(OpRem, U, x.op(OpDiv, U, col, u(P)), u(P)), x.op(OpRem, U, col, u(P))
	y := x.op(OpAdd, U, x.op(OpMul, U, py, u(P)), ky)
	xx := x.op(OpAdd, U, x.op(OpMul, U, pxi, u(P)), kx)
	pix := x.op(OpAdd, U, x.op(OpMul, U, x.op(OpAdd, U, x.op(OpMul, U, y, x.op(OpMul, U, gw, u(P))), xx), u(3)), ch)
	at := x.op(OpMin, U, x.op(OpDiv, U, pix, x.u(PW)), u(R-1))
	v := x.at(0, ish, at, x.op(OpRem, U, pix, x.u(PW)))
	v = x.sel(F, x.op(OpLess, ir.Pred, col, x.u(PW)), v, x.f(0))
	if lead > 0 {
		x.sel(F, x.op(OpLess, ir.Pred, row, u(lead)), x.f(0), v)
	}
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0]), tableRequest, tablePos}, Out: c.value(o.Out[0]), Expr: x.e,
		Facts: Facts{Extents: map[int]int{1: RequestRows + R, 2: R + 1}}}}, nil
}

// decomposeFold is each Span consecutive rows as one: output row g is
// input rows g*Span to g*Span+Span-1, each a run of the input's width.
func decomposeFold(o graph.Op, c *Context) ([]Prim, error) {
	k := o.Attr.Span
	if len(o.In) != 1 || len(o.Out) != 1 || k < 2 || c.width(o.Out[0]) != k*c.width(o.In[0]) {
		return nil, fmt.Errorf("a fold of %d operands by %d", len(o.In), k)
	}
	U := ir.U32
	sh, ish := c.rows(o.Out[0]), c.rows(o.In[0])
	E := ish[1]
	var x exprs
	g, j := x.coord(0), x.coord(1)
	row := x.op(OpAdd, U, x.op(OpMul, U, g, x.u(uint32(k))), x.op(OpDiv, U, j, x.u(uint32(E))))
	x.at(0, ish, x.op(OpMin, U, row, x.u(uint32(ish[0]-1))), x.op(OpRem, U, j, x.u(uint32(E))))
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0])}, Out: c.value(o.Out[0]), Expr: x.e}}, nil
}

// decomposeRegroup is space-to-depth over the grid of rows RequestGrid
// wide: output row g is the S x S block of rows at its place, slot (dy, dx)
// the run dy*S+dx of the input's width.
func decomposeRegroup(o graph.Op, c *Context) ([]Prim, error) {
	S := o.Attr.Span
	if len(o.In) != 1 || len(o.Out) != 1 || S < 2 || c.width(o.Out[0]) != S*S*c.width(o.In[0]) {
		return nil, fmt.Errorf("a regroup of %d operands by %d", len(o.In), S)
	}
	U := ir.U32
	sh, ish := c.rows(o.Out[0]), c.rows(o.In[0])
	R, E := sh[0], ish[1]
	var x exprs
	g, j := x.coord(0), x.coord(1)
	u := func(v int) int { return x.u(uint32(v)) }
	gw := x.op(OpMax, U, x.load(1, x.u(RequestGrid)), u(S))
	gsw := x.op(OpDiv, U, gw, u(S))
	slot, e := x.op(OpDiv, U, j, u(E)), x.op(OpRem, U, j, u(E))
	row := x.op(OpAdd, U, x.op(OpMul, U, x.op(OpDiv, U, g, gsw), u(S)), x.op(OpDiv, U, slot, u(S)))
	colm := x.op(OpAdd, U, x.op(OpMul, U, x.op(OpRem, U, g, gsw), u(S)), x.op(OpRem, U, slot, u(S)))
	p := x.op(OpAdd, U, x.op(OpMul, U, row, gw), colm)
	x.at(0, ish, x.op(OpMin, U, p, u(R-1)), e)
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0]), tableRequest}, Out: c.value(o.Out[0]), Expr: x.e,
		Facts: Facts{Extents: map[int]int{1: RequestRows + R}}}}, nil
}

// decomposeLeadRow is the input with its first Offset rows each the
// weight's row.
func decomposeLeadRow(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 || len(o.Weight) != 1 || o.Attr.Offset < 1 || o.Weight[0].K != c.width(o.In[0]) {
		return nil, fmt.Errorf("a lead row of %d operands, %d weights", len(o.In), len(o.Weight))
	}
	sh := c.rows(o.Out[0])
	w := o.Weight[0]
	var x exprs
	r, i := x.coord(0), x.coord(1)
	in, lead := x.at(0, sh, r, i), x.at(1, []int{sh[1]}, i)
	x.sel(ir.F32, x.op(OpLess, ir.Pred, r, x.u(uint32(o.Attr.Offset))), lead, in)
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0]), {Kind: OperandVector, Role: w.Role, Name: w.Role.String()}},
		Out: c.value(o.Out[0]), Expr: x.e}}, nil
}

// decomposeDropRows is the input from row Offset on.
func decomposeDropRows(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 || o.Attr.Offset < 1 || c.width(o.In[0]) != c.width(o.Out[0]) {
		return nil, fmt.Errorf("a row drop of %d operands", len(o.In))
	}
	U := ir.U32
	sh := c.rows(o.Out[0])
	var x exprs
	at := x.op(OpMin, U, x.op(OpAdd, U, x.coord(0), x.u(uint32(o.Attr.Offset))), x.u(uint32(sh[0]-1)))
	x.at(0, sh, at, x.coord(1))
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0])}, Out: c.value(o.Out[0]), Expr: x.e}}, nil
}
