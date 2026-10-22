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

// decomposePosResample is each row's position-table row resampled to the
// grid: the row's cell, on each axis the source coordinate by the op's
// chain and its filter's taps' weights and clamped indices, then the sum of
// every tap pair's weight product times its table row, in the taps' order.
func decomposePosResample(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	if len(o.In) != 0 || len(o.Out) != 1 || len(o.Weight) != 1 || a.Span < 2 || o.Weight[0].Rows != a.Span*a.Span ||
		a.Chain < graph.ChainAligned || a.Chain > graph.ChainGridSample || a.Filter < graph.FilterBilinear || a.Filter > graph.FilterCubic {
		return nil, fmt.Errorf("a position resample of %d weights, chain %d, filter %d", len(o.Weight), a.Chain, a.Filter)
	}
	wo, err := c.weight(o.Weight[0])
	if err != nil {
		return nil, err
	}
	U, F, I := ir.U32, ir.F32, ir.I32
	sh := c.rows(o.Out[0])
	R, E, S := sh[0], sh[1], a.Span
	var x exprs
	r, e := x.coord(0), x.coord(1)
	in := []Operand{tablePos, tableRequest}
	facts := Facts{Extents: map[int]int{0: R + 1, 1: RequestRows + R}}
	lead := uint32(a.Offset)
	gw := x.op(OpMax, U, x.load(1, x.u(RequestGrid)), x.u(1))
	live := x.op(OpMax, U, x.load(0, x.u(0)), x.u(lead+1))
	gh := x.op(OpMax, U, x.op(OpDiv, U, x.op(OpSub, U, live, x.u(lead)), gw), x.u(1))
	q := x.op(OpSub, U, x.op(OpMax, U, x.load(0, x.op(OpAdd, U, r, x.u(1))), x.u(lead)), x.u(lead))
	f := func(v float32) int { return x.f(v) }
	side := f(float32(S))
	type tap struct{ w, i int }
	axis := func(cell, size int) []tap {
		i, n := x.un(OpCvtF32, F, cell), x.un(OpCvtF32, F, size)
		var v int
		switch a.Chain {
		case graph.ChainAligned:
			v = x.op(OpDiv, F, x.op(OpMul, F, i, f(float32(S-1))), x.op(OpMax, F, x.op(OpSub, F, n, f(1)), f(1)))
		case graph.ChainHalfPixel:
			v = x.op(OpAdd, F, x.op(OpMul, F, x.op(OpAdd, F, i, f(0.5)), x.op(OpDiv, F, side, n)), f(-0.5))
		default:
			v = x.op(OpAdd, F, x.op(OpAdd, F, x.op(OpMul, F, x.op(OpDiv, F, x.op(OpAdd, F, i, f(0.5)), n), f(2)), f(-1)), f(1))
			v = x.op(OpDiv, F, x.op(OpAdd, F, x.op(OpMul, F, v, side), f(-1)), f(2))
		}
		if a.Filter == graph.FilterLinear {
			v = x.op(OpMax, F, v, f(0))
		}
		// floor(v): the truncation, one less where it lies above v.
		tr := x.un(OpCvtI32, I, v)
		above := x.op(OpLess, ir.Pred, v, x.un(OpCvtF32, F, tr))
		fl := x.sel(I, above, x.op(OpSub, I, tr, x.un(OpBitcast, I, x.u(1))), tr)
		t := x.op(OpSub, F, v, x.un(OpCvtF32, F, fl))
		abs := func(v int) int { return x.op(OpMax, F, v, x.op(OpSub, F, f(0), v)) }
		conv1 := func(v int) int {
			return x.op(OpAdd, F, x.op(OpMul, F, x.op(OpMul, F, x.op(OpSub, F, x.op(OpMul, F, f(1.25), v), f(2.25)), v), v), f(1))
		}
		conv2 := func(v int) int {
			b := x.op(OpAdd, F, x.op(OpMul, F, f(-0.75), v), f(3.75))
			return x.op(OpAdd, F, x.op(OpMul, F, x.op(OpSub, F, x.op(OpMul, F, b, v), f(6)), v), f(3))
		}
		var ws []int
		off := 0
		switch a.Filter {
		case graph.FilterBilinear:
			ws = []int{x.op(OpMax, F, x.op(OpSub, F, f(1), abs(t)), f(0)), x.op(OpMax, F, x.op(OpSub, F, f(1), abs(x.op(OpSub, F, t, f(1)))), f(0))}
		case graph.FilterLinear:
			ws = []int{x.op(OpSub, F, f(1), t), t}
		default:
			off = -1
			ws = []int{conv2(x.op(OpAdd, F, t, f(1))), conv1(t), conv1(x.op(OpSub, F, f(1), t)), conv2(x.op(OpSub, F, f(2), t))}
		}
		out := make([]tap, len(ws))
		for k, w := range ws {
			j := x.op(OpAdd, I, fl, x.un(OpBitcast, I, x.u(uint32(int32(k+off)))))
			j = x.op(OpMin, I, x.op(OpMax, I, j, x.un(OpBitcast, I, x.u(0))), x.un(OpBitcast, I, x.u(uint32(S-1))))
			out[k] = tap{w, x.un(OpBitcast, U, j)}
		}
		return out
	}
	ys := axis(x.op(OpDiv, U, q, gw), gh)
	xs := axis(x.op(OpRem, U, q, gw), gw)
	acc := -1
	for _, ty := range ys {
		for _, tx := range xs {
			row := x.op(OpAdd, U, x.op(OpMul, U, ty.i, x.u(uint32(S))), tx.i)
			v, err := weightElement(&x, wo, S*S, E, &in, &facts, row, e)
			if err != nil {
				return nil, fmt.Errorf("a position table of %s: %w", wo.Format, err)
			}
			term := x.op(OpMul, F, x.op(OpMul, F, ty.w, tx.w), v)
			if acc < 0 {
				acc = term
			} else {
				acc = x.op(OpAdd, F, acc, term)
			}
		}
	}
	return []Prim{{Kind: PrimMap, Shape: sh, In: in, Out: c.value(o.Out[0]), Expr: x.e, Facts: facts}}, nil
}
