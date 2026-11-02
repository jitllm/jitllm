package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/ir"
)

// selectBounds is a selection's block bounds on a row: its first and end
// positions, its own block, and the first of the blocks forced in.
type selectBounds struct{ first, end, own, lo int }

// selectPooled is a pooled selection's block scores (graph.Attrs.Pooled):
// each block's keys summed, the mean's sum of squares, the rotary's table at
// each block's first position, the mean normed by Weight[0] and turned by
// it, each query head's dot with it, and each group's sum of its heads'
// scores through Fn and the weights, into best. A block a row does not hold
// whole scores from its clamped reads; the ranking never counts it.
func selectPooled(o graph.Op, c *Context, pool Pool, keys Operand, tables func(...Operand) []Operand, flat Facts, best Operand) ([]Prim, error) {
	a := o.Attr
	if a.NRot < 0 || a.NRot%2 != 0 || a.NRot > a.HeadDim || a.NRot > 0 && a.Layout != graph.RopeNorm && a.Layout != graph.RopeNeox {
		return nil, fmt.Errorf("a pooled selection turned by a rotary of %d of %d, layout %d", a.NRot, a.HeadDim, a.Layout)
	}
	var fn Op
	switch a.Fn {
	case 0:
	case graph.MapMax:
		fn = OpMax
	case graph.MapMin:
		fn = OpMin
	default:
		return nil, fmt.Errorf("a selection's score through %v", a.Fn)
	}
	U, F := ir.U32, ir.F32
	R, H, D, G, B, C := c.Step.Rows, a.NHead, a.HeadDim, a.NKVHead, a.Span, c.Step.Keys
	J, NB := H/G, (c.Step.Keys+a.Span-1)/a.Span
	hs := history{a: a, pageSize: c.Step.Page}
	name := c.value(o.Out[0]).Name
	tmp := func(n string) Operand {
		return Operand{Kind: OperandScratch, Name: name + "." + n, Elem: F, Temp: true}
	}
	sums, sumsq, pooled, dots := tmp("pool"), tmp("poolsq"), tmp("pooled"), tmp("pooldots")
	wt := Operand{Kind: OperandVector, Role: o.Weight[0].Role, Name: o.Weight[0].Role.String()}

	// Each block's mean key: every attended position's key weighted by
	// 1/B where it is of the block, the attention's read of its values.
	var px exprs
	pos, first := hs.position(&px)
	pid, off := hs.page(&px, hs.held(&px, pos, first))
	in := px.sel(F, px.op(OpLess, ir.Pred, px.op(OpDiv, U, pos, px.u(uint32(B))), px.op(OpAdd, U, px.coord(1), px.u(1))),
		px.sel(F, px.op(OpLess, ir.Pred, px.op(OpDiv, U, pos, px.u(uint32(B))), px.coord(1)), px.f(0), px.f(1/float32(B))), px.f(0))
	px.op(OpMul, F, in, c.poolRead(&px, 0, pool, a.Width, pid, off, px.op(OpAdd, U, px.coord(2), px.u(uint32(a.Offset)))))

	var sx exprs
	mean := sx.at(0, []int{R, NB, D}, sx.coord(0), sx.coord(1), sx.coord(2))
	sx.op(OpMul, F, mean, mean)

	rng := Facts{Range: hs.span(3), Extents: map[int]int{1: R * rowWords, 2: c.pages()}}
	prims := []Prim{
		{Kind: PrimContract, Shape: []int{R, NB, D, C}, In: tables(keys),
			Out: sums, Expr: px.e, Facts: rng},
		{Kind: PrimReduce, Shape: []int{R, NB, D}, In: []Operand{sums}, Out: sumsq, Combine: OpAdd, Expr: sx.e},
	}

	// The mean normed, then turned at the block's first position: a pair an
	// element of the map, as OpRope turns a head, the rest of the key the
	// normed mean's.
	var kx exprs
	r, b, d := kx.coord(0), kx.coord(1), kx.coord(2)
	inv := kx.op(OpDiv, F, kx.f(1), kx.un(OpSqrt, F, kx.op(OpAdd, F,
		kx.op(OpMul, F, kx.at(1, []int{R, NB}, r, b), kx.f(1/float32(D))), kx.f(float32(a.Eps)))))
	kx.op(OpMul, F, kx.op(OpMul, F, kx.at(0, []int{R, NB, D}, r, b, d), inv), kx.at(2, []int{D}, d))
	normed := pooled
	if a.NRot > 0 {
		normed = tmp("normed")
	}
	prims = append(prims, Prim{Kind: PrimMap, Shape: []int{R, NB, D}, In: []Operand{sums, sumsq, wt}, Out: normed, Expr: kx.e})
	if a.NRot > 0 {
		P := a.NRot / 2
		stride := (P + ropeLanes - 1) &^ (ropeLanes - 1)
		table, err := c.rotaryTable(a)
		if err != nil {
			return nil, err
		}
		planes := tableOp(TablePlanes, table.Instance, ir.Void)
		tab := tmp("pooltab")
		var tx exprs
		tb, tp := tx.coord(0), tx.coord(1)
		cvs, svs, quarter := ropeTurn(&tx, tx.op(OpMul, U, tb, tx.u(uint32(B))), tp, 0, 1, stride)
		at := tx.op(OpMul, U, tx.op(OpAdd, U, tx.op(OpMul, U, tb, tx.u(uint32(P))), tp), tx.u(2))
		prims = append(prims, Prim{Kind: PrimMap, Shape: []int{NB, P}, In: []Operand{planes, tableRopeKon}, Expr: tx.e,
			Stores: []PrimStore{{Out: tab, Index: at, Value: cvs, Extent: 2 * NB * P},
				{Out: tab, Index: tx.op(OpAdd, U, at, tx.u(1)), Value: svs, Extent: 2 * NB * P}},
			Facts: Facts{Words: map[int][]uint32{1: ropeConsts()}, Extents: map[int]int{0: 8*stride + 1}, Quarter: quarter, Whole: true}})

		// Each element of the mean turned with its pair's partner: NEOX
		// pairs d and d+P, NORM pairs neighbours; the first of a pair
		// takes -sin, the second +sin, and past NRot the normed mean.
		var rx exprs
		rr, rb, rd := rx.coord(0), rx.coord(1), rx.coord(2)
		turned := rx.op(OpLess, ir.Pred, rd, rx.u(uint32(a.NRot)))
		var firstOf, partner, pair int
		if a.Layout == graph.RopeNeox {
			firstOf = rx.op(OpLess, ir.Pred, rd, rx.u(uint32(P)))
			partner = rx.sel(U, firstOf, rx.op(OpAdd, U, rd, rx.u(uint32(P))), rx.op(OpSub, U, rd, rx.u(uint32(P))))
			pair = rx.sel(U, firstOf, rd, rx.op(OpSub, U, rd, rx.u(uint32(P))))
		} else {
			firstOf = rx.op(OpLess, ir.Pred, rx.op(OpAnd, U, rd, rx.u(1)), rx.u(1))
			partner = rx.op(OpXor, U, rd, rx.u(1))
			pair = rx.op(OpShr, U, rd, rx.u(1))
		}
		partner = rx.sel(U, turned, partner, rd)
		pair = rx.sel(U, turned, pair, rx.u(0))
		sh := []int{R, NB, D}
		x0, x1 := rx.at(0, sh, rr, rb, rd), rx.at(0, sh, rr, rb, partner)
		e := rx.op(OpMul, U, rx.op(OpAdd, U, rx.op(OpMul, U, rb, rx.u(uint32(P))), pair), rx.u(2))
		co, si := rx.at(1, []int{NB * P * 2}, e), rx.at(1, []int{NB * P * 2}, rx.op(OpAdd, U, e, rx.u(1)))
		si = rx.sel(F, firstOf, rx.op(OpSub, F, rx.f(0), si), si)
		y := rx.op(OpAdd, F, rx.op(OpMul, F, x0, co), rx.op(OpMul, F, x1, si))
		rx.sel(F, turned, y, x0)
		prims = append(prims, Prim{Kind: PrimMap, Shape: sh, In: []Operand{normed, tab}, Out: pooled, Expr: rx.e})
	}

	var dx exprs
	dx.op(OpMul, F, dx.at(0, []int{R, H, D}, dx.coord(0), dx.coord(1), dx.coord(3)), dx.at(1, []int{R, NB, D}, dx.coord(0), dx.coord(2), dx.coord(3)))

	var gx exprs
	gh := gx.op(OpAdd, U, gx.op(OpMul, U, gx.coord(1), gx.u(uint32(J))), gx.coord(3))
	s := gx.at(0, []int{R, H, NB}, gx.coord(0), gh, gx.coord(2))
	if fn != 0 {
		s = gx.op(fn, F, s, gx.f(float32(a.Const)))
	}
	sumIn := []Operand{dots}
	if len(o.In) == 2 {
		gx.op(OpMul, F, s, gx.at(1, []int{R, H}, gx.coord(0), gh))
		sumIn = append(sumIn, c.value(o.In[1]))
	}
	return append(prims,
		Prim{Kind: PrimContract, Shape: []int{R, H, NB, D}, In: []Operand{c.value(o.In[0]), pooled}, Out: dots, Expr: dx.e},
		Prim{Kind: PrimReduce, Shape: []int{R, G, NB, J}, In: sumIn, Out: best, Combine: OpAdd, Expr: gx.e},
	), nil
}
