package jit

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/ir"
)

// decomposeAttend is decode attention over the paged history: per row and
// query head, each key's dot with the query, the scores masked outside what
// the row attends, their maximum, the weights and their sum, the weighted
// values and the normalized row.
func decomposeAttend(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	if len(o.In) != 1 || len(o.Out) != 1 || len(o.Weight) != 0 || a.Window != 0 || a.Softcap != 0 || a.TempScale != 0 ||
		a.NHead <= 0 || a.NKVHead <= 0 || a.NHead%a.NKVHead != 0 || c.Step.Page <= 0 || c.Step.Keys <= 0 {
		return nil, fmt.Errorf("an attention of %d heads over %d, window %d, softcap %v", a.NHead, a.NKVHead, a.Window, a.Softcap)
	}
	U, F := ir.U32, ir.F32
	R, H, D := c.Step.Rows, a.NHead, a.HeadDim
	C := c.Step.Keys
	gqa, kw, pg := a.NHead/a.NKVHead, a.NKVHead*a.HeadDim, c.Step.Page
	out := c.value(o.Out[0])
	tmp := func(name string) Operand {
		return Operand{Kind: OperandScratch, Name: out.Name + "." + name, Temp: true}
	}
	dots, scores, maxes, weights, sums, values := tmp("dots"), tmp("scores"), tmp("max"), tmp("weights"), tmp("sum"), tmp("values")
	minus := math.Float32frombits(0xff7fffff)
	keys, heads, dims := []int{R, H, C}, []int{R, H}, []int{R, H, D}

	// Every primitive reads the rows' descriptors as operand 1 and the
	// pages as 2, row r its coordinate 0.
	row := func(x *exprs, w int) int {
		return x.load(1, x.op(OpAdd, U, x.op(OpMul, U, x.coord(0), x.u(rowWords)), x.u(uint32(w))))
	}
	bound := func(w int) []Expr {
		var x exprs
		row(&x, w)
		return x.e
	}
	rng := func(axis int) Facts {
		return Facts{Range: &Range{Axis: axis, First: bound(rowStart), End: bound(rowEnd)},
			Heads: &Heads{Rows: R, Query: H, Key: a.NKVHead, Dim: D}}
	}
	// position is the key's position and the row's first.
	position := func(x *exprs) (pos, first int) {
		return x.pos(), row(x, rowStart)
	}
	// held is a position kept inside what the row attends.
	held := func(x *exprs, pos, first int) int {
		last := x.op(OpSub, U, x.op(OpMax, U, x.live(), x.u(1)), x.u(1))
		return x.op(OpMin, U, x.op(OpMax, U, pos, first), last)
	}
	page := func(x *exprs, t int) (pid, off int) {
		pid = x.load(2, x.op(OpAdd, U, row(x, rowTab), x.op(OpDiv, U, t, x.u(uint32(pg)))))
		return pid, x.op(OpRem, U, t, x.u(uint32(pg)))
	}

	var dx exprs
	h, d := dx.coord(1), dx.coord(3)
	pos, first := position(&dx)
	pid, off := page(&dx, held(&dx, pos, first))
	kh := dx.op(OpDiv, U, h, dx.u(uint32(gqa)))
	pool := []int{c.pages(), pg, kw}
	kd := dx.op(OpAdd, U, dx.op(OpMul, U, kh, dx.u(uint32(D))), d)
	qat := dx.op(OpAdd, U, dx.op(OpMul, U, dx.op(OpAdd, U, dx.op(OpMul, U, dx.coord(0), dx.u(uint32(H))), h), dx.u(uint32(D))), d)
	dx.op(OpMul, F, dx.load(0, qat), dx.at(3, pool, pid, off, kd))

	var sx exprs
	spos, sfirst := position(&sx)
	valid := sx.op(OpLess, ir.Pred, sx.op(OpSub, U, spos, sfirst), sx.op(OpSub, U, sx.live(), sfirst))
	dot := sx.at(0, keys, sx.coord(0), sx.coord(1), sx.coord(2))
	sx.sel(F, valid, sx.op(OpMul, F, dot, sx.f(float32(a.Scale))), sx.f(minus))

	var mx exprs
	mx.at(0, keys, mx.coord(0), mx.coord(1), mx.coord(2))

	var wx exprs
	wpos, _ := position(&wx)
	below := wx.op(OpLess, ir.Pred, wpos, wx.live())
	s := wx.at(0, keys, wx.coord(0), wx.coord(1), wx.coord(2))
	m := wx.at(3, heads, wx.coord(0), wx.coord(1))
	wx.sel(F, below, wx.un(OpExp, F, wx.op(OpSub, F, s, m)), wx.f(0))

	var lx exprs
	lx.at(0, keys, lx.coord(0), lx.coord(1), lx.coord(2))

	var vx exprs
	vh, vd, vk := vx.coord(1), vx.coord(2), vx.coord(3)
	vpos, vfirst := position(&vx)
	vpid, voff := page(&vx, held(&vx, vpos, vfirst))
	vkh := vx.op(OpDiv, U, vh, vx.u(uint32(gqa)))
	vd2 := vx.op(OpAdd, U, vx.op(OpMul, U, vkh, vx.u(uint32(D))), vd)
	w := vx.at(0, keys, vx.coord(0), vh, vk)
	vx.op(OpMul, F, w, vx.at(3, pool, vpid, voff, vd2))

	var ox exprs
	nv := ox.at(0, dims, ox.coord(0), ox.coord(1), ox.coord(2))
	l := ox.at(1, heads, ox.coord(0), ox.coord(1))
	ox.op(OpMul, F, nv, ox.op(OpDiv, F, ox.f(1), ox.op(OpMax, F, l, ox.f(1e-30))))

	tables := func(ops ...Operand) []Operand { return append([]Operand{ops[0], tableRows, tablePages}, ops[1:]...) }
	return []Prim{
		{Kind: PrimContract, Shape: []int{R, H, C, D}, In: tables(c.value(o.In[0]), tableKeys), Out: dots, Expr: dx.e, Facts: rng(2)},
		{Kind: PrimMap, Shape: keys, In: tables(dots), Out: scores, Expr: sx.e, Facts: rng(2)},
		{Kind: PrimReduce, Shape: keys, In: tables(scores), Out: maxes, Combine: OpMax, Init: int64(math.Float32bits(minus)),
			Expr: mx.e, Facts: rng(2)},
		{Kind: PrimMap, Shape: keys, In: tables(scores, maxes), Out: weights, Expr: wx.e, Facts: rng(2)},
		{Kind: PrimReduce, Shape: keys, In: tables(weights), Out: sums, Combine: OpAdd, Expr: lx.e, Facts: rng(2)},
		{Kind: PrimContract, Shape: []int{R, H, D, C}, In: tables(weights, tableValues), Out: values, Expr: vx.e, Facts: rng(3)},
		{Kind: PrimMap, Shape: dims, In: []Operand{values, sums}, Out: out, Expr: ox.e, Facts: Facts{Heads: &Heads{Rows: R, Query: H, Key: a.NKVHead, Dim: D}}},
	}, nil
}
