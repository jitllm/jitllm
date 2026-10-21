package jit

import (
	"math"

	"github.com/jitllm/jitllm/jit/ir"
)

// keyAlign is where a row's keys are counted from: its first attended
// position rounded down to it. keyGrain is what a step's longest history is
// rounded up to.
const (
	keyAlign = 32
	keyGrain = 128
)

// attention is decode attention over the paged history, unsplit: per row
// and head, the scores of capacity keys from the row's aligned first
// position, masked outside what the row attends, their maximum, the weights
// and their sum, the weighted values and the normalized row.
type attention struct {
	rows, heads, kv, dim, page, capacity int
	scale                                float32
	q, out                               Operand
}

func (x *exprs) coord(a int) int { return x.add(Expr{Op: ExprCoord, A: a}) }

func (x *exprs) at(j int, ext []int, ax ...int) int {
	return x.add(Expr{Op: ExprAt, In: j, Ax: ax, Ext: ext})
}

func (x *exprs) load(j, ix int) int { return x.add(Expr{Op: ExprLoad, In: j, A: ix}) }

func (a attention) prims() []Prim {
	U, F := ir.U32, ir.F32
	R, H, C, D := a.rows, a.heads, a.capacity, a.dim
	gqa, kw := a.heads/a.kv, a.kv*a.dim
	tmp := func(name string) Operand {
		return Operand{Kind: OperandScratch, Name: a.out.Name + "." + name, Temp: true}
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
	rng := func(axis int) *Range {
		return &Range{Axis: axis, Align: keyAlign, First: bound(rowStart), End: bound(rowEnd)}
	}
	// position is key k's position, its first the row's first rounded down.
	position := func(x *exprs, k int) (pos, first int) {
		first = row(x, rowStart)
		return x.op(OpAdd, U, x.op(OpMul, U, x.op(OpDiv, U, first, x.u(keyAlign)), x.u(keyAlign)), k), first
	}
	// held is a position kept inside what the row attends.
	held := func(x *exprs, pos, first int) int {
		last := x.op(OpSub, U, x.op(OpMax, U, row(x, rowEnd), x.u(1)), x.u(1))
		return x.op(OpMin, U, x.op(OpMax, U, pos, first), last)
	}
	page := func(x *exprs, t int) (pid, off int) {
		pid = x.load(2, x.op(OpAdd, U, row(x, rowTab), x.op(OpDiv, U, t, x.u(uint32(a.page)))))
		return pid, x.op(OpRem, U, t, x.u(uint32(a.page)))
	}

	var dx exprs
	h, k, d := dx.coord(1), dx.coord(2), dx.coord(3)
	pos, first := position(&dx, k)
	pid, off := page(&dx, held(&dx, pos, first))
	kh := dx.op(OpDiv, U, h, dx.u(uint32(gqa)))
	kat := dx.op(OpAdd, U, dx.op(OpMul, U, pid, dx.u(uint32(a.page*kw))),
		dx.op(OpAdd, U, dx.op(OpMul, U, dx.op(OpAdd, U, dx.op(OpMul, U, kh, dx.u(uint32(D))), d), dx.u(uint32(a.page))), off))
	qat := dx.op(OpAdd, U, dx.op(OpMul, U, dx.op(OpAdd, U, dx.op(OpMul, U, dx.coord(0), dx.u(uint32(H))), h), dx.u(uint32(D))), d)
	dx.op(OpMul, F, dx.load(0, qat), dx.load(3, kat))

	var sx exprs
	spos, sfirst := position(&sx, sx.coord(2))
	live := sx.add(Expr{Op: ExprLive})
	valid := sx.op(OpLess, ir.Pred, sx.op(OpSub, U, spos, sfirst), sx.op(OpSub, U, live, sfirst))
	dot := sx.at(0, keys, sx.coord(0), sx.coord(1), sx.coord(2))
	sx.sel(F, valid, sx.op(OpMul, F, dot, sx.f(a.scale)), sx.f(minus))

	var mx exprs
	mx.at(0, keys, mx.coord(0), mx.coord(1), mx.coord(2))

	var wx exprs
	wpos, _ := position(&wx, wx.coord(2))
	below := wx.op(OpLess, ir.Pred, wpos, wx.add(Expr{Op: ExprLive}))
	s := wx.at(0, keys, wx.coord(0), wx.coord(1), wx.coord(2))
	m := wx.at(3, heads, wx.coord(0), wx.coord(1))
	wx.sel(F, below, wx.un(OpExp, F, wx.op(OpSub, F, s, m)), wx.f(0))

	var lx exprs
	lx.at(0, keys, lx.coord(0), lx.coord(1), lx.coord(2))

	var vx exprs
	vh, vd, vk := vx.coord(1), vx.coord(2), vx.coord(3)
	vpos, vfirst := position(&vx, vk)
	vpid, voff := page(&vx, held(&vx, vpos, vfirst))
	vkh := vx.op(OpDiv, U, vh, vx.u(uint32(gqa)))
	vat := vx.op(OpAdd, U, vx.op(OpMul, U, vx.op(OpAdd, U, vx.op(OpMul, U, vpid, vx.u(uint32(a.page))), voff), vx.u(uint32(kw))),
		vx.op(OpAdd, U, vx.op(OpMul, U, vkh, vx.u(uint32(D))), vd))
	w := vx.at(0, keys, vx.coord(0), vh, vk)
	vx.op(OpMul, F, w, vx.load(3, vat))

	var ox exprs
	nv := ox.at(0, dims, ox.coord(0), ox.coord(1), ox.coord(2))
	l := ox.at(1, heads, ox.coord(0), ox.coord(1))
	ox.op(OpMul, F, nv, ox.op(OpDiv, F, ox.f(1), ox.op(OpMax, F, l, ox.f(1e-30))))

	tables := func(ops ...Operand) []Operand { return append([]Operand{ops[0], tableRows, tablePages}, ops[1:]...) }
	hd := &Heads{Rows: R, Query: H, Key: a.kv, Dim: D}
	return []Prim{
		{Kind: PrimReduce, Shape: []int{R, H, C, D}, N: R * H * C, K: D, In: tables(a.q, tableKeys), Out: dots, Combine: OpFma,
			Expr: dx.e, Range: rng(2), Heads: hd},
		{Kind: PrimMap, Shape: keys, N: R * H * C, In: tables(dots), Out: scores, Expr: sx.e, Range: rng(2), Heads: hd},
		{Kind: PrimReduce, Shape: keys, N: R * H, K: C, In: tables(scores), Out: maxes, Combine: OpMax,
			Init: int64(math.Float32bits(minus)), Expr: mx.e, Range: rng(2), Heads: hd},
		{Kind: PrimMap, Shape: keys, N: R * H * C, In: tables(scores, maxes), Out: weights, Expr: wx.e, Range: rng(2), Heads: hd},
		{Kind: PrimReduce, Shape: keys, N: R * H, K: C, In: tables(weights), Out: sums, Combine: OpAdd, Expr: lx.e, Range: rng(2), Heads: hd},
		{Kind: PrimReduce, Shape: []int{R, H, D, C}, N: R * H * D, K: C, In: tables(weights, tableValues), Out: values, Combine: OpFma,
			Expr: vx.e, Range: rng(3), Heads: hd},
		{Kind: PrimMap, Shape: dims, N: R * H * D, In: []Operand{values, sums}, Out: a.out, Expr: ox.e, Heads: hd},
	}
}
