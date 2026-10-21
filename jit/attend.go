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

func (a attention) prims() []Prim {
	U, F := ir.U32, ir.F32
	R, H, C, D := a.rows, a.heads, a.capacity, a.dim
	gqa, kw := a.heads/a.kv, a.kv*a.dim
	tmp := func(name string) Operand {
		return Operand{Kind: OperandScratch, Name: a.out.Name + "." + name, Temp: true}
	}
	dots, scores, maxes, weights, sums, values := tmp("dots"), tmp("scores"), tmp("max"), tmp("weights"), tmp("sum"), tmp("values")
	minus := math.Float32frombits(0xff7fffff)

	// key is position j of row r's keys, held inside what the row attends,
	// and whether the row attends it.
	key := func(x *exprs, r, j int) (t, valid int) {
		at := func(w int) int {
			return x.add(Expr{Op: ExprLoad, In: 1, A: x.op(OpAdd, U, x.op(OpMul, U, r, x.u(rowWords)), x.u(uint32(w)))})
		}
		first, end := at(rowStart), at(rowEnd)
		base := x.op(OpMul, U, x.op(OpDiv, U, first, x.u(keyAlign)), x.u(keyAlign))
		raw := x.op(OpAdd, U, base, j)
		valid = x.op(OpLess, ir.Pred, x.op(OpSub, U, raw, first), x.op(OpSub, U, end, first))
		last := x.op(OpSub, U, x.op(OpMax, U, end, x.u(1)), x.u(1))
		return x.op(OpMin, U, x.op(OpMax, U, raw, first), last), valid
	}
	page := func(x *exprs, r, t int) (pid, off int) {
		tab := x.add(Expr{Op: ExprLoad, In: 1, A: x.op(OpAdd, U, x.op(OpMul, U, r, x.u(rowWords)), x.u(rowTab))})
		pid = x.add(Expr{Op: ExprLoad, In: 2, A: x.op(OpAdd, U, tab, x.op(OpDiv, U, t, x.u(uint32(a.page))))})
		return pid, x.op(OpRem, U, t, x.u(uint32(a.page)))
	}
	// split is an element of (row, head, key) as its parts.
	split := func(x *exprs, e int) (r, h, j int) {
		j = x.op(OpRem, U, e, x.u(uint32(C)))
		h = x.op(OpRem, U, x.op(OpDiv, U, e, x.u(uint32(C))), x.u(uint32(H)))
		return x.op(OpDiv, U, e, x.u(uint32(C*H))), h, j
	}

	var dx exprs
	el := dx.add(Expr{Op: ExprElem})
	d := dx.op(OpRem, U, el, dx.u(uint32(D)))
	r, h, j := split(&dx, dx.op(OpDiv, U, el, dx.u(uint32(D))))
	t, _ := key(&dx, r, j)
	pid, off := page(&dx, r, t)
	kh := dx.op(OpDiv, U, h, dx.u(uint32(gqa)))
	kat := dx.op(OpAdd, U, dx.op(OpMul, U, pid, dx.u(uint32(a.page*kw))),
		dx.op(OpAdd, U, dx.op(OpMul, U, dx.op(OpAdd, U, dx.op(OpMul, U, kh, dx.u(uint32(D))), d), dx.u(uint32(a.page))), off))
	qat := dx.op(OpAdd, U, dx.op(OpMul, U, dx.op(OpAdd, U, dx.op(OpMul, U, r, dx.u(uint32(H))), h), dx.u(uint32(D))), d)
	dx.op(OpMul, F, dx.add(Expr{Op: ExprLoad, In: 0, A: qat}), dx.add(Expr{Op: ExprLoad, In: 3, A: kat}))

	var sx exprs
	se := sx.add(Expr{Op: ExprElem})
	sr, _, sj := split(&sx, se)
	_, svalid := key(&sx, sr, sj)
	sx.sel(F, svalid, sx.op(OpMul, F, sx.add(Expr{Op: ExprLoad, In: 0, A: se}), sx.f(a.scale)), sx.f(minus))

	var mx exprs
	mx.in(0)

	var wx exprs
	we := wx.add(Expr{Op: ExprElem})
	wr, _, wj := split(&wx, we)
	_, wvalid := key(&wx, wr, wj)
	m := wx.add(Expr{Op: ExprLoad, In: 3, A: wx.op(OpDiv, U, we, wx.u(uint32(C)))})
	wx.sel(F, wvalid, wx.un(OpExp, F, wx.op(OpSub, F, wx.add(Expr{Op: ExprLoad, In: 0, A: we}), m)), wx.f(0))

	var lx exprs
	lx.in(0)

	var vx exprs
	ve := vx.add(Expr{Op: ExprElem})
	vd := vx.op(OpRem, U, vx.op(OpDiv, U, ve, vx.u(uint32(C))), vx.u(uint32(D)))
	vrh := vx.op(OpDiv, U, vx.op(OpDiv, U, ve, vx.u(uint32(C))), vx.u(uint32(D)))
	vj := vx.op(OpRem, U, ve, vx.u(uint32(C)))
	vr, vh := vx.op(OpDiv, U, vrh, vx.u(uint32(H))), vx.op(OpRem, U, vrh, vx.u(uint32(H)))
	vt, _ := key(&vx, vr, vj)
	vpid, voff := page(&vx, vr, vt)
	vkh := vx.op(OpDiv, U, vh, vx.u(uint32(gqa)))
	vat := vx.op(OpAdd, U, vx.op(OpMul, U, vx.op(OpAdd, U, vx.op(OpMul, U, vpid, vx.u(uint32(a.page))), voff), vx.u(uint32(kw))),
		vx.op(OpAdd, U, vx.op(OpMul, U, vkh, vx.u(uint32(D))), vd))
	w := vx.add(Expr{Op: ExprLoad, In: 0, A: vx.op(OpAdd, U, vx.op(OpMul, U, vrh, vx.u(uint32(C))), vj)})
	vx.op(OpMul, F, w, vx.add(Expr{Op: ExprLoad, In: 3, A: vat}))

	var ox exprs
	oe := ox.add(Expr{Op: ExprElem})
	l := ox.add(Expr{Op: ExprLoad, In: 1, A: ox.op(OpDiv, U, oe, ox.u(uint32(D)))})
	ox.op(OpMul, F, ox.add(Expr{Op: ExprLoad, In: 0, A: oe}), ox.op(OpDiv, F, ox.f(1), ox.op(OpMax, F, l, ox.f(1e-30))))

	tables := func(ops ...Operand) []Operand { return append([]Operand{ops[0], tableRows, tablePages}, ops[1:]...) }
	return []Prim{
		{Kind: PrimReduce, N: R * H * C, K: D, In: tables(a.q, tableKeys), Out: dots, Combine: OpAdd, Expr: dx.e},
		{Kind: PrimMap, N: R * H * C, In: tables(dots), Out: scores, Expr: sx.e},
		{Kind: PrimReduce, N: R * H, K: C, In: []Operand{scores}, At: []Access{atElement}, Out: maxes, Combine: OpMax,
			Init: int64(math.Float32bits(minus)), Expr: mx.e},
		{Kind: PrimMap, N: R * H * C, In: tables(scores, maxes), Out: weights, Expr: wx.e},
		{Kind: PrimReduce, N: R * H, K: C, In: []Operand{weights}, At: []Access{atElement}, Out: sums, Combine: OpAdd, Expr: lx.e},
		{Kind: PrimReduce, N: R * H * D, K: C, In: tables(weights, tableValues), Out: values, Combine: OpAdd, Expr: vx.e},
		{Kind: PrimMap, N: R * H * D, In: []Operand{values, sums}, Out: a.out, Expr: ox.e},
	}
}
