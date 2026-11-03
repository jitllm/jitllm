package composer

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/engine/graph"
)

// decomposeColPool is OpColPool over the paged history: for each row, column
// and slot of its window (the row's own position the last), the slot's gate
// where its position is held (a position before the history's first
// leaves it out), each column's largest, the softmax's weights and their
// sum, the weighted values' sum, and the quotient. A row of Overlap reads
// the Span before's first halves, then its own Span's second halves.
func decomposeColPool(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	halves := 1
	if a.Overlap {
		halves = 2
	}
	if len(o.Out) != 1 || len(o.Reads) != 1 || a.HeadDim <= 0 || a.Span <= 0 || a.Offset < 0 || a.GateAt < 0 ||
		a.Offset+halves*a.HeadDim > a.Width || a.GateAt+halves*a.HeadDim > a.Width || c.Step.Page <= 0 {
		return nil, fmt.Errorf("a pool of %d columns at %d and gates at %d of rows of %d, windows of %d", a.HeadDim, a.Offset, a.GateAt, a.Width, a.Span)
	}
	pool := c.History.Keys
	if err := pool.check(a.Width); err != nil {
		return nil, err
	}
	keys := pool.operand(TableKeys.op())
	U, F := ir.U32, ir.F32
	R, D, S := c.Step.Rows, a.HeadDim, halves*a.Span
	hs := history{a: a, pageSize: c.Step.Page}
	out := c.value(o.Out[0])
	tmp := func(n string) Operand {
		return Operand{Kind: OperandScratch, Name: out.Name + "." + n, Elem: F, Temp: true}
	}
	gmax, wts, sums, acc := tmp("gmax"), tmp("w"), tmp("sum"), tmp("acc")
	minus := math.Float32frombits(0xff7fffff)
	tables := func(ops ...Operand) []Operand {
		return append([]Operand{ops[0], TableRows.op(), TablePages.op()}, ops[1:]...)
	}
	flat := Facts{Extents: map[int]int{1: R * rowWords, 2: c.pages()}}

	// slot is slot j's (coordinate 2) hold and its column's element at at of
	// the history row, read from operand j of x.
	slot := func(x *exprs, at, op int) (held, v int) {
		j := x.coord(2)
		p1 := x.op(OpAdd, U, hs.row(x, rowWrite), x.u(1))
		tp := x.op(OpAdd, U, p1, j)
		lo := x.op(OpAdd, U, x.u(uint32(S)), hs.row(x, rowStart))
		held = x.sel(U, x.op(OpLess, ir.Pred, tp, lo), x.u(0), x.u(1))
		pid, off := hs.page(x, x.op(OpSub, U, x.op(OpMax, U, tp, lo), x.u(uint32(S))))
		col := x.op(OpAdd, U, x.coord(1), x.u(uint32(at)))
		if a.Overlap {
			col = x.op(OpAdd, U, col, x.op(OpMul, U, x.op(OpDiv, U, j, x.u(uint32(a.Span))), x.u(uint32(D))))
		}
		return held, c.poolRead(x, op, pool, a.Width, pid, off, col)
	}
	gate := func(x *exprs, op int) (held, g int) {
		held, g = slot(x, a.GateAt, op)
		return held, x.sel(F, x.op(OpLess, ir.Pred, x.u(0), held), g, x.f(minus))
	}

	var mx exprs
	gate(&mx, 0)

	var wx exprs
	wh, wg := gate(&wx, 0)
	wx.sel(F, wx.op(OpLess, ir.Pred, wx.u(0), wh), wx.un(OpExp, F, wx.op(OpSub, F, wg, wx.at(3, []int{R, D}, wx.coord(0), wx.coord(1)))), wx.f(0))

	var sx exprs
	sx.at(0, []int{R, D, S}, sx.coord(0), sx.coord(1), sx.coord(2))

	var ax exprs
	_, v := slot(&ax, a.Offset, 0)
	ax.op(OpMul, F, ax.at(3, []int{R, D, S}, ax.coord(0), ax.coord(1), ax.coord(2)), v)

	var ox exprs
	ox.op(OpDiv, F, ox.at(0, []int{R, D}, ox.coord(0), ox.coord(1)), ox.at(1, []int{R, D}, ox.coord(0), ox.coord(1)))

	return []Prim{
		{Kind: PrimReduce, Shape: []int{R, D, S}, In: tables(keys), Out: gmax, Combine: OpMax, Init: int64(math.Float32bits(minus)), Expr: mx.e, Facts: flat},
		{Kind: PrimMap, Shape: []int{R, D, S}, In: tables(keys, gmax), Out: wts, Expr: wx.e, Facts: flat},
		{Kind: PrimReduce, Shape: []int{R, D, S}, In: []Operand{wts}, Out: sums, Combine: OpAdd, Expr: sx.e},
		{Kind: PrimReduce, Shape: []int{R, D, S}, In: tables(keys, wts), Out: acc, Combine: OpAdd, Expr: ax.e, Facts: flat},
		{Kind: PrimMap, Shape: []int{R, D}, In: []Operand{acc, sums}, Out: out, Expr: ox.e},
	}, nil
}
