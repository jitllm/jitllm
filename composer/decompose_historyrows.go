package composer

import (
	"fmt"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/graph"
)

// decomposeHistoryRows is OpHistoryRows: for each row, position of its
// window (Local windows before the one holding the row's position) and
// element, the element of the position's history row, or Const where the
// position is before the first or past the row's own -- read at the row's
// own position then, so every read is of a page the row holds.
func decomposeHistoryRows(o graph.Op, c *Context) ([]Prim, error) {
	undo, err := c.onHistory(o)
	if err != nil {
		return nil, err
	}
	defer undo()
	a := o.Attr
	if len(o.In) != 0 || len(o.Out) != 1 || len(o.Weight) != 0 || a.Span < 1 || a.Local < 0 || c.width(o.Out[0])%a.Span != 0 ||
		a.Offset < 0 || a.Offset+c.width(o.Out[0])/a.Span > a.Width || c.Step.Page <= 0 {
		return nil, fmt.Errorf("history rows of windows of %d, %d back, %d wide from %d of %d", a.Span, a.Local, c.width(o.Out[0]), a.Offset, a.Width)
	}
	if c.History.Period > 1 {
		return nil, fmt.Errorf("history rows over entries of a history of one each %d positions", c.History.Period)
	}
	if err := c.History.Keys.check(a.Width); err != nil {
		return nil, err
	}
	U, F := ir.U32, ir.F32
	R, P := c.Step.Rows, a.Span
	W := c.width(o.Out[0]) / P
	hs := history{pageSize: c.Step.Page}
	pool := c.History.Keys.operand(c.tab(TableKeys))

	var x exprs
	t := hs.row(&x, rowWrite)
	j, e := x.op(OpDiv, U, x.coord(1), x.u(uint32(W))), x.op(OpRem, U, x.coord(1), x.u(uint32(W)))
	back := x.u(uint32(a.Local * P))
	ws := x.op(OpMul, U, x.op(OpDiv, U, t, x.u(uint32(P))), x.u(uint32(P)))
	p := x.op(OpAdd, U, x.op(OpSub, U, ws, back), j)
	one := x.sel(U, x.op(OpLess, ir.Pred, ws, back), x.u(0), x.sel(U, x.op(OpLess, ir.Pred, t, p), x.u(0), x.u(1)))
	ok := x.op(OpLess, ir.Pred, x.u(0), one)
	pid, off := hs.page(&x, x.sel(U, ok, p, t))
	v := c.poolRead(&x, 0, c.History.Keys, a.Width, pid, off, x.op(OpAdd, U, x.u(uint32(a.Offset)), e))
	x.sel(F, ok, v, x.f(float32(a.Const)))
	return []Prim{{Kind: PrimMap, Shape: []int{R, P * W}, In: []Operand{pool, TableRows.op(), c.tab(TablePages)}, Out: c.value(o.Out[0]),
		Expr: x.e, Facts: Facts{Extents: map[int]int{1: R * rowWords, 2: c.pages()}}}}, nil
}
