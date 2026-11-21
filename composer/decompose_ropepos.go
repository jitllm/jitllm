package composer

import (
	"fmt"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/graph"
)

// The rotary positions' inputs the engine writes: the step's rows (its live
// row count, then a row's cache position and the word its sequence's marks
// start at) and the sequences' marks (each a count, then Step.Marks marks of
// graph.MarkWords words, ordered by first position).

// decomposeRopePos is each row's position on the rotary axes: how many of
// its sequence's marks start at or before its cache position (a scan over
// the sequence's live marks), then from the last of them each axis's
// coordinate as the mark states it (graph.MarkPosition), and the position
// itself before every mark.
func decomposeRopePos(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 0 || len(o.Out) != 1 || c.width(o.Out[0]) != ropeAxes {
		return nil, fmt.Errorf("rotary positions of %d inputs into %d outputs of %d", len(o.In), len(o.Out), c.width(o.Out[0]))
	}
	U, P := ir.U32, ir.Pred
	R, M, W := c.Step.Rows, c.Step.Marks, graph.MarkWords
	out := c.value(o.Out[0])
	rowWords := map[int]int{0: 1 + 2*R}
	pos := func(x *exprs) int { return x.load(0, x.op(OpAdd, U, x.op(OpMul, U, x.coord(0), x.u(2)), x.u(1))) }
	if M == 0 {
		var x exprs
		pos(&x)
		return []Prim{{Kind: PrimMap, Shape: []int{R, ropeAxes}, In: []Operand{TableRopeRows.op()}, Out: out, Expr: x.e,
			Facts: Facts{Extents: rowWords}}}, nil
	}
	ext := map[int]int{0: 1 + 2*R, 1: c.Step.Histories() * (1 + M*W)}
	off := func(x *exprs) int { return x.load(0, x.op(OpAdd, U, x.op(OpMul, U, x.coord(0), x.u(2)), x.u(2))) }
	word := func(x *exprs, m, w int) int {
		return x.load(1, x.op(OpAdd, U, off(x), x.op(OpAdd, U, x.op(OpMul, U, m, x.u(uint32(W))), x.u(uint32(1+w)))))
	}
	// The marks at or before the row's position: their count is one past
	// the last one's index, the marks ordered by first position.
	last := Operand{Kind: OperandScratch, Name: out.Name + ".mark", Elem: U, Temp: true}
	var sx exprs
	mk, live := sx.pos(), sx.live()
	at := word(&sx, sx.op(OpMin, U, mk, sx.op(OpSub, U, sx.op(OpMax, U, live, sx.u(1)), sx.u(1))), graph.MarkAt)
	held := sx.sel(U, sx.op(OpLess, P, mk, live), sx.sel(U, sx.op(OpLess, P, pos(&sx), at), sx.u(0), sx.u(1)), sx.u(0))
	sx.op(OpAdd, U, held, sx.u(0))
	var lo, hi exprs
	lo.u(0)
	hi.load(1, off(&hi))
	scan := Prim{Kind: PrimReduce, Shape: []int{R, M}, In: []Operand{TableRopeRows.op(), TableRopeMarks.op()}, Out: last, Combine: OpAdd, Type: U,
		Expr: sx.e, Facts: Facts{Range: &Range{Axis: 1, First: lo.e, End: hi.e}, Extents: ext}}

	var x exprs
	r := x.coord(0)
	p := pos(&x)
	k := x.load(2, r)
	m := x.op(OpSub, U, x.op(OpMax, U, k, x.u(1)), x.u(1))
	f := func(w int) int { return word(&x, m, w) }
	at, skip, count := f(graph.MarkAt), f(graph.MarkSkip), f(graph.MarkCount)
	j0 := x.op(OpSub, U, p, at)
	j := x.op(OpSub, U, j0, skip)
	// The mark's span: past its skip and short of its count, where it has
	// one.
	before := x.op(OpLess, P, j0, skip)
	past := x.sel(U, x.op(OpLess, P, x.u(0), count), x.sel(U, x.op(OpLess, P, j, count), x.u(0), x.u(1)), x.u(0))
	inside := x.op(OpLess, P, x.op(OpAdd, U, x.sel(U, before, x.u(1), x.u(0)), past), x.u(1))
	marked := x.op(OpLess, P, x.u(0), k)
	var stores []PrimStore
	for a := range ropeAxes {
		w := func(i int) int { return f(graph.MarkAxis(a, i)) }
		q := x.op(OpDiv, U, j, x.op(OpMax, U, w(graph.MarkDiv), x.u(1)))
		mod := w(graph.MarkMod)
		q = x.sel(U, x.op(OpLess, P, x.u(0), mod), x.op(OpRem, U, q, x.op(OpMax, U, mod, x.u(1))), q)
		v := x.op(OpAdd, U, w(graph.MarkBase), x.op(OpAdd, U, x.op(OpMul, U, w(graph.MarkPos), p), x.op(OpMul, U, w(graph.MarkMul), q)))
		v = x.sel(U, inside, v, p)
		v = x.sel(U, marked, v, p)
		stores = append(stores, PrimStore{Out: out, Index: x.op(OpAdd, U, x.op(OpMul, U, r, x.u(uint32(ropeAxes))), x.u(uint32(a))), Value: v, Extent: R * ropeAxes})
	}
	turn := Prim{Kind: PrimMap, Shape: []int{R}, In: []Operand{TableRopeRows.op(), TableRopeMarks.op(), last}, Expr: x.e, Stores: stores,
		Facts: Facts{Extents: map[int]int{0: ext[0], 1: ext[1], 2: R}}}
	return []Prim{scan, turn}, nil
}
