package composer

import (
	"fmt"
	"maps"
	"slices"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/graph"
)

// decomposeNGramWindow is each row's n-gram window and its run's
// predecessors carried on: the row's token (the image id for a row naming
// none), then its predecessors, an earlier row's of its run as its id plus
// one and past the run's first row the slot's; then each run's
// predecessors after its last row into a scratch, and from it to the slot
// written, so no kernel reads the slot while writing it. One row is its own
// run, its slots the descriptor's (stateConvWindow's).
func decomposeNGramWindow(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	n, R := a.Taps, c.Step.Rows
	if len(o.In) != 1 || len(o.Out) != 1 || n < 2 || len(a.Words) != 1 || a.Words[0] >= 1<<32 || R < 1 ||
		a.Offset < 0 || a.Span < a.Offset+n-1 {
		return nil, fmt.Errorf("an n-gram window of %d over %d rows at %d of a slot of %d", n, R, a.Offset, a.Span)
	}
	U, F := ir.U32, ir.F32
	tok, win := c.value(o.In[0]), c.value(o.Out[0])
	win.Elem = U
	img := uint32(a.Words[0])
	// id is a row's token as the window reads it.
	id := func(x *exprs, row int) int {
		t := x.at(0, []int{R, 1}, row, x.u(0))
		return x.sel(U, x.op(OpLess, ir.Pred, t, x.u(^uint32(0))), t, x.u(img))
	}
	// held is the slot's predecessor i of the run at word w of its slot.
	held := func(x *exprs, slot, i int) int {
		return x.un(OpBitcast, U, x.load(1, x.op(OpAdd, U, slot, x.op(OpAdd, U, x.u(uint32(a.Offset)), i))))
	}
	next := Operand{Kind: OperandScratch, Name: win.Name + ".next", Elem: U}
	if R == 1 && !a.Keeps {
		var x exprs
		k := x.coord(1)
		st := held(&x, x.recSlot(2, 1, a.Span), x.op(OpSub, U, x.op(OpMax, U, k, x.u(1)), x.u(1)))
		x.sel(U, x.op(OpLess, ir.Pred, k, x.u(1)), id(&x, x.coord(0)), st)
		var y exprs
		j := y.coord(1)
		yst := held(&y, y.recSlot(2, 1, a.Span), y.op(OpSub, U, y.op(OpMax, U, j, y.u(1)), y.u(1)))
		y.sel(U, y.op(OpLess, ir.Pred, j, y.u(1)), y.op(OpAdd, U, id(&y, y.coord(0)), y.u(1)), yst)
		var z exprs
		zj := z.coord(1)
		zv := z.un(OpBitcast, F, z.load(0, zj))
		to := z.op(OpAdd, U, z.recSlot(1, 2, a.Span), z.op(OpAdd, U, z.u(uint32(a.Offset)), zj))
		ext := Facts{Extents: map[int]int{1: a.Span, 2: 3}}
		return []Prim{
			{Kind: PrimMap, Shape: []int{1, n}, In: []Operand{tok, TableRecState.op(), TableRecRows.op()}, Out: win, Expr: x.e, Facts: ext},
			{Kind: PrimMap, Shape: []int{1, n - 1}, In: []Operand{tok, TableRecState.op(), TableRecRows.op()}, Out: next, Expr: y.e, Facts: ext},
			{Kind: PrimMap, Shape: []int{1, n - 1}, In: []Operand{next, TableRecRows.op()}, Expr: z.e,
				Facts: Facts{Extents: map[int]int{0: n - 1, 1: 3}}, Stores: []PrimStore{{Out: TableRecStateOut.op(), Index: to, Value: zv, Extent: a.Span}}},
		}, nil
	}

	// The run descriptor, as stateConvRuns reads it: the live rows, each
	// run's slots, its first place and count, the rows in run order, each
	// row's place and run.
	const desc = 2
	live := func(x *exprs) int { return x.load(desc, x.u(0)) }
	last := func(x *exprs, j int) int {
		return x.op(OpMin, U, j, x.op(OpSub, U, x.op(OpMax, U, live(x), x.u(1)), x.u(1)))
	}
	word := func(x *exprs, p, q, j int) int {
		return x.load(desc, x.op(OpAdd, U, x.op(OpAdd, U, x.u(uint32(p)), x.op(OpMul, U, live(x), x.u(uint32(q)))), j))
	}
	// run is a run's first entry of the order list, the order list's base
	// with it, its rows, and its slots read and written past Offset.
	type run struct{ start, base, count, in, out int }
	runAt := func(x *exprs, j int) run {
		j = last(x, j)
		twoJ := x.op(OpAdd, U, j, j)
		start := word(x, 1, 2, twoJ)
		slot := func(w int) int {
			s := x.load(desc, x.op(OpAdd, U, twoJ, x.u(uint32(w))))
			return x.op(OpAdd, U, x.op(OpMul, U, s, x.u(uint32(a.Span))), x.u(uint32(a.Offset)))
		}
		base := x.op(OpAdd, U, x.op(OpAdd, U, x.u(1), x.op(OpMul, U, live(x), x.u(4))), start)
		return run{start: start, base: base, count: word(x, 2, 2, twoJ), in: slot(1), out: slot(2)}
	}
	ext := map[int]int{1: a.Span, desc: RunDescWords(R)}
	in := []Operand{tok, TableRecState.op(), TableRecRows.op()}

	// pred is predecessor k (from 1) of place at of run r: its run's row
	// at-k where there is one, else the slot's k-at-1. Both indices are
	// clamped rather than selected, so they stay off the pick's mask.
	pred := func(x *exprs, r run, at int, k uint32) int {
		back := x.op(OpSub, U, x.op(OpMax, U, at, x.u(k)), x.u(k))
		slot := x.op(OpSub, U, x.u(k-1), x.op(OpMin, U, at, x.u(k-1)))
		row := x.load(desc, x.op(OpAdd, U, r.base, back))
		st := x.un(OpBitcast, U, x.load(1, x.op(OpAdd, U, r.in, slot)))
		return x.sel(U, x.op(OpLess, ir.Pred, at, x.u(k)), st, x.op(OpAdd, U, id(x, row), x.u(1)))
	}

	// Each row's window, a map over the rows a word.
	var prims []Prim
	for k := range n {
		var x exprs
		r := last(&x, x.coord(0))
		v := id(&x, r)
		if k > 0 {
			xr := runAt(&x, word(&x, 1, 6, r))
			v = pred(&x, xr, x.op(OpSub, U, word(&x, 1, 5, r), xr.start), uint32(k))
		}
		st := PrimStore{Out: win, Ax: []int{x.coord(0), x.u(uint32(k))}, Ext: []int{R, n}, Value: v}
		prims = append(prims, Prim{Kind: PrimMap, Shape: []int{R, 1}, In: in, Expr: x.e, Facts: Facts{Extents: maps.Clone(ext)},
			Stores: []PrimStore{st}})
	}
	// With Keeps, each entry's predecessors after its row -- place local+1
	// of its run's sequence -- to the entry's slot of the keep table, read
	// before the slot written is.
	for k := 0; a.Keeps && k < n-1; k++ {
		var q exprs
		e := last(&q, q.coord(0))
		qr := runAt(&q, word(&q, 1, 6, word(&q, 1, 4, e)))
		v := q.un(OpBitcast, F, pred(&q, qr, q.op(OpAdd, U, q.op(OpSub, U, e, qr.start), q.u(1)), uint32(k+1)))
		at := q.op(OpAdd, U, q.op(OpMul, U, e, q.u(uint32(a.Span))), q.u(uint32(a.Offset+k)))
		prims = append(prims, Prim{Kind: PrimMap, Shape: []int{R, 1}, In: in, Expr: q.e, Facts: Facts{Extents: maps.Clone(ext)},
			Stores: []PrimStore{{Out: TableRecKeep.op(), Index: at, Value: v, Extent: R * a.Span}}})
	}
	// Each run's predecessors after it, place count of its sequence's, into
	// a scratch; then from it to the run's slot written.
	for k := 0; k < n-1; k++ {
		var y exprs
		yr := runAt(&y, y.coord(0))
		v := pred(&y, yr, yr.count, uint32(k+1))
		st := PrimStore{Out: next, Ax: []int{y.coord(0), y.u(uint32(k))}, Ext: []int{R, n - 1}, Value: v}
		prims = append(prims, Prim{Kind: PrimMap, Shape: []int{R, 1}, In: in, Expr: y.e, Facts: Facts{Extents: maps.Clone(ext)},
			Stores: []PrimStore{st}})
	}
	for k := 0; k < n-1; k++ {
		var z exprs
		zr := runAt(&z, z.coord(0))
		v := z.un(OpBitcast, F, z.at(3, []int{R, n - 1}, z.coord(0), z.u(uint32(k))))
		st := PrimStore{Out: TableRecStateOut.op(), Index: z.op(OpAdd, U, zr.out, z.u(uint32(k))), Value: v, Extent: a.Span}
		prims = append(prims, Prim{Kind: PrimMap, Shape: []int{R, 1}, In: append(slices.Clone(in), next), Expr: z.e,
			Facts:  Facts{Extents: map[int]int{1: a.Span, desc: RunDescWords(R), 3: R * (n - 1)}},
			Stores: []PrimStore{st}})
	}
	return prims, nil
}
