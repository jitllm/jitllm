package jit

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/ir"
)

// tableIdxKeys is an indexer's history, its keys a part's own.
var tableIdxKeys = Operand{Kind: OperandTable, Name: "idx.k", PerPart: true}

// maskOf is the keep mask an indexer's selection sel writes and its
// attention reads: one a row and attended position, 1 kept and 0 not.
func maskOf(c *Context, sel graph.ValueID) Operand {
	return Operand{Kind: OperandScratch, Name: c.value(sel).Name + ".mask"}
}

// decomposeIndexSelect is an indexer's selection: the row's key into its
// history's page, each head's dot with every attended position's key, the
// sum over heads of each weight times its dot's relu, each position's rank
// (the positions scoring higher, or as high and earlier) and the mask
// keeping those ranked below TopK.
func decomposeIndexSelect(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	if len(o.In) != 3 || len(o.Out) != 1 || a.NHead < 1 || a.HeadDim < 1 || a.TopK < 1 || c.Step.Page <= 0 || c.Step.Keys <= 0 {
		return nil, fmt.Errorf("an index selection of %d operands", len(o.In))
	}
	H, D := a.NHead, a.HeadDim
	if c.width(o.In[0]) != H*D || c.width(o.In[1]) != H || c.width(o.In[2]) != D {
		return nil, fmt.Errorf("an index selection of %d heads of %d", H, D)
	}
	U, F := ir.U32, ir.F32
	R, C, pg := c.Step.Rows, c.Step.Keys, c.Step.Page
	hs := history{pageSize: pg}
	pool := []int{c.pages(), pg, D}
	ext := func(f Facts) Facts {
		f.Extents = map[int]int{1: R * rowWords, 2: c.pages()}
		return f
	}
	name := c.value(o.Out[0]).Name
	tmp := func(s string) Operand { return Operand{Kind: OperandScratch, Name: name + "." + s, Temp: true} }
	dots, scores, ranks := tmp("dots"), tmp("scores"), tmp("ranks")
	tables := func(ops ...Operand) []Operand { return append([]Operand{ops[0], tableRows, tablePages}, ops[1:]...) }

	var dx exprs
	h, d := dx.coord(1), dx.coord(3)
	pos, first := hs.position(&dx)
	pid, off := hs.page(&dx, hs.held(&dx, pos, first))
	qat := dx.op(OpAdd, U, dx.op(OpMul, U, dx.op(OpAdd, U, dx.op(OpMul, U, dx.coord(0), dx.u(uint32(H))), h), dx.u(uint32(D))), d)
	dx.op(OpMul, F, dx.load(0, qat), dx.at(3, pool, pid, off, d))

	var sx exprs
	sh := sx.coord(2)
	dot := sx.at(0, []int{R, H, C}, sx.coord(0), sh, sx.coord(1))
	sx.op(OpMul, F, sx.op(OpMax, F, dot, sx.f(0)), sx.at(3, []int{H}, sh))

	// The attended count: the row's end past its first.
	count := func(x *exprs) int { return x.op(OpSub, U, hs.row(x, rowEnd), hs.start(x)) }

	var rx exprs
	rc, ru := rx.coord(1), rx.coord(2)
	sc := rx.at(0, []int{R, C}, rx.coord(0), rc)
	su := rx.at(0, []int{R, C}, rx.coord(0), ru)
	one, zero := rx.f(1), rx.f(0)
	// Of two equal scores the earlier position is ahead: equal is neither
	// below the other.
	earlier := rx.sel(F, rx.op(OpLess, ir.Pred, ru, rc), one, zero)
	tie := rx.sel(F, rx.op(OpLess, ir.Pred, su, sc), zero, earlier)
	ahead := rx.sel(F, rx.op(OpLess, ir.Pred, sc, su), one, tie)
	rx.sel(F, rx.op(OpLess, ir.Pred, ru, count(&rx)), ahead, zero)

	var mx exprs
	mc := mx.coord(1)
	rank := mx.at(0, []int{R, C}, mx.coord(0), mc)
	live := mx.sel(F, mx.op(OpLess, ir.Pred, mc, count(&mx)), mx.f(1), mx.f(0))
	mx.sel(F, mx.op(OpLess, ir.Pred, rank, mx.f(float32(a.TopK))), live, mx.f(0))

	ranged := ext(Facts{Range: hs.span(2)})
	ranged.Extents[0] = R * H * D
	return []Prim{
		appendKey(c, c.value(o.In[2]), D),
		{Kind: PrimContract, Shape: []int{R, H, C, D}, In: tables(c.value(o.In[0]), tableIdxKeys), Out: dots, Expr: dx.e, Facts: ranged},
		{Kind: PrimReduce, Shape: []int{R, C, H}, In: tables(dots, c.value(o.In[1])), Out: scores, Combine: OpAdd, Expr: sx.e,
			Facts: ext(Facts{})},
		{Kind: PrimReduce, Shape: []int{R, C, C}, In: tables(scores), Out: ranks, Combine: OpAdd, Expr: rx.e, Facts: ext(Facts{})},
		{Kind: PrimMap, Shape: []int{R, C}, In: tables(ranks), Out: maskOf(c, o.Out[0]), Expr: mx.e, Facts: ext(Facts{})},
	}, nil
}

// appendKey writes each row's indexer key, d wide, into its page of the
// indexer's history.
func appendKey(c *Context, key Operand, d int) Prim {
	R, pg := c.Step.Rows, c.Step.Page
	hs := history{pageSize: pg}
	var x exprs
	at := x.coord(1)
	pid, off := hs.page(&x, hs.row(&x, rowWrite))
	v := x.at(0, []int{R, d}, x.coord(0), at)
	return Prim{Kind: PrimMap, Shape: []int{R, d}, In: []Operand{key, tableRows, tablePages}, Expr: x.e,
		Facts:  Facts{Extents: map[int]int{1: R * rowWords, 2: c.pages()}},
		Stores: []PrimStore{{Out: tableIdxKeys, Ax: []int{pid, off, at}, Ext: []int{c.pages(), pg, d}, Value: v}}}
}

// decomposeBlockSelect is a block-selecting indexer's selection: the row's
// key into its history, each group's dot with every attended position's
// key, each block's best score and the earliest position holding it, each
// block's rank among the blocks not forced in, and the mask a group keeps:
// every block where there are at most TopK, else the Local blocks ending at
// the query's and the best of the rest up to TopK in all.
func decomposeBlockSelect(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	if len(o.In) != 2 || len(o.Out) != 1 || a.NHead < 1 || a.HeadDim < 1 || a.TopK < 1 || a.Block < 1 || a.Local < 0 ||
		c.Step.Page <= 0 || c.Step.Keys <= 0 {
		return nil, fmt.Errorf("a block selection of %d operands", len(o.In))
	}
	G, D, B := a.NHead, a.HeadDim, a.Block
	if c.width(o.In[0]) != G*D || c.width(o.In[1]) != D {
		return nil, fmt.Errorf("a block selection of %d groups of %d", G, D)
	}
	U, F := ir.U32, ir.F32
	R, C, pg := c.Step.Rows, c.Step.Keys, c.Step.Page
	NB := (C + B - 1) / B
	hs := history{pageSize: pg}
	minus := math.Float32frombits(0xff7fffff)
	ext := func() Facts { return Facts{Extents: map[int]int{1: R * rowWords, 2: c.pages()}} }
	name := c.value(o.Out[0]).Name
	tmp := func(s string) Operand { return Operand{Kind: OperandScratch, Name: name + "." + s, Temp: true} }
	scores, best, first, ranks := tmp("scores"), tmp("best"), tmp("first"), tmp("ranks")
	tables := func(ops ...Operand) []Operand { return append([]Operand{ops[0], tableRows, tablePages}, ops[1:]...) }
	fl := func(x *exprs, v int) int { return x.un(OpCvtF32, F, x.un(OpBitcast, ir.I32, v)) }
	// n is the attended count; nb its blocks; own the query's block.
	n := func(x *exprs) int { return x.op(OpSub, U, hs.row(x, rowEnd), hs.start(x)) }
	nb := func(x *exprs) int { return x.op(OpDiv, U, x.op(OpAdd, U, n(x), x.u(uint32(B-1))), x.u(uint32(B))) }
	own := func(x *exprs) int { return x.op(OpDiv, U, x.op(OpSub, U, x.op(OpMax, U, n(x), x.u(1)), x.u(1)), x.u(uint32(B))) }
	// forced is 1 where block b is one of the Local ending at the query's.
	forced := func(x *exprs, b int) int {
		in := x.sel(F, x.op(OpLess, ir.Pred, x.op(OpSub, U, own(x), b), x.u(uint32(a.Local))), x.f(1), x.f(0))
		return x.sel(F, x.op(OpLess, ir.Pred, own(x), b), x.f(0), in)
	}

	var dx exprs
	g, d := dx.coord(1), dx.coord(3)
	pos, start := hs.position(&dx)
	pid, off := hs.page(&dx, hs.held(&dx, pos, start))
	qat := dx.op(OpAdd, U, dx.op(OpMul, U, dx.op(OpAdd, U, dx.op(OpMul, U, dx.coord(0), dx.u(uint32(G))), g), dx.u(uint32(D))), d)
	dx.op(OpMul, F, dx.load(0, qat), dx.at(3, []int{c.pages(), pg, D}, pid, off, d))
	ranged := ext()
	ranged.Range = hs.span(2)
	ranged.Extents[0] = R * G * D

	// A position of block b: its score where it is attended, else the
	// least float.
	at := func(x *exprs, b, j int) (t, s int) {
		t = x.op(OpAdd, U, x.op(OpMul, U, b, x.u(uint32(B))), j)
		v := x.at(0, []int{R, G, C}, x.coord(0), x.coord(1), x.op(OpMin, U, t, x.u(uint32(C-1))))
		return t, x.sel(F, x.op(OpLess, ir.Pred, t, n(x)), v, x.f(minus))
	}
	var bx exprs
	at(&bx, bx.coord(2), bx.coord(3))

	var fx exprs
	ft, fs := at(&fx, fx.coord(2), fx.coord(3))
	top := fx.at(3, []int{R, G, NB}, fx.coord(0), fx.coord(1), fx.coord(2))
	fx.sel(F, fx.op(OpLess, ir.Pred, fs, top), fx.f(float32(C)), fl(&fx, ft))

	var rx exprs
	rb, ru := rx.coord(2), rx.coord(3)
	// Operand 0 is the blocks' best, 3 their first positions.
	cell := func(j, b int) int { return rx.at(j, []int{R, G, NB}, rx.coord(0), rx.coord(1), b) }
	mb, mu := cell(0, rb), cell(0, ru)
	ab, au := cell(3, rb), cell(3, ru)
	one, zero := rx.f(1), rx.f(0)
	earlier := rx.sel(F, rx.op(OpLess, ir.Pred, au, ab), one, zero)
	ahead := rx.sel(F, rx.op(OpLess, ir.Pred, mb, mu), one, rx.sel(F, rx.op(OpLess, ir.Pred, mu, mb), zero, earlier))
	counted := rx.sel(F, rx.op(OpLess, ir.Pred, rx.f(0.5), forced(&rx, ru)), zero, ahead)
	rx.sel(F, rx.op(OpLess, ir.Pred, ru, nb(&rx)), counted, zero)

	var mx exprs
	mt := mx.coord(2)
	mbk := mx.op(OpDiv, U, mt, mx.u(uint32(B)))
	rank := mx.at(0, []int{R, G, NB}, mx.coord(0), mx.coord(1), mbk)
	// The blocks the rest may take: TopK less those forced in.
	nforced := mx.op(OpMin, U, mx.u(uint32(a.Local)), mx.op(OpAdd, U, own(&mx), mx.u(1)))
	room := fl(&mx, mx.op(OpSub, U, mx.op(OpMax, U, mx.u(uint32(a.TopK)), nforced), nforced))
	chosen := mx.sel(F, mx.op(OpLess, ir.Pred, rank, room), mx.f(1), forced(&mx, mbk))
	all := mx.sel(F, mx.op(OpLess, ir.Pred, mx.u(uint32(a.TopK)), nb(&mx)), chosen, mx.f(1))
	mx.sel(F, mx.op(OpLess, ir.Pred, mt, n(&mx)), all, mx.f(0))

	return []Prim{
		appendKey(c, c.value(o.In[1]), D),
		{Kind: PrimContract, Shape: []int{R, G, C, D}, In: tables(c.value(o.In[0]), tableIdxKeys), Out: scores, Expr: dx.e, Facts: ranged},
		{Kind: PrimReduce, Shape: []int{R, G, NB, B}, In: tables(scores), Out: best, Combine: OpMax,
			Init: int64(math.Float32bits(minus)), Expr: bx.e, Facts: ext()},
		{Kind: PrimReduce, Shape: []int{R, G, NB, B}, In: tables(scores, best), Out: first, Combine: OpMin,
			Init: int64(math.Float32bits(float32(C))), Expr: fx.e, Facts: ext()},
		{Kind: PrimReduce, Shape: []int{R, G, NB, NB}, In: tables(best, first), Out: ranks, Combine: OpAdd, Expr: rx.e, Facts: ext()},
		{Kind: PrimMap, Shape: []int{R, G, C}, In: tables(ranks), Out: maskOf(c, o.Out[0]), Expr: mx.e, Facts: ext()},
	}, nil
}
