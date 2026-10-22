package jit

import (
	"fmt"

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

	var ax exprs
	ad := ax.coord(1)
	apid, aoff := hs.page(&ax, hs.row(&ax, rowWrite))
	av := ax.at(0, []int{R, D}, ax.coord(0), ad)

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
		{Kind: PrimMap, Shape: []int{R, D}, In: tables(c.value(o.In[2])), Expr: ax.e, Facts: ext(Facts{}),
			Stores: []PrimStore{{Out: tableIdxKeys, Ax: []int{apid, aoff, ad}, Ext: pool, Value: av}}},
		{Kind: PrimContract, Shape: []int{R, H, C, D}, In: tables(c.value(o.In[0]), tableIdxKeys), Out: dots, Expr: dx.e, Facts: ranged},
		{Kind: PrimReduce, Shape: []int{R, C, H}, In: tables(dots, c.value(o.In[1])), Out: scores, Combine: OpAdd, Expr: sx.e,
			Facts: ext(Facts{})},
		{Kind: PrimReduce, Shape: []int{R, C, C}, In: tables(scores), Out: ranks, Combine: OpAdd, Expr: rx.e, Facts: ext(Facts{})},
		{Kind: PrimMap, Shape: []int{R, C}, In: tables(ranks), Out: maskOf(c, o.Out[0]), Expr: mx.e, Facts: ext(Facts{})},
	}, nil
}
