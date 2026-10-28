package jit

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/ir"
)

// decomposeSelect is OpSelect over the paged history: each query head's dot
// with every attended position's key (its HeadDim elements at Offset of the
// pool row), each group's sum of its heads' scores through Fn and the
// weights, each block's best, each block's place among the blocks a group
// ranks (the Local ones ending at the row's own kept apart), then the keep
// mask an OpAttend reads, one a row, group and attended position, and the
// blocks each group keeps. Its ranking compares every pair of blocks: the
// basic form.
func decomposeSelect(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	if len(o.In) < 1 || len(o.In) > 2 || len(o.Out) != 1 || len(o.Reads) != 1 || a.Window != 0 || a.Bidir ||
		a.NHead <= 0 || a.NKVHead <= 0 || a.NHead%a.NKVHead != 0 || a.Span <= 0 || a.TopK <= 0 || a.Local < 0 ||
		a.HeadDim <= 0 || a.Offset < 0 || a.Offset+a.HeadDim > a.Width || c.Step.Page <= 0 || c.Step.Keys <= 0 {
		return nil, fmt.Errorf("a selection of %d heads in %d groups, keys of %d at %d of %d, blocks of %d, window %d",
			a.NHead, a.NKVHead, a.HeadDim, a.Offset, a.Width, a.Span, a.Window)
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
	pool := c.History.Keys
	if err := pool.check(a.Width); err != nil {
		return nil, err
	}
	keys := pool.operand(tableKeys)
	if r := o.Reads[0].Name; strings.HasPrefix(r, "kv@") {
		n, err := strconv.Atoi(strings.TrimPrefix(r, "kv@"))
		if err != nil || c.Graph.Block < 0 {
			return nil, fmt.Errorf("a selection over the history %q", r)
		}
		keys.Of = c.blockPart(n)
	}
	U, F := ir.U32, ir.F32
	R, H, D, G, C, B := c.Step.Rows, a.NHead, a.HeadDim, a.NKVHead, c.Step.Keys, a.Span
	J, NB := H/G, (c.Step.Keys+a.Span-1)/a.Span
	out := c.value(o.Out[0])
	tmp := func(name string, t ir.Type) Operand {
		return Operand{Kind: OperandScratch, Name: out.Name + "." + name, Elem: t, Temp: true}
	}
	dots, scores, best, rank, keep := tmp("dots", F), tmp("scores", F), tmp("best", F), tmp("rank", U), tmp("keep", U)
	mask := maskOf(c, o.Out[0])
	minus := math.Float32frombits(0xff7fffff)
	hs := history{a: a, pageSize: c.Step.Page}
	tables := func(ops ...Operand) []Operand { return append([]Operand{ops[0], tableRows, tablePages}, ops[1:]...) }
	ext := func() map[int]int { return map[int]int{1: R * rowWords, 2: c.pages()} }
	rng := Facts{Range: hs.span(2), Extents: ext()}
	rng.Extents[0] = R * H * D

	// The block's bounds on the row: its first and end positions, its own
	// block, the first of the blocks forced in and whether block b is one,
	// and whether block b holds an attended position.
	type bounds struct{ first, end, own, lo int }
	boundsOf := func(x *exprs) bounds {
		first, end := hs.row(x, rowStart), hs.row(x, rowEnd)
		own := x.op(OpDiv, U, x.op(OpSub, U, x.op(OpMax, U, end, x.u(1)), x.u(1)), x.u(uint32(B)))
		forced := x.op(OpMin, U, x.u(uint32(a.Local)), x.op(OpAdd, U, own, x.u(1)))
		return bounds{first, end, own, x.op(OpSub, U, x.op(OpAdd, U, own, x.u(1)), forced)}
	}
	isForced := func(x *exprs, bd bounds, b int) int {
		return x.sel(U, x.op(OpLess, ir.Pred, b, bd.lo), x.u(0), x.sel(U, x.op(OpLess, ir.Pred, bd.own, b), x.u(0), x.u(1)))
	}
	held := func(x *exprs, bd bounds, b int) int {
		lo, hi := x.op(OpMul, U, b, x.u(uint32(B))), x.op(OpMul, U, x.op(OpAdd, U, b, x.u(1)), x.u(uint32(B)))
		return x.sel(U, x.op(OpLess, ir.Pred, lo, bd.end), x.sel(U, x.op(OpLess, ir.Pred, bd.first, hi), x.u(1), x.u(0)), x.u(0))
	}
	// kept is whether block b is kept: a held block forced in, or ranked
	// within what the forced leave of TopK -- every held block where they
	// number at most TopK, a block's rank among the rest below their count.
	kept := func(x *exprs, bd bounds, b, rk int) int {
		forced := x.op(OpSub, U, x.op(OpAdd, U, bd.own, x.u(1)), bd.lo)
		ranked := x.sel(U, x.op(OpLess, ir.Pred, x.op(OpAdd, U, rk, forced), x.u(uint32(a.TopK))), x.u(1), x.u(0))
		return x.op(OpAnd, U, x.op(OpOr, U, isForced(x, bd, b), ranked), held(x, bd, b))
	}

	var dx exprs
	h, d := dx.coord(1), dx.coord(3)
	pos, first := hs.position(&dx)
	pid, off := hs.page(&dx, hs.held(&dx, pos, first))
	qat := dx.op(OpAdd, U, dx.op(OpMul, U, dx.op(OpAdd, U, dx.op(OpMul, U, dx.coord(0), dx.u(uint32(H))), h), dx.u(uint32(D))), d)
	dx.op(OpMul, F, dx.load(0, qat), c.poolRead(&dx, 3, pool, a.Width, pid, off, dx.op(OpAdd, U, d, dx.u(uint32(a.Offset)))))

	var gx exprs
	gh := gx.op(OpAdd, U, gx.op(OpMul, U, gx.coord(1), gx.u(uint32(J))), gx.coord(3))
	s := gx.at(0, []int{R, H, C}, gx.coord(0), gh, gx.coord(2))
	if fn != 0 {
		s = gx.op(fn, F, s, gx.f(float32(a.Const)))
	}
	sumIn := []Operand{dots}
	if len(o.In) == 2 {
		s = gx.op(OpMul, F, s, gx.at(1, []int{R, H}, gx.coord(0), gh))
		sumIn = append(sumIn, c.value(o.In[1]))
	}

	var bx exprs
	bbd := boundsOf(&bx)
	t := bx.op(OpAdd, U, bx.op(OpMul, U, bx.coord(2), bx.u(uint32(B))), bx.coord(3))
	in := bx.sel(U, bx.op(OpLess, ir.Pred, t, bbd.first), bx.u(0), bx.sel(U, bx.op(OpLess, ir.Pred, t, bbd.end), bx.u(1), bx.u(0)))
	rel := bx.op(OpMin, U, bx.op(OpSub, U, bx.op(OpMax, U, t, bbd.first), bbd.first), bx.u(uint32(C-1)))
	v := bx.at(0, []int{R, G, C}, bx.coord(0), bx.coord(1), rel)
	bx.sel(F, bx.op(OpLess, ir.Pred, bx.u(0), in), v, bx.f(minus))

	var rx exprs
	rbd := boundsOf(&rx)
	b, b2 := rx.coord(2), rx.coord(3)
	vb, vb2 := rx.at(0, []int{R, G, NB}, rx.coord(0), rx.coord(1), b), rx.at(0, []int{R, G, NB}, rx.coord(0), rx.coord(1), b2)
	ahead := rx.sel(U, rx.op(OpLess, ir.Pred, vb, vb2), rx.u(1),
		rx.sel(U, rx.op(OpLess, ir.Pred, vb2, vb), rx.u(0), rx.sel(U, rx.op(OpLess, ir.Pred, b2, b), rx.u(1), rx.u(0))))
	counts := rx.op(OpAnd, U, ahead, rx.op(OpAnd, U, held(&rx, rbd, b2), rx.op(OpXor, U, isForced(&rx, rbd, b2), rx.u(1))))
	rx.op(OpAdd, U, counts, rx.u(0))

	var kx exprs
	kbd := boundsOf(&kx)
	kept(&kx, kbd, kx.coord(2), kx.at(0, []int{R, G, NB}, kx.coord(0), kx.coord(1), kx.coord(2)))

	var mx exprs
	mpos := mx.op(OpAdd, U, hs.row(&mx, rowStart), mx.coord(2))
	mb := mx.op(OpMin, U, mx.op(OpDiv, U, mpos, mx.u(uint32(B))), mx.u(uint32(NB-1)))
	live := mx.sel(U, mx.op(OpLess, ir.Pred, mpos, hs.row(&mx, rowEnd)), mx.u(1), mx.u(0))
	mx.un(OpCvtF32, F, mx.op(OpAnd, U, live, mx.at(0, []int{R, G, NB}, mx.coord(0), mx.coord(1), mb)))

	var cx exprs
	cx.at(0, []int{R, G, C}, cx.coord(0), cx.coord(1), cx.coord(2))

	flat := Facts{Extents: ext()}
	return []Prim{
		{Kind: PrimContract, Shape: []int{R, H, C, D}, In: tables(c.value(o.In[0]), keys), Out: dots, Expr: dx.e, Facts: rng},
		{Kind: PrimReduce, Shape: []int{R, G, C, J}, In: sumIn, Out: scores, Combine: OpAdd, Expr: gx.e},
		{Kind: PrimReduce, Shape: []int{R, G, NB, B}, In: tables(scores), Out: best, Combine: OpMax,
			Init: int64(math.Float32bits(minus)), Expr: bx.e, Facts: flat},
		{Kind: PrimReduce, Shape: []int{R, G, NB, NB}, In: tables(best), Out: rank, Combine: OpAdd, Type: U, Expr: rx.e, Facts: flat},
		{Kind: PrimMap, Shape: []int{R, G, NB}, In: tables(rank), Out: keep, Expr: kx.e, Facts: flat},
		{Kind: PrimMap, Shape: []int{R, G, C}, In: tables(keep), Out: mask, Expr: mx.e, Facts: flat},
		{Kind: PrimReduce, Shape: []int{R, G, C}, In: []Operand{mask}, Out: out, Combine: OpAdd, Expr: cx.e},
	}, nil
}
