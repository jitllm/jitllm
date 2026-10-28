package jit

import (
	"fmt"
	"math/bits"

	"github.com/jitllm/jitllm/jit/ir"
)

// selectMin is the group past which a top-k is selected by a threshold
// rather than by ranking the whole group: ranking compares every pair of
// the group, the threshold reads it a few times and ranks only the k kept.
const selectMin = 1024

// selectBlock is the block the selection's prefix counts run over, for a
// group of S: a power of two near the cube root of 2S, where the counts
// before an element (S times the block) and those before a block (the
// blocks squared) cost about the same.
func selectBlock(S int) int {
	b := 16
	for b*b*b < 2*S {
		b *= 2
	}
	return b
}

// orderKey is x's float order as an unsigned order: the sign bit flipped on
// a positive float, every bit on a negative one, -0 taken as +0 so that the
// two compare equal as floats do.
func (x *exprs) orderKey(v int) int {
	U := ir.U32
	b := x.un(OpBitcast, U, v)
	b = x.sel(U, x.op(OpLess, ir.Pred, x.op(OpXor, U, b, x.u(0x80000000)), x.u(1)), x.u(0), b)
	neg := x.op(OpLess, ir.Pred, x.u(0x7fffffff), b)
	return x.sel(U, neg, x.op(OpXor, U, b, x.u(0xffffffff)), x.op(OpOr, U, b, x.u(0x80000000)))
}

// countsAt is the packed count of key k against threshold t: one where k is
// above t, 1<<16 where it equals t.
func (x *exprs) countsAt(k, t int) int {
	U := ir.U32
	above := x.op(OpLess, ir.Pred, t, k)
	below := x.op(OpLess, ir.Pred, k, t)
	return x.sel(U, above, x.u(1), x.sel(U, below, x.u(0), x.u(1<<16)))
}

// decomposeTopKSelect is the K largest of each row of S, largest first and
// equal values to the lower index, for a row too long to rank whole: the
// K-th largest key found four bits a round (each round counts the keys at
// or above fifteen candidates and keeps the largest that K reach), the keys
// above it and the first of those equal to it -- K in all -- written in
// index order by their prefix counts (counted a block at a time), then the
// K ranked as decomposeTopKRanked ranks a group. Its work is a few passes
// over the row and K squared compares, whatever K is.
func decomposeTopKSelect(c *Context, in, sel, val Operand, S, K int) []Prim {
	U, I, F := ir.U32, ir.I32, ir.F32
	R := c.Step.Rows
	B := selectBlock(S)
	sh := bits.TrailingZeros(uint(B))
	nb := (S + B - 1) / B
	name := val.Name
	// Not Temp: a count is read past the kernel after its producer (the
	// prefix counts by the scatter, the kept by the ranking), which a chain
	// of the two that drops a Temp value's store would lose.
	scratch := func(s string, t ir.Type) Operand {
		return Operand{Kind: OperandScratch, Name: name + "." + s, Elem: t}
	}
	keyAt := func(x *exprs, r, i int) int { return x.orderKey(x.at(0, []int{R, S}, r, i)) }
	keyLoad := func(x *exprs, r, j int) int {
		return x.orderKey(x.load(0, x.op(OpAdd, U, x.op(OpMul, U, r, x.u(uint32(S))), j)))
	}
	ext := map[int]int{0: R * S}
	// A round's counts are sums of ones in floats, exact under 2^24, so a
	// device sums them as it sums floats, then made words.
	words := func(from, to Operand, n int) Prim {
		var x exprs
		x.un(OpBitcast, U, x.un(OpCvtI32, I, x.at(0, []int{R, n}, x.coord(0), x.coord(1))))
		return Prim{Kind: PrimMap, Shape: []int{R, n}, In: []Operand{from}, Out: to, Expr: x.e}
	}

	var prims []Prim
	var t Operand
	for round := range 8 {
		s := uint32(28 - 4*round)
		cnt, cw := scratch(fmt.Sprintf("cnt%d", round), F), scratch(fmt.Sprintf("cnt%d.w", round), U)
		var cx exprs
		r, d, i := cx.coord(0), cx.coord(1), cx.coord(2)
		base := cx.u(0)
		if round > 0 {
			base = cx.at(1, []int{R}, r)
		}
		thr := cx.op(OpAdd, U, base, cx.op(OpShl, U, cx.op(OpAdd, U, d, cx.u(1)), cx.u(s)))
		cx.sel(F, cx.op(OpLess, ir.Pred, keyAt(&cx, r, i), thr), cx.f(0), cx.f(1))
		cin := []Operand{in}
		if round > 0 {
			cin = append(cin, t)
		}
		prims = append(prims, Prim{Kind: PrimReduce, Shape: []int{R, 15, S}, In: cin, Out: cnt, Combine: OpAdd, Type: F, Expr: cx.e},
			words(cnt, cw, 15))

		next := scratch(fmt.Sprintf("t%d", round+1), U)
		var ux exprs
		r, j := ux.coord(0), ux.coord(1)
		prev := ux.u(0)
		if round > 0 {
			prev = ux.at(1, []int{R}, r)
		}
		jm := ux.op(OpSub, U, ux.op(OpMax, U, j, ux.u(1)), ux.u(1))
		reach := ux.sel(U, ux.op(OpLess, ir.Pred, ux.at(0, []int{R, 15}, r, jm), ux.u(uint32(K))), ux.u(0), ux.u(1<<s))
		ux.sel(U, ux.op(OpLess, ir.Pred, j, ux.u(1)), prev, reach)
		uin := []Operand{cw}
		if round > 0 {
			uin = append(uin, t)
		}
		prims = append(prims, Prim{Kind: PrimReduce, Shape: []int{R, 16}, In: uin, Out: next, Combine: OpAdd, Type: U, Expr: ux.e})
		t = next
	}

	// The keys above the threshold.
	above := scratch("above", U)
	var ax exprs
	r, i := ax.coord(0), ax.coord(1)
	ax.sel(U, ax.op(OpLess, ir.Pred, ax.at(1, []int{R}, r), keyAt(&ax, r, i)), ax.u(1), ax.u(0))
	prims = append(prims, Prim{Kind: PrimReduce, Shape: []int{R, S}, In: []Operand{in, t}, Out: above, Combine: OpAdd, Type: U, Expr: ax.e})

	// Each block's counts above and at the threshold, packed.
	blk := scratch("blocks", U)
	var bx exprs
	r, b, u := bx.coord(0), bx.coord(1), bx.coord(2)
	j := bx.op(OpAdd, U, bx.op(OpShl, U, b, bx.u(uint32(sh))), u)
	jc := bx.op(OpMin, U, j, bx.u(uint32(S-1)))
	bx.sel(U, bx.op(OpLess, ir.Pred, j, bx.u(uint32(S))), bx.countsAt(keyLoad(&bx, r, jc), bx.at(1, []int{R}, r)), bx.u(0))
	prims = append(prims, Prim{Kind: PrimReduce, Shape: []int{R, nb, B}, In: []Operand{in, t}, Out: blk, Combine: OpAdd, Type: U,
		Expr: bx.e, Facts: Facts{Extents: ext}})

	// The counts before each block, above and at the threshold.
	pre := [2]Operand{scratch("above.before", U), scratch("at.before", U)}
	for h, p := range pre {
		var px exprs
		r, b, v := px.coord(0), px.coord(1), px.coord(2)
		w := px.at(0, []int{R, nb}, r, v)
		if h == 0 {
			w = px.op(OpAnd, U, w, px.u(0xffff))
		} else {
			w = px.op(OpShr, U, w, px.u(16))
		}
		px.sel(U, px.op(OpLess, ir.Pred, v, b), w, px.u(0))
		prims = append(prims, Prim{Kind: PrimReduce, Shape: []int{R, nb, nb}, In: []Operand{blk}, Out: p, Combine: OpAdd, Type: U, Expr: px.e})
	}

	// The counts before each element within its block, packed: above the
	// threshold plus 2^16 times at it, each below B, a sum of floats made
	// words. A row whose blocks are whole reads no element past it.
	wf, within := scratch("within.f", F), scratch("within", U)
	var wx exprs
	r, i, u = wx.coord(0), wx.coord(1), wx.coord(2)
	j = wx.op(OpAdd, U, wx.op(OpShl, U, wx.op(OpShr, U, i, wx.u(uint32(sh))), wx.u(uint32(sh))), u)
	jc = j
	if S%B != 0 {
		jc = wx.op(OpMin, U, j, wx.u(uint32(S-1)))
	}
	k := keyLoad(&wx, r, jc)
	tt := wx.at(1, []int{R}, r)
	w := wx.sel(F, wx.op(OpLess, ir.Pred, tt, k), wx.f(1), wx.sel(F, wx.op(OpLess, ir.Pred, k, tt), wx.f(0), wx.f(1<<16)))
	wx.sel(F, wx.op(OpLess, ir.Pred, j, i), w, wx.f(0))
	prims = append(prims, Prim{Kind: PrimReduce, Shape: []int{R, S, B}, In: []Operand{in, t}, Out: wf, Combine: OpAdd, Type: F,
		Expr: wx.e, Facts: Facts{Extents: ext}}, words(wf, within, S))

	// Every kept element written at its place among its row's kept, in
	// index order; the others at the place after them, a row K+1 wide.
	ids, vals := scratch("kept", U), scratch("kept.v", F)
	var sx exprs
	r, i = sx.coord(0), sx.coord(1)
	v := sx.at(0, []int{R, S}, r, i)
	k = sx.orderKey(v)
	tt = sx.at(1, []int{R}, r)
	bi := sx.op(OpShr, U, i, sx.u(uint32(sh)))
	w = sx.at(5, []int{R, S}, r, i)
	ab := sx.op(OpAdd, U, sx.at(3, []int{R, nb}, r, bi), sx.op(OpAnd, U, w, sx.u(0xffff)))
	eb := sx.op(OpAdd, U, sx.at(4, []int{R, nb}, r, bi), sx.op(OpShr, U, w, sx.u(16)))
	need := sx.op(OpSub, U, sx.u(uint32(K)), sx.at(2, []int{R}, r))
	isAbove := sx.op(OpLess, ir.Pred, tt, k)
	isBelow := sx.op(OpLess, ir.Pred, k, tt)
	kept := sx.sel(U, isAbove, sx.u(1), sx.sel(U, isBelow, sx.u(0), sx.sel(U, sx.op(OpLess, ir.Pred, eb, need), sx.u(1), sx.u(0))))
	row := sx.op(OpMul, U, r, sx.u(uint32(K+1)))
	place := sx.op(OpAdd, U, row, sx.op(OpAdd, U, ab, sx.op(OpMin, U, eb, need)))
	dest := sx.sel(U, sx.op(OpLess, ir.Pred, sx.u(0), kept), place, sx.op(OpAdd, U, row, sx.u(uint32(K))))
	prims = append(prims, Prim{Kind: PrimMap, Shape: []int{R, S}, In: []Operand{in, t, above, pre[0], pre[1], within}, Out: ids, Expr: sx.e,
		Stores: []PrimStore{
			{Out: ids, Index: dest, Value: i, Extent: R * (K + 1)},
			{Out: vals, Index: dest, Value: v, Extent: R * (K + 1)},
		}})

	return append(prims, rankedPlaces(c, vals, &ids, sel, val, 1, K, K, K+1)...)
}
