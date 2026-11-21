package composer

import (
	"fmt"
	"math/bits"

	"github.com/jitllm/jitllm/composer/ir"
)

// sortPays says whether ordering k elements by a radix sort (sortedPlaces)
// costs less than ranking them pairwise (rankedPlaces): k squared compares
// against eight passes of k times a block, the blocks squared by sixteen
// digits, and k by sixteen.
func sortPays(k int) bool {
	b := selectBlock(k)
	nb := (k + b - 1) / b
	return k*k > 8*(k*b+16*nb*nb+16*k)
}

// sortedPlaces is the K elements of each row of vals -- their indices ids,
// or their places where ids is nil, a row of both w wide -- in descending
// order of value, equal values in the order they stand: sel their indices,
// val their values. A stable radix sort of the values' order keys, four
// bits a pass from the lowest, a larger digit first: each pass puts an
// element after the elements of the larger digits, those of its digit in
// the blocks before its own, and those of its digit before it in its block.
// Its work is eight passes of K times a block and the blocks squared,
// whatever K is.
func sortedPlaces(c *Context, vals Operand, ids *Operand, sel, val Operand, K, w int) []Prim {
	U, F := ir.U32, ir.F32
	R := c.Step.Rows
	B := selectBlock(K)
	sh := uint32(bits.TrailingZeros(uint(B)))
	nb := (K + B - 1) / B
	// Not Temp: each pass's places are read past the kernels that write
	// them. A pass's counts are its own and the next pass's overwrite
	// them; its keys, indices and values are read by the next pass alone,
	// so two of each serve all eight.
	scratch := func(s string, t ir.Type) Operand {
		return Operand{Kind: OperandScratch, Name: val.Name + ".sort." + s, Elem: t}
	}
	digit := func(x *exprs, k int, s uint32) int {
		return x.op(OpSub, U, x.u(15), x.op(OpAnd, U, x.op(OpShr, U, k, x.u(s)), x.u(15)))
	}
	same := func(x *exprs, t ir.Type, a, b, one, zero int) int {
		return x.sel(t, x.op(OpLess, ir.Pred, a, b), zero, x.sel(t, x.op(OpLess, ir.Pred, b, a), zero, one))
	}
	ext := map[int]int{0: R * K}

	key := scratch("key0", U)
	var kx exprs
	kx.orderKey(kx.at(0, []int{R, w}, kx.coord(0), kx.coord(1)))
	prims := []Prim{{Kind: PrimMap, Shape: []int{R, K}, In: []Operand{vals}, Out: key, Expr: kx.e}}

	curV, curID, cw := vals, ids, w
	for p := range 8 {
		s := uint32(4 * p)
		blk, tot := scratch("blk", U), scratch("tot", U)
		start, before := scratch("start", U), scratch("before", U)
		wf, within := scratch("within.f", F), scratch("within", U)

		// Each block's count of each digit.
		var bx exprs
		r, b, d, u := bx.coord(0), bx.coord(1), bx.coord(2), bx.coord(3)
		j := bx.op(OpAdd, U, bx.op(OpShl, U, b, bx.u(sh)), u)
		jc := bx.op(OpMin, U, j, bx.u(uint32(K-1)))
		dg := digit(&bx, bx.load(0, bx.op(OpAdd, U, bx.op(OpMul, U, r, bx.u(uint32(K))), jc)), s)
		hit := same(&bx, U, dg, d, bx.u(1), bx.u(0))
		bx.sel(U, bx.op(OpLess, ir.Pred, j, bx.u(uint32(K))), hit, bx.u(0))
		prims = append(prims, Prim{Kind: PrimReduce, Shape: []int{R, nb, 16, B}, In: []Operand{key}, Out: blk, Combine: OpAdd, Type: U,
			Expr: bx.e, Facts: Facts{Extents: ext}})

		// Each digit's count in the row, and the counts of the larger digits.
		var tx exprs
		tx.at(0, []int{R, nb, 16}, tx.coord(0), tx.coord(2), tx.coord(1))
		prims = append(prims, Prim{Kind: PrimReduce, Shape: []int{R, 16, nb}, In: []Operand{blk}, Out: tot, Combine: OpAdd, Type: U, Expr: tx.e})
		var sx exprs
		sx.sel(U, sx.op(OpLess, ir.Pred, sx.coord(2), sx.coord(1)), sx.at(0, []int{R, 16}, sx.coord(0), sx.coord(2)), sx.u(0))
		prims = append(prims, Prim{Kind: PrimReduce, Shape: []int{R, 16, 16}, In: []Operand{tot}, Out: start, Combine: OpAdd, Type: U, Expr: sx.e})

		// Each digit's count in the blocks before each block.
		var px exprs
		pr, pb, pd, pb2 := px.coord(0), px.coord(1), px.coord(2), px.coord(3)
		px.sel(U, px.op(OpLess, ir.Pred, pb2, pb), px.at(0, []int{R, nb, 16}, pr, pb2, pd), px.u(0))
		prims = append(prims, Prim{Kind: PrimReduce, Shape: []int{R, nb, 16, nb}, In: []Operand{blk}, Out: before, Combine: OpAdd, Type: U, Expr: px.e})

		// Each element's count of its digit before it in its block, a sum of
		// ones in floats, exact under 2^24, made words.
		var wx exprs
		r, i, u := wx.coord(0), wx.coord(1), wx.coord(2)
		j = wx.op(OpAdd, U, wx.op(OpShl, U, wx.op(OpShr, U, i, wx.u(sh)), wx.u(sh)), u)
		jc = j
		if K%B != 0 {
			jc = wx.op(OpMin, U, j, wx.u(uint32(K-1)))
		}
		dj := digit(&wx, wx.load(0, wx.op(OpAdd, U, wx.op(OpMul, U, r, wx.u(uint32(K))), jc)), s)
		di := digit(&wx, wx.at(0, []int{R, K}, r, i), s)
		hitF := same(&wx, F, dj, di, wx.f(1), wx.f(0))
		wx.sel(F, wx.op(OpLess, ir.Pred, j, i), hitF, wx.f(0))
		var mx exprs
		mx.un(OpBitcast, U, mx.un(OpCvtI32, ir.I32, mx.at(0, []int{R, K}, mx.coord(0), mx.coord(1))))
		prims = append(prims, Prim{Kind: PrimReduce, Shape: []int{R, K, B}, In: []Operand{key}, Out: wf, Combine: OpAdd, Type: F,
			Expr: wx.e, Facts: Facts{Extents: ext}},
			Prim{Kind: PrimMap, Shape: []int{R, K}, In: []Operand{wf}, Out: within, Expr: mx.e})

		// Each element at its place: after the larger digits', its digit's
		// in the blocks before, and its digit's before it in its block.
		var ox exprs
		r, i = ox.coord(0), ox.coord(1)
		k := ox.at(0, []int{R, K}, r, i)
		dk := digit(&ox, k, s)
		place := ox.op(OpAdd, U, ox.at(1, []int{R, 16}, r, dk),
			ox.op(OpAdd, U, ox.at(2, []int{R, nb, 16}, r, ox.op(OpShr, U, i, ox.u(sh)), dk), ox.at(3, []int{R, K}, r, i)))
		dest := ox.op(OpAdd, U, ox.op(OpMul, U, r, ox.u(uint32(K))), place)
		v := ox.at(4, []int{R, cw}, r, i)
		in := []Operand{key, start, before, within, curV}
		id := i
		if curID != nil {
			in = append(in, *curID)
			id = ox.at(5, []int{R, cw}, r, i)
		}
		nk, nid, nv := scratch(fmt.Sprintf("key%d", (p+1)%2), U), scratch(fmt.Sprintf("id%d", (p+1)%2), U), scratch(fmt.Sprintf("val%d", (p+1)%2), F)
		stores := []PrimStore{{Out: nk, Index: dest, Value: k, Extent: R * K}, {Out: nid, Index: dest, Value: id, Extent: R * K},
			{Out: nv, Index: dest, Value: v, Extent: R * K}}
		out := nid
		if p == 7 {
			stores, out = []PrimStore{{Out: sel, Index: dest, Value: id, Extent: R * K}, {Out: val, Index: dest, Value: v, Extent: R * K}}, sel
		}
		prims = append(prims, Prim{Kind: PrimMap, Shape: []int{R, K}, In: in, Out: out, Expr: ox.e, Stores: stores})
		key, curV, curID, cw = nk, nv, &nid, K
	}
	return prims
}
