package compose

import "math"

// Paged is a paged attention's shape: query and KV heads, the head width,
// a page's positions, the splits a row's keys run in and the slots a split
// holds (docs/design/device-kv-paging.md).
type Paged struct {
	Rows, Heads, KVHeads, Dim, Page, Splits, Chunk int
	// MLA > 0 is a latent row: one row-major region of Dim a position (one
	// KV head), its first MLA elements the value.
	MLA int
	// Lanes > 1 takes an item over a subgroup's lanes: a group row an item,
	// its element the lane.
	Lanes int
}

// valueDim is the value's width: the latent's prefix, or the head.
func (p Paged) valueDim() int {
	if p.MLA > 0 {
		return p.MLA
	}
	return p.Dim
}

// Descriptor words of a row (the paged ABI): where its table starts, its
// first key, one past its last, where its new position is written.
const (
	DescTab = iota
	DescStart
	DescEnd
	DescWrite
	DescWords
)

// pagedSplit is split sp's keys [st, en) of a row with keys [ks, ke): the
// row's keys from the 32-aligned position at or below ks, divided into
// whole 32-key runs, each at most a chunk; last is the last key the row may
// address and live whether it has any.
type pagedSplit struct{ tab, ks, st, en, last, live int }

func (g *Row) pagedSplitOf(desc, row, sp, splits int) pagedSplit {
	u, ub := g.UConst, g.UBin
	w := func(i int) int { return g.LoadAt(desc, ub(OpAdd, ub(OpMul, row, u(DescWords)), u(i))) }
	tab, ks, ke := w(DescTab), w(DescStart), w(DescEnd)
	a := ub(OpMul, ub(OpDiv, ks, u(32)), u(32))
	live := g.Less(ks, ke)
	span := g.Select(live, ub(OpSub, ke, a), u(0))
	per := ub(OpDiv, ub(OpAdd, span, u(splits-1)), u(splits))
	per = ub(OpMul, ub(OpDiv, ub(OpAdd, per, u(31)), u(32)), u(32))
	st := ub(OpMin, ke, ub(OpAdd, a, ub(OpMul, sp, per)))
	en := g.Select(live, ub(OpMin, ke, ub(OpAdd, st, per)), st)
	last := ub(OpSub, ub(OpMax, ke, u(1)), u(1))
	return pagedSplit{tab: tab, ks: ks, st: st, en: en, last: last, live: live}
}

// valid is whether slot holds one of the row's keys: st+slot in [ks, en).
func (g *Row) valid(p pagedSplit, slot int) int {
	ub := g.UBin
	return g.Less(ub(OpSub, ub(OpAdd, p.st, slot), p.ks), ub(OpSub, p.en, p.ks))
}

// HeadGroup is how many query heads of one KV head an item serves, so one K
// or V load feeds that many products: one per 64 slots of a chunk up to 8,
// rounded down to a divisor of the KV head's group.
func (p Paged) HeadGroup() int {
	gqa := p.Heads / p.KVHeads
	g := min(8, gqa, max(1, p.Chunk/64))
	for gqa%g != 0 {
		g--
	}
	return g
}

// pageOf is the page holding position t of the row whose table starts at
// tab: word tab + t/page of the table.
func (g *Row) pageOf(table, tab, t, page int) int {
	return g.LoadAt(table, g.UBin(OpAdd, tab, g.pageDiv(t, page)))
}

// pageDiv and pageRem are t/page and t%page, a shift and a mask where the
// page is a power of two.
func (g *Row) pageDiv(t, page int) int {
	if page&(page-1) == 0 {
		return g.UBin(OpShr, t, g.UConst(log2(page)))
	}
	return g.UBin(OpDiv, t, g.UConst(page))
}

func (g *Row) pageRem(t, page int) int {
	if page&(page-1) == 0 {
		return g.UBin(OpAnd, t, g.UConst(page-1))
	}
	return g.UBin(OpRem, t, g.UConst(page))
}

func log2(n int) int {
	k := 0
	for 1<<k < n {
		k++
	}
	return k
}

// PagedMerge is the group element's output (an element a (row, head,
// value element)) from the splits' partials at part: each split's maximum
// and denominator after its numerators, the splits weighted by exp of their
// maximum less the greatest, the sum of the weighted denominators (and the
// sink's exp, where sink >= 0: one logit a head) the normaliser.
func (g *Row) PagedMerge(part, out, sink int, p Paged) {
	u, ub := g.UConst, g.UBin
	flat := g.Column()
	vd := p.valueDim()
	item := ub(OpDiv, flat, u(vd))
	dim := ub(OpRem, flat, u(vd))
	padded := (vd + 31) / 32 * 32
	stride := padded + 64
	base := ub(OpMul, item, u(p.Splits*stride))
	mx := g.Const(-math.MaxFloat32)
	ms, ns := make([]int, p.Splits), make([]int, p.Splits)
	for j := range ms {
		ms[j] = g.LoadAt(part, ub(OpAdd, base, u(j*stride+padded)))
		ns[j] = g.LoadAt(part, ub(OpAdd, base, u(j*stride+padded+32)))
		mx = g.Bin(OpMax, mx, ms[j])
	}
	var sv int
	if sink >= 0 {
		sv = g.LoadAt(sink, ub(OpRem, item, u(p.Heads)))
		mx = g.Bin(OpMax, mx, sv)
	}
	sum, acc := g.Const(0), g.Const(0)
	src := ub(OpAdd, base, dim)
	exp := func(x int) int { return g.Add(Node{Kind: NUnary, Op: OpExp, A: x}) }
	fma := func(a, b, c int) int { return g.Add(Node{Kind: NBinary, Op: OpFma, A: a, B: b, C: c}) }
	for j := range ms {
		w := exp(g.Bin(OpSub, ms[j], mx))
		sum = fma(ns[j], w, sum)
		acc = fma(g.LoadAt(part, ub(OpAdd, src, u(j*stride))), w, acc)
	}
	if sink >= 0 {
		sum = g.Bin(OpAdd, sum, exp(g.Bin(OpSub, sv, mx)))
	}
	inv := g.Bin(OpDiv, g.Const(1), g.Bin(OpMax, sum, g.Const(1e-30)))
	g.StoreAt(out, g.Bin(OpMul, acc, inv), flat)
}
