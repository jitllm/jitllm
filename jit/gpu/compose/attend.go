package compose

import "math"

// Paged is a paged attention's shape: query and KV heads, the head width,
// a page's positions, the splits a row's keys run in and the slots a split
// holds (docs/design/device-kv-paging.md).
type Paged struct {
	Rows, Heads, KVHeads, Dim, Page, Splits, Chunk int
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

// PagedSoftmax is the group element's split of a paged attention's scores
// (an element an item: a row's head's split, Rows*Heads*Splits of them):
// over the split's keys their maximum m, then over every slot of its chunk
// the weight exp(score - m), 0 where the slot holds no key, stored
// unnormalised at out, and their sum l; m and l to the merge's partials at
// part, 32 copies of each after the value's padded width. The buffers:
// the scores (BufRow, read by index), the outputs (BufOut), the
// descriptors (BufU32).
func (g *Row) PagedSoftmax(scores, out, part, desc int, p Paged) {
	u, ub := g.UConst, g.UBin
	item := g.Column()
	sp := ub(OpRem, item, u(p.Splits))
	row := ub(OpDiv, item, u(p.Heads*p.Splits))
	ps := g.pagedSplitOf(desc, row, sp, p.Splits)
	keys := ub(OpSub, ps.en, ps.st)
	base := ub(OpMul, item, u(p.Chunk))
	lane, step := u(0), u(1)
	cnt := ub(OpDiv, ub(OpAdd, ub(OpSub, ub(OpMax, keys, lane), lane), ub(OpSub, step, u(1))), step)
	c1 := g.LoopN(cnt)
	mi := ub(OpAdd, lane, ub(OpMul, g.Iter(c1), step))
	m := g.Fold(c1, g.Const(-math.MaxFloat32))
	g.SetFold(m, g.Bin(OpMax, m, g.LoadAt(scores, ub(OpAdd, base, mi))))
	cntC := ub(OpDiv, ub(OpAdd, ub(OpSub, u(p.Chunk), lane), ub(OpSub, step, u(1))), step)
	c2 := g.LoopN(cntC)
	si := ub(OpAdd, lane, ub(OpMul, g.Iter(c2), step))
	at := ub(OpAdd, base, si)
	w := g.Select(g.valid(ps, si), g.Add(Node{Kind: NUnary, Op: OpExp, A: g.Bin(OpSub, g.LoadAt(scores, at), m)}), g.Const(0))
	g.StoreAt(out, w, at)
	l := g.Fold(c2, g.Const(0))
	g.SetFold(l, g.Bin(OpAdd, l, w))
	padded := (p.Dim + 31) / 32 * 32
	pb := ub(OpMul, item, u(padded+64))
	for r := 0; r < 32; r++ {
		g.StoreAt(part, m, ub(OpAdd, pb, u(padded+r)))
		g.StoreAt(part, l, ub(OpAdd, pb, u(padded+32+r)))
	}
}
