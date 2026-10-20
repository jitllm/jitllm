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

// PagedScores is the scaled q.k of the group element's slot (an element a
// (row, head group, split, slot), slot fastest) for each query head of its
// group, stored into the score plane at out, [row][head][split][chunk], and
// -MaxFloat32 where the slot holds no key of the row; softcap > 0 caps a
// score at c*tanh(score/c). K is the paged pool
// [page][kvRow][Page] (an element of position t of page pid at
// (pid*kvRow + e)*Page + t%Page); the key read is the slot's clamped into
// the row's, so a masked slot reads the row's own pages. The dot runs in
// chains of the head's dimensions, over blocks of 32 where the group's
// products pass 512.
func (g *Row) PagedScores(q, k, out, table, desc int, p Paged, scale, softcap float32) {
	u, ub := g.UConst, g.UBin
	G := p.HeadGroup()
	gqa, S, C, P := p.Heads/p.KVHeads, p.Splits, p.Chunk, p.Page
	kvRow := p.KVHeads * p.Dim
	flat := g.Column()
	slot := ub(OpRem, flat, u(C))
	rest := ub(OpDiv, flat, u(C))
	sp := ub(OpRem, rest, u(S))
	rest = ub(OpDiv, rest, u(S))
	hg := ub(OpRem, rest, u(p.Heads/G))
	row := ub(OpDiv, rest, u(p.Heads/G))
	ps := g.pagedSplitOf(desc, row, sp, S)
	t := g.Select(ps.live, ub(OpMin, ub(OpMax, ub(OpAdd, ps.st, slot), ps.ks), ps.last), u(0))
	pid := g.pageOf(table, ps.tab, t, P)
	h0 := ub(OpMul, hg, u(G))
	kvh := ub(OpMul, ub(OpDiv, h0, u(gqa)), u(p.Dim))
	rh0 := ub(OpAdd, ub(OpMul, row, u(p.Heads)), h0)
	qb := ub(OpMul, rh0, u(p.Dim))
	kb := ub(OpAdd, ub(OpAdd, ub(OpMul, pid, u(P*kvRow)), ub(OpMul, kvh, u(P))), g.pageRem(t, P))
	chains := max(1, 4/G)
	acc := make([][]int, G)
	for h := range acc {
		acc[h] = make([]int, chains)
		for c := range acc[h] {
			acc[h][c] = g.Const(0)
		}
	}
	// dims folds dimensions d0 + 0..n-1 into chains from cur, d0 + the block
	// base at blk (an index, or -1 for none).
	dims := func(cur [][]int, blk, n int) [][]int {
		nx := make([][]int, G)
		for h := range nx {
			nx[h] = append([]int(nil), cur[h]...)
		}
		for d := 0; d < n; d++ {
			dd := u(d)
			if blk >= 0 {
				dd = ub(OpAdd, blk, dd)
			}
			kv := g.LoadAt(k, ub(OpAdd, kb, ub(OpMul, dd, u(P))))
			for h := range nx {
				qv := g.LoadAt(q, ub(OpAdd, qb, ub(OpAdd, u(h*p.Dim), dd)))
				nx[h][d%chains] = g.Add(Node{Kind: NBinary, Op: OpFma, A: qv, B: kv, C: nx[h][d%chains]})
			}
		}
		return nx
	}
	const blk = 32
	done := 0
	if nb := p.Dim / blk; G*p.Dim > 512 && nb > 1 {
		done = nb * blk
		c := g.Loop(nb)
		folds := make([][]int, G)
		for h := range folds {
			folds[h] = make([]int, chains)
			for x := range folds[h] {
				folds[h][x] = g.Fold(c, acc[h][x])
			}
		}
		nx := dims(folds, ub(OpMul, g.Iter(c), u(blk)), blk)
		for h := range folds {
			for x := range folds[h] {
				g.SetFold(folds[h][x], nx[h][x])
			}
		}
		acc = folds
	}
	if done < p.Dim {
		rest := dims(acc, u(done), p.Dim-done)
		for h := range acc {
			// Dimensions past the loop continue its chains: d%chains of
			// the whole head.
			acc[h] = rest[h]
		}
	}
	valid := g.valid(ps, slot)
	ob := ub(OpAdd, ub(OpMul, ub(OpAdd, ub(OpMul, rh0, u(S)), sp), u(C)), slot)
	for h := range acc {
		dot := acc[h][0]
		for x := 1; x < chains; x++ {
			dot = g.Bin(OpAdd, dot, acc[h][x])
		}
		score := g.Bin(OpMul, dot, g.Const(scale))
		if softcap > 0 {
			// c*tanh(score/c), as c*(1 - 2/(exp(2*score/c) + 1)).
			e := g.Add(Node{Kind: NUnary, Op: OpExp, A: g.Bin(OpMul, g.Const(2/softcap), score)})
			score = g.Bin(OpMul, g.Const(softcap), g.Bin(OpSub, g.Const(1), g.Bin(OpDiv, g.Const(2), g.Bin(OpAdd, e, g.Const(1)))))
		}
		g.StoreAt(out, g.Select(valid, score, g.Const(-math.MaxFloat32)), ub(OpAdd, ob, u(h*S*C)))
	}
}

// PagedAcc is the group element's weighted sum of V over its split's keys
// (an element a (row, head group, split, value element), the element
// fastest) for each query head of its group, into the merge's partials at
// part. It walks the split's keys in blocks of 8, which start at a multiple
// of 8 from a 32-aligned partition and so lie in one page: a block's page
// is read once. A key outside the row's live range reads V within the
// block's page and contributes 0 (its weight is 0 too, and 0*NaN is NaN).
// V is the paged pool [page][Page][kvRow].
func (g *Row) PagedAcc(wts, v, part, table, desc int, p Paged) {
	const block = 8
	u, ub := g.UConst, g.UBin
	G := p.HeadGroup()
	gqa, S, C, P := p.Heads/p.KVHeads, p.Splits, p.Chunk, p.Page
	kvRow := p.KVHeads * p.Dim
	flat := g.Column()
	j := ub(OpRem, flat, u(p.Dim))
	rest := ub(OpDiv, flat, u(p.Dim))
	sp := ub(OpRem, rest, u(S))
	rest = ub(OpDiv, rest, u(S))
	hg := ub(OpRem, rest, u(p.Heads/G))
	row := ub(OpDiv, rest, u(p.Heads/G))
	h0 := ub(OpMul, hg, u(G))
	ps := g.pagedSplitOf(desc, row, sp, S)
	keys := ub(OpSub, ps.en, ps.st)
	blocks := ub(OpDiv, ub(OpAdd, keys, u(block-1)), u(block))
	col := ub(OpAdd, ub(OpMul, ub(OpDiv, h0, u(gqa)), u(p.Dim)), j)
	lastTile := ub(OpMul, ub(OpDiv, ps.last, u(32)), u(32))
	wb := ub(OpMul, ub(OpAdd, ub(OpMul, ub(OpAdd, ub(OpMul, row, u(p.Heads)), h0), u(S)), sp), u(C))
	d0 := ub(OpSub, ps.st, ps.ks)
	span := ub(OpSub, ps.en, ps.ks)
	c := g.LoopN(blocks)
	k0 := ub(OpMul, g.Iter(c), u(block))
	bs := ub(OpAdd, ps.st, k0)
	pid := g.pageOf(table, ps.tab, g.Select(ps.live, ub(OpMin, bs, lastTile), u(0)), P)
	vb := ub(OpAdd, ub(OpMul, ub(OpAdd, ub(OpMul, pid, u(P)), g.pageRem(bs, P)), u(kvRow)), col)
	dk := ub(OpAdd, d0, k0)
	wk := ub(OpAdd, wb, k0)
	acc := make([]int, G)
	next := make([]int, G)
	for h := range acc {
		acc[h] = g.Fold(c, g.Const(0))
		next[h] = acc[h]
	}
	for kk := 0; kk < block; kk++ {
		live := g.Less(ub(OpAdd, dk, u(kk)), span)
		v0 := g.Select(live, g.LoadAt(v, ub(OpAdd, vb, u(kk*kvRow))), g.Const(0))
		for h := range next {
			w := g.LoadAt(wts, ub(OpAdd, wk, u(h*S*C+kk)))
			next[h] = g.Add(Node{Kind: NBinary, Op: OpFma, A: w, B: v0, C: next[h]})
		}
	}
	for h := range acc {
		g.SetFold(acc[h], next[h])
	}
	padded := (p.Dim + 31) / 32 * 32
	pb := ub(OpMul, ub(OpAdd, ub(OpMul, ub(OpAdd, ub(OpMul, row, u(p.Heads)), h0), u(S)), sp), u(padded+64))
	for h := range acc {
		g.StoreAt(part, acc[h], ub(OpAdd, ub(OpAdd, pb, j), u(h*S*(padded+64))))
	}
}

// PagedMerge is the group element's output (an element a (row, head,
// value element)) from the splits' partials at part: each split's maximum
// and denominator after its numerators, the splits weighted by exp of their
// maximum less the greatest, the sum of the weighted denominators (and the
// sink's exp, where sink >= 0: one logit a head) the normaliser.
func (g *Row) PagedMerge(part, out, sink int, p Paged) {
	u, ub := g.UConst, g.UBin
	flat := g.Column()
	item := ub(OpDiv, flat, u(p.Dim))
	dim := ub(OpRem, flat, u(p.Dim))
	padded := (p.Dim + 31) / 32 * 32
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
