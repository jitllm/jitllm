package compose

import (
	"math"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// Prefill is a prompt chunk's paged attention staged: each row's scores
// into a plane of Splits*Chunk slots a (row, head), each (row, head,
// split)'s softmax, each split's weighted sum of V, merged after. A slot is
// key base+slot, base the 64-aligned position at or below row 0's first key
// (with a run table, Map, each slot's row and base read from two tables).
// Splits 0 is the direct form: one split, the weights normalised and the
// sums the output.
type Prefill struct {
	Rows, Heads, KVHeads, Dim, Page, Splits, Chunk int
	Scale, Softcap                                 float32
	Sink, Map                                      bool
}

func (s Prefill) splits() int { return max(1, s.Splits) }

// prefillSpan is what a query tile of rows attends: its table, the
// partition base, its keys [lo, hi), and whether it has any.
type prefillSpan struct{ tab, base, lo, hi, live int }

// rowOf is slot's row: itself, or the run table's entry.
func (g *Row) rowOf(s Prefill, mp, slot int) int {
	if !s.Map {
		return slot
	}
	return g.LoadAt(mp, slot)
}

// descOf is word w of row r's descriptor.
func (g *Row) descOf(desc, r, w int) int {
	return g.LoadAt(desc, g.UBin(OpAdd, g.UBin(OpMul, r, g.UConst(kernels.PRowWords)), g.UConst(w)))
}

// baseOf is slot's partition base: row 0's first key rounded down to 64, or
// the base table's entry.
func (g *Row) baseOf(s Prefill, desc, bt, slot int) int {
	if s.Map {
		return g.LoadAt(bt, slot)
	}
	u, ub := g.UConst, g.UBin
	return ub(OpMul, ub(OpDiv, g.descOf(desc, u(0), kernels.PRowStart), u(64)), u(64))
}

func (g *Row) spanOf(s Prefill, desc, mp, bt, first, last int) prefillSpan {
	f, l := g.rowOf(s, mp, first), g.rowOf(s, mp, last)
	lo, hi := g.descOf(desc, f, kernels.PRowStart), g.descOf(desc, l, kernels.PRowEnd)
	return prefillSpan{tab: g.descOf(desc, f, kernels.PRowTab), base: g.baseOf(s, desc, bt, first), lo: lo, hi: hi, live: g.Less(lo, hi)}
}

// clamp is t moved into the span's keys (aligned to align where it is
// over one), position 0 for an empty span.
func (g *Row) clamp(p prefillSpan, t, align int) int {
	u, ub := g.UConst, g.UBin
	lo, last := p.lo, ub(OpSub, ub(OpMax, p.hi, u(1)), u(1))
	if align > 1 {
		lo, last = ub(OpMul, ub(OpDiv, lo, u(align)), u(align)), ub(OpMul, ub(OpDiv, last, u(align)), u(align))
	}
	return g.Select(p.live, ub(OpMin, ub(OpMax, t, lo), last), u(0))
}

// capped is a dot's score: scaled, then capped where there is a softcap.
func (g *Row) capped(dot int, scale, softcap float32) int {
	x := g.Bin(OpMul, dot, g.Const(scale))
	if softcap > 0 {
		e := g.exp(g.Bin(OpMul, g.Const(2/softcap), x))
		x = g.Bin(OpMul, g.Const(softcap), g.Bin(OpSub, g.Const(1), g.Bin(OpDiv, g.Const(2), g.Bin(OpAdd, e, g.Const(1)))))
	}
	return x
}

// PrefillScores is a score an element: row R's (query row R/Heads, head
// R%Heads) dot with slot j's key, its page looked up through the row's
// table, -inf where the row does not attend; a group of Rows*Heads rows of
// Splits*Chunk elements. mp and bt are the run tables (-1 without).
func (g *Row) PrefillScores(s Prefill, q, k, out, tab, desc, mp, bt int) {
	u, ub := g.UConst, g.UBin
	add := func(a, b int) int { return ub(OpAdd, a, b) }
	mul := func(a, b int) int { return ub(OpMul, a, b) }
	P := s.Page
	kvRow, gqa := s.KVHeads*s.Dim, s.Heads/s.KVHeads
	j, R := g.Column(), g.RowIdx()
	row, h := ub(OpDiv, R, u(s.Heads)), ub(OpRem, R, u(s.Heads))
	dr := g.rowOf(s, mp, row)
	base := g.SpanUniform(g.baseOf(s, desc, bt, row))
	tw := g.SpanUniform(g.descOf(desc, dr, kernels.PRowTab))
	t := g.SpanClamp(desc, kernels.PRowWords, dr, kernels.PRowStart, kernels.PRowEnd, add(base, j), 1)
	pid := g.LoadAt(tab, add(tw, ub(OpDiv, t, u(P))))
	kb := add(add(mul(pid, u(P*kvRow)), mul(mul(ub(OpDiv, h, u(gqa)), u(s.Dim)), u(P))), ub(OpRem, t, u(P)))
	qb := mul(add(mul(dr, u(s.Heads)), h), u(s.Dim))
	acc := g.Const(0)
	for i := 0; i < s.Dim; i++ {
		acc = g.fma(g.LoadAt(q, add(qb, u(i))), g.LoadAt(k, add(kb, u(i*P))), acc)
	}
	ks, ke := g.descOf(desc, dr, kernels.PRowStart), g.descOf(desc, dr, kernels.PRowEnd)
	ok := g.Less(ub(OpSub, add(base, j), ks), ub(OpSub, ke, ks))
	g.Store(out, g.Select(ok, g.capped(acc, s.Scale, s.Softcap), g.Add(Node{Kind: NUnary, Op: OpBits, A: u(0xFF800000)})))
}

// PrefillSoftmax is each (row, head, split) item's softmax over the slots of
// its split its row attends: an item a row of 32 lanes over the items' rows,
// or a thread an item of one row. The weights go to out over the slots the item's query
// tile of qt rows attends, in whole blocks of 8; the maximum and the
// denominator to the merge's partials, or in the direct form the weights
// normalised (by the sink where sink >= 0).
func (g *Row) PrefillSoftmax(s Prefill, a, out, part, desc, sink, mp, bt, qt int) {
	u, ub := g.UConst, g.UBin
	add := func(x, y int) int { return ub(OpAdd, x, y) }
	C, S := s.Chunk, s.splits()
	lanes := 1
	if g.Rows > 1 {
		lanes = g.Width
	}
	var item, lane, step int
	if lanes == 32 {
		item, lane, step = g.RowIdx(), g.Column(), u(32)
	} else {
		item, lane, step = g.Column(), u(0), u(1)
	}
	sp := ub(OpRem, item, u(S))
	row := ub(OpDiv, item, u(s.Heads*S))
	c0 := add(g.baseOf(s, desc, bt, row), ub(OpMul, sp, u(C)))
	cEnd := add(c0, u(C))
	in := func(t int) int { return ub(OpSub, ub(OpMin, ub(OpMax, t, c0), cEnd), c0) }
	dr := g.rowOf(s, mp, row)
	lo := in(g.descOf(desc, dr, kernels.PRowStart))
	e := ub(OpMax, in(g.descOf(desc, dr, kernels.PRowEnd)), lo)
	first := ub(OpMul, ub(OpDiv, row, u(qt)), u(qt))
	ts := in(g.descOf(desc, g.rowOf(s, mp, first), kernels.PRowStart))
	te := ub(OpMax, in(g.descOf(desc, g.rowOf(s, mp, add(first, u(qt-1))), kernels.PRowEnd)), ts)
	z0 := ub(OpMul, ub(OpDiv, ts, u(8)), u(8))
	z1 := ub(OpMin, ub(OpMul, ub(OpDiv, add(te, u(7)), u(8)), u(8)), u(C))
	z1 = g.Select(g.Less(ts, te), z1, z0)
	pb := ub(OpMul, item, u(C))
	count := func(x, y int) int {
		n := ub(OpSub, y, x)
		if lanes == 1 {
			return n
		}
		return ub(OpShr, add(ub(OpSub, ub(OpMax, n, lane), lane), u(31)), u(5))
	}
	minus := g.Const(-math.MaxFloat32)
	first0 := add(add(pb, lo), lane)
	reduce := func(x int, red Red) int {
		if lanes != 32 {
			return x
		}
		return g.Add(Node{Kind: NReduce, Red: red, A: x, At: 1})
	}
	c1 := g.LoopN(count(lo, e))
	mx := g.Fold(c1, minus)
	g.SetFold(mx, g.Bin(OpMax, mx, g.LoadAt(a, add(first0, ub(OpMul, g.Iter(c1), step)))))
	m := reduce(mx, RedMax)
	startW := add(add(pb, z0), lane)
	if s.Splits == 0 {
		c2 := g.LoopN(count(lo, e))
		sum := g.Fold(c2, g.Const(0))
		g.SetFold(sum, g.Bin(OpAdd, sum, g.exp(g.Bin(OpSub, g.LoadAt(a, add(first0, ub(OpMul, g.Iter(c2), step))), m))))
		l, mm := reduce(sum, RedSum), m
		if s.Sink && sink >= 0 {
			sv := g.LoadAt(sink, ub(OpRem, ub(OpRem, item, u(s.Heads*S)), u(s.Heads)))
			mm = g.Bin(OpMax, m, sv)
			l = g.Bin(OpAdd, g.Bin(OpMul, l, g.exp(g.Bin(OpSub, m, mm))), g.exp(g.Bin(OpSub, sv, mm)))
		}
		inv := g.Bin(OpDiv, g.Const(1), g.Bin(OpMax, l, g.Const(1e-30)))
		c3 := g.LoopN(count(z0, z1))
		wi := add(startW, ub(OpMul, g.Iter(c3), step))
		g.StoreAt(out, g.Bin(OpMul, g.exp(g.Bin(OpSub, g.LoadAt(a, wi), mm)), inv), wi)
		return
	}
	c2 := g.LoopN(count(z0, z1))
	si := add(startW, ub(OpMul, g.Iter(c2), step))
	sum := g.Fold(c2, g.Const(0))
	w := g.exp(g.Bin(OpSub, g.LoadAt(a, si), m))
	g.StoreAt(out, w, si)
	g.SetFold(sum, g.Bin(OpAdd, sum, w))
	l := reduce(sum, RedSum)
	padded := (s.Dim + 31) / 32 * 32
	pp := ub(OpMul, item, u(padded+64))
	if lanes == 32 {
		g.StoreAt(part, m, add(add(pp, lane), u(padded)))
		g.StoreAt(part, l, add(add(pp, lane), u(padded+32)))
		return
	}
	for r := 0; r < 32; r++ {
		g.StoreAt(part, m, add(pp, u(padded+r)))
		g.StoreAt(part, l, add(pp, u(padded+32+r)))
	}
}

// PrefillAcc is each split's weighted sum of V a thread a (query tile of
// qt rows, head, split, value element): the split's part of the tile's keys
// in blocks of 8 from a multiple of 8, a block's page read once (a block
// ahead between the first and the last), the first and the last block
// masked to the keys; into the merge's partials, or the output in the
// direct form.
func (g *Row) PrefillAcc(s Prefill, a, v, part, tab, desc, mp, bt, qt int) {
	u, ub := g.UConst, g.UBin
	add := func(x, y int) int { return ub(OpAdd, x, y) }
	mul := func(x, y int) int { return ub(OpMul, x, y) }
	const blk = 8
	P, C, S := s.Page, s.Chunk, s.splits()
	kvRow, gqa := s.KVHeads*s.Dim, s.Heads/s.KVHeads
	per := s.Dim
	flat := g.Column()
	j, rest := ub(OpRem, flat, u(per)), ub(OpDiv, flat, u(per))
	spl := ub(OpRem, rest, u(S))
	rest = ub(OpDiv, rest, u(S))
	h := ub(OpRem, rest, u(s.Heads))
	qrow := mul(ub(OpDiv, rest, u(s.Heads)), u(qt))
	sp := g.spanOf(s, desc, mp, bt, qrow, add(qrow, u(qt-1)))
	c0 := add(sp.base, mul(spl, u(C)))
	st := ub(OpMin, ub(OpMax, sp.lo, c0), add(c0, u(C)))
	en := ub(OpMax, ub(OpMin, sp.hi, add(c0, u(C))), st)
	a8 := add(c0, mul(ub(OpDiv, ub(OpSub, st, c0), u(blk)), u(blk)))
	blocks := ub(OpDiv, add(ub(OpSub, en, a8), u(blk-1)), u(blk))
	blocks = g.Select(g.Less(st, en), blocks, u(0))
	col := add(mul(ub(OpDiv, h, u(gqa)), u(s.Dim)), j)
	blockPage := func(t int) int { return g.LoadAt(tab, add(sp.tab, ub(OpDiv, g.clamp(sp, t, blk), u(P)))) }
	wb := add(mul(add(mul(qrow, u(s.Heads)), h), u(S*C)), mul(spl, u(C)))
	block := func(bs, pid int, acc []int, masked bool) []int {
		next := append([]int(nil), acc...)
		vb := add(mul(add(mul(pid, u(P)), ub(OpRem, bs, u(P))), u(kvRow)), col)
		slot := ub(OpSub, bs, c0)
		if masked {
			slot = ub(OpMin, slot, u(C-blk))
		}
		wk := add(wb, slot)
		for kk := 0; kk < blk; kk++ {
			var live int
			if masked {
				live = g.Less(ub(OpSub, add(bs, u(kk)), st), ub(OpSub, en, st))
			}
			v0 := g.LoadAt(v, add(vb, u(kk*kvRow)))
			if masked {
				v0 = g.Select(live, v0, g.Const(0))
			}
			for r := 0; r < qt; r++ {
				w := g.LoadAt(a, add(wk, u(r*s.Heads*S*C+kk)))
				if masked {
					w = g.Select(live, w, g.Const(0))
				}
				next[r] = g.fma(w, v0, next[r])
			}
		}
		return next
	}
	acc := make([]int, qt)
	for i := range acc {
		acc[i] = g.Const(0)
	}
	acc = block(a8, blockPage(a8), acc, true)
	mid := ub(OpSub, ub(OpMax, blocks, u(2)), u(2))
	b1 := add(a8, u(blk))
	c := g.LoopN(mid)
	it := g.Iter(c)
	ph := make([]int, qt)
	for i := range ph {
		ph[i] = g.Fold(c, acc[i])
	}
	bs := add(b1, mul(it, u(blk)))
	pid := g.Fold(c, blockPage(b1))
	next := block(bs, pid, ph, false)
	for i := range ph {
		g.SetFold(ph[i], next[i])
	}
	g.SetFold(pid, blockPage(add(bs, u(blk))))
	last := add(a8, mul(add(mid, u(1)), u(blk)))
	acc = block(last, blockPage(last), ph, true)
	padded := (s.Dim + 31) / 32 * 32
	pb := mul(add(mul(add(mul(qrow, u(s.Heads)), h), u(S)), spl), u(padded+64))
	rstep := s.Heads * S * (padded + 64)
	if s.Splits == 0 {
		pb, rstep = mul(add(mul(qrow, u(s.Heads)), h), u(s.Dim)), s.Heads*s.Dim
	}
	for r := 0; r < qt; r++ {
		at := add(add(pb, j), u(r*rstep))
		if s.Map {
			at = add(mul(add(mul(g.rowOf(s, mp, add(qrow, u(r))), u(s.Heads)), h), u(s.Dim)), j)
		}
		g.StoreAt(part, acc[r], at)
	}
}
