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
		return g.Add(Node{Kind: NHint, Hint: HintMultiple, A: g.LoadAt(bt, slot), At: 64})
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
	base := g.TileFirst(g.baseOf(s, desc, bt, row))
	tw := g.TileFirst(g.descOf(desc, dr, kernels.PRowTab))
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

// PrefillSoftmax is each (row, head, split) item's softmax over the slots
// of its split its row attends, an item a row of one element: the maximum
// and the denominator to the merge's partials (its 32 replicas), the
// weights to out over the slots the reader's tile of qt rows attends, in
// whole blocks of 8; in the direct form the weights normalised, by the sink
// where sink >= 0, and no partials.
func (g *Row) PrefillSoftmax(s Prefill, a, out, part, desc, sink, mp, bt, qt int) {
	u, ub := g.UConst, g.UBin
	add := func(x, y int) int { return ub(OpAdd, x, y) }
	C, S := s.Chunk, s.splits()
	item := g.RowIdx()
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
	over := func(x, y int, body func(at int)) int {
		c := g.LoopN(ub(OpSub, y, x))
		body(add(add(pb, x), g.Iter(c)))
		return c
	}
	var m int
	c1 := over(lo, e, func(at int) {})
	m = g.Fold(c1, g.Const(-math.MaxFloat32))
	g.SetFold(m, g.Bin(OpMax, m, g.LoadAt(a, add(add(pb, lo), g.Iter(c1)))))
	if s.Splits == 0 {
		c2 := g.LoopN(ub(OpSub, e, lo))
		sum := g.Fold(c2, g.Const(0))
		g.SetFold(sum, g.Bin(OpAdd, sum, g.exp(g.Bin(OpSub, g.LoadAt(a, add(add(pb, lo), g.Iter(c2))), m))))
		l, mm := sum, m
		if s.Sink && sink >= 0 {
			sv := g.LoadAt(sink, ub(OpRem, ub(OpRem, item, u(s.Heads*S)), u(s.Heads)))
			mm = g.Bin(OpMax, m, sv)
			l = g.Bin(OpAdd, g.Bin(OpMul, l, g.exp(g.Bin(OpSub, m, mm))), g.exp(g.Bin(OpSub, sv, mm)))
		}
		inv := g.Bin(OpDiv, g.Const(1), g.Bin(OpMax, l, g.Const(1e-30)))
		over(z0, z1, func(at int) { g.StoreAt(out, g.Bin(OpMul, g.exp(g.Bin(OpSub, g.LoadAt(a, at), mm)), inv), at) })
		return
	}
	c2 := g.LoopN(ub(OpSub, z1, z0))
	at := add(add(pb, z0), g.Iter(c2))
	sum := g.Fold(c2, g.Const(0))
	w := g.exp(g.Bin(OpSub, g.LoadAt(a, at), m))
	g.StoreAt(out, w, at)
	g.SetFold(sum, g.Bin(OpAdd, sum, w))
	padded := (s.Dim + 31) / 32 * 32
	pp := ub(OpMul, item, u(padded+64))
	r := g.Loop(32)
	g.StoreAt(part, m, add(add(pp, g.Iter(r)), u(padded)))
	g.StoreAt(part, sum, add(add(pp, g.Iter(r)), u(padded+32)))
}

// PrefillAcc is a split's weighted sum of V an element: row R's (query
// row, head, split) value element j summed over the keys of its split its
// row attends, each key's weight at its slot and its V through the row's
// table; into the merge's partials, or the output in the direct form. The
// group is Rows*Heads*Splits rows of Dim elements.
func (g *Row) PrefillAcc(s Prefill, a, v, part, tab, desc, mp, bt int) {
	u, ub := g.UConst, g.UBin
	add := func(x, y int) int { return ub(OpAdd, x, y) }
	mul := func(x, y int) int { return ub(OpMul, x, y) }
	P, C, S := s.Page, s.Chunk, s.splits()
	kvRow, gqa := s.KVHeads*s.Dim, s.Heads/s.KVHeads
	j, R := g.Column(), g.RowIdx()
	spl, rest := ub(OpRem, R, u(S)), ub(OpDiv, R, u(S))
	h, row := ub(OpRem, rest, u(s.Heads)), ub(OpDiv, rest, u(s.Heads))
	dr := g.rowOf(s, mp, row)
	base := g.TileFirst(g.baseOf(s, desc, bt, row))
	tw := g.TileFirst(g.descOf(desc, dr, kernels.PRowTab))
	lo := g.TileFirst(g.descOf(desc, dr, kernels.PRowStart))
	hi := g.TileLast(g.descOf(desc, dr, kernels.PRowEnd))
	c0 := add(base, mul(spl, u(C)))
	st := ub(OpMin, ub(OpMax, lo, c0), add(c0, u(C)))
	en := ub(OpMax, ub(OpMin, hi, add(c0, u(C))), st)
	c, t := g.RangeLoop(c0, st, en, C)
	pid := g.LoadAt(tab, add(tw, ub(OpDiv, g.KeyClamp(t, lo, hi), u(P))))
	col := add(mul(ub(OpDiv, h, u(gqa)), u(s.Dim)), j)
	vv := g.LoadAt(v, add(mul(add(mul(pid, u(P)), ub(OpRem, t, u(P))), u(kvRow)), col))
	w := g.LoadAt(a, add(mul(R, u(C)), ub(OpSub, t, c0)))
	acc := g.Fold(c, g.Const(0))
	g.SetFold(acc, g.fma(w, vv, acc))
	if s.Splits == 0 {
		g.StoreAt(part, acc, add(mul(add(mul(dr, u(s.Heads)), h), u(s.Dim)), j))
		return
	}
	g.StoreAt(part, acc, add(mul(R, u((s.Dim+31)/32*32+64)), j))
}
