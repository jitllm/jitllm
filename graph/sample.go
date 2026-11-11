package graph

import "math"

// Sample is greedy sampling: the token of the largest logit, the lower of
// two equal ones.
type Sample struct {
	NVocab int
}

func (s *Sample) Build() *Graph {
	g := &Graph{Block: -1}
	l := g.value("logits", s.NVocab)
	tok := g.value("token", 1)
	best := g.value("best", 1)
	g.In, g.Out = []ValueID{l}, []ValueID{tok}
	g.op(Op{Kind: OpTopK, In: []ValueID{l}, Out: []ValueID{tok, best}, Attr: Attrs{TopK: 1}})
	return g
}

// SampleTopK is the sampler's candidates: the penalties over the row
// (penalizedRow), then its K largest, largest first and the lower id on a
// tie. Its inputs are the logits, sample.window (each token's count in the
// repeat window, as floats) and sample.penalty (the repeat, presence and
// frequency penalties, nn.Penalties); its outputs the K values and their
// ids. A request for k <= K reads the first k, and none of the form
// depends on k or on the window's length.
type SampleTopK struct {
	NVocab, K int
}

func (s *SampleTopK) Build() *Graph {
	b := New()
	vals, ids := s.emit(b)
	b.Out(vals, ids)
	return b.G
}

// emit is SampleTopK on b: its inputs, then the K values and their ids.
func (s *SampleTopK) emit(b *Builder) (vals, ids ValueID) {
	K := min(s.K, s.NVocab)
	l := b.In("logits", s.NVocab)
	win := b.In("sample.window", s.NVocab)
	pen := b.In("sample.penalty", 3)
	v := penalizedRow(b, l, win, pen)
	cand := b.Emit(OpTopK, []ValueID{v}, []Result{{Name: "sample.ids", Width: K, Elem: ElemU32}, {Name: "sample.vals", Width: K}}, nil,
		Attrs{TopK: K}, nil, nil)
	return cand[1], cand[0]
}

// penalizedRow is llama.cpp's penalties over logits l, on each token the
// window holds (its count in win above zero): the repeat penalty once,
// whatever its count -- a positive logit divided by it, correctly rounded,
// any other multiplied by it -- then frequency times its count plus
// presence subtracted. Every other token is as it is. pen is the three
// penalties' words, in that order.
func penalizedRow(b *Builder, l, win, pen ValueID) ValueID {
	e := sampleEmitter{b}
	rep, pres, freq := e.word(pen, 0, "sample.repeat"), e.word(pen, 1, "sample.presence"), e.word(pen, 2, "sample.frequency")
	in := func(i int) Arg { return Arg{Kind: ArgIn, Index: i} }
	term := func(i int) Arg { return Arg{Kind: ArgTerm, Index: i} }
	terms := []Term{
		{Fn: MapDiv, Args: [3]Arg{in(0), in(2)}},
		{Fn: MapMul, Args: [3]Arg{in(0), in(2)}},
		{Fn: MapWhere, Args: [3]Arg{in(0), term(0), term(1)}},
		{Fn: MapLess, Args: [3]Arg{{Kind: ArgConst, Const: 0}, in(1)}},
		{Fn: MapWhere, Args: [3]Arg{term(3), term(2), in(0)}},
		{Fn: MapMul, Args: [3]Arg{in(1), in(4)}},
		{Fn: MapMul, Args: [3]Arg{term(3), in(3)}},
		{Fn: MapAdd, Args: [3]Arg{term(5), term(6)}},
		{Fn: MapSub, Args: [3]Arg{term(4), term(7)}},
	}
	return b.Emit(OpExpr, []ValueID{l, win, rep, pres, freq}, []Result{{Name: "sample.penalized", Width: b.G.Values[l].Width}}, nil,
		Attrs{Expr: terms, Exact: true}, nil, nil)[0]
}

// SampleWalk is SampleTopK's candidates drawn as the host's sampler draws
// them: of the first k, top-p and then min-p over their softmax at
// temperature 1, the best always kept, then the inverse-CDF walk at a uniform
// u the caller draws over the survivors' softmax at the temperature -- the
// first survivor whose mass up to and including it passes u times the
// survivors' mass, the last survivor where rounding leaves none. Its inputs
// are SampleTopK's, then sample.walk (k, 1/T, top-p +Inf where off, min-p 0
// where off, then u); its output the token. The mass before each survivor is a KxK sum,
// as top-p's. With Prob the outputs are the distribution the draw draws
// from (u unread): each candidate's share of the survivors' softmax at the
// temperature (0 where it is cut), then the candidates' ids.
type SampleWalk struct {
	SampleTopK
	Prob bool
}

func (s *SampleWalk) Build() *Graph {
	b := New()
	vals, ids := s.emit(b)
	walk := b.In("sample.walk", 5)
	K := b.G.Values[vals].Width
	e := sampleEmitter{b}
	place, d, keep := e.cuts(vals, walk)
	invT, u := e.word(walk, 1, "sample.invtemp"), e.word(walk, 4, "sample.u")
	et := e.mapc(MapMul, 0, "sample.et", K, e.mapc(MapExp, 0, "sample.etd", K, e.mapc(MapMul, 0, "sample.dt", K, d, invT)), keep)
	mass := e.one(OpReduce, "sample.mass", 1, Attrs{Fn: MapAdd}, et)
	if s.Prob {
		b.Out(e.mapc(MapDiv, 0, "sample.pt", K, et, mass), ids)
		return b.G
	}
	upto := e.mapc(MapAdd, 0, "sample.upto", K, e.before(et, place, "sample.walk"), et)
	// past is upto > u*mass, as -upto < -(u*mass).
	target := e.mapc(MapMul, -1, "sample.target.neg", 1, e.mapc(MapMul, 0, "sample.target", 1, u, mass))
	past := e.mapc(MapLess, 0, "sample.past", K, e.mapc(MapMul, -1, "sample.upto.neg", K, upto), target)
	land := e.mapc(MapMul, 0, "sample.land", K, past, keep)
	// The first place landed on outranks every other (2K - place), and with
	// none the last survivor (its place) outranks the cut (-1).
	fall := e.mapc(MapWhere, -1, "sample.fall", K, keep, place)
	hi := e.mapc(MapAdd, float64(2*K), "sample.hi", K, e.mapc(MapMul, -1, "sample.place.neg", K, place))
	notLand := e.mapc(MapAdd, 1, "sample.notland", K, e.mapc(MapMul, -1, "sample.land.neg", K, land))
	score := e.mapc(MapAdd, 0, "sample.walkscore", K, e.mapc(MapMul, 0, "sample.landscore", K, land, hi),
		e.mapc(MapMul, 0, "sample.fallscore", K, notLand, fall))
	b.Out(e.token(ids, score))
	return b.G
}

// SampleRow is the host sampler's draw with no top-k and no cut: the
// penalized row's softmax at the temperature, in its own order (the larger
// first, the lower id of two equal), walked at a uniform u the caller
// draws -- the first token whose mass up to and including it passes u times
// the row's mass, the last where rounding leaves none. Its inputs are
// SampleTopK's logits, sample.window and sample.penalty, sample.ids (the row's
// ids in order, u32) and sample.row (1/T, then u); its output the token. The
// walk is a weighted selection over the row (sampleEmitter.selection): no
// order of it is formed.
type SampleRow struct {
	NVocab int
}

func (s *SampleRow) Build() *Graph {
	b := New()
	l := b.In("logits", s.NVocab)
	win := b.In("sample.window", s.NVocab)
	pen := b.In("sample.penalty", 3)
	ids := b.In("sample.ids", s.NVocab)
	b.G.Values[ids].Elem = ElemU32
	row := b.In("sample.row", 2)
	e := sampleEmitter{b}
	v := penalizedRow(b, l, win, pen)
	invT, u := e.word(row, 0, "sample.invtemp"), e.word(row, 1, "sample.u")
	q := e.softmax(e.mapc(MapMul, 0, "sample.scaled", s.NVocab, v, invT), "sample.q")
	target := e.mapc(MapMul, 0, "sample.target", 1, u, e.one(OpReduce, "sample.mass", 1, Attrs{Fn: MapAdd}, q))
	all := e.mapc(MapLess, 2, "sample.all", s.NVocab, q)
	b.Out(e.argmax(e.selection(q, e.mapc(MapMul, 1, "sample.idf", s.NVocab, ids), q, all, target, false)))
	return b.G
}

// SampleCut is the host sampler's draw with no top-k and a cut: over the
// penalized row's softmax at temperature 1 in its order (the larger first,
// the lower id of two equal), min-p keeps the tokens at least min-p times
// the best's probability and top-p those before the first at which the mass
// up to and including it reaches top-p, that one kept, the best always; the
// survivors' softmax at the temperature is walked in that order at a uniform
// u the caller draws, as SampleRow walks the row. Where neither cut fires
// anywhere in the row, the draw is SampleRow's. Its inputs are SampleRow's,
// with sample.cut (1/T, top-p +Inf where off, min-p 0 where off, then u) for
// sample.row; its output the token. With Prob the output is the
// distribution the draw draws from (u unread), each token's probability:
// its share of the survivors' softmax at the temperature where a cut fires,
// of the row's where none does. With TopK, sample.cut has a fifth word, k
// (0 < k < NVocab), and the draw is the host's top-k draw past the bound
// SampleWalk is formed for: the cuts read the softmax over the k best (the
// k-th found by a selection counting tokens), and the survivors are walked
// whether or not a cut fires.
type SampleCut struct {
	NVocab     int
	Prob, TopK bool
}

func (s *SampleCut) Build() *Graph {
	b := New()
	n := s.NVocab
	l := b.In("logits", n)
	win := b.In("sample.window", s.NVocab)
	pen := b.In("sample.penalty", 3)
	ids := b.In("sample.ids", n)
	b.G.Values[ids].Elem = ElemU32
	words := 4
	if s.TopK {
		words = 5
	}
	cut := b.In("sample.cut", words)
	e := sampleEmitter{b}
	v := penalizedRow(b, l, win, pen)
	invT, topp, minp, u := e.word(cut, 0, "sample.invtemp"), e.word(cut, 1, "sample.topp"), e.word(cut, 2, "sample.minp"), e.word(cut, 3, "sample.u")
	id := e.mapc(MapMul, 1, "sample.idf", n, ids)
	notv := func(name string, w int, x ValueID) ValueID {
		return e.mapc(MapAdd, 1, name, w, e.mapc(MapMul, -1, name+".neg", w, x))
	}
	order := e.softmax(v, "sample.q1")
	all := e.mapc(MapLess, 2, "sample.all", n, order)
	// q1 is the probabilities the cuts read, over pool: the row, or the k
	// best renormalized.
	q1, pool := order, all
	if s.TopK {
		k := e.word(cut, 4, "sample.k")
		kth := e.selection(order, id, all, all, e.mapc(MapAdd, -1, "sample.k.target", 1, k), false)
		pool = e.upTo(order, id, kth, "sample.topk")
		inK := e.mapc(MapMul, 0, "sample.q1k", n, order, pool)
		q1 = e.mapc(MapDiv, 0, "sample.q1kn", n, inK, e.one(OpReduce, "sample.q1ksum", 1, Attrs{Fn: MapAdd}, inK))
	}
	// The best: the largest probability, the lowest id of those.
	p0 := e.one(OpReduce, "sample.p0", 1, Attrs{Fn: MapMax}, q1)
	atBest := e.mapc(MapEqual, 0, "sample.atbest", n, q1, p0)
	bestID := e.one(OpReduce, "sample.bestid", 1, Attrs{Fn: MapMin}, e.mapc(MapWhere, math.Inf(1), "sample.bestids", n, atBest, id))
	best := e.mapc(MapMul, 0, "sample.best", n, atBest, e.mapc(MapEqual, 0, "sample.isbestid", n, id, bestID))
	// Min-p: at least min-p times the best's probability.
	floor := e.mapc(MapMul, 0, "sample.floor", 1, p0, minp)
	below := e.mapc(MapLess, 0, "sample.below", n, q1, floor)
	minFired := e.one(OpReduce, "sample.minfired", 1, Attrs{Fn: MapMax}, below)
	// Top-p: up to the first token whose mass up to it reaches top-p, where
	// the row's mass does.
	mass1 := e.one(OpReduce, "sample.mass1", 1, Attrs{Fn: MapAdd}, q1)
	topFired := notv("sample.topfired", 1, e.mapc(MapLess, 0, "sample.short", 1, mass1, topp))
	edge := e.selection(order, id, q1, pool, topp, true)
	keepP := e.mapc(MapMax, 0, "sample.keepp", n, e.upTo(order, id, edge, "sample.edge"), notv("sample.topoff", 1, topFired))
	keep := e.mapc(MapMax, 0, "sample.keep", n,
		e.mapc(MapMul, 0, "sample.keeppmk", n, e.mapc(MapMul, 0, "sample.keeppm", n, keepP, notv("sample.above.floor", n, below)), pool), best)
	fired := e.mapc(MapMax, 0, "sample.fired", 1, topFired, minFired)
	if s.TopK {
		// A top-k draw walks its survivors whether or not a cut fires.
		fired = e.mapc(MapLess, 2, "sample.fired.k", 1, fired)
	}
	// The survivors' softmax at the temperature, walked in the cut's order.
	scaled := e.mapc(MapMul, 0, "sample.scaled", n, v, invT)
	top := e.one(OpReduce, "sample.top", 1, Attrs{Fn: MapMax}, e.mapc(MapWhere, math.Inf(-1), "sample.keptscaled", n, keep, scaled))
	et := e.mapc(MapWhere, 0, "sample.et", n, keep, e.mapc(MapExp, 0, "sample.ete", n, e.mapc(MapSub, 0, "sample.etd", n, scaled, top)))
	pt := e.mapc(MapDiv, 0, "sample.pt", n, et, e.one(OpReduce, "sample.etsum", 1, Attrs{Fn: MapAdd}, et))
	q := e.softmax(scaled, "sample.q")
	if s.Prob {
		b.Out(e.mapc(MapAdd, 0, "sample.prob", n, q, e.mapc(MapMul, 0, "sample.prob.cut", n, e.mapc(MapSub, 0, "sample.prob.d", n, pt, q), fired)))
		return b.G
	}
	walkT := e.mapc(MapMul, 0, "sample.walk.target", 1, u, e.one(OpReduce, "sample.ptsum", 1, Attrs{Fn: MapAdd}, pt))
	walk := e.selection(order, id, pt, keep, walkT, false)
	if s.TopK {
		b.Out(e.argmax(walk))
		return b.G
	}
	// No cut anywhere: the row's walk.
	rowT := e.mapc(MapMul, 0, "sample.row.target", 1, u, e.one(OpReduce, "sample.mass", 1, Attrs{Fn: MapAdd}, q))
	row := e.selection(q, id, q, all, rowT, false)
	chosen := e.mapc(MapAdd, 0, "sample.chosen", n, row, e.mapc(MapMul, 0, "sample.chosen.cut", n, e.mapc(MapSub, 0, "sample.chosen.d", n, walk, row), fired))
	b.Out(e.argmax(chosen))
	return b.G
}

// sampleEmitter is the ops SampleWalk, SampleRow and SampleCut share on b.
type sampleEmitter struct{ b *Builder }

func (e sampleEmitter) one(op OpKind, name string, w int, a Attrs, in ...ValueID) ValueID {
	return e.b.Emit(op, in, []Result{{Name: name, Width: w}}, nil, a, nil, nil)[0]
}

func (e sampleEmitter) mapc(fn MapFn, c float64, name string, w int, in ...ValueID) ValueID {
	return e.one(OpMap, name, w, Attrs{Fn: fn, Const: c}, in...)
}

func (e sampleEmitter) word(v ValueID, i int, name string) ValueID {
	return e.one(OpSlice, name, 1, Attrs{Offset: i, Width: 1}, v)
}

// before is each candidate's sum of x over the candidates of a lower place: a
// KxK sum, each candidate's row of x with those of a lower place kept.
func (e sampleEmitter) before(x, place ValueID, name string) ValueID {
	K := e.b.G.Values[x].Width
	tx := e.one(OpTile, name+".rows", K*K, Attrs{Span: K}, x)
	tr := e.one(OpTile, name+".place.rows", K*K, Attrs{Span: K}, place)
	lower := e.mapc(MapMul, 0, name+".lower", K*K, tx, e.mapc(MapLess, 0, name+".lt", K*K, tr, place))
	return e.one(OpReduce, name+".before", K, Attrs{Fn: MapAdd, Span: K}, lower)
}

// cuts is the K candidates' (vals, best first) places, their distances from
// the best, and whether each survives: among the first k (params' word 0),
// top-p over the softmax at temperature 1 (word 2: a candidate survives while
// the mass before it is under top-p) and min-p (word 3: while its
// probability is at least min-p times the best's), the best always.
func (e sampleEmitter) cuts(vals, params ValueID) (place, d, keep ValueID) {
	K := e.b.G.Values[vals].Width
	k, topp, minp := e.word(params, 0, "sample.k"), e.word(params, 2, "sample.topp"), e.word(params, 3, "sample.minp")
	rank := e.b.Emit(OpRank, []ValueID{vals}, []Result{{Name: "sample.rank", Width: K, Elem: ElemU32}}, nil, Attrs{}, nil, nil)[0]
	place = e.mapc(MapMul, 1, "sample.place.f", K, rank)
	inK := e.mapc(MapLess, 0, "sample.ink", K, place, k)
	v0 := e.one(OpReduce, "sample.v0", 1, Attrs{Fn: MapMax}, vals)
	d = e.mapc(MapSub, 0, "sample.d", K, vals, v0)
	ex := e.mapc(MapMul, 0, "sample.e", K, e.mapc(MapExp, 0, "sample.ed", K, d), inK)
	p := e.mapc(MapDiv, 0, "sample.p", K, ex, e.one(OpReduce, "sample.sum", 1, Attrs{Fn: MapAdd}, ex))
	keepP := e.mapc(MapLess, 0, "sample.keep.p", K, e.before(p, place, "sample"), topp)
	floor := e.mapc(MapMul, 0, "sample.floor", 1, e.one(OpReduce, "sample.pmax", 1, Attrs{Fn: MapMax}, p), minp)
	under := e.mapc(MapLess, 0, "sample.under", K, p, floor)
	keepM := e.mapc(MapAdd, 1, "sample.keep.m", K, e.mapc(MapMul, -1, "sample.under.neg", K, under))
	first := e.mapc(MapLess, 1, "sample.first", K, place)
	keep = e.mapc(MapMax, 0, "sample.keep", K, e.mapc(MapMul, 0, "sample.keep.pmk", K, e.mapc(MapMul, 0, "sample.keep.pm", K, keepP, keepM), inK), first)
	return place, d, keep
}

// where is the term where(c, a, b): a where c > 0, else b.
func (e sampleEmitter) where(name string, w int, c, a, b ValueID) ValueID {
	t := []Term{{Fn: MapWhere, Args: [3]Arg{{Kind: ArgIn, Index: 0}, {Kind: ArgIn, Index: 1}, {Kind: ArgIn, Index: 2}}}}
	return e.b.Emit(OpExpr, []ValueID{c, a, b}, []Result{{Name: name, Width: w}}, nil, Attrs{Expr: t}, nil, nil)[0]
}

// softmax is x's softmax: each exp(x - max) over their sum.
func (e sampleEmitter) softmax(x ValueID, name string) ValueID {
	w := e.b.G.Values[x].Width
	m := e.one(OpReduce, name+".max", 1, Attrs{Fn: MapMax}, x)
	ex := e.mapc(MapExp, 0, name+".exp", w, e.mapc(MapSub, 0, name+".d", w, x, m))
	return e.mapc(MapDiv, 0, name, w, ex, e.one(OpReduce, name+".sum", 1, Attrs{Fn: MapAdd}, ex))
}

// upTo is the tokens at or before the one edge marks (a 0/1 over the row)
// in key's order: a larger key, or the edge's and an id not past its.
func (e sampleEmitter) upTo(key, id, edge ValueID, name string) ValueID {
	n := e.b.G.Values[key].Width
	edgeQ := e.one(OpReduce, name+".q", 1, Attrs{Fn: MapMax}, e.mapc(MapMul, 0, name+".qs", n, edge, key))
	edgeID := e.one(OpReduce, name+".id", 1, Attrs{Fn: MapMax}, e.mapc(MapMul, 0, name+".ids", n, edge, id))
	above := e.mapc(MapLess, 0, name+".above", n, e.mapc(MapMul, -1, name+".keyneg", n, key), e.mapc(MapMul, -1, name+".qneg", 1, edgeQ))
	notPast := e.mapc(MapAdd, 1, name+".notpast", n, e.mapc(MapMul, -1, name+".pastneg", n, e.mapc(MapLess, 0, name+".past", n, edgeID, id)))
	tie := e.mapc(MapMul, 0, name+".tie", n, e.mapc(MapEqual, 0, name+".attie", n, key, edgeQ), notPast)
	return e.mapc(MapMax, 0, name, n, above, tie)
}

// argmax is the index of the largest of x, the lower of two equal (the
// fast argmax: OpTopK of one, its index not a u32 row).
func (e sampleEmitter) argmax(x ValueID) ValueID {
	return e.b.Emit(OpTopK, []ValueID{x}, []Result{{Name: "sample.token", Width: 1}, {Name: "sample.token.v", Width: 1}}, nil, Attrs{TopK: 1}, nil, nil)[0]
}

// selection is the active candidate (a 0/1 over the row) at which key's
// order -- the larger first, the lower id of two equal; key at most 4 and
// never negative -- reaches target in w's mass: with reach false the first
// whose w up to and including it passes target, with reach true the first
// at which it is at least target; the order's last active candidate where
// none does. Its output is the candidates left active: that one alone. Each
// pass partitions the active ones and keeps
// the earlier part where the target is within its mass, else the later with
// the target less that mass: 8 over the exponents [-126, 129], 24 within
// the binade found (or [0, 2^-125) below the first), then one an id bit.
func (e sampleEmitter) selection(key, id, w, active, target ValueID, reach bool) ValueID {
	n := e.b.G.Values[key].Width
	// The scalars' zero, from the 0/1 mask: the target may be +Inf (top-p off).
	zero := e.one(OpReduce, "sample.sel.zero", 1, Attrs{Fn: MapMin}, e.mapc(MapLess, -1, "sample.sel.none", n, active))
	one := func(name string, c float64) ValueID {
		return e.mapc(MapAdd, c, name, 1, zero)
	}
	notv := func(name string, w int, x ValueID) ValueID {
		return e.mapc(MapAdd, 1, name, w, e.mapc(MapMul, -1, name+".neg", w, x))
	}
	pass := func(cond ValueID) ValueID {
		pref := e.mapc(MapMul, 0, "sample.sel.pref", n, active, cond)
		mass := e.one(OpReduce, "sample.sel.mass", 1, Attrs{Fn: MapAdd}, e.mapc(MapMul, 0, "sample.sel.w", n, pref, w))
		rest := e.mapc(MapSub, 0, "sample.sel.rest", n, active, pref)
		left := e.one(OpReduce, "sample.sel.left", 1, Attrs{Fn: MapAdd}, rest)
		in := e.mapc(MapLess, 0, "sample.sel.in", 1, target, mass)
		if reach {
			in = notv("sample.sel.in", 1, e.mapc(MapLess, 0, "sample.sel.past", 1, mass, target))
		}
		choose := e.mapc(MapMax, 0, "sample.sel.choose", 1, in, e.mapc(MapLess, 0.5, "sample.sel.empty", 1, left))
		active = e.mapc(MapAdd, 0, "sample.sel.active", n, rest,
			e.mapc(MapMul, 0, "sample.sel.move", n, e.mapc(MapSub, 0, "sample.sel.diff", n, pref, rest), choose))
		target = e.mapc(MapSub, 0, "sample.sel.target", 1, target,
			e.mapc(MapMul, 0, "sample.sel.taken", 1, notv("sample.sel.not", 1, choose), mass))
		return choose
	}
	atLeast := func(t ValueID) ValueID {
		return notv("sample.sel.atleast", n, e.mapc(MapLess, 0, "sample.sel.below", n, key, t))
	}
	// The binade: lo is 2^e of the exponents [e, e+2^k) still in play, the
	// split the upper half's first, a power of two times lo.
	lo := one("sample.sel.lo", math.Ldexp(1, -126))
	for k := 8; k >= 1; k-- {
		th := lo
		for r := 1 << (k - 1); r > 0; r -= min(r, 64) {
			th = e.mapc(MapMul, math.Ldexp(1, min(r, 64)), "sample.sel.split", 1, th)
		}
		lo = e.where("sample.sel.lo", 1, pass(atLeast(th)), th, lo)
	}
	// Within it by halves, from lo (0 below the first binade).
	bottom := e.mapc(MapLess, math.Ldexp(1, -125), "sample.sel.bottom", 1, lo)
	step := e.where("sample.sel.step", 1, bottom, one("sample.sel.sub", math.Ldexp(1, -126)), e.mapc(MapMul, 0.5, "sample.sel.half", 1, lo))
	lo = e.where("sample.sel.from", 1, bottom, one("sample.sel.zero", 0), lo)
	for range 24 {
		th := e.mapc(MapAdd, 0, "sample.sel.mid", 1, lo, step)
		lo = e.where("sample.sel.lo", 1, pass(atLeast(th)), th, lo)
		step = e.mapc(MapMul, 0.5, "sample.sel.step", 1, step)
	}
	// Equal keys, the lower id first.
	bits := 0
	for 1<<bits < n {
		bits++
	}
	base := one("sample.sel.idlo", 0)
	for k := bits - 1; k >= 0; k-- {
		th := e.mapc(MapAdd, math.Ldexp(1, k), "sample.sel.idmid", 1, base)
		choose := pass(e.mapc(MapLess, 0, "sample.sel.idbelow", n, id, th))
		base = e.mapc(MapAdd, 0, "sample.sel.idlo", 1, base,
			e.mapc(MapMul, math.Ldexp(1, k), "sample.sel.idstep", 1, notv("sample.sel.idnot", 1, choose)))
	}
	return active
}

// token is the id of the largest of score.
func (e sampleEmitter) token(ids, score ValueID) ValueID {
	best := e.b.Emit(OpTopK, []ValueID{score}, []Result{{Name: "sample.best", Width: 1, Elem: ElemU32}, {Name: "sample.best.v", Width: 1}}, nil, Attrs{TopK: 1}, nil, nil)
	return e.b.Emit(OpTake, []ValueID{ids, best[0]}, []Result{{Name: "sample.token", Width: 1, Elem: ElemU32}}, nil, Attrs{}, nil, nil)[0]
}

// MaskRow is a constrained draw's logits: each logit plus its token's bias,
// 0 where a constraint allows the token and -inf where it does not. Its
// inputs are the logits and the mask; its output the masked row.
type MaskRow struct {
	NVocab int
}

func (s *MaskRow) Build() *Graph {
	b := New()
	l := b.In("logits", s.NVocab)
	m := b.In("mask", s.NVocab)
	e := sampleEmitter{b}
	b.Out(e.mapc(MapAdd, 0, "masked", s.NVocab, l, m))
	return b.G
}

// Logprobs is a step's log-probabilities: the logits' log-softmax, each x -
// max - log(sum(exp(x - max))), and the K most likely tokens, the larger
// logit first and the lower id on a tie. Its input is the logits; its
// outputs the log row and the K ids.
type Logprobs struct {
	NVocab, K int
}

func (s *Logprobs) Build() *Graph {
	b := New()
	n := s.NVocab
	l := b.In("logits", n)
	e := sampleEmitter{b}
	d := e.mapc(MapSub, 0, "logprobs.d", n, l, e.one(OpReduce, "logprobs.max", 1, Attrs{Fn: MapMax}, l))
	sum := e.one(OpReduce, "logprobs.sum", 1, Attrs{Fn: MapAdd}, e.mapc(MapExp, 0, "logprobs.exp", n, d))
	row := e.mapc(MapSub, 0, "logprobs.row", n, d, e.mapc(MapLog, 0, "logprobs.lse", 1, sum))
	top := b.Emit(OpTopK, []ValueID{l}, []Result{{Name: "logprobs.ids", Width: min(s.K, n), Elem: ElemU32}, {Name: "logprobs.vals", Width: min(s.K, n)}}, nil,
		Attrs{TopK: min(s.K, n)}, nil, nil)
	b.Out(row, top[0])
	return b.G
}

// Pick is the label pick over the head's logits: the logits at N ids, the
// graph's second input (an id past the row read as the row's last).
type Pick struct {
	NVocab, N int
}

func (p *Pick) Build() *Graph {
	b := New()
	l := b.In("logits", p.NVocab)
	ids := b.In("pick.ids", p.N)
	b.G.Values[ids].Elem = ElemU32
	b.Out(b.Emit(OpTake, []ValueID{l, ids}, []Result{{Name: "picked", Width: p.N}}, nil, Attrs{}, nil, nil)[0])
	return b.G
}
