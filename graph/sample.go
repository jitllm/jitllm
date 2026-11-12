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
// them: of the first k, typical-p, then top-p and min-p over the survivors'
// softmax at temperature 1, the best survivor always kept, then the
// inverse-CDF walk at a uniform u the caller draws over the survivors'
// softmax at the temperature -- the first survivor whose mass up to and
// including it passes u times the survivors' mass, the last survivor where
// rounding leaves none. XTC cuts the survivors before the walk
// (sampleEmitter.xtc), and a dynamic temperature replaces 1/T over what is
// left (sampleEmitter.dynTemp). Its inputs are SampleTopK's, then
// sample.walk (k, 1/T, top-p +Inf where off, min-p 0 where off, u,
// typical-p +Inf where off, XTC's threshold +Inf where off, XTC's
// probability, which only Prob reads, then the dynamic temperature's range,
// on above 0, and exponent); its output the token. With Prob the distribution is the mixture
// the draw is: the XTC cut's at its probability, the survivors' before it at
// the rest. The mass before each survivor is a KxK sum,
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
	walk := b.In("sample.walk", 10)
	K := b.G.Values[vals].Width
	e := sampleEmitter{b}
	place, d, cut := e.cuts(vals, walk)
	keep := e.xtc(e.mapc(MapExp, 0, "sample.xtc.ed", K, d), cut, place, e.word(walk, 6, "sample.xtc"))
	invT, u := e.word(walk, 1, "sample.invtemp"), e.word(walk, 4, "sample.u")
	rng, ex := e.word(walk, 8, "sample.dyn.range"), e.word(walk, 9, "sample.dyn.exp")
	ed := e.mapc(MapExp, 0, "sample.dyn.ed", K, d)
	edt := e.mapc(MapExp, 0, "sample.etd", K, e.mapc(MapMul, 0, "sample.dt", K, d, e.dynTemp(ed, keep, invT, rng, ex, "sample.dyn")))
	et := e.mapc(MapMul, 0, "sample.et", K, edt, keep)
	mass := e.one(OpReduce, "sample.mass", 1, Attrs{Fn: MapAdd}, et)
	if s.Prob {
		pt := e.mapc(MapDiv, 0, "sample.pt", K, et, mass)
		edc := e.mapc(MapExp, 0, "sample.etd.cut", K, e.mapc(MapMul, 0, "sample.dt.cut", K, d, e.dynTemp(ed, cut, invT, rng, ex, "sample.dyn.cut")))
		ec := e.mapc(MapMul, 0, "sample.et.cut", K, edc, cut)
		pc := e.mapc(MapDiv, 0, "sample.pt.cut", K, ec, e.one(OpReduce, "sample.mass.cut", 1, Attrs{Fn: MapAdd}, ec))
		mix := e.word(walk, 7, "sample.xtc.p")
		b.Out(e.mapc(MapAdd, 0, "sample.pt.mix", K, pc, e.mapc(MapMul, 0, "sample.pt.mixd", K, e.mapc(MapSub, 0, "sample.pt.d", K, pt, pc), mix)), ids)
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
// anywhere in the row, the draw is SampleRow's. With Typical, sample.cut's
// last word is typical-p, and before the cuts the row keeps its locally
// typical tokens (llama.cpp's typical sampler): those whose surprise -log p
// is nearest the row's entropy, in that order, up to the first whose mass up
// to and including it passes typical-p; the cuts read their softmax
// renormalized, and the survivors are walked whether or not a cut fires.
// Before them all, top-n-sigma: every logit under the largest less n
// standard deviations of the row's finite logits goes to -Inf. sample.cut's
// words past its base are n (on above 0), typical-p, XTC's threshold, its
// probability, and the dynamic temperature's range (on above 0) and
// exponent, which sets the walk's temperature over the survivors
// (sampleEmitter.dynTemp). XTC follows the cuts: of the survivors at or above the threshold
// in their softmax, all but the last in the order, where two or more are;
// with Prob the distribution is the mixture, XTC's at its probability. Its inputs are SampleRow's,
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
	NVocab              int
	Prob, TopK, Typical bool
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
	// x is the first of the Typical forms' words.
	x := words
	if s.Typical {
		words += 6
	}
	cut := b.In("sample.cut", words)
	e := sampleEmitter{b}
	v := penalizedRow(b, l, win, pen)
	invT, topp, minp, u := e.word(cut, 0, "sample.invtemp"), e.word(cut, 1, "sample.topp"), e.word(cut, 2, "sample.minp"), e.word(cut, 3, "sample.u")
	id := e.mapc(MapMul, 1, "sample.idf", n, ids)
	notv := func(name string, w int, x ValueID) ValueID {
		return e.mapc(MapAdd, 1, name, w, e.mapc(MapMul, -1, name+".neg", w, x))
	}
	if s.Typical {
		// Top-n-sigma: the finite logits' mean and deviation, the threshold
		// the largest less n of them (-Inf where n is not above 0).
		ns := e.word(cut, x, "sample.sigma")
		fin := notv("sample.sig.fin", n, e.mapc(MapLess, -math.MaxFloat32, "sample.sig.inf", n, v))
		cnt := e.one(OpReduce, "sample.sig.count", 1, Attrs{Fn: MapAdd}, fin)
		mean := e.mapc(MapDiv, 0, "sample.sig.mean", 1, e.one(OpReduce, "sample.sig.sum", 1, Attrs{Fn: MapAdd},
			e.mapc(MapWhere, 0, "sample.sig.fv", n, fin, v)), cnt)
		dv := e.mapc(MapWhere, 0, "sample.sig.dv", n, fin, e.mapc(MapSub, 0, "sample.sig.d", n, v, mean))
		dev := e.mapc(MapSqrt, 0, "sample.sig.dev", 1, e.mapc(MapDiv, 0, "sample.sig.var", 1,
			e.one(OpReduce, "sample.sig.ss", 1, Attrs{Fn: MapAdd}, e.mapc(MapMul, 0, "sample.sig.d2", n, dv, dv)), cnt))
		vmax := e.one(OpReduce, "sample.sig.max", 1, Attrs{Fn: MapMax}, v)
		on := notv("sample.sig.on", 1, e.mapc(MapLess, math.SmallestNonzeroFloat32, "sample.sig.off", 1, ns))
		thr := e.mapc(MapWhere, math.Inf(-1), "sample.sig.thr", 1, on,
			e.mapc(MapSub, 0, "sample.sig.edge", 1, vmax, e.mapc(MapMul, 0, "sample.sig.ndev", 1, ns, dev)))
		v = e.mapc(MapWhere, math.Inf(-1), "sample.sig.v", n, notv("sample.sig.keep", n, e.mapc(MapLess, 0, "sample.sig.under", n, v, thr)), v)
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
	if s.Typical {
		// The pool's entropy H, each token's key exp(-|log q1 + H|) -- the
		// nearest the entropy first -- and the pool cut where the mass up to
		// one passes typical-p, q1 renormalized over what is left.
		typ := e.word(cut, x+1, "sample.typical")
		lq := e.mapc(MapLog, 0, "sample.typ.log", n, q1)
		plogp := e.mapc(MapWhere, 0, "sample.typ.plogp", n, q1, e.mapc(MapMul, 0, "sample.typ.pl", n, q1, lq))
		h := e.mapc(MapMul, -1, "sample.typ.h", 1, e.one(OpReduce, "sample.typ.hsum", 1, Attrs{Fn: MapAdd}, plogp))
		key := e.mapc(MapExp, 0, "sample.typ.key", n, e.mapc(MapMul, -1, "sample.typ.shiftneg", n,
			e.mapc(MapAbs, 0, "sample.typ.shift", n, e.mapc(MapAdd, 0, "sample.typ.d", n, lq, h))))
		edge := e.selection(key, id, q1, pool, typ, false)
		pool = e.mapc(MapMul, 0, "sample.typ.keep", n, e.upTo(key, id, edge, "sample.typ.edge"), pool)
		kq := e.mapc(MapMul, 0, "sample.typ.q", n, q1, pool)
		q1 = e.mapc(MapDiv, 0, "sample.typ.qn", n, kq, e.one(OpReduce, "sample.typ.qsum", 1, Attrs{Fn: MapAdd}, kq))
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
	if s.TopK || s.Typical {
		// A top-k or typical draw walks its survivors whether or not a cut
		// fires.
		fired = e.mapc(MapLess, 2, "sample.fired.k", 1, fired)
	}
	// The survivors' softmax at the temperature, walked in the cut's order.
	scaled := e.mapc(MapMul, 0, "sample.scaled", n, v, invT)
	share := func(keep ValueID, name string) ValueID {
		scaled := scaled
		if s.Typical {
			it := e.dynTemp(q1, keep, invT, e.word(cut, x+4, name+".dyn.range"), e.word(cut, x+5, name+".dyn.exp"), name+".dyn")
			scaled = e.mapc(MapMul, 0, name+".scaled", n, v, it)
		}
		top := e.one(OpReduce, name+".top", 1, Attrs{Fn: MapMax}, e.mapc(MapWhere, math.Inf(-1), name+".keptscaled", n, keep, scaled))
		et := e.mapc(MapWhere, 0, name+".et", n, keep, e.mapc(MapExp, 0, name+".ete", n, e.mapc(MapSub, 0, name+".etd", n, scaled, top)))
		return e.mapc(MapDiv, 0, name+".pt", n, et, e.one(OpReduce, name+".etsum", 1, Attrs{Fn: MapAdd}, et))
	}
	pt := share(keep, "sample")
	if s.Typical {
		// XTC: of the survivors' softmax at temperature 1, those at or above
		// the threshold but the last in the order -- the least, the higher id
		// of two equal.
		thr := e.word(cut, x+2, "sample.xtc")
		kq := e.mapc(MapMul, 0, "sample.xtc.q", n, q1, keep)
		ps := e.mapc(MapDiv, 0, "sample.xtc.p", n, kq, e.one(OpReduce, "sample.xtc.sum", 1, Attrs{Fn: MapAdd}, kq))
		above := e.mapc(MapMul, 0, "sample.xtc.above", n, keep, notv("sample.xtc.notbelow", n, e.mapc(MapLess, 0, "sample.xtc.below", n, ps, thr)))
		least := e.one(OpReduce, "sample.xtc.least", 1, Attrs{Fn: MapMin}, e.mapc(MapWhere, math.Inf(1), "sample.xtc.qs", n, above, q1))
		atLeast := e.mapc(MapMul, 0, "sample.xtc.atleast", n, above, e.mapc(MapEqual, 0, "sample.xtc.isleast", n, q1, least))
		lastID := e.one(OpReduce, "sample.xtc.lastid", 1, Attrs{Fn: MapMax}, e.mapc(MapWhere, -1, "sample.xtc.ids", n, atLeast, id))
		last := e.mapc(MapMul, 0, "sample.xtc.last", n, atLeast, e.mapc(MapEqual, 0, "sample.xtc.islast", n, id, lastID))
		cutPt := pt
		keep = e.mapc(MapAdd, 0, "sample.xtc.keep", n, e.mapc(MapSub, 0, "sample.xtc.rest", n, keep, above), last)
		pt = share(keep, "sample.xtc")
		if s.Prob {
			mix := e.word(cut, x+3, "sample.xtc.mix")
			pt = e.mapc(MapAdd, 0, "sample.pt.mix", n, cutPt, e.mapc(MapMul, 0, "sample.pt.mixd", n, e.mapc(MapSub, 0, "sample.pt.mixe", n, pt, cutPt), mix))
		}
	}
	q := e.softmax(scaled, "sample.q")
	if s.Prob {
		b.Out(e.mapc(MapAdd, 0, "sample.prob", n, q, e.mapc(MapMul, 0, "sample.prob.cut", n, e.mapc(MapSub, 0, "sample.prob.d", n, pt, q), fired)))
		return b.G
	}
	walkT := e.mapc(MapMul, 0, "sample.walk.target", 1, u, e.one(OpReduce, "sample.ptsum", 1, Attrs{Fn: MapAdd}, pt))
	walk := e.selection(order, id, pt, keep, walkT, false)
	if s.TopK || s.Typical {
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

// SampleMirostat is llama.cpp's Mirostat draw: over the logits' softmax at
// the temperature, in its order (the larger first, the lower id of two
// equal), version 2 keeps the tokens whose surprise -log2 p is at most mu,
// the best always; version 1 (V1) the k most likely, k the integer part of
// ((e 2^mu)/(1 - N^-e))^(1/s), at least 1 and at most the row, where s is
// the Zipf exponent its m = 100 most likely estimate and e = s - 1. The
// survivors' renormalized probabilities are walked at a uniform u the caller
// draws, as SampleRow walks the row. Its inputs are the logits,
// sample.window and sample.penalty (read as SampleTopK reads them; a
// Mirostat draw passes them off), sample.ids and sample.miro (1/T, tau, eta,
// mu, then u); its outputs the token and mu less eta times the token's
// surprise past tau.
type SampleMirostat struct {
	NVocab int
	V1     bool
}

func (s *SampleMirostat) Build() *Graph {
	b := New()
	n := s.NVocab
	l := b.In("logits", n)
	win := b.In("sample.window", n)
	pen := b.In("sample.penalty", 3)
	ids := b.In("sample.ids", n)
	b.G.Values[ids].Elem = ElemU32
	miro := b.In("sample.miro", 5)
	e := sampleEmitter{b}
	invT, tau, eta, mu, u := e.word(miro, 0, "miro.invtemp"), e.word(miro, 1, "miro.tau"), e.word(miro, 2, "miro.eta"),
		e.word(miro, 3, "miro.mu"), e.word(miro, 4, "miro.u")
	v := penalizedRow(b, l, win, pen)
	id := e.mapc(MapMul, 1, "miro.idf", n, ids)
	q := e.softmax(e.mapc(MapMul, 0, "miro.scaled", n, v, invT), "miro.q")
	all := e.mapc(MapLess, 2, "miro.all", n, q)
	var keep ValueID
	if s.V1 {
		m := min(100, n)
		top := b.Emit(OpTopK, []ValueID{q}, []Result{{Name: "miro.top.ids", Width: m, Elem: ElemU32}, {Name: "miro.top", Width: m}}, nil,
			Attrs{TopK: m}, nil, nil)[1]
		pi := e.one(OpSlice, "miro.pi", m-1, Attrs{Offset: 0, Width: m - 1}, top)
		pn := e.one(OpSlice, "miro.pn", m-1, Attrs{Offset: 1, Width: m - 1}, top)
		bi := e.mapc(MapLog, 0, "miro.b", m-1, e.mapc(MapDiv, 0, "miro.ratio", m-1, pi, pn))
		ii := e.one(OpSlice, "miro.i", m-1, Attrs{Offset: 0, Width: m - 1}, id)
		ti := e.mapc(MapLog, 0, "miro.t", m-1, e.mapc(MapDiv, 0, "miro.tr", m-1, e.mapc(MapAdd, 2, "miro.i2", m-1, ii), e.mapc(MapAdd, 1, "miro.i1", m-1, ii)))
		sh := e.mapc(MapDiv, 0, "miro.s", 1, e.one(OpReduce, "miro.tb", 1, Attrs{Fn: MapAdd}, e.mapc(MapMul, 0, "miro.tbi", m-1, ti, bi)),
			e.one(OpReduce, "miro.tt", 1, Attrs{Fn: MapAdd}, e.mapc(MapMul, 0, "miro.tti", m-1, ti, ti)))
		eps := e.mapc(MapAdd, -1, "miro.eps", 1, sh)
		num := e.mapc(MapMul, 0, "miro.num", 1, eps, e.mapc(MapExp, 0, "miro.pow2", 1, e.mapc(MapMul, math.Ln2, "miro.mu2", 1, mu)))
		den := e.mapc(MapAdd, 1, "miro.den", 1, e.mapc(MapMul, -1, "miro.denn", 1,
			e.mapc(MapExp, 0, "miro.npow", 1, e.mapc(MapMul, -math.Log(float64(n)), "miro.nlog", 1, eps))))
		k := e.mapc(MapExp, 0, "miro.k", 1, e.mapc(MapDiv, 0, "miro.klog", 1, e.mapc(MapLog, 0, "miro.kl", 1, e.mapc(MapDiv, 0, "miro.kb", 1, num, den)), sh))
		// The integer part where k is past 1 (-k < -1, false for no number),
		// else 1; at most the row.
		kk := e.mapc(MapMin, float64(n), "miro.kk", 1, e.mapc(MapRound, 0, "miro.kfloor", 1, e.mapc(MapAdd, -0.5, "miro.kh", 1,
			e.mapc(MapWhere, 1, "miro.kok", 1, e.mapc(MapLess, -1, "miro.kgt", 1, e.mapc(MapMul, -1, "miro.kneg", 1, k)), k))))
		kth := e.selection(q, id, all, all, e.mapc(MapAdd, -1, "miro.k.target", 1, kk), false)
		keep = e.upTo(q, id, kth, "miro.topk")
	} else {
		// At most mu bits: p at least 2^-mu, the best always.
		floor := e.mapc(MapExp, 0, "miro.floor", 1, e.mapc(MapMul, -math.Ln2, "miro.mubits", 1, mu))
		in := e.mapc(MapAdd, 1, "miro.in", n, e.mapc(MapMul, -1, "miro.out", n, e.mapc(MapLess, 0, "miro.below", n, q, floor)))
		p0 := e.one(OpReduce, "miro.p0", 1, Attrs{Fn: MapMax}, q)
		atBest := e.mapc(MapEqual, 0, "miro.atbest", n, q, p0)
		bestID := e.one(OpReduce, "miro.bestid", 1, Attrs{Fn: MapMin}, e.mapc(MapWhere, math.Inf(1), "miro.bestids", n, atBest, id))
		best := e.mapc(MapMul, 0, "miro.best", n, atBest, e.mapc(MapEqual, 0, "miro.isbest", n, id, bestID))
		keep = e.mapc(MapMax, 0, "miro.keep", n, in, best)
	}
	kq := e.mapc(MapMul, 0, "miro.kq", n, q, keep)
	pt := e.mapc(MapDiv, 0, "miro.pt", n, kq, e.one(OpReduce, "miro.ksum", 1, Attrs{Fn: MapAdd}, kq))
	walkT := e.mapc(MapMul, 0, "miro.walk.target", 1, u, e.one(OpReduce, "miro.ptsum", 1, Attrs{Fn: MapAdd}, pt))
	chosen := e.selection(q, id, pt, keep, walkT, false)
	ptok := e.one(OpReduce, "miro.ptok", 1, Attrs{Fn: MapAdd}, e.mapc(MapMul, 0, "miro.ptokv", n, pt, chosen))
	// Its surprise in bits, and mu less eta times its excess over tau.
	bits := e.mapc(MapMul, -1/math.Ln2, "miro.bits", 1, e.mapc(MapLog, 0, "miro.ln", 1, ptok))
	mu2 := e.mapc(MapSub, 0, "miro.mu.next", 1, mu, e.mapc(MapMul, 0, "miro.step", 1, eta, e.mapc(MapSub, 0, "miro.err", 1, bits, tau)))
	b.Out(e.argmax(chosen), mu2)
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

// dynTemp is the reciprocal temperature a draw over the survivors keep
// (0/1, w their weights at temperature 1, any scale) runs at: invT, or where
// rng is above 0 and two or more survive llama.cpp's dynamic temperature,
// lo + (hi - lo)*(H/Hmax)^ex with T = 1/invT, lo = max(0, T - rng), hi = T +
// rng, H the survivors' entropy and Hmax the log of their count; its
// reciprocal held to 1e30, which a temperature of 0 makes greedy.
func (e sampleEmitter) dynTemp(w, keep, invT, rng, ex ValueID, name string) ValueID {
	K := e.b.G.Values[w].Width
	wk := e.mapc(MapMul, 0, name+".w", K, w, keep)
	ps := e.mapc(MapDiv, 0, name+".p", K, wk, e.one(OpReduce, name+".sum", 1, Attrs{Fn: MapAdd}, wk))
	plogp := e.mapc(MapWhere, 0, name+".plogp", K, ps, e.mapc(MapMul, 0, name+".pl", K, ps, e.mapc(MapLog, 0, name+".log", K, ps)))
	h := e.mapc(MapMul, -1, name+".h", 1, e.one(OpReduce, name+".hsum", 1, Attrs{Fn: MapAdd}, plogp))
	size := e.one(OpReduce, name+".size", 1, Attrs{Fn: MapAdd}, keep)
	norm := e.mapc(MapDiv, 0, name+".norm", 1, h, e.mapc(MapLog, 0, name+".hmax", 1, size))
	pw := e.mapc(MapWhere, 0, name+".pow", 1, norm, e.mapc(MapExp, 0, name+".powe", 1,
		e.mapc(MapMul, 0, name+".powl", 1, ex, e.mapc(MapLog, 0, name+".normlog", 1, norm))))
	t := e.mapc(MapRecip, 0, name+".t", 1, invT)
	lo := e.mapc(MapMax, 0, name+".lo", 1, e.mapc(MapSub, 0, name+".lod", 1, t, rng), e.mapc(MapMul, 0, name+".zero", 1, rng, e.mapc(MapLess, -1, name+".nil", 1, rng)))
	span := e.mapc(MapSub, 0, name+".span", 1, e.mapc(MapAdd, 0, name+".hi", 1, t, rng), lo)
	dyn := e.mapc(MapAdd, 0, name+".dyn", 1, lo, e.mapc(MapMul, 0, name+".dynd", 1, span, pw))
	inv := e.mapc(MapMin, 1e30, name+".inv", 1, e.mapc(MapRecip, 0, name+".invd", 1, dyn))
	not := func(x ValueID, nm string) ValueID {
		return e.mapc(MapAdd, 1, nm, 1, e.mapc(MapMul, -1, nm+".neg", 1, x))
	}
	on := e.mapc(MapMul, 0, name+".on", 1, not(e.mapc(MapLess, 1.5, name+".single", 1, size), name+".multi"),
		not(e.mapc(MapLess, math.SmallestNonzeroFloat32, name+".rangeoff", 1, rng), name+".ranged"))
	return e.where(name+".invt", 1, on, inv, invT)
}

// xtc is keep less XTC's cut at threshold thr (+Inf where off): of the
// survivors' softmax at temperature 1 (w their unnormalized weights, keep
// 0/1), those at or above thr, every one but the last in place, where two or
// more are.
func (e sampleEmitter) xtc(w, keep, place, thr ValueID) ValueID {
	K := e.b.G.Values[w].Width
	wk := e.mapc(MapMul, 0, "sample.xtc.w", K, w, keep)
	ps := e.mapc(MapDiv, 0, "sample.xtc.p", K, wk, e.one(OpReduce, "sample.xtc.sum", 1, Attrs{Fn: MapAdd}, wk))
	above := e.mapc(MapMul, 0, "sample.xtc.above", K, keep,
		e.mapc(MapAdd, 1, "sample.xtc.notbelow", K, e.mapc(MapMul, -1, "sample.xtc.below.neg", K, e.mapc(MapLess, 0, "sample.xtc.below", K, ps, thr))))
	last := e.one(OpReduce, "sample.xtc.last", 1, Attrs{Fn: MapMax}, e.mapc(MapWhere, -1, "sample.xtc.places", K, above, place))
	drop := e.mapc(MapMul, 0, "sample.xtc.drop", K, above, e.mapc(MapLess, 0, "sample.xtc.before", K, place, last))
	return e.mapc(MapSub, 0, "sample.xtc.keep", K, keep, drop)
}

// cuts is the K candidates' (vals, best first) places, their distances from
// the best, and whether each survives: among the first k (params' word 0),
// typical-p over their softmax at temperature 1 (word 5, +Inf where off: in
// the order of their surprise's distance from the entropy, the nearest
// first, a candidate survives while the mass before it is at most
// typical-p, the nearest always), then over the survivors' softmax top-p
// (word 2: while the mass before it is under top-p) and min-p (word 3: while
// its probability is at least min-p times the best survivor's), the best
// survivor always.
func (e sampleEmitter) cuts(vals, params ValueID) (place, d, keep ValueID) {
	K := e.b.G.Values[vals].Width
	k, topp, minp := e.word(params, 0, "sample.k"), e.word(params, 2, "sample.topp"), e.word(params, 3, "sample.minp")
	typ := e.word(params, 5, "sample.typical")
	rank := e.b.Emit(OpRank, []ValueID{vals}, []Result{{Name: "sample.rank", Width: K, Elem: ElemU32}}, nil, Attrs{}, nil, nil)[0]
	place = e.mapc(MapMul, 1, "sample.place.f", K, rank)
	inK := e.mapc(MapLess, 0, "sample.ink", K, place, k)
	v0 := e.one(OpReduce, "sample.v0", 1, Attrs{Fn: MapMax}, vals)
	d = e.mapc(MapSub, 0, "sample.d", K, vals, v0)
	ex := e.mapc(MapMul, 0, "sample.e", K, e.mapc(MapExp, 0, "sample.ed", K, d), inK)
	p := e.mapc(MapDiv, 0, "sample.p", K, ex, e.one(OpReduce, "sample.sum", 1, Attrs{Fn: MapAdd}, ex))
	// Typical-p: the entropy, each candidate's key -|log p + H| (-Inf past
	// the k), their order by it and the mass before each in that order.
	lp := e.mapc(MapLog, 0, "sample.typ.log", K, p)
	plogp := e.mapc(MapWhere, 0, "sample.typ.plogp", K, p, e.mapc(MapMul, 0, "sample.typ.pl", K, p, lp))
	h := e.mapc(MapMul, -1, "sample.typ.h", 1, e.one(OpReduce, "sample.typ.hsum", 1, Attrs{Fn: MapAdd}, plogp))
	key := e.mapc(MapWhere, math.Inf(-1), "sample.typ.key", K, inK, e.mapc(MapMul, -1, "sample.typ.shiftneg", K,
		e.mapc(MapAbs, 0, "sample.typ.shift", K, e.mapc(MapAdd, 0, "sample.typ.d", K, lp, h))))
	trank := e.b.Emit(OpRank, []ValueID{key}, []Result{{Name: "sample.typ.rank", Width: K, Elem: ElemU32}}, nil, Attrs{}, nil, nil)[0]
	tplace := e.mapc(MapMul, 1, "sample.typ.place", K, trank)
	tbefore := e.before(p, tplace, "sample.typ")
	// over is before > typical-p, as -before < -typical-p.
	over := e.mapc(MapLess, 0, "sample.typ.over", K, e.mapc(MapMul, -1, "sample.typ.before.neg", K, tbefore),
		e.mapc(MapMul, -1, "sample.typ.neg", 1, typ))
	keepT := e.mapc(MapMax, 0, "sample.typ.keep", K,
		e.mapc(MapMul, 0, "sample.typ.keepk", K, e.mapc(MapAdd, 1, "sample.typ.notover", K, e.mapc(MapMul, -1, "sample.typ.over.neg", K, over)), inK),
		e.mapc(MapLess, 1, "sample.typ.first", K, tplace))
	// The survivors renormalized where typical-p is on (under 2), p as it
	// is where it is off.
	pk := e.mapc(MapMul, 0, "sample.typ.p", K, p, keepT)
	on := e.mapc(MapLess, 2, "sample.typ.on", 1, typ)
	p = e.mapc(MapAdd, 0, "sample.p2", K, p, e.mapc(MapMul, 0, "sample.p2.on", K,
		e.mapc(MapSub, 0, "sample.p2.d", K, e.mapc(MapDiv, 0, "sample.typ.pn", K, pk, e.one(OpReduce, "sample.typ.psum", 1, Attrs{Fn: MapAdd}, pk)), p), on))
	keepP := e.mapc(MapLess, 0, "sample.keep.p", K, e.before(p, place, "sample"), topp)
	floor := e.mapc(MapMul, 0, "sample.floor", 1, e.one(OpReduce, "sample.pmax", 1, Attrs{Fn: MapMax}, p), minp)
	under := e.mapc(MapLess, 0, "sample.under", K, p, floor)
	keepM := e.mapc(MapAdd, 1, "sample.keep.m", K, e.mapc(MapMul, -1, "sample.under.neg", K, under))
	// The best survivor: the lowest place typical-p keeps.
	fp := e.one(OpReduce, "sample.first.place", 1, Attrs{Fn: MapMin}, e.mapc(MapWhere, math.Inf(1), "sample.first.places", K, keepT, place))
	first := e.mapc(MapEqual, 0, "sample.first", K, place, fp)
	keep = e.mapc(MapMax, 0, "sample.keep", K, e.mapc(MapMul, 0, "sample.keep.pmk", K, e.mapc(MapMul, 0, "sample.keep.pm", K, keepP, keepM), keepT), first)
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
	in := func(i int) Arg { return Arg{Kind: ArgIn, Index: i} }
	term := func(i int) Arg { return Arg{Kind: ArgTerm, Index: i} }
	cst := func(c float64) Arg { return Arg{Kind: ArgConst, Const: c} }
	expr := func(name string, w int, ins []ValueID, terms ...Term) ValueID {
		return e.b.Emit(OpExpr, ins, []Result{{Name: name, Width: w}}, nil, Attrs{Expr: terms}, nil, nil)[0]
	}
	// A pass is nine ops whatever the partition: each step is one
	// expression, its terms in the order of the maps it stands for, so the
	// words are theirs. pref is active*cond, rest active - pref; the earlier
	// part is kept where the target is within its mass (reach: its mass is
	// not below the target) or nothing is left past it.
	pass := func(cond ValueID) ValueID {
		mass := e.one(OpReduce, "sample.sel.mass", 1, Attrs{Fn: MapAdd}, expr("sample.sel.w", n, []ValueID{active, cond, w},
			Term{Fn: MapMul, Args: [3]Arg{in(0), in(1)}}, Term{Fn: MapMul, Args: [3]Arg{term(0), in(2)}}))
		left := e.one(OpReduce, "sample.sel.left", 1, Attrs{Fn: MapAdd}, expr("sample.sel.rest", n, []ValueID{active, cond},
			Term{Fn: MapMul, Args: [3]Arg{in(0), in(1)}}, Term{Fn: MapSub, Args: [3]Arg{in(0), term(0)}}))
		within := []Term{{Fn: MapLess, Args: [3]Arg{in(0), in(1)}}}
		if reach {
			within = []Term{{Fn: MapLess, Args: [3]Arg{in(1), in(0)}}, {Fn: MapMul, Args: [3]Arg{term(0), cst(-1)}},
				{Fn: MapAdd, Args: [3]Arg{term(1), cst(1)}}}
		}
		nw := len(within)
		choose := expr("sample.sel.choose", 1, []ValueID{target, mass, left}, append(within,
			Term{Fn: MapLess, Args: [3]Arg{in(2), cst(0.5)}}, Term{Fn: MapMax, Args: [3]Arg{term(nw - 1), term(nw)}})...)
		active = expr("sample.sel.active", n, []ValueID{active, cond, choose},
			Term{Fn: MapMul, Args: [3]Arg{in(0), in(1)}}, Term{Fn: MapSub, Args: [3]Arg{in(0), term(0)}},
			Term{Fn: MapSub, Args: [3]Arg{term(0), term(1)}}, Term{Fn: MapMul, Args: [3]Arg{term(2), in(2)}},
			Term{Fn: MapAdd, Args: [3]Arg{term(1), term(3)}})
		target = expr("sample.sel.target", 1, []ValueID{target, choose, mass},
			Term{Fn: MapMul, Args: [3]Arg{in(1), cst(-1)}}, Term{Fn: MapAdd, Args: [3]Arg{term(0), cst(1)}},
			Term{Fn: MapMul, Args: [3]Arg{term(1), in(2)}}, Term{Fn: MapSub, Args: [3]Arg{in(0), term(2)}})
		return choose
	}
	// atLeast is the candidates whose key is at least the split the terms
	// over ins make (their last), and moved lo where the pass keeps them.
	atLeast := func(ins []ValueID, split []Term) ValueID {
		s := len(split) - 1
		return expr("sample.sel.atleast", n, append([]ValueID{key}, ins...), append(shiftIns(split, 1),
			Term{Fn: MapLess, Args: [3]Arg{in(0), term(s)}}, Term{Fn: MapMul, Args: [3]Arg{term(s + 1), cst(-1)}},
			Term{Fn: MapAdd, Args: [3]Arg{term(s + 2), cst(1)}})...)
	}
	moved := func(choose ValueID, lo ValueID, ins []ValueID, split []Term) ValueID {
		s := len(split) - 1
		return expr("sample.sel.lo", 1, append([]ValueID{choose, lo}, ins...), append(shiftIns(split, 2),
			Term{Fn: MapWhere, Args: [3]Arg{in(0), term(s), in(1)}})...)
	}
	// The binade: lo is 2^e of the exponents [e, e+2^k) still in play, the
	// split the upper half's first, a power of two times lo.
	lo := one("sample.sel.lo", math.Ldexp(1, -126))
	for k := 8; k >= 1; k-- {
		var split []Term
		for r := 1 << (k - 1); r > 0; r -= min(r, 64) {
			x := in(0)
			if len(split) > 0 {
				x = term(len(split) - 1)
			}
			split = append(split, Term{Fn: MapMul, Args: [3]Arg{x, cst(math.Ldexp(1, min(r, 64)))}})
		}
		lo = moved(pass(atLeast([]ValueID{lo}, split)), lo, []ValueID{lo}, split)
	}
	// Within it by halves, from lo (0 below the first binade): the j-th
	// split is lo plus the first half-width over 2^j, a power of two halved
	// as the halvings it stands for round it.
	bottom := e.mapc(MapLess, math.Ldexp(1, -125), "sample.sel.bottom", 1, lo)
	step := e.where("sample.sel.step", 1, bottom, one("sample.sel.sub", math.Ldexp(1, -126)), e.mapc(MapMul, 0.5, "sample.sel.half", 1, lo))
	lo = e.where("sample.sel.from", 1, bottom, one("sample.sel.zero", 0), lo)
	for j := range 24 {
		split := []Term{{Fn: MapMul, Args: [3]Arg{in(1), cst(math.Ldexp(1, -j))}}, {Fn: MapAdd, Args: [3]Arg{in(0), term(0)}}}
		lo = moved(pass(atLeast([]ValueID{lo, step}, split)), lo, []ValueID{lo, step}, split)
	}
	// Equal keys, the lower id first.
	bits := 0
	for 1<<bits < n {
		bits++
	}
	base := one("sample.sel.idlo", 0)
	for k := bits - 1; k >= 0; k-- {
		choose := pass(expr("sample.sel.idbelow", n, []ValueID{id, base},
			Term{Fn: MapAdd, Args: [3]Arg{in(1), cst(math.Ldexp(1, k))}}, Term{Fn: MapLess, Args: [3]Arg{in(0), term(0)}}))
		base = expr("sample.sel.idlo", 1, []ValueID{base, choose},
			Term{Fn: MapMul, Args: [3]Arg{in(1), cst(-1)}}, Term{Fn: MapAdd, Args: [3]Arg{term(0), cst(1)}},
			Term{Fn: MapMul, Args: [3]Arg{term(1), cst(math.Ldexp(1, k))}}, Term{Fn: MapAdd, Args: [3]Arg{in(0), term(2)}})
	}
	return active
}

// shiftIns is terms with every operand In[j] made In[j+by]: terms written
// over their own inputs, placed after by others.
func shiftIns(terms []Term, by int) []Term {
	out := append([]Term(nil), terms...)
	for i := range out {
		for a := range out[i].Args {
			if out[i].Args[a].Kind == ArgIn {
				out[i].Args[a].Index += by
			}
		}
	}
	return out
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
