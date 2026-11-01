package graph

import (
	"fmt"
	"math"
)

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

// SampleTopK is the sampler's candidates: the repeat penalty over a history
// of at most Hist ids, applied once an entry as the host's sampler applies
// it (a positive logit divided by the penalty, correctly rounded, any other
// multiplied), then the K largest of the penalized row, largest first and
// the lower id on a tie. Its inputs are the logits, the sampler's argument
// words (a count, the penalty's bits, then the history's ids, an id no row
// has past the count) and the penalty; its outputs the K values and their
// ids. A request for k <= K reads the first k, and none of the form depends
// on k or on the history's length: a penalty is above one and never raises
// a value, so the K largest after it are among the K+Hist largest before it,
// and those alone are penalized.
type SampleTopK struct {
	NVocab, K, Hist int
}

// sampleSteps is how many of the history's entries one expression of the
// penalty applies.
const sampleSteps = 16

func (s *SampleTopK) Build() *Graph {
	b := New()
	vals, ids := s.emit(b)
	b.Out(vals, ids)
	return b.G
}

// emit is SampleTopK on b: its inputs, then the K values and their ids.
func (s *SampleTopK) emit(b *Builder) (vals, ids ValueID) {
	C := min(s.K+s.Hist, s.NVocab)
	K := min(s.K, C)
	l := b.In("logits", s.NVocab)
	args := b.In("sample.args", 2+s.Hist)
	b.G.Values[args].Elem = ElemU32
	pen := b.In("sample.penalty", 1)
	u32 := func(name string, w int) Result { return Result{Name: name, Width: w, Elem: ElemU32} }
	f32 := func(name string, w int) Result { return Result{Name: name, Width: w} }
	hist := b.Emit(OpSlice, []ValueID{args}, []Result{u32("sample.history", s.Hist)}, nil, Attrs{Offset: 2, Width: s.Hist}, nil, nil)[0]
	cand := b.Emit(OpTopK, []ValueID{l}, []Result{u32("sample.cand", C), f32("sample.cand.v", C)}, nil, Attrs{TopK: C}, nil, nil)
	tiled := b.Emit(OpTile, []ValueID{hist}, []Result{u32("sample.history.tiled", C*s.Hist)}, nil, Attrs{Span: s.Hist}, nil, nil)[0]
	seen := b.Emit(OpMap, []ValueID{cand[0], tiled}, []Result{f32("sample.seen", C*s.Hist)}, nil, Attrs{Fn: MapEqual}, nil, nil)[0]
	count := b.Emit(OpReduce, []ValueID{seen}, []Result{f32("sample.count", C)}, nil, Attrs{Fn: MapAdd, Span: s.Hist}, nil, nil)[0]
	v := penalize(b, cand[1], count, pen, s.Hist)
	// Ranked by id first, so that the lower place of two equal values is the
	// lower id.
	negID := b.Emit(OpMap, []ValueID{cand[0]}, []Result{f32("sample.cand.negid", C)}, nil, Attrs{Fn: MapMul, Const: -1}, nil, nil)[0]
	byID := b.Emit(OpTopK, []ValueID{negID}, []Result{u32("sample.byid", C), f32("sample.byid.v", C)}, nil, Attrs{TopK: C}, nil, nil)
	pv := b.Emit(OpTake, []ValueID{v, byID[0]}, []Result{f32("sample.byid.pen", C)}, nil, Attrs{}, nil, nil)[0]
	byIDs := b.Emit(OpTake, []ValueID{cand[0], byID[0]}, []Result{u32("sample.byid.ids", C)}, nil, Attrs{}, nil, nil)[0]
	best := b.Emit(OpTopK, []ValueID{pv}, []Result{u32("sample.place", K), f32("sample.vals", K)}, nil, Attrs{TopK: K}, nil, nil)
	out := b.Emit(OpTake, []ValueID{byIDs, best[0]}, []Result{u32("sample.ids", K)}, nil, Attrs{}, nil, nil)[0]
	return best[1], out
}

// penalize is v with the repeat penalty applied count times an element, as
// the host's sampler applies it once a history entry: a positive value
// divided by the penalty, correctly rounded, any other multiplied. Counts
// past hist are not reached.
func penalize(b *Builder, v, count, pen ValueID, hist int) ValueID {
	w := b.G.Values[v].Width
	for t0 := 0; t0 < hist; t0 += sampleSteps {
		var terms []Term
		cur := Arg{Kind: ArgIn, Index: 0}
		for t := t0; t < min(t0+sampleSteps, hist); t++ {
			n := len(terms)
			terms = append(terms,
				Term{Fn: MapDiv, Args: [3]Arg{cur, {Kind: ArgIn, Index: 2}}},
				Term{Fn: MapMul, Args: [3]Arg{cur, {Kind: ArgIn, Index: 2}}},
				Term{Fn: MapWhere, Args: [3]Arg{cur, {Kind: ArgTerm, Index: n}, {Kind: ArgTerm, Index: n + 1}}},
				Term{Fn: MapLess, Args: [3]Arg{{Kind: ArgConst, Const: float64(t)}, {Kind: ArgIn, Index: 1}}},
				Term{Fn: MapWhere, Args: [3]Arg{{Kind: ArgTerm, Index: n + 3}, {Kind: ArgTerm, Index: n + 2}, cur}})
			cur = Arg{Kind: ArgTerm, Index: n + 4}
		}
		v = b.Emit(OpExpr, []ValueID{v, count, pen}, []Result{{Name: fmt.Sprintf("sample.penalized%d", t0/sampleSteps), Width: w}}, nil,
			Attrs{Expr: terms, Exact: true}, nil, nil)[0]
	}
	return v
}

// SampleDraw is SampleTopK's candidates drawn from where they are: of the
// first k, top-p and then min-p over their softmax at
// temperature 1, the best always kept, then a Gumbel-max at the temperature
// -- the largest of (v - v0)/T - log(-log u) over the survivors, u each
// candidate's OpUniform under the counter keyed by its id. Its inputs are
// SampleTopK's, then sample.draw (k, 1/T, top-p and min-p: top-p +Inf and
// min-p 0 where off) and sample.counter (u32: the seed, then the position);
// its output the token. Top-p's mass before each candidate is a KxK sum:
// each candidate's row of the probabilities, those of a lower place kept.
type SampleDraw struct {
	SampleTopK
}

func (s *SampleDraw) Build() *Graph {
	b := New()
	vals, ids := s.emit(b)
	draw := b.In("sample.draw", 4)
	ctr := b.In("sample.counter", 2)
	b.G.Values[ctr].Elem = ElemU32
	K := b.G.Values[vals].Width
	e := sampleEmitter{b}
	_, d, keep := e.cuts(vals, draw)
	invT := e.word(draw, 1, "sample.invtemp")
	u := e.one(OpUniform, "sample.u", K, Attrs{}, ids, ctr)
	g := e.mapc(MapMul, -1, "sample.gumbel", K, e.mapc(MapLog, 0, "sample.lnlnu", K, e.mapc(MapMul, -1, "sample.nlnu", K, e.mapc(MapLog, 0, "sample.lnu", K, u))))
	score := e.mapc(MapAdd, 0, "sample.score", K, e.mapc(MapMul, 0, "sample.dt", K, d, invT), g)
	kept := e.mapc(MapWhere, math.Inf(-1), "sample.kept", K, keep, score)
	b.Out(e.token(ids, kept))
	return b.G
}

// SampleWalk is SampleTopK's candidates drawn as the host's sampler draws
// them: of the first k, top-p and then min-p over their softmax at
// temperature 1, the best always kept, then the inverse-CDF walk at a uniform
// u the caller draws over the survivors' softmax at the temperature -- the
// first survivor whose mass up to and including it passes u times the
// survivors' mass, the last survivor where rounding leaves none. Its inputs
// are SampleTopK's, then sample.walk (k, 1/T, top-p, min-p as SampleDraw's,
// then u); its output the token. The mass before each survivor is a KxK sum,
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
// SampleTopK's logits, sample.args and sample.penalty, sample.ids (the row's
// ids in order, u32) and sample.row (1/T, then u); its output the token. The
// walk is a weighted selection over the row (sampleEmitter.selection): no
// order of it is formed.
type SampleRow struct {
	NVocab, Hist int
}

func (s *SampleRow) Build() *Graph {
	b := New()
	l := b.In("logits", s.NVocab)
	args := b.In("sample.args", 2+s.Hist)
	b.G.Values[args].Elem = ElemU32
	pen := b.In("sample.penalty", 1)
	ids := b.In("sample.ids", s.NVocab)
	b.G.Values[ids].Elem = ElemU32
	row := b.In("sample.row", 2)
	e := sampleEmitter{b}
	v := e.penalizedRow(l, ids, args, pen, s.Hist)
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
// of the row's where none does.
type SampleCut struct {
	NVocab, Hist int
	Prob         bool
}

func (s *SampleCut) Build() *Graph {
	b := New()
	n := s.NVocab
	l := b.In("logits", n)
	args := b.In("sample.args", 2+s.Hist)
	b.G.Values[args].Elem = ElemU32
	pen := b.In("sample.penalty", 1)
	ids := b.In("sample.ids", n)
	b.G.Values[ids].Elem = ElemU32
	cut := b.In("sample.cut", 4)
	e := sampleEmitter{b}
	v := e.penalizedRow(l, ids, args, pen, s.Hist)
	invT, topp, minp, u := e.word(cut, 0, "sample.invtemp"), e.word(cut, 1, "sample.topp"), e.word(cut, 2, "sample.minp"), e.word(cut, 3, "sample.u")
	id := e.mapc(MapMul, 1, "sample.idf", n, ids)
	notv := func(name string, w int, x ValueID) ValueID {
		return e.mapc(MapAdd, 1, name, w, e.mapc(MapMul, -1, name+".neg", w, x))
	}
	q1 := e.softmax(v, "sample.q1")
	all := e.mapc(MapLess, 2, "sample.all", n, q1)
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
	edge := e.selection(q1, id, q1, all, topp, true)
	edgeQ := e.one(OpReduce, "sample.edgeq", 1, Attrs{Fn: MapMax}, e.mapc(MapMul, 0, "sample.edgeqs", n, edge, q1))
	edgeID := e.one(OpReduce, "sample.edgeid", 1, Attrs{Fn: MapMax}, e.mapc(MapMul, 0, "sample.edgeids", n, edge, id))
	// Up to the edge: a larger probability, or the edge's and an id not past it.
	above := e.mapc(MapLess, 0, "sample.above", n, e.mapc(MapMul, -1, "sample.q1neg", n, q1), e.mapc(MapMul, -1, "sample.edgeqneg", 1, edgeQ))
	tie := e.mapc(MapMul, 0, "sample.tie", n, e.mapc(MapEqual, 0, "sample.attie", n, q1, edgeQ),
		notv("sample.notpast", n, e.mapc(MapLess, 0, "sample.past", n, edgeID, id)))
	upTo := e.mapc(MapMax, 0, "sample.upto", n, above, tie)
	keepP := e.mapc(MapMax, 0, "sample.keepp", n, upTo, notv("sample.topoff", 1, topFired))
	keep := e.mapc(MapMax, 0, "sample.keep", n, e.mapc(MapMul, 0, "sample.keeppm", n, keepP, notv("sample.above.floor", n, below)), best)
	fired := e.mapc(MapMax, 0, "sample.fired", 1, topFired, minFired)
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
	walk := e.selection(q1, id, pt, keep, walkT, false)
	// No cut anywhere: the row's walk.
	rowT := e.mapc(MapMul, 0, "sample.row.target", 1, u, e.one(OpReduce, "sample.mass", 1, Attrs{Fn: MapAdd}, q))
	row := e.selection(q, id, q, all, rowT, false)
	chosen := e.mapc(MapAdd, 0, "sample.chosen", n, row, e.mapc(MapMul, 0, "sample.chosen.cut", n, e.mapc(MapSub, 0, "sample.chosen.d", n, walk, row), fired))
	b.Out(e.argmax(chosen))
	return b.G
}

// sampleEmitter is the ops SampleDraw, SampleWalk and SampleRow share on b.
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

// penalizedRow is the logits l with the repeat penalty of SampleTopK's args
// and pen applied to the whole row: each history id's logit penalized once
// an entry of it (penalize, over the history's own entries), every other as
// it is. ids is the row's ids. The history is compared with the row a chunk
// of sampleSteps entries at a time.
func (e sampleEmitter) penalizedRow(l, ids, args, pen ValueID, hist int) ValueID {
	n := e.b.G.Values[l].Width
	u32 := func(name string, w int) Result { return Result{Name: name, Width: w, Elem: ElemU32} }
	h := e.b.Emit(OpSlice, []ValueID{args}, []Result{u32("sample.history", hist)}, nil, Attrs{Offset: 2, Width: hist}, nil, nil)[0]
	// Each entry's logit (an id past the row reads the row's last, and
	// matches no id) and how many entries share its id.
	raw := e.b.Emit(OpTake, []ValueID{l, h}, []Result{{Name: "sample.history.raw", Width: hist}}, nil, Attrs{}, nil, nil)[0]
	tiled := e.b.Emit(OpTile, []ValueID{h}, []Result{u32("sample.history.tiled", hist*hist)}, nil, Attrs{Span: hist}, nil, nil)[0]
	count := e.one(OpReduce, "sample.history.count", hist, Attrs{Fn: MapAdd, Span: hist},
		e.mapc(MapEqual, 0, "sample.history.same", hist*hist, h, tiled))
	pv := penalize(e.b, raw, count, pen, hist)
	// Each token's penalized value where an entry names it, -Inf where none.
	var m ValueID = -1
	for c0 := 0; c0 < hist; c0 += sampleSteps {
		k := min(sampleSteps, hist-c0)
		hc := e.b.Emit(OpSlice, []ValueID{h}, []Result{u32("sample.history.chunk", k)}, nil, Attrs{Offset: c0, Width: k}, nil, nil)[0]
		vc := e.one(OpSlice, "sample.history.pen", k, Attrs{Offset: c0, Width: k}, pv)
		th := e.b.Emit(OpTile, []ValueID{hc}, []Result{u32("sample.history.row", n*k)}, nil, Attrs{Span: k}, nil, nil)[0]
		tv := e.one(OpTile, "sample.history.rowpen", n*k, Attrs{Span: k}, vc)
		hit := e.mapc(MapEqual, 0, "sample.history.hit", n*k, ids, th)
		mc := e.one(OpReduce, "sample.history.named", n, Attrs{Fn: MapMax, Span: k},
			e.mapc(MapWhere, math.Inf(-1), "sample.history.pick", n*k, hit, tv))
		if m < 0 {
			m = mc
		} else {
			m = e.mapc(MapMax, 0, "sample.history.max", n, m, mc)
		}
	}
	// named is m > -Inf, as -m < +Inf.
	named := e.mapc(MapLess, math.Inf(1), "sample.history.isnamed", n, e.mapc(MapMul, -1, "sample.history.neg", n, m))
	return e.where("sample.penalizedrow", n, named, m, l)
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
