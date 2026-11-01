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
	v := cand[1]
	for t0 := 0; t0 < s.Hist; t0 += sampleSteps {
		var terms []Term
		cur := Arg{Kind: ArgIn, Index: 0}
		for t := t0; t < min(t0+sampleSteps, s.Hist); t++ {
			n := len(terms)
			terms = append(terms,
				Term{Fn: MapDiv, Args: [3]Arg{cur, {Kind: ArgIn, Index: 2}}},
				Term{Fn: MapMul, Args: [3]Arg{cur, {Kind: ArgIn, Index: 2}}},
				Term{Fn: MapWhere, Args: [3]Arg{cur, {Kind: ArgTerm, Index: n}, {Kind: ArgTerm, Index: n + 1}}},
				Term{Fn: MapLess, Args: [3]Arg{{Kind: ArgConst, Const: float64(t)}, {Kind: ArgIn, Index: 1}}},
				Term{Fn: MapWhere, Args: [3]Arg{{Kind: ArgTerm, Index: n + 3}, {Kind: ArgTerm, Index: n + 2}, cur}})
			cur = Arg{Kind: ArgTerm, Index: n + 4}
		}
		v = b.Emit(OpExpr, []ValueID{v, count, pen}, []Result{f32(fmt.Sprintf("sample.penalized%d", t0/sampleSteps), C)}, nil,
			Attrs{Expr: terms, Exact: true}, nil, nil)[0]
	}
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
// as top-p's.
type SampleWalk struct {
	SampleTopK
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

// sampleEmitter is the ops SampleDraw and SampleWalk share on b.
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
