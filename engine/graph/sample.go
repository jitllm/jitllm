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
	f32 := func(name string, w int) Result { return Result{Name: name, Width: w} }
	one := func(op OpKind, name string, w int, a Attrs, in ...ValueID) ValueID {
		return b.Emit(op, in, []Result{f32(name, w)}, nil, a, nil, nil)[0]
	}
	mapc := func(fn MapFn, c float64, name string, w int, in ...ValueID) ValueID {
		return one(OpMap, name, w, Attrs{Fn: fn, Const: c}, in...)
	}
	word := func(i int, name string) ValueID {
		return one(OpSlice, name, 1, Attrs{Offset: i, Width: 1}, draw)
	}
	k, invT, topp, minp := word(0, "sample.k"), word(1, "sample.invtemp"), word(2, "sample.topp"), word(3, "sample.minp")
	rank := b.Emit(OpRank, []ValueID{vals}, []Result{{Name: "sample.rank", Width: K, Elem: ElemU32}}, nil, Attrs{}, nil, nil)[0]
	place := mapc(MapMul, 1, "sample.place.f", K, rank)
	inK := mapc(MapLess, 0, "sample.ink", K, place, k)
	v0 := one(OpReduce, "sample.v0", 1, Attrs{Fn: MapMax}, vals)
	d := mapc(MapSub, 0, "sample.d", K, vals, v0)
	e := mapc(MapMul, 0, "sample.e", K, mapc(MapExp, 0, "sample.ed", K, d), inK)
	p := mapc(MapDiv, 0, "sample.p", K, e, one(OpReduce, "sample.sum", 1, Attrs{Fn: MapAdd}, e))
	tp := one(OpTile, "sample.p.rows", K*K, Attrs{Span: K}, p)
	tr := one(OpTile, "sample.place.rows", K*K, Attrs{Span: K}, place)
	lower := mapc(MapMul, 0, "sample.lower", K*K, tp, mapc(MapLess, 0, "sample.lt", K*K, tr, place))
	before := one(OpReduce, "sample.before", K, Attrs{Fn: MapAdd, Span: K}, lower)
	keepP := mapc(MapLess, 0, "sample.keep.p", K, before, topp)
	floor := mapc(MapMul, 0, "sample.floor", 1, one(OpReduce, "sample.pmax", 1, Attrs{Fn: MapMax}, p), minp)
	under := mapc(MapLess, 0, "sample.under", K, p, floor)
	keepM := mapc(MapAdd, 1, "sample.keep.m", K, mapc(MapMul, -1, "sample.under.neg", K, under))
	first := mapc(MapLess, 1, "sample.first", K, place)
	keep := mapc(MapMax, 0, "sample.keep", K, mapc(MapMul, 0, "sample.keep.pmk", K, mapc(MapMul, 0, "sample.keep.pm", K, keepP, keepM), inK), first)
	u := one(OpUniform, "sample.u", K, Attrs{}, ids, ctr)
	g := mapc(MapMul, -1, "sample.gumbel", K, mapc(MapLog, 0, "sample.lnlnu", K, mapc(MapMul, -1, "sample.nlnu", K, mapc(MapLog, 0, "sample.lnu", K, u))))
	score := mapc(MapAdd, 0, "sample.score", K, mapc(MapMul, 0, "sample.dt", K, d, invT), g)
	kept := mapc(MapWhere, math.Inf(-1), "sample.kept", K, keep, score)
	best := b.Emit(OpTopK, []ValueID{kept}, []Result{{Name: "sample.best", Width: 1, Elem: ElemU32}, f32("sample.best.v", 1)}, nil, Attrs{TopK: 1}, nil, nil)
	tok := b.Emit(OpTake, []ValueID{ids, best[0]}, []Result{{Name: "sample.token", Width: 1, Elem: ElemU32}}, nil, Attrs{}, nil, nil)[0]
	b.Out(tok)
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
