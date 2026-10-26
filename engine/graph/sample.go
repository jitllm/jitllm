package graph

import "fmt"

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
	ids := b.Emit(OpTake, []ValueID{cand[0], byID[0]}, []Result{u32("sample.byid.ids", C)}, nil, Attrs{}, nil, nil)[0]
	best := b.Emit(OpTopK, []ValueID{pv}, []Result{u32("sample.place", K), f32("sample.vals", K)}, nil, Attrs{TopK: K}, nil, nil)
	out := b.Emit(OpTake, []ValueID{ids, best[0]}, []Result{u32("sample.ids", K)}, nil, Attrs{}, nil, nil)[0]
	b.Out(best[1], out)
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
