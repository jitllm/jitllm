package probe

import (
	"github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/graph"
)

// gpuStep is class cl's chained step on a GPU, the terms appended after
// last, the row's element type, and false for a class with no step: a float
// multiply-add of the value by itself, an integer shift and exclusive or, an
// integer multiply-add of the value by itself, an exponential, and for the
// lanes and group classes the value summed over its run of the row
// (gpuRun), an iteration's alone. None of them folds across a chain; a
// multiply-add by the row's element and a constant does on Vulkan, whose
// launch of a deeper chain of them took no longer.
func gpuStep(cl cost.Class) (func(terms []graph.Term, last graph.Arg) []graph.Term, graph.Elem, bool) {
	in := graph.Arg{Kind: graph.ArgIn}
	k := func(v float64) graph.Arg { return graph.Arg{Kind: graph.ArgConst, Const: v} }
	switch cl {
	case cost.ClassFloat:
		return func(terms []graph.Term, last graph.Arg) []graph.Term {
			return append(terms, graph.Term{Fn: graph.MapFma, Args: [3]graph.Arg{last, last, in}})
		}, graph.ElemF32, true
	case cost.ClassInt:
		return func(terms []graph.Term, last graph.Arg) []graph.Term {
			terms = append(terms, graph.Term{Fn: graph.MapShr, Args: [3]graph.Arg{last, k(1)}})
			return append(terms, graph.Term{Fn: graph.MapXor, Args: [3]graph.Arg{last, {Kind: graph.ArgTerm, Index: len(terms) - 1}}})
		}, graph.ElemU32, true
	case cost.ClassIndex:
		return func(terms []graph.Term, last graph.Arg) []graph.Term {
			terms = append(terms, graph.Term{Fn: graph.MapMul, Args: [3]graph.Arg{last, last}})
			return append(terms, graph.Term{Fn: graph.MapAdd, Args: [3]graph.Arg{{Kind: graph.ArgTerm, Index: len(terms) - 1}, k(1)}})
		}, graph.ElemU32, true
	case cost.ClassSpecial:
		return func(terms []graph.Term, last graph.Arg) []graph.Term {
			return append(terms, graph.Term{Fn: graph.MapExp, Args: [3]graph.Arg{last}})
		}, graph.ElemF32, true
	case cost.ClassLanes, cost.ClassGroup:
		return func(terms []graph.Term, last graph.Arg) []graph.Term {
			if last.Kind != graph.ArgTerm {
				terms = append(terms, graph.Term{Fn: graph.MapF32, Args: [3]graph.Arg{last}})
				last = graph.Arg{Kind: graph.ArgTerm, Index: len(terms) - 1}
			}
			return append(terms, graph.Term{Fn: graph.MapF32, Args: [3]graph.Arg{{Kind: graph.ArgSum, Index: last.Index}}})
		}, graph.ElemF32, true
	}
	return nil, 0, false
}

// lanesRun is the elements a lanes step sums over: a pair, its sum an
// exchange of the pair's lanes beside an add. groupRun is a group step's:
// the probe's workgroup.
const (
	lanesRun = 2
	groupRun = 256
)

// gpuRun is the run class cl's step sums over, 0 for a class whose step
// sums nothing.
func gpuRun(cl cost.Class) int {
	switch cl {
	case cost.ClassLanes:
		return lanesRun
	case cost.ClassGroup:
		return groupRun
	}
	return 0
}

// countedAs is the class the estimator counts class cl's chain in: on a GPU
// integer work on words is index work, since the words it counts as data
// are the ones a host holds in vector registers (composer's VectorWords).
func countedAs(cl cost.Class) cost.Class {
	if cl == cost.ClassInt {
		return cost.ClassIndex
	}
	return cl
}

// chainTerms is chains chains of deep steps each, chain c starting at the
// row's element plus c+1, summed pairwise: the last term their sum.
func chainTerms(chains, deep int, step func(terms []graph.Term, last graph.Arg) []graph.Term) []graph.Term {
	var terms []graph.Term
	var ends []int
	for c := range chains {
		terms = append(terms, graph.Term{Fn: graph.MapAdd, Args: [3]graph.Arg{{Kind: graph.ArgIn}, {Kind: graph.ArgConst, Const: float64(c + 1)}}})
		for range deep {
			terms = step(terms, graph.Arg{Kind: graph.ArgTerm, Index: len(terms) - 1})
		}
		ends = append(ends, len(terms)-1)
	}
	for len(ends) > 1 {
		var next []int
		for i := 0; i+1 < len(ends); i += 2 {
			terms = append(terms, graph.Term{Fn: graph.MapAdd, Args: [3]graph.Arg{{Kind: graph.ArgTerm, Index: ends[i]}, {Kind: graph.ArgTerm, Index: ends[i+1]}}})
			next = append(next, len(terms)-1)
		}
		ends = append(next, ends[len(ends)&^1:]...)
	}
	return terms
}

// rateGraph is a row of n through chains independent chains of deep steps
// of class cl, summed: one chain is the class's delay, several its rate.
func rateGraph(cl cost.Class, n, chains, deep int) (*graph.Graph, bool) {
	step, elem, ok := gpuStep(cl)
	if !ok {
		return nil, false
	}
	b := graph.New()
	x := b.In("x", n)
	b.G.Values[x].Elem = elem
	y := b.Emit(graph.OpExpr, []graph.ValueID{x}, []graph.Result{{Name: "chains", Width: n}}, nil,
		graph.Attrs{Expr: chainTerms(chains, deep, step)}, nil, nil)[0]
	b.G.Values[y].Elem = elem
	b.Out(y)
	return b.G, true
}

// iterateGraph is a row of n through trips trips of an iteration carrying
// chains values, each through deep steps of class cl a trip, the value c
// starting at the row's element plus c+1: one value is the class's delay, a
// few a thread's rate, the trips a loop the code keeps rather than code laid
// out straight.
func iterateGraph(cl cost.Class, n, chains, deep, trips int) (*graph.Graph, bool) {
	step, elem, ok := gpuStep(cl)
	if !ok {
		return nil, false
	}
	return iterateSteps(step, elem, gpuRun(cl), n, chains, deep, trips), true
}

// addsGraph is the lanes class's chain (iterateGraph) of one value with
// each sum over a pair an add of the row's element: the same adds, no
// exchange, and none a device's compiler can fold into the next, as it may
// a chain of doublings.
func addsGraph(n, deep, trips int) *graph.Graph {
	step := func(terms []graph.Term, last graph.Arg) []graph.Term {
		return append(terms, graph.Term{Fn: graph.MapAdd, Args: [3]graph.Arg{last, {Kind: graph.ArgIn}}})
	}
	return iterateSteps(step, graph.ElemF32, 0, n, 1, deep, trips)
}

// iterateSteps is iterateGraph through step, of elements elem, its sums
// over runs of run.
func iterateSteps(step func(terms []graph.Term, last graph.Arg) []graph.Term, elem graph.Elem, run, n, chains, deep, trips int) *graph.Graph {
	var terms []graph.Term
	for c := range chains {
		terms = append(terms, graph.Term{Fn: graph.MapAdd, Args: [3]graph.Arg{{Kind: graph.ArgIn}, {Kind: graph.ArgConst, Const: float64(c + 1)}}})
	}
	outs := make([]uint64, chains)
	for c := range chains {
		last := graph.Arg{Kind: graph.ArgState, Index: c}
		for range deep {
			terms = step(terms, last)
			last = graph.Arg{Kind: graph.ArgTerm, Index: len(terms) - 1}
		}
		outs[c] = uint64(len(terms) - 1)
	}
	b := graph.New()
	x := b.In("x", n)
	b.G.Values[x].Elem = elem
	y := b.Emit(graph.OpIterate, []graph.ValueID{x}, []graph.Result{{Name: "chains", Width: n * chains}}, nil,
		graph.Attrs{Taps: chains, Span: trips, Width: run, Words: outs, Expr: terms}, nil, nil)[0]
	b.G.Values[y].Elem = elem
	b.Out(y)
	return b.G
}

// wrapGraph is threads columns, each summing per floats of a table of
// words floats: column c from the float its row element at names, stride
// floats apart, wrapped at mask. One row's iteration carrying a word, the
// index, beside a float, the sum, each to an output of its own.
func wrapGraph(words, threads, per, stride, mask int) (*graph.Graph, graph.ValueID, graph.ValueID) {
	in := graph.Arg{Kind: graph.ArgIn}
	k := func(v int) graph.Arg { return graph.Arg{Kind: graph.ArgConst, Const: float64(v)} }
	st := func(j int) graph.Arg { return graph.Arg{Kind: graph.ArgState, Index: j} }
	term := func(j int) graph.Arg { return graph.Arg{Kind: graph.ArgTerm, Index: j} }
	terms := []graph.Term{
		{Fn: graph.MapAdd, Args: [3]graph.Arg{in, k(0)}},
		{Fn: graph.MapMul, Args: [3]graph.Arg{in, k(0)}},
		{Fn: graph.MapTake, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 1}, st(0)}},
		{Fn: graph.MapAdd, Args: [3]graph.Arg{st(1), term(2)}},
		{Fn: graph.MapAdd, Args: [3]graph.Arg{st(0), k(stride)}},
		{Fn: graph.MapAnd, Args: [3]graph.Arg{term(4), k(mask)}},
	}
	b := graph.New()
	at := b.In("at", threads)
	b.G.Values[at].Elem = graph.ElemU32
	tab := b.In("table", words)
	ys := b.Emit(graph.OpIterate, []graph.ValueID{at, tab}, []graph.Result{{Name: "reached", Width: threads}, {Name: "read", Width: threads}}, nil,
		graph.Attrs{Taps: 2, Span: per, Words: []uint64{5, 3}, Expr: terms}, nil, nil)
	b.G.Values[ys[0]].Elem = graph.ElemU32
	b.Out(ys[0])
	b.Out(ys[1])
	return b.G, tab, at
}

// streamStarts is threads columns' first words of a set of words, a power
// of two, each warp's 32 columns a line, streamed through the set from a
// phase of its own: warp w's line w*g of the set's lines, g the nearest odd
// number to the lines over the golden ratio. Warps near each other in
// number, a workgroup's, start far apart, and so do any few of them,
// whatever workgroups a multiprocessor holds.
func streamStarts(threads, words int) []uint32 {
	const line = 32
	lines := max(1, words/line)
	g := int(float64(lines)*0.6180339887498949) | 1
	at := make([]uint32, threads)
	for c := range at {
		at[c] = uint32((c/line*g%lines)*line + c%line)
	}
	return at
}

// chaseGraph is a walk of trips steps through a table of words, one row's
// iteration: each trip takes the word the one before named, the first
// named by the row at.
func chaseGraph(words, trips int) (*graph.Graph, graph.ValueID, graph.ValueID) {
	b := graph.New()
	at := b.In("at", 1)
	b.G.Values[at].Elem = graph.ElemU32
	tab := b.In("table", words)
	b.G.Values[tab].Elem = graph.ElemU32
	terms := []graph.Term{
		{Fn: graph.MapAdd, Args: [3]graph.Arg{{Kind: graph.ArgIn}, {Kind: graph.ArgConst}}},
		{Fn: graph.MapTake, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 1}, {Kind: graph.ArgState}}},
	}
	v := b.Emit(graph.OpIterate, []graph.ValueID{at, tab}, []graph.Result{{Name: "reached", Width: 1}}, nil,
		graph.Attrs{Taps: 1, Span: trips, Expr: terms}, nil, nil)[0]
	b.G.Values[v].Elem = graph.ElemU32
	b.Out(v)
	return b.G, tab, at
}
