//go:build amd64 || arm64

package probe

import (
	"fmt"
	"math"
	"runtime"
	"unsafe"

	"github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/graph"
)

// readGraph sums a row of n words: the composer vectorizes the reduction
// and unrolls it, a sum for each copy, so the loads bind the sum and not
// the adds. The words are integers because a core issues integer adds on
// more ports than float ones (Broadwell adds floats on one), and a read's
// rate is its loads'.
func readGraph(n int) *graph.Graph {
	b := graph.New()
	x := b.In("x", n)
	b.G.Values[x].Elem = graph.ElemU32
	s := b.Emit(graph.OpReduce, []graph.ValueID{x}, []graph.Result{{Name: "sum", Width: 1}}, nil,
		graph.Attrs{Fn: graph.MapAdd, Span: n}, nil, nil)[0]
	b.G.Values[s].Elem = graph.ElemU32
	b.Out(s)
	return b.G
}

// reader reads buffers through the composed sum, one form a buffer size.
type reader struct {
	fm    former
	forms map[int]*formed
}

// on is a run reading buf whole, a whole number of words; the word it
// stores the sum in is the last it keeps.
func (r *reader) on(buf []byte) (*bound, error) {
	n := len(buf) / 4
	f, ok := r.forms[n]
	if !ok {
		var err error
		if f, err = r.fm.form(readGraph(n), 1); err != nil {
			return nil, err
		}
		if f.count(cost.ClassMem) == 0 {
			f.close()
			return nil, fmt.Errorf("probe: the read form issues no load")
		}
		r.forms[n] = f
	}
	// A line of its own: threads storing their sums on one line would pass
	// it between cores every call.
	out := make([]uint32, 16)
	b, err := f.bind(map[graph.ValueID]unsafe.Pointer{0: unsafe.Pointer(&buf[0]), 1: unsafe.Pointer(&out[0])})
	if err != nil {
		return nil, err
	}
	b.keep = append(b.keep, out)
	return b, nil
}

func (r *reader) close() {
	for _, f := range r.forms {
		f.close()
	}
}

// The host's rates and delays are iterations: a row's values carried
// through trips in order, timed between two trip counts so what a call
// costs beside its trips cancels, over the instructions of the class the
// estimator counts in them.
const (
	tripsShort, tripsLong = 1 << 10, 1 << 13
	// iterateCalls is the calls each trip count is timed over.
	iterateCalls = 16
	// rateChains is the values a rate's iteration carries, each a chain of
	// its own: as many as a float rate needs to hide its latency.
	rateChains = 8
	// delaySteps is a delay's chained steps a trip.
	delaySteps = 8
)

// The integer class's rate is a row's: vectors of it, each through
// rateChains chains of rowSteps steps, between rows of rowShort and rowLong
// vectors, each timed over rowCalls calls.
const (
	rowShort, rowLong = 32, 256
	rowSteps          = 4
	rowCalls          = 256
)

// hostStep is class cl's step on the host, the next of the last, the row's
// element type and the row's value as bits: the row's element times itself
// added onto the value, as a dot's sum takes its products, so that the
// multiply-add accumulates in the value's register; a divide by 1; an add
// of the row's word. False for a class the composed probe does not step.
func hostStep(cl cost.Class) (func(last graph.Arg) graph.Term, graph.Elem, uint32, bool) {
	in := graph.Arg{Kind: graph.ArgIn}
	switch cl {
	case cost.ClassFloat:
		return func(last graph.Arg) graph.Term {
			return graph.Term{Fn: graph.MapFma, Args: [3]graph.Arg{in, in, last}}
		}, graph.ElemF32, math.Float32bits(0.5), true
	case cost.ClassSpecial:
		return func(last graph.Arg) graph.Term {
			return graph.Term{Fn: graph.MapDiv, Args: [3]graph.Arg{last, in}}
		}, graph.ElemF32, math.Float32bits(1), true
	case cost.ClassInt, cost.ClassIndex:
		return func(last graph.Arg) graph.Term {
			return graph.Term{Fn: graph.MapAdd, Args: [3]graph.Arg{last, in}}
		}, graph.ElemU32, 1, true
	}
	return nil, 0, 0, false
}

// stepRow is the row class cl's iteration runs over: a vector of lanes for
// the integer class, whose words an iteration holds in a vector register;
// one element for the rest, a float of which an iteration holds in a vector
// register and a word in a general one, the index class's.
func stepRow(cl cost.Class, lanes int) int {
	if cl == cost.ClassInt {
		return lanes
	}
	return 1
}

// iterateRow is a row of n through trips trips of an iteration carrying
// chains values, each through deep steps a trip and each starting at the
// row's element: values that start alike hold no start of their own beside
// them before the first trip.
func iterateRow(step func(last graph.Arg) graph.Term, elem graph.Elem, n, chains, deep, trips int) *graph.Graph {
	var terms []graph.Term
	for range chains {
		terms = append(terms, graph.Term{Fn: graph.MapAdd, Args: [3]graph.Arg{{Kind: graph.ArgIn}, {Kind: graph.ArgConst}}})
	}
	outs := make([]uint64, chains)
	for c := range chains {
		last := graph.Arg{Kind: graph.ArgState, Index: c}
		for range deep {
			terms = append(terms, step(last))
			last = graph.Arg{Kind: graph.ArgTerm, Index: len(terms) - 1}
		}
		outs[c] = uint64(len(terms) - 1)
	}
	b := graph.New()
	x := b.In("x", n)
	b.G.Values[x].Elem = elem
	y := b.Emit(graph.OpIterate, []graph.ValueID{x}, []graph.Result{{Name: "chains", Width: n * chains}}, nil,
		graph.Attrs{Taps: chains, Span: trips, Words: outs, Expr: terms}, nil, nil)[0]
	b.G.Values[y].Elem = elem
	b.Out(y)
	return b.G
}

// rowGraph is a row of n through rateChains chains of rowSteps steps each,
// summed: an iteration carrying several words holds them in the general
// registers, so the integer class's rate is a row's vectors, each
// independent of the next.
func rowGraph(step func(last graph.Arg) graph.Term, elem graph.Elem, n int) *graph.Graph {
	b := graph.New()
	x := b.In("x", n)
	b.G.Values[x].Elem = elem
	terms := chainTerms(rateChains, rowSteps, func(terms []graph.Term, last graph.Arg) []graph.Term { return append(terms, step(last)) })
	y := b.Emit(graph.OpExpr, []graph.ValueID{x}, []graph.Result{{Name: "chains", Width: n}}, nil, graph.Attrs{Expr: terms}, nil, nil)[0]
	b.G.Values[y].Elem = elem
	b.Out(y)
	return b.G
}

// mixAdds is the mix's adds a trip, and its reads.
const mixAdds = 4

// mixGraph is an iteration over one word of mixAdds adds of the row's
// word, each a chain of its own, and a read of a table at the place each
// add names: the index and memory classes, which take ports of their own,
// so that the rate is what the core issues a second rather than any class's
// units. Its values are words: the float class is not in it.
func mixGraph(trips int) *graph.Graph {
	b := graph.New()
	x := b.In("x", 1)
	b.G.Values[x].Elem = graph.ElemU32
	tab := b.In("table", 64)
	b.G.Values[tab].Elem = graph.ElemU32
	var terms []graph.Term
	for range 2 * mixAdds {
		terms = append(terms, graph.Term{Fn: graph.MapAdd, Args: [3]graph.Arg{{Kind: graph.ArgConst}, {Kind: graph.ArgConst}}})
	}
	for j := range mixAdds {
		terms = append(terms, graph.Term{Fn: graph.MapAdd, Args: [3]graph.Arg{{Kind: graph.ArgState, Index: j}, {Kind: graph.ArgIn}}})
	}
	for j := range mixAdds {
		terms = append(terms, graph.Term{Fn: graph.MapTake, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 1}, {Kind: graph.ArgTerm, Index: 2*mixAdds + j}}})
	}
	v := b.Emit(graph.OpIterate, []graph.ValueID{x, tab}, []graph.Result{{Name: "mix", Width: 2 * mixAdds}}, nil,
		graph.Attrs{Taps: 2 * mixAdds, Span: trips, Expr: terms}, nil, nil)[0]
	b.G.Values[v].Elem = graph.ElemU32
	b.Out(v)
	return b.G
}

// hostRate is the instructions a second threads issue together through
// the forms of graphAt at two sizes, short and long, as count counts them,
// each thread on rows of its own (bindRow) and each form timed over calls
// calls: what a call costs beside its work cancels in the difference. The
// larger form counts at least least more, else it lost the work it was
// asked for.
func hostRate(fm former, graphAt func(size int) *graph.Graph, v uint32, short, long, calls, threads int, count func(*formed) float64, least float64) (float64, error) {
	var issued, secs [2]float64
	for i, size := range [2]int{short, long} {
		g := graphAt(size)
		f, err := fm.form(g, 1)
		if err != nil {
			return 0, err
		}
		issued[i] = count(f)
		runs := make([]*bound, threads)
		for t := range runs {
			if runs[t], err = bindRow(f, g, v); err != nil {
				f.close()
				return 0, err
			}
		}
		secs[i] = parallel(threads, func(t int) {
			for range calls {
				runs[t].run()
			}
		})
		runtime.KeepAlive(runs)
		f.close()
	}
	if !(issued[1]-issued[0] >= least) || !(secs[1] > secs[0]) {
		return 0, fmt.Errorf("probe: forms at %d and %d count %v and took %v seconds", short, long, issued, secs)
	}
	return (issued[1] - issued[0]) * float64(calls*threads) / (secs[1] - secs[0]), nil
}

// bindRow is f, formed from g, bound to its first input holding v, a
// table of zeros for each input after it, and an output of its own, each on
// cache lines of its own.
func bindRow(f *formed, g *graph.Graph, v uint32) (*bound, error) {
	vals := map[graph.ValueID]unsafe.Pointer{}
	var keep [][]uint32
	for i, in := range g.In {
		w := lineWords(g.Values[in].Width)
		if i == 0 {
			for j := range w {
				w[j] = v
			}
		}
		vals[in], keep = unsafe.Pointer(&w[0]), append(keep, w)
	}
	out := lineWords(g.Values[g.Out[0]].Width)
	vals[g.Out[0]], keep = unsafe.Pointer(&out[0]), append(keep, out)
	b, err := f.bind(vals)
	if err != nil {
		return nil, err
	}
	b.keep = append(b.keep, keep...)
	return b, nil
}

// lineWords is n words starting a cache line, the rest of their last line
// past them unused.
func lineWords(n int) []uint32 {
	w := make([]uint32, n+32)
	first := (64 - int(uintptr(unsafe.Pointer(&w[0]))%64)) % 64 / 4
	return w[first : first+n]
}

// classRate is class cl's rate, one thread's and every worker's together,
// and its delay, the seconds an instruction waits for the one before; zero
// for a class the composed probe does not step. A rate is rateChains values
// of one element through a step each a trip, the integer class's a row's
// (rowGraph); a delay is one value through delaySteps steps a trip, over
// the row stepRow names.
func classRate(fm former, cl cost.Class, lanes, workers int) (cost.Rate, float64, error) {
	if _, _, _, ok := hostStep(cl); !ok {
		return cost.Rate{}, 0, nil
	}
	one, err := stepRate(fm, cl, lanes, 1)
	if err != nil {
		return cost.Rate{}, 0, err
	}
	all, err := stepRate(fm, cl, lanes, workers)
	if err != nil {
		return cost.Rate{}, 0, err
	}
	chain, err := chainRate(fm, cl, lanes)
	if err != nil {
		return cost.Rate{}, 0, err
	}
	return cost.Rate{Thread: one, Total: all}, 1 / chain, nil
}

// stepRate is the instructions of class cl a second threads issue
// together through its rate's form.
func stepRate(fm former, cl cost.Class, lanes, threads int) (float64, error) {
	step, elem, v, ok := hostStep(cl)
	if !ok {
		return 0, fmt.Errorf("probe: no step of the %v class", cl)
	}
	count := func(f *formed) float64 { return f.count(cl) }
	if cl == cost.ClassInt {
		return hostRate(fm, func(vectors int) *graph.Graph { return rowGraph(step, elem, lanes*vectors) }, v,
			rowShort, rowLong, rowCalls, threads, count, float64((rowLong-rowShort)*rateChains*rowSteps))
	}
	return hostRate(fm, func(trips int) *graph.Graph { return iterateRow(step, elem, 1, rateChains, 1, trips) }, v,
		tripsShort, tripsLong, iterateCalls, threads, count, float64((tripsLong-tripsShort)*rateChains))
}

// chainRate is the instructions of class cl a second one value issues
// through delaySteps chained steps a trip, over the row stepRow names.
func chainRate(fm former, cl cost.Class, lanes int) (float64, error) {
	step, elem, v, ok := hostStep(cl)
	if !ok {
		return 0, fmt.Errorf("probe: no step of the %v class", cl)
	}
	n := stepRow(cl, lanes)
	return hostRate(fm, func(trips int) *graph.Graph { return iterateRow(step, elem, n, 1, delaySteps, trips) }, v,
		tripsShort, tripsLong, iterateCalls, 1, func(f *formed) float64 { return f.count(cl) }, float64((tripsLong-tripsShort)*delaySteps))
}

// issued is every instruction of f the estimator counts.
func issued(f *formed) float64 {
	var n float64
	for cl := range cost.NumClasses {
		n += f.count(cl)
	}
	return n
}

// mixRate is the instructions one thread and every worker together issue
// a second through the mix (mixGraph), every class the estimator counts.
func mixRate(fm former, workers int) (cost.Rate, error) {
	one, err := mixAt(fm, 1)
	if err != nil {
		return cost.Rate{}, err
	}
	all, err := mixAt(fm, workers)
	return cost.Rate{Thread: one, Total: all}, err
}

// mixAt is the instructions threads issue together a second through the
// mix.
func mixAt(fm former, threads int) (float64, error) {
	return hostRate(fm, mixGraph, 0, tripsShort, tripsLong, iterateCalls, threads, issued, float64((tripsLong-tripsShort)*2*mixAdds))
}

// callGraph is the smallest form: one element doubled.
func callGraph() *graph.Graph {
	b := graph.New()
	x := b.In("x", 1)
	b.Out(b.Map("probe_call", graph.Attrs{Fn: graph.MapMul, Const: 2}, nil, 1, x))
	return b.G
}

// callCost is the seconds a call of the smallest form costs (callGraph).
func callCost(fm former) (float64, error) {
	g := callGraph()
	f, err := fm.form(g, 1)
	if err != nil {
		return 0, err
	}
	defer f.close()
	b, err := bindRow(f, g, math.Float32bits(1))
	if err != nil {
		return 0, err
	}
	const calls = 1 << 20
	secs := parallel(1, func(int) {
		for range calls {
			b.run()
		}
	})
	runtime.KeepAlive(b)
	return secs / calls, nil
}
