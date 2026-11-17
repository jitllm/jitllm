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

// chainWidth is a chain's row, one vector of the composer's, and
// chainDepth its terms: deep enough that a call's own cost is a small part
// of the chain's.
const (
	chainWidth = 8
	chainDepth = 2048
)

// chainGraph is a row of chainWidth through chainDepth terms of step, each
// a function of the one before.
func chainGraph(step func(last graph.Arg) graph.Term) *graph.Graph {
	b := graph.New()
	x := b.In("x", chainWidth)
	terms := []graph.Term{{Fn: graph.MapAdd, Args: [3]graph.Arg{{Kind: graph.ArgIn}, {Kind: graph.ArgConst}}}}
	for range chainDepth {
		terms = append(terms, step(graph.Arg{Kind: graph.ArgTerm, Index: len(terms) - 1}))
	}
	b.Out(b.Emit(graph.OpExpr, []graph.ValueID{x}, []graph.Result{{Name: "chain", Width: chainWidth}}, nil,
		graph.Attrs{Expr: terms}, nil, nil)[0])
	return b.G
}

// chainStep is class cl's chained term, the next of the last, and the
// row's value that keeps the chain finite; false for a class the composed
// probe does not measure yet.
func chainStep(cl cost.Class) (func(last graph.Arg) graph.Term, float32, bool) {
	in := graph.Arg{Kind: graph.ArgIn}
	switch cl {
	case cost.ClassSpecial:
		return func(last graph.Arg) graph.Term {
			return graph.Term{Fn: graph.MapDiv, Args: [3]graph.Arg{last, in}}
		}, 1, true
	}
	return nil, 0, false
}

// chainDelay is the seconds an instruction of class cl waits for the one
// before: the composer's form of its chain, each call's output the next
// call's input so the calls wait for one another too, over the
// instructions of the class the estimator counts in it. False for a class
// the composed probe does not measure yet.
func chainDelay(fm former, cl cost.Class) (float64, bool, error) {
	step, v, ok := chainStep(cl)
	if !ok {
		return 0, false, nil
	}
	f, err := fm.form(chainGraph(step), 1)
	if err != nil {
		return 0, false, err
	}
	defer f.close()
	n := f.count(cl)
	if n == 0 {
		return 0, false, fmt.Errorf("probe: the %v chain counts none of the class", cl)
	}
	x := make([]uint32, chainWidth+16)
	for i := range x {
		x[i] = math.Float32bits(v)
	}
	b, err := f.bind(map[graph.ValueID]unsafe.Pointer{0: unsafe.Pointer(&x[0]), 1: unsafe.Pointer(&x[0])})
	if err != nil {
		return 0, false, err
	}
	b.keep = append(b.keep, x)
	const calls = 1 << 13
	secs := parallel(1, func(int) {
		for range calls {
			b.run()
		}
	})
	runtime.KeepAlive(b)
	if !(secs > 0) || math.IsInf(secs, 0) {
		return 0, false, fmt.Errorf("probe: %d chains took %v seconds", calls, secs)
	}
	return secs / (calls * n), true, nil
}
