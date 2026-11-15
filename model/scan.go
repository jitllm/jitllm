package model

import "github.com/jitllm/jitllm/graph"

func scanArg(k graph.ArgKind, i int) graph.Arg { return graph.Arg{Kind: k, Index: i} }

func scanTerm(fn graph.MapFn, a, b graph.Arg) graph.Term {
	return graph.Term{Fn: fn, Args: [3]graph.Arg{a, b}}
}

// gatedDeltaTerms is the gated delta rule over q, k, v, decay and beta
// (operands 0 to 4) as a scan's terms: the state times the decay, plus k
// times (v less the decayed state's sum against k) times beta; read
// against q.
func gatedDeltaTerms() ([]graph.Term, int) {
	in, t := func(i int) graph.Arg { return scanArg(graph.ArgIn, i) }, func(i int) graph.Arg { return scanArg(graph.ArgTerm, i) }
	return []graph.Term{
		scanTerm(graph.MapMul, scanArg(graph.ArgState, 0), in(3)),
		scanTerm(graph.MapMul, t(0), in(1)),
		scanTerm(graph.MapSub, in(2), scanArg(graph.ArgSum, 1)),
		scanTerm(graph.MapMul, t(2), in(4)),
		scanTerm(graph.MapMul, in(1), t(3)),
		scanTerm(graph.MapAdd, t(0), t(4)),
		scanTerm(graph.MapMul, t(5), in(0)),
	}, 5
}

// selectiveTerms is the selective state update over q, k, v, decay and dt
// (operands 0 to 4) as a scan's terms: the state times the decay, plus k
// times v times dt; read against q.
func selectiveTerms() ([]graph.Term, int) {
	in, t := func(i int) graph.Arg { return scanArg(graph.ArgIn, i) }, func(i int) graph.Arg { return scanArg(graph.ArgTerm, i) }
	return []graph.Term{
		scanTerm(graph.MapMul, scanArg(graph.ArgState, 0), in(3)),
		scanTerm(graph.MapMul, in(2), in(4)),
		scanTerm(graph.MapMul, in(1), t(1)),
		scanTerm(graph.MapAdd, t(0), t(2)),
		scanTerm(graph.MapMul, t(3), in(0)),
	}, 3
}

// stateScan is one OpStateScan of terms over in, each operand along its
// axes, into a row of width named name, advancing the block's recurrent
// state.
func stateScan(b *graph.Builder, name string, width int, in []graph.ValueID, axes []graph.ScanAxis, terms []graph.Term, update int, a graph.Attrs) graph.ValueID {
	a.Expr, a.Update = terms, update
	for _, m := range axes {
		a.Words = append(a.Words, uint64(m))
	}
	at := b.G.StateVersion(graph.StateRecState)
	return b.Emit(graph.OpStateScan, in, []graph.Result{{Name: name, Width: width}}, nil, a,
		[]graph.StateRef{{State: graph.StateRecState, Version: at}}, []graph.StateRef{{State: graph.StateRecState, Version: at + 1}})[0]
}
