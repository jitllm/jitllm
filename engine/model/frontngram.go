package model

import (
	"math"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// ngramStreams is the n-gram embedding's block (ngram.go) over the streams
// xs, stated as ops: the rows' gathered table rows, the block's side input,
// project to a key per stream and one value; each stream's normed key
// against its normed residual, over sqrt(n_embd), gives a gate, the
// sigmoid of its signed square root (the magnitude floored at
// kernels.SqrtSigmoidFloor); the gated values, and their normed
// convolution through silu -- depthwise, dilated by the n-gram, its history
// in the delta rule's state slot after the predecessors -- are added to the
// streams.
func (m *Model) ngramStreams(b *graph.Builder, xs []graph.ValueID,
	matvec func(graph.ValueID, jlm.Role, int, ...jlm.Role) graph.ValueID) []graph.ValueID {
	c := m.Cfg
	h, d := len(xs), c.NEmbd
	hd := h * d
	emb := b.In("ngram", c.NGramWidth())
	norm := func(name string, v graph.ValueID, role jlm.Role) graph.ValueID {
		n := b.Emit(graph.OpHeadNorm, []graph.ValueID{v}, []graph.Result{{Name: name, Width: hd}}, nil,
			graph.Attrs{Eps: c.RMSEps, HeadDim: d}, nil, nil)[0]
		return mapRow(b, name+".w", graph.Attrs{Fn: graph.MapMul}, []graph.Binding{{Role: role, K: hd}}, hd, n)
	}
	kn := norm("ng.key", matvec(emb, jlm.RoleNGramKey, hd), jlm.RoleNGramNormKey)
	val := matvec(emb, jlm.RoleNGramValue, d)
	cat := b.Emit(graph.OpConcat, xs, []graph.Result{{Name: "ng.streams", Width: hd}}, nil, graph.Attrs{}, nil, nil)[0]
	qn := norm("ng.query", cat, jlm.RoleNGramNormQuery)
	dot := b.Emit(graph.OpReduce, []graph.ValueID{mapRow(b, "ng.kq", graph.Attrs{Fn: graph.MapMul}, nil, hd, kn, qn)},
		[]graph.Result{{Name: "ng.dot", Width: h}}, nil, graph.Attrs{Fn: graph.MapAdd, Span: d}, nil, nil)[0]
	in0 := graph.Arg{Kind: graph.ArgIn, Index: 0}
	term := func(i int) graph.Arg { return graph.Arg{Kind: graph.ArgTerm, Index: i} }
	konst := func(v float64) graph.Arg { return graph.Arg{Kind: graph.ArgConst, Const: v} }
	gate := b.Expr("ng.gate", h, []graph.Term{
		{Fn: graph.MapMul, Args: [3]graph.Arg{in0, konst(1 / math.Sqrt(float64(d)))}},
		{Fn: graph.MapAbs, Args: [3]graph.Arg{term(0)}},
		{Fn: graph.MapMax, Args: [3]graph.Arg{term(1), konst(float64(graph.SqrtSigmoidFloor))}},
		{Fn: graph.MapSqrt, Args: [3]graph.Arg{term(2)}},
		{Fn: graph.MapLess, Args: [3]graph.Arg{konst(0), term(0)}},
		{Fn: graph.MapLess, Args: [3]graph.Arg{term(0), konst(0)}},
		{Fn: graph.MapSub, Args: [3]graph.Arg{term(4), term(5)}},
		{Fn: graph.MapMul, Args: [3]graph.Arg{term(6), term(3)}},
		{Fn: graph.MapSigmoid, Args: [3]graph.Arg{term(7)}},
	}, nil, dot)
	gv := make([]graph.ValueID, h)
	for k := range h {
		gv[k] = mapRow(b, "ng.gv", graph.Attrs{Fn: graph.MapMul}, nil, d, slice1(b, "ng.gk", gate, k), val)
	}
	gvs := b.Emit(graph.OpConcat, gv, []graph.Result{{Name: "ng.gvs", Width: hd}}, nil, graph.Attrs{}, nil, nil)[0]
	g := c.delta()
	at := b.G.StateVersion(graph.StateRecState)
	convIn := norm("ng.conv.in", gvs, jlm.RoleNGramNormConv)
	conv := b.Emit(graph.OpStateConv, []graph.ValueID{convIn},
		[]graph.Result{{Name: "ng.conv", Width: hd}},
		[]graph.Binding{{Role: jlm.RoleNGramConv, K: c.NGramConv, Rows: hd}},
		graph.Attrs{Taps: c.NGramConv, Dil: c.NGram, Offset: g.vHeads*g.vDim*g.kDim + c.NGram - 1,
			Span: stateSpan(c, g.vHeads*g.vDim*g.kDim)},
		[]graph.StateRef{{State: graph.StateRecState, Version: at}}, []graph.StateRef{{State: graph.StateRecState, Version: at + 1}})[0]
	act := siluOf(b, "ng.conv.act", conv)
	if c.q4eFault == q4eFaultNoNGramConv {
		act = mapRow(b, "ng.conv.off", graph.Attrs{Fn: graph.MapMul, Const: 0}, nil, hd, act)
	}
	out := make([]graph.ValueID, h)
	for k, x := range xs {
		ck := b.Emit(graph.OpSlice, []graph.ValueID{act}, []graph.Result{{Name: "ng.ck", Width: d}}, nil,
			graph.Attrs{Offset: k * d, Width: d}, nil, nil)[0]
		y := b.Add(gv[k], ck)
		if c.q4eFault == q4eFaultNoNGram {
			y = mapRow(b, "ng.off", graph.Attrs{Fn: graph.MapMul, Const: 0}, nil, d, y)
		}
		out[k] = b.Add(x, y)
	}
	return out
}
