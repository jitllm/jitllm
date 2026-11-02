package model

import (
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// sinkhornMix is a sublayer's input from the streams xs through a
// Sinkhorn-normalised hyper-connection (mixesStreamsSinkhorn): the streams
// side by side, RMS-normed with no weight, project through fn to (2+S)*S
// mixes, each third scaled by its own of scale's three and offset by base.
// The collapse weights (a sigmoid, plus HCEps) weigh the streams into the
// input; post (twice a sigmoid) places the sublayer's output into each
// stream; comb (a softmax along each row plus HCEps, then normalised by
// columns, then HCIters-1 rounds of rows then columns, each over its sum
// plus HCEps) mixes them, comb[j][k] stream j into stream k (row-major).
func (m *Model) sinkhornMix(b *graph.Builder, xs []graph.ValueID, fn, base, scale jlm.Role) (in, post, comb graph.ValueID) {
	c := m.Cfg
	s, d := len(xs), c.NEmbd
	nm := (2 + s) * s
	eps := c.HCEps
	cat := b.Emit(graph.OpConcat, xs, []graph.Result{{Name: "hc.streams", Width: s * d}}, nil, graph.Attrs{}, nil, nil)[0]
	flat := b.Emit(graph.OpHeadNorm, []graph.ValueID{cat}, []graph.Result{{Name: "hc.flat", Width: s * d}}, nil,
		graph.Attrs{Eps: c.RMSEps, HeadDim: s * d}, nil, nil)[0]
	mix := b.MatVec(flat, fn, nm)
	part := func(name string, off, w, si int) graph.ValueID {
		v := b.Emit(graph.OpSlice, []graph.ValueID{mix}, []graph.Result{{Name: name, Width: w}}, nil,
			graph.Attrs{Offset: off, Width: w}, nil, nil)[0]
		v = mapRow(b, name+".scaled", graph.Attrs{Fn: graph.MapMul, Offset: si, Width: 1}, []graph.Binding{{Role: scale, K: 3}}, w, v)
		return mapRow(b, name+".based", graph.Attrs{Fn: graph.MapAdd, Offset: off, Width: w}, []graph.Binding{{Role: base, K: nm}}, w, v)
	}
	pre := mapRow(b, "hc.pre", graph.Attrs{Fn: graph.MapAdd, Const: eps}, nil, s,
		mapRow(b, "hc.pre.sig", graph.Attrs{Fn: graph.MapSigmoid}, nil, s, part("hc.pre", 0, s, 0)))
	post = mapRow(b, "hc.post", graph.Attrs{Fn: graph.MapMul, Const: 2}, nil, s,
		mapRow(b, "hc.post.sig", graph.Attrs{Fn: graph.MapSigmoid}, nil, s, part("hc.post", s, s, 1)))
	reduce := func(name string, v graph.ValueID, fn graph.MapFn) graph.ValueID {
		return b.Emit(graph.OpReduce, []graph.ValueID{v}, []graph.Result{{Name: name, Width: s}}, nil,
			graph.Attrs{Fn: fn, Span: s}, nil, nil)[0]
	}
	plusEps := func(name string, v graph.ValueID, w int) graph.ValueID {
		return mapRow(b, name, graph.Attrs{Fn: graph.MapAdd, Const: eps}, nil, w, v)
	}
	logits := part("hc.comb", 2*s, s*s, 2)
	e := b.Expr("hc.comb.exp", s*s, []graph.Term{
		{Fn: graph.MapSub, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}, {Kind: graph.ArgIn, Index: 1}}},
		{Fn: graph.MapExp, Args: [3]graph.Arg{{Kind: graph.ArgTerm, Index: 0}}},
	}, nil, logits, reduce("hc.comb.max", logits, graph.MapMax))
	comb = plusEps("hc.comb.soft", mapRow(b, "hc.comb.p", graph.Attrs{Fn: graph.MapDiv}, nil, s*s, e, reduce("hc.comb.sum", e, graph.MapAdd)), s*s)
	// rows divides each row by its sum plus eps; cols each column by its.
	rows := func(v graph.ValueID) graph.ValueID {
		return mapRow(b, "hc.comb.rows", graph.Attrs{Fn: graph.MapDiv}, nil, s*s, v, plusEps("hc.comb.rsum", reduce("hc.comb.rsum", v, graph.MapAdd), s))
	}
	cols := func(v graph.ValueID) graph.ValueID {
		var sum graph.ValueID
		for j := range s {
			r := b.Emit(graph.OpSlice, []graph.ValueID{v}, []graph.Result{{Name: "hc.comb.row", Width: s}}, nil,
				graph.Attrs{Offset: j * s, Width: s}, nil, nil)[0]
			if j == 0 {
				sum = r
				continue
			}
			sum = b.Add(sum, r)
		}
		t := b.Emit(graph.OpTile, []graph.ValueID{plusEps("hc.comb.csum", sum, s)}, []graph.Result{{Name: "hc.comb.cols", Width: s * s}}, nil,
			graph.Attrs{Span: s}, nil, nil)[0]
		return mapRow(b, "hc.comb.c", graph.Attrs{Fn: graph.MapDiv}, nil, s*s, v, t)
	}
	if c.ds4Fault != ds4FaultCombSoftmax {
		comb = cols(comb)
		for range c.HCIters - 1 {
			comb = cols(rows(comb))
		}
	}
	in = mapRow(b, "hc.in", graph.Attrs{Fn: graph.MapMul}, nil, d, slice1(b, "hc.prek", pre, 0), xs[0])
	for k := 1; k < s; k++ {
		in = mapRow(b, "hc.in", graph.Attrs{Fn: graph.MapFma}, nil, d, slice1(b, "hc.prek", pre, k), xs[k], in)
	}
	return in, post, comb
}

// sinkhornPlace is the streams after a sublayer of output y under the
// hyper-connection's post and comb (sinkhornMix): stream k is post_k*y plus
// each stream j weighted by comb[j][k].
func (m *Model) sinkhornPlace(b *graph.Builder, xs []graph.ValueID, post, comb, y graph.ValueID) []graph.ValueID {
	s, d := len(xs), m.Cfg.NEmbd
	out := make([]graph.ValueID, s)
	for k := range s {
		acc := mapRow(b, "hc.placed", graph.Attrs{Fn: graph.MapMul}, nil, d, slice1(b, "hc.postk", post, k), y)
		for j, x := range xs {
			at := j*s + k
			if m.Cfg.ds4Fault == ds4FaultCombRows {
				at = k*s + j
			}
			acc = mapRow(b, "hc.placed", graph.Attrs{Fn: graph.MapFma}, nil, d, slice1(b, "hc.combjk", comb, at), x, acc)
		}
		out[k] = acc
	}
	return out
}
