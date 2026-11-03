package model

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// deltaMixer is a linear block's mixer over its normed row n, as the
// container states it: q, k and v in one projection, through the causal
// convolution over the block's window and SiLU; q and k each normalised to
// unit length per key head (an unweighted head norm at eps over the head's
// width, then the head's 1/sqrt width), q scaled by it once more; the
// output gate's projection, through a low-rank pair where the block stores
// one; the decay's logits -- one a value head, deinterleaved from the beta
// projection's groups, or one a key channel of each, through the stored
// low-rank pair -- with the stored dt bias, softplus and the stored log
// decay, or the stored lower bound's sigmoid form, exponentiated; beta the
// sigmoid of its logits; the delta rule; each value head normed against the
// stored norm, gated by its gate (SiLU, or sigmoid where the container
// states it or the decay is a channel's).
func (m *Model) deltaMixer(b *graph.Builder, f graph.Facts, n graph.ValueID, has func(jlm.Role) bool,
	matvec func(graph.ValueID, jlm.Role, int, ...jlm.Role) graph.ValueID) (graph.ValueID, error) {
	c := m.Cfg
	g := c.delta()
	vh, kh, dk, dv := g.vHeads, g.kHeads, g.kDim, g.vDim
	if vh < 1 || kh < 1 || dk < 1 || dv < 1 || vh%kh != 0 || g.conv < 2 {
		return 0, fmt.Errorf("a delta rule of %d value heads of %d over %d key heads of %d, a window of %d", vh, dv, kh, dk, g.conv)
	}
	vec := func(r jlm.Role) []graph.Binding {
		s := f.Inventory[r]
		return []graph.Binding{{Role: r, K: s.K, Rows: s.Rows}}
	}
	mapc := func(name string, fn graph.MapFn, k float64, width int, w []graph.Binding, in ...graph.ValueID) graph.ValueID {
		return mapRow(b, name, graph.Attrs{Fn: fn, Const: k}, w, width, in...)
	}
	slice := func(name string, x graph.ValueID, head, off, width, out int) graph.ValueID {
		return b.Emit(graph.OpSlice, []graph.ValueID{x}, []graph.Result{{Name: name, Width: out}}, nil,
			graph.Attrs{HeadDim: head, Offset: off, Width: width}, nil, nil)[0]
	}
	silu := func(name string, x graph.ValueID) graph.ValueID {
		return b.Expr(name, b.G.Values[x].Width, []graph.Term{
			{Fn: graph.MapSigmoid, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}}},
			{Fn: graph.MapMul, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}, {Kind: graph.ArgTerm, Index: 0}}},
		}, nil, x)
	}
	lowRank := func(x graph.ValueID, down, up jlm.Role, rows int) graph.ValueID {
		return matvec(matvec(x, down, f.Inventory[down].Rows), up, rows)
	}

	qk := kh * dk
	mixed := matvec(n, jlm.RoleAttnQKV, g.chans)
	conv := b.Emit(graph.OpStateConv, []graph.ValueID{mixed}, []graph.Result{{Name: "conv", Width: g.chans}},
		[]graph.Binding{{Role: jlm.RoleSSMConv1d, K: g.conv, Rows: g.chans}}, graph.Attrs{Taps: g.conv},
		[]graph.StateRef{{State: graph.StateRecConv}}, []graph.StateRef{{State: graph.StateRecConv, Version: 1}})[0]
	act := silu("conv.act", conv)
	q := slice("q", act, 0, 0, qk, qk)
	k := slice("k", act, 0, qk, qk, qk)
	v := slice("v", act, 0, 2*qk, vh*dv, vh*dv)
	eps := c.RMSEps
	if g.chanDecay {
		eps = kdaL2Eps
	}
	unit := func(name string, x graph.ValueID, scale float64) graph.ValueID {
		y := b.Emit(graph.OpHeadNorm, []graph.ValueID{x}, []graph.Result{{Name: name, Width: qk}}, nil,
			graph.Attrs{Eps: eps / float64(dk), HeadDim: dk}, nil, nil)[0]
		return mapc(name+".scaled", graph.MapMul, scale, qk, nil, y)
	}
	s := float64(g.scale)
	k = unit("k.unit", k, s)
	q = mapc("q.scaled", graph.MapMul, s, qk, nil, unit("q.unit", q, s))

	var z, alpha, braw graph.ValueID
	nd := vh
	switch {
	case g.chanDecay:
		nd = vh * dk
		if has(jlm.RoleSSMGA) {
			z = lowRank(n, jlm.RoleSSMGA, jlm.RoleSSMGB, vh*dv)
		} else {
			z = matvec(n, jlm.RoleAttnGate, vh*dv)
		}
		alpha = lowRank(n, jlm.RoleSSMFA, jlm.RoleSSMFB, nd)
		braw = matvec(n, jlm.RoleSSMBA, vh)
	default:
		z = matvec(n, jlm.RoleAttnGate, vh*dv)
		// Each key head's group of the projection is its value heads' beta
		// logits, then their decay logits.
		rep := vh / kh
		ba := matvec(n, jlm.RoleSSMBA, 2*vh)
		braw, alpha = slice("beta.logit", ba, 2*rep, 0, rep, vh), slice("decay.logit", ba, 2*rep, rep, rep, vh)
	}
	t := mapc("decay.dt", graph.MapAdd, 0, nd, vec(jlm.RoleSSMDtBias), alpha)
	if lb := c.KDALowerBound; lb != 0 {
		t = mapc("decay.a", graph.MapMul, 0, nd, vec(jlm.RoleSSMA), t)
		t = mapc("decay.bound", graph.MapMul, float64(lb), nd, nil, mapc("decay.sig", graph.MapSigmoid, 0, nd, nil, mapc("decay.neg", graph.MapMul, -1, nd, nil, t)))
	} else {
		t = mapc("decay.a", graph.MapMul, 0, nd, vec(jlm.RoleSSMA), mapc("decay.sp", graph.MapSoftplus, 0, nd, nil, t))
	}
	decay := mapc("decay", graph.MapExp, 0, nd, nil, t)
	beta := mapc("beta", graph.MapSigmoid, 0, vh, nil, braw)

	at := b.G.StateVersion(graph.StateRecState)
	o := b.Emit(graph.OpDeltaRule, []graph.ValueID{q, k, v, decay, beta}, []graph.Result{{Name: "delta", Width: vh * dv}}, nil,
		graph.Attrs{NHead: vh, NKVHead: kh, HeadDim: dk, Tiled: g.tiled, Span: stateSpan(c, vh*dv*dk)},
		[]graph.StateRef{{State: graph.StateRecState, Version: at}}, []graph.StateRef{{State: graph.StateRecState, Version: at + 1}})[0]
	o = b.HeadNorm(o, jlm.RoleSSMNorm, dv, c.RMSEps)
	if c.k3Fault == k3FaultNoKDAGate {
		return keepBound(b, o, z), nil
	}
	if g.chanDecay || c.q4eSigmoidGate() {
		return sigmoidMul(b, "gated", z, o), nil
	}
	return b.Expr("gated", vh*dv, []graph.Term{
		{Fn: graph.MapSigmoid, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}}},
		{Fn: graph.MapMul, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}, {Kind: graph.ArgTerm, Index: 0}}},
		{Fn: graph.MapMul, Args: [3]graph.Arg{{Kind: graph.ArgTerm, Index: 1}, {Kind: graph.ArgIn, Index: 1}}},
	}, nil, z, o), nil
}

// stateSpan is a delta rule's state floats a sequence where the block's
// state carries a tail past the rule's n (Config.recTailLen), 0 where it is
// the rule's alone.
func stateSpan(c *Config, n int) int {
	if t := c.recTailLen(); t != 0 {
		return n + t
	}
	return 0
}
