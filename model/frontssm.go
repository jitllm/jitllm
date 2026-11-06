package model

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// stateConv is the causal convolution of x over the block's window of
// taps, its stored bias added where the block stores one.
func stateConv(b *graph.Builder, f graph.Facts, x graph.ValueID, taps int) graph.ValueID {
	w := b.G.Values[x].Width
	y := b.Emit(graph.OpStateConv, []graph.ValueID{x}, []graph.Result{{Name: "conv", Width: w}},
		[]graph.Binding{{Role: jlm.RoleSSMConv1d, K: taps, Rows: w}}, graph.Attrs{Taps: taps},
		[]graph.StateRef{{State: graph.StateRecConv}}, []graph.StateRef{{State: graph.StateRecConv, Version: 1}})[0]
	if s, ok := f.Inventory[jlm.RoleSSMConvBias]; ok {
		y = mapRow(b, "conv.biased", graph.Attrs{Fn: graph.MapAdd}, []graph.Binding{{Role: jlm.RoleSSMConvBias, K: s.K, Rows: s.Rows}}, w, y)
	}
	return y
}

// siluOf is x times its sigmoid.
func siluOf(b *graph.Builder, name string, x graph.ValueID) graph.ValueID {
	return b.Expr(name, b.G.Values[x].Width, []graph.Term{
		{Fn: graph.MapSigmoid, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}}},
		{Fn: graph.MapMul, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}, {Kind: graph.ArgTerm, Index: 0}}},
	}, nil, x)
}

// siluGated is y times SiLU of z.
func siluGated(b *graph.Builder, name string, z, y graph.ValueID) graph.ValueID {
	return b.Expr(name, b.G.Values[y].Width, []graph.Term{
		{Fn: graph.MapSigmoid, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}}},
		{Fn: graph.MapMul, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}, {Kind: graph.ArgTerm, Index: 0}}},
		{Fn: graph.MapMul, Args: [3]graph.Arg{{Kind: graph.ArgTerm, Index: 1}, {Kind: graph.ArgIn, Index: 1}}},
	}, nil, z, y)
}

// selectiveUpdate is the additive state update over the block's state:
// for value head h of heads, its rows of v's width over heads by the key's
// width, paired with key head h over the heads a key head serves, decayed
// and plus (v times dt) along k, then read against q; plus the stored skip
// D times v, one a value head.
func selectiveUpdate(b *graph.Builder, f graph.Facts, q, k, v, decay, dt graph.ValueID, heads, kHeads, kDim int) graph.ValueID {
	w := b.G.Values[v].Width
	y := b.Emit(graph.OpDeltaRule, []graph.ValueID{q, k, v, decay, dt}, []graph.Result{{Name: "ssm.y", Width: w}}, nil,
		graph.Attrs{NHead: heads, NKVHead: kHeads, HeadDim: kDim, Additive: true},
		[]graph.StateRef{{State: graph.StateRecState}}, []graph.StateRef{{State: graph.StateRecState, Version: 1}})[0]
	s := f.Inventory[jlm.RoleSSMD]
	skip := mapRow(b, "ssm.skip", graph.Attrs{Fn: graph.MapMul}, []graph.Binding{{Role: jlm.RoleSSMD, K: s.K, Rows: s.Rows}}, w, v)
	return b.Add(y, skip)
}

// ssdMixer is a selective state-space mixer over its normed row n, with
// one decay a value head, as the container states it: the mixed
// projection of C, B and x (C and B a group each of key heads, x the value
// heads), through the causal convolution with its stored bias and SiLU;
// dt the softplus of its projection plus the stored bias, the decay
// exp(A dt) of the stored A; the additive update with C read against the
// state and the stored skip; then y times SiLU of the gate's projection,
// through the grouped norm before or after the gate where the block
// stores one (each group of the value channels over the stored groups
// normed alone, against its slice of the weight).
func (m *Model) ssdMixer(b *graph.Builder, f graph.Facts, n graph.ValueID, has func(jlm.Role) bool,
	matvec func(graph.ValueID, jlm.Role, int, ...jlm.Role) graph.ValueID) (graph.ValueID, error) {
	c := m.Cfg
	g := c.Delta()
	vh, kh, dk, dv := g.VHeads, g.KHeads, g.KDim, g.VDim
	if vh < 1 || kh < 1 || dk < 1 || dv < 1 || vh%kh != 0 || g.Conv < 2 || g.Tiled || g.chanDecay {
		return 0, fmt.Errorf("a state update of %d value heads of %d over %d key heads of %d, a window of %d", vh, dv, kh, dk, g.Conv)
	}
	inner, qk := vh*dv, kh*dk
	vec := func(r jlm.Role) []graph.Binding {
		s := f.Inventory[r]
		return []graph.Binding{{Role: r, K: s.K, Rows: s.Rows}}
	}
	slice := func(name string, x graph.ValueID, off, width int) graph.ValueID {
		return b.Emit(graph.OpSlice, []graph.ValueID{x}, []graph.Result{{Name: name, Width: width}}, nil,
			graph.Attrs{Offset: off, Width: width}, nil, nil)[0]
	}
	mixed := matvec(n, jlm.RoleAttnQKV, g.Chans)
	z := matvec(n, jlm.RoleAttnGate, inner)
	dtRaw := matvec(n, jlm.RoleSSMBA, vh)
	act := siluOf(b, "conv.act", stateConv(b, f, mixed, g.Conv))
	cq, bk, x := slice("c", act, 0, qk), slice("b", act, qk, qk), slice("x", act, 2*qk, inner)
	dt := mapRow(b, "dt", graph.Attrs{Fn: graph.MapSoftplus}, nil, vh,
		mapRow(b, "dt.biased", graph.Attrs{Fn: graph.MapAdd}, vec(jlm.RoleSSMDtBias), vh, dtRaw))
	decay := mapRow(b, "decay", graph.Attrs{Fn: graph.MapExp}, nil, vh,
		mapRow(b, "decay.a", graph.Attrs{Fn: graph.MapMul}, vec(jlm.RoleSSMA), vh, dt))
	y := selectiveUpdate(b, f, cq, bk, x, decay, dt, vh, kh, dk)
	if !has(jlm.RoleSSMNorm) {
		return siluGated(b, "gated", z, y), nil
	}
	groups := c.SSMNormGroups
	if groups < 1 || inner%groups != 0 {
		return 0, fmt.Errorf("a grouped norm of %d groups over %d channels", groups, inner)
	}
	norm := func(v graph.ValueID) graph.ValueID {
		u := b.Emit(graph.OpHeadNorm, []graph.ValueID{v}, []graph.Result{{Name: "ssm.norm", Width: inner}}, nil,
			graph.Attrs{Eps: c.RMSEps, HeadDim: inner / groups}, nil, nil)[0]
		return mapRow(b, "ssm.normed", graph.Attrs{Fn: graph.MapMul}, vec(jlm.RoleSSMNorm), inner, u)
	}
	if c.SSMNormBeforeGate {
		return siluGated(b, "gated", z, norm(y)), nil
	}
	return norm(siluGated(b, "gated", z, y)), nil
}

// scanMixer is a selective state-space mixer with a decay a state element
// of each channel, over its normed row n, as the container states it: x's
// projection through the causal convolution with its stored bias and
// SiLU; dt's bottleneck, B and C off the convolved x, each normed against
// its stored weight where the block stores the three; dt the softplus of
// the bottleneck's projection plus the stored bias, each element's decay
// exp(dt A) of the stored A (a row of the state a channel); the additive
// update, every channel a value head of one row over B and C, and the
// stored skip; then y times SiLU of the gate's projection.
func (m *Model) scanMixer(b *graph.Builder, f graph.Facts, n graph.ValueID, has func(jlm.Role) bool,
	matvec func(graph.ValueID, jlm.Role, int, ...jlm.Role) graph.ValueID) (graph.ValueID, error) {
	c := m.Cfg
	g := c.Delta()
	inner, dk, rank := g.VHeads, g.KDim, g.Rank
	if inner < 1 || dk < 1 || rank < 1 || g.Conv < 2 {
		return 0, fmt.Errorf("a selective scan of %d channels of a state of %d through a rank of %d, a window of %d", inner, dk, rank, g.Conv)
	}
	vec := func(r jlm.Role) []graph.Binding {
		s := f.Inventory[r]
		return []graph.Binding{{Role: r, K: s.K, Rows: s.Rows}}
	}
	xm := matvec(n, jlm.RoleAttnQKV, inner)
	z := matvec(n, jlm.RoleAttnGate, inner)
	x := siluOf(b, "conv.act", stateConv(b, f, xm, g.Conv))
	dtl, bk, cq := matvec(x, jlm.RoleSSMXDt, rank), matvec(x, jlm.RoleSSMXB, dk), matvec(x, jlm.RoleSSMXC, dk)
	if has(jlm.RoleSSMDtNorm) || has(jlm.RoleSSMBNorm) || has(jlm.RoleSSMCNorm) {
		dtl, bk, cq = b.Norm(dtl, jlm.RoleSSMDtNorm, c.RMSEps), b.Norm(bk, jlm.RoleSSMBNorm, c.RMSEps), b.Norm(cq, jlm.RoleSSMCNorm, c.RMSEps)
	}
	dt := mapRow(b, "dt", graph.Attrs{Fn: graph.MapSoftplus}, nil, inner,
		mapRow(b, "dt.biased", graph.Attrs{Fn: graph.MapAdd}, vec(jlm.RoleSSMDtBias), inner, matvec(dtl, jlm.RoleSSMDtProj, inner)))
	decay := mapRow(b, "decay", graph.Attrs{Fn: graph.MapExp}, nil, inner*dk,
		mapRow(b, "decay.a", graph.Attrs{Fn: graph.MapMul}, vec(jlm.RoleSSMA), inner*dk, dt))
	return siluGated(b, "gated", z, selectiveUpdate(b, f, cq, bk, x, decay, dt, inner, 1, dk)), nil
}

// shortConvMixer is a gated short convolution over its normed row n: x's
// projection times B's, through the causal convolution, times C's -- or, in
// a non-causal model, whose rows see the rows after them, a centred one
// across the call's rows, each row's taps inside its window (RowWindowed).
func (m *Model) shortConvMixer(b *graph.Builder, n graph.ValueID,
	matvec func(graph.ValueID, jlm.Role, int, ...jlm.Role) graph.ValueID) (graph.ValueID, error) {
	g := m.Cfg.Delta()
	w := g.Chans
	if w < 1 || g.Conv < 2 {
		return 0, fmt.Errorf("a short convolution of %d channels over a window of %d", w, g.Conv)
	}
	mul := func(name string, x, y graph.ValueID) graph.ValueID {
		return mapRow(b, name, graph.Attrs{Fn: graph.MapMul}, nil, w, x, y)
	}
	bx := mul("bx", matvec(n, jlm.RoleSCB, w), matvec(n, jlm.RoleSCX, w))
	taps := []graph.Binding{{Role: jlm.RoleSSMConv1d, K: g.Conv, Rows: w}}
	if c := m.Cfg; c.NonCausal {
		conv := b.Emit(graph.OpRowConv, []graph.ValueID{bx}, []graph.Result{{Name: "conv", Width: w}}, taps,
			graph.Attrs{Taps: g.Conv, RowWindows: c.RowWindowed}, nil, nil)[0]
		return mul("cy", matvec(n, jlm.RoleSCC, w), conv), nil
	}
	conv := b.Emit(graph.OpStateConv, []graph.ValueID{bx}, []graph.Result{{Name: "conv", Width: w}}, taps,
		graph.Attrs{Taps: g.Conv},
		[]graph.StateRef{{State: graph.StateRecConv}}, []graph.StateRef{{State: graph.StateRecConv, Version: 1}})[0]
	return mul("cy", matvec(n, jlm.RoleSCC, w), conv), nil
}
