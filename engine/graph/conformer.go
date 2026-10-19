package graph

import "github.com/jitllm/jitllm/format/jlm"

// Conformer describes a FastConformer block (the audio tower's), stated in
// the op language: a half feed-forward, relative-position self-attention
// over the call's rows, a convolution module, a second half feed-forward and
// an output LayerNorm, each sublayer's output added into the residual (the
// feed-forwards' at one half). The block reads its rows (In[0]) and the
// relative positions' sinusoids (In[1], 2R-1 rows of the width for R rows,
// positions R-1 down to -(R-1)).
type Conformer struct {
	Block       int
	NEmbd, NFFN int
	NHead       int
	HeadDim     int
	// Taps is the depthwise convolution's kernel, centred over the rows.
	Taps int
	Eps  float64
}

// ConformerHalf is the feed-forwards' residual scale.
const ConformerHalf = 0.5

// Build is the block's graph.
func (c *Conformer) Build() *Graph {
	g := &Graph{Version: Version, Block: c.Block}
	d := c.NEmbd
	x := g.value("x", d)
	pe := g.value("pos", d)
	g.In = []ValueID{x, pe}
	vec := func(r jlm.Role, n int) Binding { return Binding{Role: r, K: n} }
	mat := func(r jlm.Role, k, rows int) Binding { return Binding{Role: r, K: k, Rows: rows} }
	ln := func(name string, in ValueID, w, b jlm.Role) ValueID {
		out := g.value(name, d)
		g.op(Op{Kind: OpLayerNorm, In: []ValueID{in}, Out: []ValueID{out}, Weight: []Binding{vec(w, d), vec(b, d)},
			Attr: Attrs{Eps: c.Eps}})
		return out
	}
	mv := func(name string, in ValueID, w, b jlm.Role, rows int) ValueID {
		out := g.value(name, rows)
		ws := []Binding{mat(w, g.Values[in].Width, rows)}
		if b != 0 {
			ws = append(ws, vec(b, rows))
		}
		g.op(Op{Kind: OpMatVec, In: []ValueID{in}, Out: []ValueID{out}, Weight: ws})
		return out
	}
	add := func(name string, a, b ValueID, alpha float64) ValueID {
		out := g.value(name, d)
		g.op(Op{Kind: OpAdd, In: []ValueID{a, b}, Out: []ValueID{out}, Attr: Attrs{Alpha: alpha}})
		return out
	}
	act := func(name string, in ValueID) ValueID {
		out := g.value(name, g.Values[in].Width)
		g.op(Op{Kind: OpAct, In: []ValueID{in}, Out: []ValueID{out}, Attr: Attrs{Act: ActSiLU}})
		return out
	}
	ffn := func(pre string, in ValueID, norm, normB, up, upB, down, downB jlm.Role) ValueID {
		h := ln(pre+"_norm", in, norm, normB)
		u := act(pre+"_act", mv(pre+"_up", h, up, upB, c.NFFN))
		return add(pre+"_out", in, mv(pre+"_down", u, down, downB, d), ConformerHalf)
	}
	x1 := ffn("ff1", x, jlm.RoleAFF1Norm, jlm.RoleAFF1NormB, jlm.RoleAFF1Up, jlm.RoleAFF1UpB,
		jlm.RoleAFF1Down, jlm.RoleAFF1DownB)
	// Relative-position attention: q, k, v and the positions projected, the
	// scores (q+u)k + (q+v)p shifted to each pair's distance, over the rows.
	h := ln("attn_norm", x1, jlm.RoleAAttnNorm, jlm.RoleAAttnNormB)
	q := mv("q", h, jlm.RoleAQ, jlm.RoleAQB, d)
	k := mv("k", h, jlm.RoleAK, jlm.RoleAKB, d)
	v := mv("v", h, jlm.RoleAV, jlm.RoleAVB, d)
	p := mv("pos_proj", pe, jlm.RoleAPos, 0, d)
	a := g.value("attn", d)
	g.op(Op{Kind: OpRelAttend, In: []ValueID{q, k, v, p}, Out: []ValueID{a},
		Weight: []Binding{mat(jlm.RoleAPosU, c.HeadDim, c.NHead), mat(jlm.RoleAPosV, c.HeadDim, c.NHead)},
		Attr:   Attrs{NHead: c.NHead, HeadDim: c.HeadDim}})
	x2 := add("x_attn", x1, mv("attn_out", a, jlm.RoleAOut, jlm.RoleAOutB, d), 0)
	// The convolution module: a GLU, the depthwise convolution, the folded
	// BatchNorm, SiLU and the second pointwise projection.
	h = ln("conv_norm", x2, jlm.RoleAConvNorm, jlm.RoleAConvNormB)
	gv := mv("glu_val", h, jlm.RoleAGLUVal, jlm.RoleAGLUValB, d)
	gg := mv("glu_gate", h, jlm.RoleAGLUGate, jlm.RoleAGLUGateB, d)
	gl := g.value("glu", d)
	g.op(Op{Kind: OpActMul, In: []ValueID{gg, gv}, Out: []ValueID{gl}, Attr: Attrs{Act: ActSigmoid}})
	cv := g.value("dw", d)
	g.op(Op{Kind: OpConv, In: []ValueID{gl}, Out: []ValueID{cv}, Weight: []Binding{mat(jlm.RoleADW, d, c.Taps)},
		Attr: Attrs{Taps: c.Taps}})
	bn := g.value("bn", d)
	g.op(Op{Kind: OpAffine, In: []ValueID{cv}, Out: []ValueID{bn},
		Weight: []Binding{vec(jlm.RoleABNScale, d), vec(jlm.RoleABNShift, d)}})
	x3 := add("x_conv", x2, mv("pw2", act("conv_act", bn), jlm.RoleAPW2, jlm.RoleAPW2B, d), 0)
	x4 := ffn("ff2", x3, jlm.RoleAFF2Norm, jlm.RoleAFF2NormB, jlm.RoleAFF2Up, jlm.RoleAFF2UpB,
		jlm.RoleAFF2Down, jlm.RoleAFF2DownB)
	g.Out = []ValueID{ln("out", x4, jlm.RoleAOutNorm, jlm.RoleAOutNormB)}
	return g
}
