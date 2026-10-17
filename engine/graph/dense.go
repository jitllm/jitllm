package graph

import "github.com/jitllm/jitllm/format/jlm"

// Dense describes a dense pre-norm attention block: llama's, with the
// switches the covered variants turn (qwen3's per-head q/k norm before the
// rotary, gemma's norms over each sublayer's output before the residual).
// It is what the container states about the block; Build turns it into the
// block's graph.
type Dense struct {
	Block                          int
	NEmbd, NHead, NKVHead, HeadDim int
	NFFN                           int
	Eps                            float64
	NRot                           int
	Base                           float64
	Layout                         RopeLayout
	RopeFreqs                      bool
	Scale                          float64
	Window                         int
	Softcap                        float64
	Act                            Act
	QKNorm                         bool // RoleAttnQNorm and RoleAttnKNorm, over each head before the rotary
	PostNorm                       bool // RolePostAttnNorm and RolePostFFNNorm, over each sublayer's output
	QKVBias                        bool // RoleAttnQBias/KBias/VBias
}

// Build is the block's graph.
func (d *Dense) Build() *Graph {
	g := &Graph{Version: Version, Block: d.Block}
	qd, kvd := d.NHead*d.HeadDim, d.NKVHead*d.HeadDim
	x := g.value("x", d.NEmbd)
	g.In = []ValueID{x}
	vec := func(r jlm.Role, n int) Binding { return Binding{Role: r, K: n} }
	mat := func(r jlm.Role, k, rows int) Binding { return Binding{Role: r, K: k, Rows: rows} }
	norm := func(name string, in ValueID, w jlm.Role) ValueID {
		out := g.value(name, g.Values[in].Width)
		g.op(Op{Kind: OpNorm, In: []ValueID{in}, Out: []ValueID{out}, Weight: []Binding{vec(w, g.Values[in].Width)},
			Attr: Attrs{Eps: d.Eps}})
		return out
	}
	mv := func(name string, in ValueID, w jlm.Role, rows int, bias jlm.Role) ValueID {
		out := g.value(name, rows)
		ws := []Binding{mat(w, g.Values[in].Width, rows)}
		if bias != 0 {
			ws = append(ws, vec(bias, rows))
		}
		g.op(Op{Kind: OpMatVec, In: []ValueID{in}, Out: []ValueID{out}, Weight: ws})
		return out
	}
	bias := func(r jlm.Role) jlm.Role {
		if d.QKVBias {
			return r
		}
		return 0
	}

	h := norm("attn_in", x, jlm.RoleAttnNorm)
	q := mv("q", h, jlm.RoleAttnQ, qd, bias(jlm.RoleAttnQBias))
	k := mv("k", h, jlm.RoleAttnK, kvd, bias(jlm.RoleAttnKBias))
	v := mv("v", h, jlm.RoleAttnV, kvd, bias(jlm.RoleAttnVBias))
	if d.QKNorm {
		hn := func(name string, in ValueID, w jlm.Role) ValueID {
			out := g.value(name, g.Values[in].Width)
			g.op(Op{Kind: OpHeadNorm, In: []ValueID{in}, Out: []ValueID{out}, Weight: []Binding{vec(w, d.HeadDim)},
				Attr: Attrs{Eps: d.Eps, HeadDim: d.HeadDim}})
			return out
		}
		q = hn("q_normed", q, jlm.RoleAttnQNorm)
		k = hn("k_normed", k, jlm.RoleAttnKNorm)
	}
	rope := func(name string, in ValueID) ValueID {
		out := g.value(name, g.Values[in].Width)
		g.op(Op{Kind: OpRope, In: []ValueID{in}, Out: []ValueID{out},
			Attr: Attrs{HeadDim: d.HeadDim, NRot: d.NRot, Base: d.Base, Layout: d.Layout, Freqs: d.RopeFreqs}})
		return out
	}
	qr, kr := rope("q_rot", q), rope("k_rot", k)
	g.op(Op{Kind: OpKVAppend, In: []ValueID{kr, v}, Reads: []StateRef{{"kv", 0}}, Writes: []StateRef{{"kv", 1}}})
	a := g.value("attn", qd)
	g.op(Op{Kind: OpAttend, In: []ValueID{qr}, Out: []ValueID{a}, Reads: []StateRef{{"kv", 1}},
		Attr: Attrs{HeadDim: d.HeadDim, NHead: d.NHead, NKVHead: d.NKVHead, Scale: d.Scale, Window: d.Window, Softcap: d.Softcap}})
	o := mv("attn_out", a, jlm.RoleAttnOut, d.NEmbd, 0)
	if d.PostNorm {
		o = norm("attn_post", o, jlm.RolePostAttnNorm)
	}
	x1 := g.value("x_attn", d.NEmbd)
	g.op(Op{Kind: OpAdd, In: []ValueID{x, o}, Out: []ValueID{x1}})

	f := norm("ffn_in", x1, jlm.RoleFFNNorm)
	gt := mv("gate", f, jlm.RoleFFNGate, d.NFFN, 0)
	up := mv("up", f, jlm.RoleFFNUp, d.NFFN, 0)
	p := g.value("act", d.NFFN)
	g.op(Op{Kind: OpActMul, In: []ValueID{gt, up}, Out: []ValueID{p}, Attr: Attrs{Act: d.Act}})
	dn := mv("ffn_out", p, jlm.RoleFFNDown, d.NEmbd, 0)
	if d.PostNorm {
		dn = norm("ffn_post", dn, jlm.RolePostFFNNorm)
	}
	x2 := g.value("x_out", d.NEmbd)
	g.op(Op{Kind: OpAdd, In: []ValueID{x1, dn}, Out: []ValueID{x2}})
	g.Out = []ValueID{x2}
	return g
}
