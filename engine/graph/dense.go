package graph

import "github.com/jitllm/jitllm/format/jlm"

// MixerRoles are every tensor an OpRecurrent binds: a recurrent mixer's
// projections, convolution, decay, skip, norms and output, whichever the
// block stores.
var MixerRoles = []jlm.Role{jlm.RoleAttnQKV, jlm.RoleAttnGate, jlm.RoleSSMInProj, jlm.RoleSSMBA, jlm.RoleSSMConv1d,
	jlm.RoleSSMConvBias, jlm.RoleSSMA, jlm.RoleSSMD, jlm.RoleSSMDtBias, jlm.RoleSSMNorm, jlm.RoleSSMOut,
	jlm.RoleSSMXDt, jlm.RoleSSMXB, jlm.RoleSSMXC, jlm.RoleSSMDtProj, jlm.RoleSSMDtNorm, jlm.RoleSSMBNorm,
	jlm.RoleSSMCNorm, jlm.RoleSCB, jlm.RoleSCC, jlm.RoleSCX}

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
	RopeSections                   [4]int  // Attrs.Sections
	RopeInterleaved                bool    // Attrs.Interleaved
	Mscale                         float64 // the rotary's cos/sin multiplier; 0 is one
	Scale                          float64
	Window                         int
	Chunked                        bool // Window is an aligned chunk (Attrs.Chunked)
	Softcap                        float64
	Sinks                          bool
	// TempScale, TempFloor, TempOffs: the attention's query temperature
	// (Attrs); TempScale 0 is none.
	TempScale, TempFloor, TempOffs float64 // RoleAttnSinks: a per-head logit in every softmax's denominator
	Act                            Act
	QKNorm                         bool // RoleAttnQNorm and RoleAttnKNorm, over each head before the rotary
	PostNorm                       bool // RolePostAttnNorm and RolePostFFNNorm, over each sublayer's output
	// NoPreNorm is a block whose sublayers read their input unnormed (OLMo
	// 2's, its post-norms standing in), NoRope one with no rotary.
	NoPreNorm, NoRope bool
	QKVBias           bool // RoleAttnQBias/KBias/VBias
	// QKNormWhole norms q and k each over its whole projection at once
	// (OLMoE's), an OpNorm on each, where QKNorm norms each head.
	QKNormWhole bool
	// QKNormPost moves QKNorm's q norm after the rotary, k's staying before
	// it (Hunyuan's, its k weight folded into q's at conversion:
	// jlm.FlagQKNormPostRope). QKL2Norm norms each head of q and k with no
	// weight after the rotary, on the rotating blocks only (Llama 4's).
	QKNormPost, QKL2Norm bool
	// The classic block (C6). LayerNorm makes the attention and FFN norms
	// LayerNorms (OpLayerNorm); NormBias and FFNNormBias bind their biases
	// (RoleAttnNormBias, RoleFFNNormBias). Parallel feeds the FFN the block
	// input, normed by its own norm or, with SharedNorm (no RoleFFNNorm), by
	// the attention's. OutBias, UpBias and DownBias are the projections'
	// biases; Ungated is an FFN of up, Act and down with no gate. Clamp,
	// where not 0, clamps q, k and v to [-Clamp, Clamp] after their
	// projections.
	LayerNorm, NormBias, FFNNormBias, Parallel, SharedNorm bool
	OutBias, UpBias, DownBias, Ungated                     bool
	Clamp                                                  float64
	// ResidScale scales each sublayer's output into the residual (Alpha on
	// the two residual adds); 0 is one.
	ResidScale float64
	// MLA, where set, is the block's attention: latent attention in place of
	// q, k, v and the attention over per-head keys (the output projection,
	// the residual and the FFN are the block's).
	MLA *MLA
	// Mixer, where not 0, replaces the attention with a recurrent mixer of
	// that kind (OpRecurrent) binding MixerW; NoFFN ends the block at the
	// mixer's residual add (a mixer-only layer).
	Mixer  uint8
	MixerW []Binding
	NoFFN  bool
	// Mixture, where set, is the block's FFN: a routed mixture over the FFN
	// norm's row in place of gate, up and down (NFFN is then unused).
	Mixture *Mixture
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
	// preNorm is a sublayer's input norm: an RMSNorm, or the classic block's
	// LayerNorm with its bias where the block stores one.
	preNorm := func(name string, in ValueID, w, b jlm.Role) ValueID {
		if d.NoPreNorm {
			return in
		}
		if !d.LayerNorm {
			return norm(name, in, w)
		}
		out := g.value(name, g.Values[in].Width)
		ws := []Binding{vec(w, g.Values[in].Width)}
		if b == jlm.RoleAttnNormBias && d.NormBias || b == jlm.RoleFFNNormBias && d.FFNNormBias {
			ws = append(ws, vec(b, g.Values[in].Width))
		}
		g.op(Op{Kind: OpLayerNorm, In: []ValueID{in}, Out: []ValueID{out}, Weight: ws, Attr: Attrs{Eps: d.Eps}})
		return out
	}
	opt := func(on bool, r jlm.Role) jlm.Role {
		if on {
			return r
		}
		return 0
	}
	clamp := func(name string, in ValueID) ValueID {
		if d.Clamp == 0 {
			return in
		}
		out := g.value(name, g.Values[in].Width)
		g.op(Op{Kind: OpClamp, In: []ValueID{in}, Out: []ValueID{out}, Attr: Attrs{Bound: d.Clamp}})
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

	h := preNorm("attn_in", x, jlm.RoleAttnNorm, jlm.RoleAttnNormBias)
	hf := h
	var x1 ValueID
	if d.Mixer != 0 {
		// A recurrent mixer in place of the attention.
		r := g.value("mixer", d.NEmbd)
		g.op(Op{Kind: OpRecurrent, In: []ValueID{h}, Out: []ValueID{r}, Weight: d.MixerW,
			Reads: []StateRef{{"rec", 0}}, Writes: []StateRef{{"rec", 1}}, Attr: Attrs{Mixer: d.Mixer}})
		x1 = g.value("x_mixer", d.NEmbd)
		g.op(Op{Kind: OpAdd, In: []ValueID{x, r}, Out: []ValueID{x1}, Attr: Attrs{Alpha: d.ResidScale}})
		if d.NoFFN {
			g.Out = []ValueID{x1}
			return g
		}
	} else {
		// A parallel block's FFN reads the block input: its norm is the
		// attention's (SharedNorm) or its own over x.
		if d.Parallel && !d.SharedNorm {
			hf = preNorm("ffn_in", x, jlm.RoleFFNNorm, jlm.RoleFFNNormBias)
		}
		var a ValueID
		if d.MLA != nil {
			a = d.MLA.emit(g, d, h)
		} else {
			q := mv("q", h, jlm.RoleAttnQ, qd, bias(jlm.RoleAttnQBias))
			k := mv("k", h, jlm.RoleAttnK, kvd, bias(jlm.RoleAttnKBias))
			v := mv("v", h, jlm.RoleAttnV, kvd, bias(jlm.RoleAttnVBias))
			q, k, v = clamp("q_clamped", q), clamp("k_clamped", k), clamp("v_clamped", v)
			// hn norms each head of in; w 0 is a weightless norm (a scale of ones).
			hn := func(name string, in ValueID, w jlm.Role) ValueID {
				out := g.value(name, g.Values[in].Width)
				var ws []Binding
				if w != 0 {
					ws = []Binding{vec(w, d.HeadDim)}
				}
				g.op(Op{Kind: OpHeadNorm, In: []ValueID{in}, Out: []ValueID{out}, Weight: ws,
					Attr: Attrs{Eps: d.Eps, HeadDim: d.HeadDim}})
				return out
			}
			if d.QKNorm && d.QKNormPost {
				k = hn("k_normed", k, jlm.RoleAttnKNorm)
			} else if d.QKNorm {
				q = hn("q_normed", q, jlm.RoleAttnQNorm)
				k = hn("k_normed", k, jlm.RoleAttnKNorm)
			} else if d.QKNormWhole {
				q = norm("q_normed", q, jlm.RoleAttnQNorm)
				k = norm("k_normed", k, jlm.RoleAttnKNorm)
			}
			rope := func(name string, in ValueID) ValueID {
				out := g.value(name, g.Values[in].Width)
				g.op(Op{Kind: OpRope, In: []ValueID{in}, Out: []ValueID{out},
					Attr: Attrs{HeadDim: d.HeadDim, NRot: d.NRot, Base: d.Base, Layout: d.Layout, Freqs: d.RopeFreqs, Mscale: d.Mscale,
						Sections: d.RopeSections, Interleaved: d.RopeInterleaved}})
				return out
			}
			qr, kr := q, k
			if !d.NoRope {
				qr, kr = rope("q_rot", q), rope("k_rot", k)
				if d.QKL2Norm {
					qr, kr = hn("q_l2", qr, 0), hn("k_l2", kr, 0)
				}
			}
			if d.QKNorm && d.QKNormPost {
				qr = hn("q_normed", qr, jlm.RoleAttnQNorm)
			}
			g.op(Op{Kind: OpKVAppend, In: []ValueID{kr, v}, Reads: []StateRef{{"kv", 0}}, Writes: []StateRef{{"kv", 1}}})
			a = g.value("attn", qd)
			var sinks []Binding
			if d.Sinks {
				sinks = []Binding{vec(jlm.RoleAttnSinks, d.NHead)}
			}
			g.op(Op{Kind: OpAttend, In: []ValueID{qr}, Out: []ValueID{a}, Reads: []StateRef{{"kv", 1}}, Weight: sinks,
				Attr: Attrs{HeadDim: d.HeadDim, NHead: d.NHead, NKVHead: d.NKVHead, Scale: d.Scale, Window: d.Window, Chunked: d.Chunked, Softcap: d.Softcap,
					TempScale: d.TempScale, TempFloor: d.TempFloor, TempOffs: d.TempOffs}})
		}
		o := mv("attn_out", a, jlm.RoleAttnOut, d.NEmbd, opt(d.OutBias, jlm.RoleAttnOutBias))
		if d.PostNorm {
			o = norm("attn_post", o, jlm.RolePostAttnNorm)
		}
		x1 = g.value("x_attn", d.NEmbd)
		g.op(Op{Kind: OpAdd, In: []ValueID{x, o}, Out: []ValueID{x1}, Attr: Attrs{Alpha: d.ResidScale}})
	}

	f := hf
	if !d.Parallel {
		f = preNorm("ffn_in", x1, jlm.RoleFFNNorm, jlm.RoleFFNNormBias)
	}
	var dn ValueID
	if d.Mixture != nil {
		dn = d.Mixture.emit(g, f)
	} else if d.Ungated {
		up := mv("up", f, jlm.RoleFFNUp, d.NFFN, opt(d.UpBias, jlm.RoleFFNUpBias))
		p := g.value("act", d.NFFN)
		var ws []Binding
		if d.Act == ActXIELU {
			ws = []Binding{vec(jlm.RoleXIELU, 4)}
		}
		g.op(Op{Kind: OpAct, In: []ValueID{up}, Out: []ValueID{p}, Weight: ws, Attr: Attrs{Act: d.Act}})
		dn = mv("ffn_out", p, jlm.RoleFFNDown, d.NEmbd, opt(d.DownBias, jlm.RoleFFNDownBias))
	} else {
		gt := mv("gate", f, jlm.RoleFFNGate, d.NFFN, 0)
		up := mv("up", f, jlm.RoleFFNUp, d.NFFN, 0)
		p := g.value("act", d.NFFN)
		g.op(Op{Kind: OpActMul, In: []ValueID{gt, up}, Out: []ValueID{p}, Attr: Attrs{Act: d.Act}})
		dn = mv("ffn_out", p, jlm.RoleFFNDown, d.NEmbd, 0)
	}
	if d.PostNorm {
		dn = norm("ffn_post", dn, jlm.RolePostFFNNorm)
	}
	x2 := g.value("x_out", d.NEmbd)
	g.op(Op{Kind: OpAdd, In: []ValueID{x1, dn}, Out: []ValueID{x2}, Attr: Attrs{Alpha: d.ResidScale}})
	g.Out = []ValueID{x2}
	return g
}
