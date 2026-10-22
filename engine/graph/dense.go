package graph

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
)

// MixerRoles are every tensor an OpRecurrent binds: a recurrent mixer's
// projections, convolution, decay, skip, norms and output, whichever the
// block stores.
var MixerRoles = []jlm.Role{jlm.RoleAttnQKV, jlm.RoleAttnGate, jlm.RoleSSMInProj, jlm.RoleSSMBA, jlm.RoleSSMConv1d,
	jlm.RoleSSMConvBias, jlm.RoleSSMA, jlm.RoleSSMD, jlm.RoleSSMDtBias, jlm.RoleSSMNorm, jlm.RoleSSMOut,
	jlm.RoleSSMXDt, jlm.RoleSSMXB, jlm.RoleSSMXC, jlm.RoleSSMDtProj, jlm.RoleSSMDtNorm, jlm.RoleSSMBNorm,
	jlm.RoleSSMCNorm, jlm.RoleSCB, jlm.RoleSCC, jlm.RoleSCX, jlm.RoleSSMFA, jlm.RoleSSMFB, jlm.RoleSSMGA, jlm.RoleSSMGB}

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
	BidirOff                       bool // Attrs.BidirOff
	Softcap                        float64
	Sinks                          bool
	// TempScale, TempFloor, TempOffs: the attention's query temperature
	// (Attrs); TempScale 0 is none.
	TempScale, TempFloor, TempOffs float64 // RoleAttnSinks: a per-head logit in every softmax's denominator
	Act                            Act
	GLU                            jlm.GLU // the FFN's gated unit's bounds (Attrs.GLU)
	QKNorm                         bool    // RoleAttnQNorm and RoleAttnKNorm, over each head before the rotary
	PostNorm                       bool    // RolePostAttnNorm and RolePostFFNNorm, over each sublayer's output
	// NoPreNorm is a block whose attention reads its input unnormed (OLMo
	// 2's, its post-norms standing in; ModernBERT's first block), NoFFNNorm
	// one whose FFN does (OLMo 2's), NoRope one with no rotary.
	NoPreNorm, NoFFNNorm, NoRope bool
	QKVBias                      bool // RoleAttnQBias/KBias/VBias
	// QKNormWhole norms q and k each over its whole projection at once
	// (OLMoE's), an OpNorm on each, where QKNorm norms each head.
	QKNormWhole bool
	// QKNormPost moves QKNorm's q norm after the rotary, k's staying before
	// it (Hunyuan's, its k weight folded into q's at conversion:
	// jlm.FlagQKNormPostRope). QKL2Norm norms each head of q and k with no
	// weight after the rotary, on the rotating blocks only (Llama 4's).
	QKNormPost, QKL2Norm bool
	// VNorm norms each head of v with no weight (Gemma 4's) before the
	// cache append.
	VNorm bool
	// VFromK makes v k's projection, taken before k's norm and rotary
	// (Gemma 4's global layers, attention_k_eq_v): no RoleAttnV.
	VFromK bool
	// KVShared is a block that computes no k and v: its attention reads
	// block KVFrom's history as that block left it (the state "kv@KVFrom"
	// at version 0), Gemma 4's E-models' last layers.
	KVShared bool
	KVFrom   int
	// OutScale multiplies the block's output by its stored scalar
	// (RoleLayerOutScale, an OpScale after the second residual add).
	OutScale bool
	// PLEDim, where not 0, is a per-layer embedding after the FFN's add
	// (Gemma 4's E-models): the block's second input is this layer's
	// PLEDim-wide slice of the row's per-layer inputs, and the block adds
	// norm(W_proj (Act(W_gate x) * slice)) into the residual
	// (RolePLEGate, RolePLEProj, RolePLEPostNorm).
	PLEDim int
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
	// GateBias is a gated MLP's gate bias, RoleVFFNGateBias: a vision
	// tower's (Qwen2.5-VL's) is the only block that carries one; UpBias and
	// DownBias bind on a gated MLP too.
	GateBias bool
	Clamp                                                  float64
	// ClampTable, where set, is the role of a bounds table every linear's
	// input and output is clamped by (Gemma 4's clipped linears): matrix s
	// of q, k, v, o, gate, up, down reads its input bounds at 4s and its
	// output bounds at 4s+2.
	ClampTable jlm.Role
	// ResidScale scales each sublayer's output into the residual (Alpha on
	// the two residual adds); 0 is one.
	ResidScale float64
	// OutGate is a query projection of twice the width, each head's q then
	// its gate (Qwen3-Next's), sliced apart per head; the attention's output
	// is sigmoid(gate)*attn (an act_mul of ActSigmoid) before attn_out.
	OutGate bool
	// MLA, where set, is the block's attention: latent attention in place of
	// q, k, v and the attention over per-head keys (the output projection,
	// the residual and the FFN are the block's).
	MLA *MLA
	// Mixer, where not 0, replaces the attention with a recurrent mixer of
	// that kind (OpRecurrent) binding MixerW; NoFFN ends the block at the
	// mixer's residual add (a mixer-only layer).
	Mixer  uint8
	MixerW []Binding
	// MixerAttends keeps the attention beside the mixer (Falcon-H1's).
	MixerAttends bool
	NoFFN        bool
	// DenseMix is Gemma 4's mixture block: a gated dense MLP over the FFN
	// norm (the shared expert's tensors, Mixture.SharedFF wide) post-normed
	// by RolePostFFNNorm1, beside the Mixture routed over a norm of the
	// residual (RoleRouterNorm) with its experts reading another
	// (RoleFFNNorm2), post-normed by RolePostFFNNorm2; their sum is the
	// FFN's output (PostNorm's norm and the residual add follow).
	DenseMix bool
	// Mixture, where set, is the block's FFN: a routed mixture over the FFN
	// norm's row in place of gate, up and down (NFFN is then unused).
	Mixture *Mixture
	// Compressed, where set, is the attention: OpCompressedAttend binding
	// W, the output projection inside it.
	Compressed *Compressed
	// PredictW, where set, makes the block a prediction block: OpPredict
	// first, binding these, over the block input and a second input, the
	// next token's embedding row.
	PredictW []Binding
	// NonCausal makes the attention bidirectional (OpAttend's Bidir).
	NonCausal bool
	// PostResid is BERT's post-norm: each sublayer's residual add is followed
	// by a LayerNorm of the sum (RoleAttnOutNorm, RoleLayerOutNorm, each with
	// its bias), the block's sublayers reading their input unnormed.
	PostResid bool
	// Cross permits OpCrossAttend: a decision head's block (CrossBlock).
	Cross bool
	// SubNorm is BitNet's: an RMSNorm over the attention's output before
	// its projection (RoleAttnSubNorm) and over the FFN's product before
	// down (RoleFFNSubNorm).
	SubNorm bool
	// Select, where set, is the attention's block selection (OpIndex over a
	// GQA row: MiniMax Sparse Attention).
	Select *Select
	// Streams, where set, makes the residual several streams (In[0]
	// Streams.N rows of NEmbd, stream-major): each sublayer reads its
	// OpStreamMix in place of its input norm's input (the FFN reads the mix
	// with no norm) and adds through OpStreamPlace.
	Streams *Streams
	// NGram, where set, adds the n-gram embedding into the streams first
	// (OpNGram), binding NGramW.
	NGram  bool
	NGramW []Binding
	// NGramTaps and NGramDil are its convolution's taps and their spacing.
	NGramTaps, NGramDil int
	// AltUp, where set, is Gemma 3n's residual of AltUp streams: the block
	// runs on the active stream OpAltPredict reads out, with LAuReL beside
	// its attention, and OpAltCorrect (carrying the per-layer input, PLEDim
	// wide) writes the streams back.
	AltUp *AltUp
	// SparseStd, where not 0, sparsifies the FFN's gate (OpGaussTopK).
	SparseStd float64
}

// AltUp is the stream count and LAuReL's rank.
type AltUp struct {
	N, LaurelRank int
}

// Streams is a residual of N streams mixed per sublayer by a low-rank gated
// hyper-connection of rank LowRank.
type Streams struct {
	N, LowRank int
	// Sinkhorn is DeepSeek V4's form: each sublayer's mix (bound by
	// AttnW/FFNW) weighs the streams into its row, Sinkhorn-normalises the
	// combination over Iters rounds, and a norm follows the mix; the
	// placement scales the output per stream and recombines the streams.
	Sinkhorn    bool
	Iters       int
	AttnW, FFNW []Binding
	// Bank is Kimi-K3's residual attention: stream 0 the running residual,
	// the others checkpoints of block inputs. Each sublayer reads a softmax
	// over its normed scores of the running residual and the BankAttn
	// (BankFFN) checkpoints banked so far (RoleAttnResScore,
	// RoleFFNResScore), a norm following; a checkpoint block banks its input
	// into stream Checkpoint and restarts the residual from its attention's
	// output.
	Bank              bool
	BankAttn, BankFFN int
	Checkpoint        int
}

// mix binds a hyper-connection's weights: its norm, the low-rank pair and
// the placement.
func (x *Streams) mix(nEmbd int, norm, down, up, inj jlm.Role) []Binding {
	hd := x.N * nEmbd
	return []Binding{{Role: norm, K: hd}, {Role: down, K: hd, Rows: x.LowRank}, {Role: up, K: x.LowRank, Rows: hd},
		{Role: inj, K: hd, Rows: x.N}}
}

// Compressed is a compressed attention's form: Kind (0 none, else the
// history's pooling kind), its Rate, the output's OGroups and its weights.
type Compressed struct {
	Kind          uint8
	Rate, OGroups int
	W             []Binding
}

// LaurelJoin is the scale of LAuReL's join: (x + laurel) / sqrt(2).
const LaurelJoin = 0.7071067811865476

// Select is a block-selecting indexer's geometry: Heads heads (one a kv
// group) of HeadDim, the TopK best blocks of Block positions kept with the
// Local ending at the query's forced in.
type Select struct {
	Heads, HeadDim, TopK, Block, Local int
	// Pooled is QSA's selection: the key the raw projection, each complete
	// block's keys pooled, normed and turned at its start.
	Pooled bool
}

// bindings are the selection's weights over a row of nEmbd.
func (x *Select) bindings(nEmbd int) []Binding {
	return []Binding{{Role: jlm.RoleIdxQ, K: nEmbd, Rows: x.Heads * x.HeadDim}, {Role: jlm.RoleIdxK, K: nEmbd, Rows: x.HeadDim},
		{Role: jlm.RoleIdxQNorm, K: x.HeadDim}, {Role: jlm.RoleIdxKNorm, K: x.HeadDim}}
}

// Build is the block's graph.
func (d *Dense) Build() *Graph {
	g := &Graph{Version: Version, Block: d.Block}
	qd, kvd := d.NHead*d.HeadDim, d.NKVHead*d.HeadDim
	rw := d.NEmbd
	if d.Streams != nil {
		rw = d.Streams.N * d.NEmbd
	}
	if d.AltUp != nil {
		rw = d.AltUp.N * d.NEmbd
	}
	x := g.value("x", rw)
	g.In = []ValueID{x}
	if d.PredictW != nil {
		e, xp := g.value("next_embd", d.NEmbd), g.value("x_predict", rw)
		g.In = append(g.In, e)
		g.op(Op{Kind: OpPredict, In: []ValueID{x, e}, Out: []ValueID{xp}, Weight: d.PredictW, Attr: Attrs{Eps: d.Eps}})
		x = xp
	}
	pred := ValueID(-1)
	if a := d.AltUp; a != nil {
		pred, x = g.value("alt_pred", rw), g.value("alt_active", d.NEmbd)
		g.op(Op{Kind: OpAltPredict, In: []ValueID{g.In[0]}, Out: []ValueID{pred, x},
			Weight: []Binding{{Role: jlm.RoleAltUpRouterNorm, K: d.NEmbd}, {Role: jlm.RoleAltUpRouter, K: d.NEmbd, Rows: a.N},
				{Role: jlm.RoleAltUpPredCoef, K: a.N * a.N, Rows: a.N}}, Attr: Attrs{Streams: a.N, Eps: d.Eps}})
	}
	if d.NGram {
		xn := g.value("x_ngram", rw)
		g.op(Op{Kind: OpNGram, In: []ValueID{x}, Out: []ValueID{xn}, Weight: d.NGramW,
			Reads: []StateRef{{"ngram", 0}}, Writes: []StateRef{{"ngram", 1}},
			Attr: Attrs{Streams: d.Streams.N, Taps: d.NGramTaps, Dil: d.NGramDil}})
		x = xn
	}
	// mixOf is a sublayer's read of the streams (OpStreamMix): its row and
	// its placement weights; without streams the residual itself.
	mixOf := func(name string, in ValueID, norm, down, up, inj jlm.Role) (ValueID, ValueID) {
		x := d.Streams
		if x == nil {
			return in, -1
		}
		if x.Bank {
			r, nb := jlm.RoleAttnResScore, x.BankAttn
			if norm == jlm.RoleHCFFNNorm {
				r, nb = jlm.RoleFFNResScore, x.BankFFN
			}
			row := g.value(name, d.NEmbd)
			g.op(Op{Kind: OpStreamMix, In: []ValueID{in}, Out: []ValueID{row}, Weight: []Binding{{Role: r, K: d.NEmbd}},
				Attr: Attrs{Streams: x.N, Eps: d.Eps, Groups: nb}})
			return row, -1
		}
		if x.Sinkhorn {
			ws := x.AttnW
			if norm == jlm.RoleHCFFNNorm {
				ws = x.FFNW
			}
			row, w := g.value(name, d.NEmbd), g.value(name+"_place", (2+x.N)*x.N)
			g.op(Op{Kind: OpStreamMix, In: []ValueID{in}, Out: []ValueID{row, w}, Weight: ws,
				Attr: Attrs{Streams: x.N, Eps: d.Eps, Sinkhorn: true, Groups: x.Iters}})
			return row, w
		}
		row, w := g.value(name, d.NEmbd), g.value(name+"_place", x.N)
		g.op(Op{Kind: OpStreamMix, In: []ValueID{in}, Out: []ValueID{row, w},
			Weight: x.mix(d.NEmbd, norm, down, up, inj), Attr: Attrs{Streams: x.N, Eps: d.Eps}})
		return row, w
	}
	// resid adds a sublayer's output y into the residual in: an add, or
	// with streams a placement at w.
	resid := func(name string, in, y, w ValueID) ValueID {
		out := g.value(name, g.Values[in].Width)
		if x := d.Streams; x != nil && x.Bank {
			// The attention's placement banks a checkpoint block's input.
			k := 0
			if name == "x_attn" || name == "x_mixer" {
				k = x.Checkpoint
			}
			g.op(Op{Kind: OpStreamPlace, In: []ValueID{in, y}, Out: []ValueID{out}, Attr: Attrs{Streams: x.N, Block: k}})
		} else if d.Streams != nil {
			g.op(Op{Kind: OpStreamPlace, In: []ValueID{in, y, w}, Out: []ValueID{out},
				Attr: Attrs{Streams: d.Streams.N, Sinkhorn: d.Streams.Sinkhorn}})
		} else {
			g.op(Op{Kind: OpAdd, In: []ValueID{in, y}, Out: []ValueID{out}, Attr: Attrs{Alpha: d.ResidScale}})
		}
		return out
	}
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
		if w == jlm.RoleAttnNorm && d.NoPreNorm || w == jlm.RoleFFNNorm && d.NoFFNNorm {
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
	// layerNorm is a LayerNorm with its bias (a post-norm's).
	layerNorm := func(name string, in ValueID, w, b jlm.Role) ValueID {
		out := g.value(name, g.Values[in].Width)
		g.op(Op{Kind: OpLayerNorm, In: []ValueID{in}, Out: []ValueID{out},
			Weight: []Binding{vec(w, g.Values[in].Width), vec(b, g.Values[in].Width)}, Attr: Attrs{Eps: d.Eps}})
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
	clampAt := func(name string, in ValueID, off int) ValueID {
		out := g.value(name, g.Values[in].Width)
		g.op(Op{Kind: OpClamp, In: []ValueID{in}, Out: []ValueID{out}, Weight: []Binding{vec(d.ClampTable, clampTableLen)},
			Attr: Attrs{Offset: off}})
		return out
	}
	mv := func(name string, in ValueID, w jlm.Role, rows int, bias jlm.Role) ValueID {
		s := clampedLinear(w)
		if d.ClampTable != 0 && s >= 0 {
			in = clampAt(name+"_in", in, 4*s)
		}
		out := g.value(name, rows)
		ws := []Binding{mat(w, g.Values[in].Width, rows)}
		if bias != 0 {
			ws = append(ws, vec(bias, rows))
		}
		g.op(Op{Kind: OpMatVec, In: []ValueID{in}, Out: []ValueID{out}, Weight: ws})
		if d.ClampTable != 0 && s >= 0 {
			out = clampAt(name+"_clamped", out, 4*s+2)
		}
		return out
	}
	bias := func(r jlm.Role) jlm.Role {
		if d.QKVBias {
			return r
		}
		return 0
	}

	xa, wa := mixOf("attn_mix", x, jlm.RoleHCAttnNorm, jlm.RoleHCAttnDown, jlm.RoleHCAttnUp, jlm.RoleHCAttnInject)
	h := preNorm("attn_in", xa, jlm.RoleAttnNorm, jlm.RoleAttnNormBias)
	lr := ValueID(-1)
	if a := d.AltUp; a != nil {
		lr = g.value("laurel", d.NEmbd)
		g.op(Op{Kind: OpLaurel, In: []ValueID{h}, Out: []ValueID{lr}, Weight: []Binding{
			{Role: jlm.RoleLaurelL, K: d.NEmbd, Rows: a.LaurelRank}, {Role: jlm.RoleLaurelR, K: a.LaurelRank, Rows: d.NEmbd},
			{Role: jlm.RoleLaurelPostNorm, K: d.NEmbd}}, Attr: Attrs{Eps: d.Eps}})
	}
	hf := h
	var x1 ValueID
	// A recurrent mixer, in place of the attention or (MixerAttends)
	// beside it over the same normed row, its output added to the
	// attention's before the residual (Falcon-H1's).
	mixer := ValueID(-1)
	if d.Mixer != 0 {
		mixer = g.value("mixer", d.NEmbd)
		g.op(Op{Kind: OpRecurrent, In: []ValueID{h}, Out: []ValueID{mixer}, Weight: d.MixerW,
			Reads: []StateRef{{"rec", 0}}, Writes: []StateRef{{"rec", 1}}, Attr: Attrs{Mixer: d.Mixer}})
	}
	if d.Mixer != 0 && !d.MixerAttends {
		x1 = resid("x_mixer", x, mixer, wa)
		if d.NoFFN {
			g.Out = []ValueID{x1}
			return g
		}
	} else if cx := d.Compressed; cx != nil {
		o := g.value("attn_out", d.NEmbd)
		g.op(Op{Kind: OpCompressedAttend, In: []ValueID{h}, Out: []ValueID{o}, Weight: cx.W,
			Reads: []StateRef{{"kv", 0}}, Writes: []StateRef{{"kv", 1}},
			Attr: Attrs{NHead: d.NHead, HeadDim: d.HeadDim, NRot: d.NRot, Window: d.Window, Mixer: cx.Kind, Block: cx.Rate,
				Groups: cx.OGroups}})
		x1 = resid("x_attn", x, o, wa)
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
			qw := qd
			if d.OutGate {
				qw = 2 * qd
			}
			q := mv("q", h, jlm.RoleAttnQ, qw, bias(jlm.RoleAttnQBias))
			var k, v ValueID
			switch {
			case d.KVShared:
			case d.VFromK:
				k = mv("k", h, jlm.RoleAttnK, kvd, bias(jlm.RoleAttnKBias))
				v = k
			default:
				k = mv("k", h, jlm.RoleAttnK, kvd, bias(jlm.RoleAttnKBias))
				v = mv("v", h, jlm.RoleAttnV, kvd, bias(jlm.RoleAttnVBias))
			}
			var gate ValueID
			if d.OutGate {
				qg := q
				q, gate = g.value("q_heads", qd), g.value("gate", qd)
				g.op(Op{Kind: OpSlice, In: []ValueID{qg}, Out: []ValueID{q}, Attr: Attrs{HeadDim: 2 * d.HeadDim, Width: d.HeadDim}})
				g.op(Op{Kind: OpSlice, In: []ValueID{qg}, Out: []ValueID{gate},
					Attr: Attrs{HeadDim: 2 * d.HeadDim, Offset: d.HeadDim, Width: d.HeadDim}})
			}
			if d.KVShared {
				q = clamp("q_clamped", q)
			} else {
				q, k, v = clamp("q_clamped", q), clamp("k_clamped", k), clamp("v_clamped", v)
			}
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
			kn := !d.KVShared
			// Under XD-RoPE k's norm follows the rotary too: that rotation
			// does not keep a head's RMS.
			xd := d.Layout == RopeXD
			if d.QKNorm && d.QKNormPost {
				if kn && !xd {
					k = hn("k_normed", k, jlm.RoleAttnKNorm)
				}
			} else if d.QKNorm {
				q = hn("q_normed", q, jlm.RoleAttnQNorm)
				if kn {
					k = hn("k_normed", k, jlm.RoleAttnKNorm)
				}
			} else if d.QKNormWhole {
				q = norm("q_normed", q, jlm.RoleAttnQNorm)
				if kn {
					k = norm("k_normed", k, jlm.RoleAttnKNorm)
				}
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
				qr = rope("q_rot", q)
				if kn {
					kr = rope("k_rot", k)
				}
				if d.QKL2Norm {
					qr = hn("q_l2", qr, 0)
					if kn {
						kr = hn("k_l2", kr, 0)
					}
				}
			}
			if d.QKNorm && d.QKNormPost {
				qr = hn("q_normed", qr, jlm.RoleAttnQNorm)
				if kn && xd {
					kr = hn("k_normed", kr, jlm.RoleAttnKNorm)
				}
			}
			kv := StateRef{"kv", 1}
			if d.KVShared {
				kv = StateRef{fmt.Sprintf("kv@%d", d.KVFrom), 0}
			} else {
				if d.VNorm {
					v = hn("v_normed", v, 0)
				}
				g.op(Op{Kind: OpKVAppend, In: []ValueID{kr, v}, Reads: []StateRef{{"kv", 0}}, Writes: []StateRef{{"kv", 1}}})
			}
			ain := []ValueID{qr}
			if x := d.Select; x != nil {
				sel := g.value("idx_sel", x.TopK)
				g.op(Op{Kind: OpIndex, In: []ValueID{h}, Out: []ValueID{sel}, Weight: x.bindings(d.NEmbd),
					Reads: []StateRef{{"kv", 1}}, Writes: []StateRef{{"kv", 2}},
					Attr: Attrs{NHead: x.Heads, HeadDim: x.HeadDim, TopK: x.TopK, Block: x.Block, Local: x.Local, Pooled: x.Pooled}})
				kv = StateRef{"kv", 2}
				ain = append(ain, sel)
			}
			a = g.value("attn", qd)
			var sinks []Binding
			if d.Sinks {
				sinks = []Binding{vec(jlm.RoleAttnSinks, d.NHead)}
			}
			g.op(Op{Kind: OpAttend, In: ain, Out: []ValueID{a}, Reads: []StateRef{kv}, Weight: sinks,
				Attr: Attrs{HeadDim: d.HeadDim, NHead: d.NHead, NKVHead: d.NKVHead, Scale: d.Scale, Window: d.Window, Chunked: d.Chunked, BidirOff: d.BidirOff, Bidir: d.NonCausal, Softcap: d.Softcap,
					TempScale: d.TempScale, TempFloor: d.TempFloor, TempOffs: d.TempOffs}})
			if d.OutGate {
				gated := g.value("attn_gated", qd)
				g.op(Op{Kind: OpActMul, In: []ValueID{gate, a}, Out: []ValueID{gated}, Attr: Attrs{Act: ActSigmoid}})
				a = gated
			}
		}
		if d.SubNorm {
			a = norm("attn_sub", a, jlm.RoleAttnSubNorm)
		}
		o := mv("attn_out", a, jlm.RoleAttnOut, d.NEmbd, opt(d.OutBias, jlm.RoleAttnOutBias))
		if mixer >= 0 {
			om := g.value("attn_mixer", d.NEmbd)
			g.op(Op{Kind: OpAdd, In: []ValueID{o, mixer}, Out: []ValueID{om}})
			o = om
		}
		if d.PostNorm {
			o = norm("attn_post", o, jlm.RolePostAttnNorm)
		}
		x1 = resid("x_attn", x, o, wa)
		if d.PostResid {
			x1 = layerNorm("x_attn_normed", x1, jlm.RoleAttnOutNorm, jlm.RoleAttnOutNormBias)
		}
		if lr >= 0 {
			xl := g.value("x_laurel", d.NEmbd)
			g.op(Op{Kind: OpAdd, In: []ValueID{x1, lr}, Out: []ValueID{xl}, Attr: Attrs{Scale: LaurelJoin}})
			x1 = xl
		}
	}

	f := hf
	var wf ValueID = -1
	switch {
	case d.Streams != nil:
		f, wf = mixOf("ffn_mix", x1, jlm.RoleHCFFNNorm, jlm.RoleHCFFNDown, jlm.RoleHCFFNUp, jlm.RoleHCFFNInject)
		if d.Streams.Sinkhorn || d.Streams.Bank {
			f = norm("ffn_in", f, jlm.RoleFFNNorm)
		}
	case !d.Parallel:
		f = preNorm("ffn_in", x1, jlm.RoleFFNNorm, jlm.RoleFFNNormBias)
	}
	var dn ValueID
	if d.DenseMix && d.Mixture != nil {
		mx := *d.Mixture
		mx.SharedFF = 0
		sg := mv("dense_gate", f, jlm.RoleShExpGate, d.Mixture.SharedFF, 0)
		su := mv("dense_up", f, jlm.RoleShExpUp, d.Mixture.SharedFF, 0)
		sa := g.value("dense_act", d.Mixture.SharedFF)
		g.op(Op{Kind: OpActMul, In: []ValueID{sg, su}, Out: []ValueID{sa}, Attr: Attrs{Act: mx.Act, GLU: mx.GLU}})
		dd := norm("dense_post", mv("dense_out", sa, jlm.RoleShExpDown, d.NEmbd, 0), jlm.RolePostFFNNorm1)
		rin := norm("route_in", x1, jlm.RoleRouterNorm)
		ein := norm("experts_in", x1, jlm.RoleFFNNorm2)
		e := norm("experts_post", mx.emitFrom(g, rin, ein), jlm.RolePostFFNNorm2)
		dn = g.value("dense_mix", d.NEmbd)
		g.op(Op{Kind: OpAdd, In: []ValueID{dd, e}, Out: []ValueID{dn}})
	} else if d.Mixture != nil {
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
		gt := mv("gate", f, jlm.RoleFFNGate, d.NFFN, opt(d.GateBias, jlm.RoleVFFNGateBias))
		up := mv("up", f, jlm.RoleFFNUp, d.NFFN, opt(d.UpBias, jlm.RoleFFNUpBias))
		if d.SparseStd != 0 {
			gs := g.value("gate_sparse", d.NFFN)
			g.op(Op{Kind: OpGaussTopK, In: []ValueID{gt}, Out: []ValueID{gs}, Attr: Attrs{Bound: d.SparseStd}})
			gt = gs
		}
		p := g.value("act", d.NFFN)
		g.op(Op{Kind: OpActMul, In: []ValueID{gt, up}, Out: []ValueID{p}, Attr: Attrs{Act: d.Act, GLU: d.GLU}})
		if d.SubNorm {
			p = norm("ffn_sub", p, jlm.RoleFFNSubNorm)
		}
		dn = mv("ffn_out", p, jlm.RoleFFNDown, d.NEmbd, opt(d.DownBias, jlm.RoleFFNDownBias))
	}
	if d.PostNorm {
		dn = norm("ffn_post", dn, jlm.RolePostFFNNorm)
	}
	x2 := resid("x_out", x1, dn, wf)
	if d.PostResid {
		x2 = layerNorm("x_out_normed", x2, jlm.RoleLayerOutNorm, jlm.RoleLayerOutNormBias)
	}
	if a := d.AltUp; a != nil {
		pl := g.value("ple_in", d.PLEDim)
		g.In = append(g.In, pl)
		xs := g.value("x_streams", rw)
		g.op(Op{Kind: OpAltCorrect, In: []ValueID{pred, x2, pl}, Out: []ValueID{xs}, Weight: []Binding{
			{Role: jlm.RoleAltUpRouterNorm, K: d.NEmbd}, {Role: jlm.RoleAltUpRouter, K: d.NEmbd, Rows: a.N},
			{Role: jlm.RoleAltUpCorrCoef, K: a.N, Rows: a.N}, {Role: jlm.RoleAltUpCorrScale, K: d.NEmbd},
			{Role: jlm.RolePLEGate, K: d.NEmbd, Rows: d.PLEDim}, {Role: jlm.RolePLEProj, K: d.PLEDim, Rows: d.NEmbd},
			{Role: jlm.RolePLEPostNorm, K: d.NEmbd}}, Attr: Attrs{Streams: a.N, Eps: d.Eps, Act: d.Act}})
		x2 = xs
	} else if d.PLEDim != 0 {
		pl := g.value("ple_in", d.PLEDim)
		g.In = append(g.In, pl)
		gp := mv("ple_gate", x2, jlm.RolePLEGate, d.PLEDim, 0)
		a := g.value("ple_act", d.PLEDim)
		g.op(Op{Kind: OpActMul, In: []ValueID{gp, pl}, Out: []ValueID{a}, Attr: Attrs{Act: d.Act}})
		o := norm("ple_post", mv("ple_out", a, jlm.RolePLEProj, d.NEmbd, 0), jlm.RolePLEPostNorm)
		x3 := g.value("x_ple", g.Values[x2].Width)
		g.op(Op{Kind: OpAdd, In: []ValueID{x2, o}, Out: []ValueID{x3}})
		x2 = x3
	}
	if d.OutScale {
		y := g.value("x_scaled", g.Values[x2].Width)
		g.op(Op{Kind: OpScale, In: []ValueID{x2}, Out: []ValueID{y}, Weight: []Binding{vec(jlm.RoleLayerOutScale, 1)}})
		x2 = y
	}
	g.Out = []ValueID{x2}
	return g
}

// clampTableLen is a clipped-linear bounds table's length: a lower and upper
// bound on the input and on the output of each of seven matrices.
const clampTableLen = 28

// clampedLinear is matrix w's place among the clipped linears (q, k, v, o,
// gate, up, down), or -1.
func clampedLinear(w jlm.Role) int {
	switch w {
	case jlm.RoleAttnQ:
		return 0
	case jlm.RoleAttnK:
		return 1
	case jlm.RoleAttnV:
		return 2
	case jlm.RoleAttnOut:
		return 3
	case jlm.RoleFFNGate:
		return 4
	case jlm.RoleFFNUp:
		return 5
	case jlm.RoleFFNDown:
		return 6
	}
	return -1
}
