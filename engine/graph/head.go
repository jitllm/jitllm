package graph

import (
	"math"

	"github.com/jitllm/jitllm/format/jlm"
)

// Head is a model's head: the final norm and the vocabulary's projection,
// Out the projection's role (the embedding's where the model ties them),
// then main's order over the logits: the projection's bias, the final
// softcap, the logit scale.
type Head struct {
	NEmbd, NVocab int
	Eps           float64
	Out           jlm.Role
	// LayerNorm makes the final norm a LayerNorm, NormBias with its bias.
	LayerNorm, NormBias bool
	// OutBias adds the projection's bias.
	OutBias bool
	// Softcap bounds the logits to (-Softcap, Softcap) by Softcap*tanh(l/Softcap);
	// 0 is none.
	Softcap float64
	// LogitScale divides the logits; 0 and 1 are none.
	LogitScale float64
	// Streams, where above one, is a residual of that many streams the
	// head reads Bank checkpoints and the running residual of, by Kimi-K3's
	// residual attention (OpStreamMix's bank, RoleOutputResScore), or where
	// AltUp collapses by Gemma 3n's unembedding, before its norm.
	Streams, Bank int
	AltUp         bool
}

func (h *Head) Build() *Graph {
	g := &Graph{Block: -1}
	x := g.value("x", max(1, h.Streams)*h.NEmbd)
	g.In = []ValueID{x}
	if h.Streams > 1 && h.AltUp {
		x = g.altCollapse(x, h.Streams, h.NEmbd)
	} else if h.Streams > 1 {
		row := g.value("collapsed", h.NEmbd)
		g.op(Op{Kind: OpStreamMix, In: []ValueID{x}, Out: []ValueID{row}, Weight: []Binding{{Role: jlm.RoleOutputResScore, K: h.NEmbd}},
			Attr: Attrs{Streams: h.Streams, Eps: h.Eps, Groups: h.Bank}})
		x = row
	}
	n := g.value("normed", h.NEmbd)
	norm := Op{Kind: OpNorm, In: []ValueID{x}, Out: []ValueID{n},
		Weight: []Binding{{Role: jlm.RoleOutputNorm, K: h.NEmbd, Rows: 1}}, Attr: Attrs{Eps: h.Eps}}
	if h.LayerNorm {
		norm.Kind = OpLayerNorm
		norm.Weight[0].Rows = 0
		if h.NormBias {
			norm.Weight = append(norm.Weight, Binding{Role: jlm.RoleOutputNormBias, K: h.NEmbd})
		}
	}
	g.op(norm)
	mv := Op{Kind: OpMatVec, In: []ValueID{n}, Weight: []Binding{{Role: h.Out, K: h.NEmbd, Rows: h.NVocab}}}
	if h.OutBias {
		mv.Weight = append(mv.Weight, Binding{Role: jlm.RoleOutputBias, K: h.NVocab})
	}
	l := g.value("logits", h.NVocab)
	mv.Out = []ValueID{l}
	g.op(mv)
	m := func(name string, f MapFn, c float64, in ...ValueID) ValueID {
		v := g.value(name, h.NVocab)
		g.op(Op{Kind: OpMap, In: in, Out: []ValueID{v}, Attr: Attrs{Fn: f, Const: c}})
		return v
	}
	if c := h.Softcap; c != 0 {
		// c*tanh(l/c), tanh(y) as (e-1)/(e+1) with e = exp(2y).
		e := m("softcap.exp", MapExp, 0, m("softcap.arg", MapMul, 2/c, l))
		q := m("softcap.tanh", MapDiv, 0, m("softcap.num", MapSub, 1, e), m("softcap.den", MapAdd, 1, e))
		l = m("capped", MapMul, c, q)
	}
	if s := h.LogitScale; s != 0 && s != 1 {
		l = m("scaled", MapMul, 1/s, l)
	}
	g.Out = []ValueID{l}
	return g
}

// Pooled is an encoder's head: the step's live rows pooled into one row
// (Pooling, a jlm.Pooling), then scaled to unit L2 norm as
// sentence-transformers does, a 1e-12 floor on the norm.
type Pooled struct {
	NEmbd   int
	Pooling uint8
}

func (p *Pooled) Build() *Graph {
	g := &Graph{Block: -1}
	x := g.value("x", p.NEmbd)
	g.In = []ValueID{x}
	pooled := g.row("pooled", p.NEmbd, Op{Kind: OpPool, In: []ValueID{x}, Attr: Attrs{Pooling: p.Pooling}})
	// An RMS norm of no weight is v/sqrt(mean(v^2)); 1/sqrt(n) turns the
	// mean into the sum, the floor squared and taken per element.
	n := float64(p.NEmbd)
	normed := g.row("normed", p.NEmbd, Op{Kind: OpHeadNorm, In: []ValueID{pooled}, Attr: Attrs{Eps: 1e-24 / n, HeadDim: p.NEmbd}})
	g.Out = []ValueID{g.mapc("embedding", p.NEmbd, MapMul, float64(float32(1/math.Sqrt(n))), normed)}
	return g
}
