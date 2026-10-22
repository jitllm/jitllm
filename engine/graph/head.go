package graph

import "github.com/jitllm/jitllm/format/jlm"

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
}

func (h *Head) Build() *Graph {
	g := &Graph{Block: -1}
	x := g.value("x", h.NEmbd)
	n := g.value("normed", h.NEmbd)
	g.In = []ValueID{x}
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
