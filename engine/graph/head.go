package graph

import "github.com/jitllm/jitllm/format/jlm"

// Head is a model's head: the final norm and the vocabulary's projection,
// Out the projection's role (the embedding's where the model ties them).
type Head struct {
	NEmbd, NVocab int
	Eps           float64
	Out           jlm.Role
}

func (h *Head) Build() *Graph {
	g := &Graph{Block: -1}
	x := g.value("x", h.NEmbd)
	n := g.value("normed", h.NEmbd)
	l := g.value("logits", h.NVocab)
	g.In, g.Out = []ValueID{x}, []ValueID{l}
	g.op(Op{Kind: OpNorm, In: []ValueID{x}, Out: []ValueID{n},
		Weight: []Binding{{Role: jlm.RoleOutputNorm, K: h.NEmbd, Rows: 1}}, Attr: Attrs{Eps: h.Eps}})
	g.op(Op{Kind: OpMatVec, In: []ValueID{n}, Out: []ValueID{l},
		Weight: []Binding{{Role: h.Out, K: h.NEmbd, Rows: h.NVocab}}})
	return g
}
