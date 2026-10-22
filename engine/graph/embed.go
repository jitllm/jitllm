package graph

import "github.com/jitllm/jitllm/format/jlm"

// Embed is a step's input rows: each row's token's row of the embedding,
// times Scale where it is neither 0 nor 1 (Gemma's sqrt(n_embd)).
type Embed struct {
	NEmbd, NVocab int
	Scale         float64
}

func (e *Embed) Build() *Graph {
	g := &Graph{Block: -1}
	scaled := e.Scale != 0 && e.Scale != 1
	name := "x"
	if scaled {
		name = "embd"
	}
	x := g.value(name, e.NEmbd)
	g.op(Op{Kind: OpEmbed, Out: []ValueID{x}, Weight: []Binding{{Role: jlm.RoleTokenEmbd, K: e.NEmbd, Rows: e.NVocab}}})
	if scaled {
		y := g.value("x", e.NEmbd)
		g.op(Op{Kind: OpMap, In: []ValueID{x}, Out: []ValueID{y}, Attr: Attrs{Fn: MapMul, Const: e.Scale}})
		x = y
	}
	g.Out = []ValueID{x}
	return g
}
