package graph

import "github.com/jitllm/jitllm/format/jlm"

// Embed is a step's input rows: each row's token's row of the embedding.
type Embed struct {
	NEmbd, NVocab int
}

func (e *Embed) Build() *Graph {
	g := &Graph{Block: -1}
	x := g.value("x", e.NEmbd)
	g.Out = []ValueID{x}
	g.op(Op{Kind: OpEmbed, Out: []ValueID{x}, Weight: []Binding{{Role: jlm.RoleTokenEmbd, K: e.NEmbd, Rows: e.NVocab}}})
	return g
}
