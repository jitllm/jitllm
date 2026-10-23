package graph

import "github.com/jitllm/jitllm/format/jlm"

// MatVec is the slice y = W x: one OpMatVec of a rows x k weight, the
// request stating its format under role.
func MatVec(role jlm.Role, rows, k int) *Graph {
	g := &Graph{}
	x := g.value("x", k)
	y := g.value("y", rows)
	g.Ops = append(g.Ops, Op{Kind: OpMatVec, In: []ValueID{x}, Out: []ValueID{y}, Weight: []Binding{{Role: role, K: k, Rows: rows}}})
	g.In, g.Out = []ValueID{x}, []ValueID{y}
	return g
}
