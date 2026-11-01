package model

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
)

// cloneGraph is a copy of g no rewrite of it reaches the original through.
func cloneGraph(g *graph.Graph) *graph.Graph {
	c := *g
	c.Values, c.In, c.Out = slices.Clone(g.Values), slices.Clone(g.In), slices.Clone(g.Out)
	c.Ops = make([]graph.Op, len(g.Ops))
	for i, o := range g.Ops {
		o.In, o.Out = slices.Clone(o.In), slices.Clone(o.Out)
		o.Weight, o.Reads, o.Writes = slices.Clone(o.Weight), slices.Clone(o.Reads), slices.Clone(o.Writes)
		c.Ops[i] = o
	}
	return &c
}

// rewriteBlock puts f's rewrite of a copy of block li's graph in the front
// end's statement of the model, where a device placing the block takes it,
// and returns what puts the front end's back; nil where f found nothing to
// rewrite.
func rewriteBlock(t *testing.T, m *Model, li int, f func(*graph.Graph) bool) func() {
	t.Helper()
	set := m.frontGraphs()
	m.front.mu.Lock()
	defer m.front.mu.Unlock()
	old := set.blocks[li]
	if old == nil {
		t.Fatalf("block %d has no graph to rewrite", li)
	}
	g := cloneGraph(old)
	if !f(g) {
		return nil
	}
	set.blocks[li] = g
	return func() {
		m.front.mu.Lock()
		set.blocks[li] = old
		m.front.mu.Unlock()
	}
}
