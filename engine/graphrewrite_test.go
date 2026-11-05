package engine

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/graph"
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
func rewriteBlock(t *testing.T, m *model.Model, li int, f func(*graph.Graph) bool) func() {
	t.Helper()
	set := m.FrontGraphs()
	m.Front.Mu.Lock()
	defer m.Front.Mu.Unlock()
	old := set.Blocks[li]
	if old == nil {
		t.Fatalf("block %d has no graph to rewrite", li)
	}
	g := cloneGraph(old)
	if !f(g) {
		return nil
	}
	set.Blocks[li] = g
	return func() {
		m.Front.Mu.Lock()
		set.Blocks[li] = old
		m.Front.Mu.Unlock()
	}
}
