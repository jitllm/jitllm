package jit

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// PartKind is what a part of the model is.
type PartKind uint8

const (
	PartBlock PartKind = iota
	PartHead
	PartTable
	// PartSample is the step's sampling of the head's logits.
	PartSample
	// PartEmbed is the step's input rows, each row's token's embedding.
	PartEmbed
)

// Part is one placeable part of the model and its graph.
type Part struct {
	Kind  PartKind
	Block int
	Graph *graph.Graph
}

func (p Part) String() string {
	switch p.Kind {
	case PartHead:
		return "head"
	case PartTable:
		return "table"
	case PartSample:
		return "sample"
	case PartEmbed:
		return "embed"
	}
	return fmt.Sprintf("block %d", p.Block)
}

// Model is what the composer reads of a model: its configuration, its
// tensors' index entries, never their bytes, and its graphs as the engine
// builds them from the container -- each block's (nil for a block it has
// none of), the head's and the input rows'.
type Model interface {
	Config() *jlm.Config
	Find(role jlm.Role, block, index int32) (*jlm.Entry, bool)
	Entries() []jlm.Entry
	BlockGraph(li int) *graph.Graph
	HeadGraph() *graph.Graph
	EmbedGraph() *graph.Graph
}

// unread is the first tensor of block (jlm.DenseBlock for none) whose role
// the front end does not read: what it does not read, it refuses rather than
// composing a model without it.
func unread(f Model, block int32, read ...jlm.Role) error {
	for _, e := range f.Entries() {
		// The rotary's factors are the step's table's, built from the
		// engine's rotary as main's host runs it (rope_factors_long unread
		// there too); a vision tensor is the tower's segment.
		switch e.Role {
		case jlm.RoleRopeFreqs, jlm.RoleRopeFactorsShort, jlm.RoleRopeFactorsLong:
			continue
		}
		if e.Role.Vision() {
			continue
		}
		if e.Block == block && !slices.Contains(read, e.Role) {
			return fmt.Errorf("the container holds %s, which the front end does not read", e.Role)
		}
	}
	return nil
}

// has reports whether model m holds a tensor of role at block and index.
func has(m Model, role jlm.Role, block, index int32) bool {
	_, ok := m.Find(role, block, index)
	return ok
}

// Front is the model's graph: the step's rotary table, its input rows,
// every block's graph as the model states it, the head, then the sampling of
// its logits. A model-wide tensor no graph binds, the front end refuses.
func Front(f Model) ([]Part, error) {
	c := f.Config()
	rt := graph.RopeTable{NRot: int(c.NRot)}
	em, h := f.EmbedGraph(), f.HeadGraph()
	if em == nil || h == nil {
		return nil, fmt.Errorf("jit: the model states no head or input rows")
	}
	parts := []Part{{Kind: PartTable, Block: -1, Graph: rt.Build()}, {Kind: PartEmbed, Block: -1, Graph: em}}
	for li := 0; li < int(c.NLayer); li++ {
		g := f.BlockGraph(li)
		if g == nil {
			return nil, fmt.Errorf("jit: block %d has no graph", li)
		}
		// Main's device takes a block's composite ops as their primitives
		// (graph.Decompose, as its tier prepares the block); so does the
		// composer.
		parts = append(parts, Part{Kind: PartBlock, Block: li, Graph: graph.Decompose(g)})
	}
	var bound []jlm.Role
	for _, g := range []*graph.Graph{em, h} {
		for _, o := range g.Ops {
			for _, w := range o.Weight {
				bound = append(bound, w.Role)
			}
		}
	}
	if err := unread(f, jlm.DenseBlock, bound...); err != nil {
		return nil, fmt.Errorf("jit: head: %w", err)
	}
	sm := graph.Sample{NVocab: int(c.NVocab)}
	return append(parts, Part{Kind: PartHead, Block: -1, Graph: h}, Part{Kind: PartSample, Block: -1, Graph: sm.Build()}), nil
}

// formats is each of a part's weights' storage format.
func formats(f Model, p Part) map[jlm.Role]jlm.Type {
	out := map[jlm.Role]jlm.Type{}
	block := int32(p.Block)
	if p.Kind != PartBlock {
		block = jlm.DenseBlock
	}
	for _, o := range p.Graph.Ops {
		for _, w := range o.Weight {
			if e, ok := f.Find(w.Role, block, -1); ok {
				out[w.Role] = e.Type
			}
		}
	}
	return out
}
