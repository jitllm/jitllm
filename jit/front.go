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
	// PartSeam is a model-wide graph between two blocks: its residual in
	// and out, its weights the model's (a decision head's prologue).
	PartSeam
)

// Part is one placeable part of the model and its graph.
type Part struct {
	Kind  PartKind
	Block int
	Graph *graph.Graph
	// Feeds are the inputs past the first that an earlier part writes: a
	// block's per-layer input, the embedding's. The first is the part
	// before's first output.
	Feeds []Feed
}

// Feed is a part's input In that part From writes as its output Out.
type Feed struct {
	In, From, Out int
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
	case PartSeam:
		return fmt.Sprintf("seam before block %d", p.Block)
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
	// Blocks is how many block parts the model runs, the config's layers
	// and any blocks its head states after them (a decision head's).
	Blocks() int
	// SeamGraph is a model-wide graph the residual runs through before
	// block li, nil for none.
	SeamGraph(li int) *graph.Graph
	HeadGraph() *graph.Graph
	EmbedGraph() *graph.Graph
	// Rotaries is every rotary the blocks turn by, the model's own first;
	// each is a table of the step.
	Rotaries() []graph.Rotary
}

// tableName is rotary i's table: rope.cs for the first, rope.cs.i past it.
func tableName(i int) string {
	if i == 0 {
		return "rope.cs"
	}
	return fmt.Sprintf("rope.cs.%d", i)
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
	em, h := f.EmbedGraph(), f.HeadGraph()
	if em == nil || h == nil {
		return nil, fmt.Errorf("jit: the model states no head or input rows")
	}
	// Their composite ops as primitives, as a block's: a residual of
	// several streams filled and collapsed.
	em, h = graph.Decompose(em), graph.Decompose(h)
	rots := f.Rotaries()
	if len(rots) == 0 {
		rots = []graph.Rotary{{NRot: int(c.NRot)}}
	}
	var parts []Part
	for i, r := range rots {
		// A model that turns nothing (BERT's) has no table to fill.
		if r.NRot == 0 {
			continue
		}
		rt := graph.RopeTable{NRot: r.NRot, Name: tableName(i)}
		parts = append(parts, Part{Kind: PartTable, Block: -1, Graph: rt.Build()})
	}
	var seamRoles []jlm.Role
	embed := len(parts)
	parts = append(parts, Part{Kind: PartEmbed, Block: -1, Graph: em})
	for li := 0; li < f.Blocks(); li++ {
		g := f.BlockGraph(li)
		if g == nil {
			return nil, fmt.Errorf("jit: block %d has no graph", li)
		}
		// A second input is the block's per-layer input, the embedding's
		// output after its row.
		var feeds []Feed
		if len(g.In) > 1 {
			if len(g.In) != 2 || len(em.Out) != 1+int(c.NLayer) || em.Values[em.Out[1+li]].Width != g.Values[g.In[1]].Width {
				return nil, fmt.Errorf("jit: block %d reads %d inputs the input rows do not write", li, len(g.In))
			}
			feeds = []Feed{{In: 1, From: embed, Out: 1 + li}}
		}
		// Main's device takes a block's composite ops as their primitives
		// (graph.Decompose, as its tier prepares the block); so does the
		// composer, and its recurrent mixers too (graph.DecomposeRecurrent),
		// which main's device runs whole.
		if s := f.SeamGraph(li); s != nil {
			parts = append(parts, Part{Kind: PartSeam, Block: li, Graph: graph.Decompose(s)})
			bound := map[jlm.Role]bool{}
			for _, o := range s.Ops {
				for _, w := range o.Weight {
					bound[w.Role] = true
				}
			}
			for r := range bound {
				seamRoles = append(seamRoles, r)
			}
		}
		parts = append(parts, Part{Kind: PartBlock, Block: li, Graph: graph.DecomposeRecurrent(graph.Decompose(g), c), Feeds: feeds})
	}
	bound := seamRoles
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
	parts = append(parts, Part{Kind: PartHead, Block: -1, Graph: h})
	// A head that is not the vocabulary's logits (an encoder's embedding)
	// is the model's output: nothing is sampled from it.
	if len(h.Out) == 0 || h.Values[h.Out[0]].Width != int(c.NVocab) {
		return parts, nil
	}
	sm := graph.Sample{NVocab: int(c.NVocab)}
	return append(parts, Part{Kind: PartSample, Block: -1, Graph: sm.Build()}), nil
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
