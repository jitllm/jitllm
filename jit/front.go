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
	// and out, its weights the model's at the index its graph's Block names
	// (-1 for any) -- a decision head's prologue, a tower's deepstack tap.
	// An output past the residual is a row the head reads.
	PartSeam
)

// Segment is which of the model's segments a part is of: the text model,
// or its vision tower, whose blocks are blocks of the one model and whose
// step is a picture's.
type Segment uint8

const (
	SegmentText Segment = iota
	SegmentVision
)

// Part is one placeable part of the model and its graph.
type Part struct {
	Kind    PartKind
	Segment Segment
	Block   int
	Graph   *graph.Graph
	// Feeds are the inputs past the first that an earlier part writes: a
	// block's per-layer input, the embedding's. The first is the part
	// before's first output.
	Feeds []Feed
}

// Index is the index a part's model-wide weights are found at: a seam's
// its graph's Block, -1 (any) for every other.
func (p Part) Index() int32 {
	if p.Kind == PartSeam && p.Graph.Block >= 0 {
		return int32(p.Graph.Block)
	}
	return -1
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
	// block li (li == Blocks: before the head), nil for none.
	SeamGraph(li int) *graph.Graph
	HeadGraph() *graph.Graph
	EmbedGraph() *graph.Graph
	// Rotaries is every rotary the blocks turn by, the model's own first;
	// each is a table of the step.
	Rotaries() []graph.Rotary
	// Tower says the model is a vision tower's segment: its tensors are the
	// container's vision tensors, and the text model's are not its.
	Tower() bool
	// Segment is the model's vision segment as the composer reads it, nil
	// for a model with none: its parts are the one model's, a picture's
	// step writing the text step's input rows at the picture's place.
	Segment() Model
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
		// there too); a vision tensor is the tower's segment, and the text
		// model's tensors are not the tower's.
		switch e.Role {
		case jlm.RoleRopeFreqs, jlm.RoleRopeFactorsShort, jlm.RoleRopeFactorsLong:
			continue
		}
		if e.Role.Vision() != f.Tower() {
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
// its logits. Where vision, the model's vision segment comes first, its parts
// the one model's: its rotary, its entry over a picture, its blocks, its
// projector, whose rows the text step's input rows take at the picture's
// place. A model-wide tensor no graph binds, the front end refuses.
func Front(f Model, vision bool) ([]Part, error) {
	rots := textRotaries(f)
	var parts []Part
	picture, picW := -1, 0
	if vision {
		v := f.Segment()
		if v == nil {
			return nil, fmt.Errorf("jit: the model has no vision segment")
		}
		vp, err := segmentParts(v, SegmentVision, len(parts), len(rots), -1, 0)
		if err != nil {
			return nil, fmt.Errorf("jit: vision: %w", err)
		}
		parts = append(parts, vp...)
		picture = len(parts) - 1
		// The head writes each row's projected row, then each tap's.
		hp := parts[picture]
		if w := hp.Graph; len(w.Out) != 1 || w.Values[w.Out[0]].Width != (1+len(hp.Feeds))*int(f.Config().NEmbd) {
			return nil, fmt.Errorf("jit: the vision segment's projector does not write rows the text model's %d wide", f.Config().NEmbd)
		}
		picW = hp.Graph.Values[hp.Graph.Out[0]].Width
	}
	tp, err := segmentParts(f, SegmentText, len(parts), 0, picture, picW)
	if err != nil {
		return nil, err
	}
	return append(parts, tp...), nil
}

// textRotaries is the text model's rotaries, its configuration's where it
// states none.
func textRotaries(f Model) []graph.Rotary {
	if rots := f.Rotaries(); len(rots) > 0 {
		return rots
	}
	return []graph.Rotary{{NRot: int(f.Config().NRot)}}
}

// segmentParts is segment seg of the model f states, its parts from index
// first on, its rotaries' tables named from rotBase; where picture is a
// part's index, the input rows take that part's rows where the step names
// them (OpSplice).
func segmentParts(f Model, seg Segment, first, rotBase, picture, picW int) ([]Part, error) {
	c := f.Config()
	em, h := f.EmbedGraph(), f.HeadGraph()
	if em == nil || h == nil {
		return nil, fmt.Errorf("jit: the model states no head or input rows")
	}
	var embedFeeds []Feed
	if picture >= 0 {
		em = graph.Spliced(em, picW)
		embedFeeds = []Feed{{In: len(em.In) - 1, From: picture, Out: 0}}
	}
	rots := f.Rotaries()
	if seg == SegmentText {
		rots = textRotaries(f)
	}
	var parts []Part
	at := func() int { return first + len(parts) }
	for i, r := range rots {
		// A model that turns nothing (BERT's) has no table to fill.
		if r.NRot == 0 {
			continue
		}
		rt := graph.RopeTable{NRot: r.NRot, Name: tableName(rotBase + i), Grid: r.Grid}
		parts = append(parts, Part{Kind: PartTable, Segment: seg, Block: -1, Graph: rt.Build()})
	}
	var seamRoles []jlm.Role
	var taps []int
	seam := func(li int) {
		s := f.SeamGraph(li)
		if s == nil {
			return
		}
		if len(s.Out) > 1 {
			taps = append(taps, at())
		}
		parts = append(parts, Part{Kind: PartSeam, Segment: seg, Block: li, Graph: s})
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
	embed := at()
	parts = append(parts, Part{Kind: PartEmbed, Segment: seg, Block: -1, Graph: em, Feeds: embedFeeds})
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
		seam(li)
		parts = append(parts, Part{Kind: PartBlock, Segment: seg, Block: li, Graph: g, Feeds: feeds})
	}
	seam(f.Blocks())
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
	// The head's inputs past the first are the seams' rows, in order.
	if len(h.In) != 1+len(taps) {
		return nil, fmt.Errorf("jit: the head reads %d inputs and %d seams give rows", len(h.In), len(taps))
	}
	var hf []Feed
	for i, from := range taps {
		hf = append(hf, Feed{In: 1 + i, From: from, Out: 1})
	}
	parts = append(parts, Part{Kind: PartHead, Segment: seg, Block: -1, Graph: h, Feeds: hf})
	// A head that is not the vocabulary's logits (an encoder's embedding, a
	// tower's projector) is the segment's output: nothing is sampled from it.
	if seg != SegmentText || len(h.Out) == 0 || h.Values[h.Out[0]].Width != int(c.NVocab) {
		return parts, nil
	}
	sm := graph.Sample{NVocab: int(c.NVocab)}
	return append(parts, Part{Kind: PartSample, Block: -1, Graph: sm.Build()}), nil
}

// formats is each of a part's weights' storage format.
func formats(f Model, p Part) map[jlm.Role]jlm.Type {
	out := map[jlm.Role]jlm.Type{}
	block, index := int32(p.Block), p.Index()
	if p.Kind != PartBlock {
		block = jlm.DenseBlock
	}
	for _, o := range p.Graph.Ops {
		for _, w := range o.Weight {
			if e, ok := f.Find(w.Role, block, index); ok {
				out[w.Role] = e.Type
			}
		}
	}
	return out
}
