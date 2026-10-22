package jit

import (
	"fmt"
	"math"
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

// Model is what the composer reads of a model: its configuration and its
// tensors' index entries, never their bytes. A container is one; a model's
// header shipped to another node is another.
type Model interface {
	Config() *jlm.Config
	Find(role jlm.Role, block, index int32) (*jlm.Entry, bool)
	Entries() []jlm.Entry
}

// unread is the first tensor of block (jlm.DenseBlock for none) whose role
// the front end does not read: what it does not read, it refuses rather than
// composing a model without it.
func unread(f Model, block int32, read ...jlm.Role) error {
	for _, e := range f.Entries() {
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

// Front is the model's graph from its container: the step's rotary table,
// its input rows' embedding, every block, the head, then the sampling of
// its logits. What the
// container states, the front end reads; what it does not, the front end
// refuses.
func Front(f Model) ([]Part, error) {
	c := f.Config()
	rt := graph.RopeTable{NRot: int(c.NRot)}
	if c.EmbdScale != 0 && c.EmbdScale != 1 {
		return nil, fmt.Errorf("jit: the container scales the embedding by %v, which the front end does not read", c.EmbdScale)
	}
	em := graph.Embed{NEmbd: int(c.NEmbd), NVocab: int(c.NVocab)}
	parts := []Part{{Kind: PartTable, Block: -1, Graph: rt.Build()}, {Kind: PartEmbed, Block: -1, Graph: em.Build()}}
	for li := 0; li < int(c.NLayer); li++ {
		g, err := blockGraph(f, li)
		if err != nil {
			return nil, fmt.Errorf("jit: block %d: %w", li, err)
		}
		parts = append(parts, Part{Kind: PartBlock, Block: li, Graph: g})
	}
	h, err := headGraph(f)
	if err != nil {
		return nil, fmt.Errorf("jit: head: %w", err)
	}
	sm := graph.Sample{NVocab: int(c.NVocab)}
	return append(parts, Part{Kind: PartHead, Block: -1, Graph: h}, Part{Kind: PartSample, Block: -1, Graph: sm.Build()}), nil
}

// stated is the container's facts the front end reads; anything else a
// model states is refused until the front end reads it.
const stated = jlm.FlagTiedEmbd | jlm.FlagRopeNeox | jlm.FlagGELU

func blockGraph(f Model, li int) (*graph.Graph, error) {
	c := f.Config()
	if c.Flags&^stated != 0 || c.NExpert != 0 || len(c.LayerKinds) != 0 || c.SWAWindow != 0 || c.YarnFactor != 0 || c.YarnLogMul != 0 {
		return nil, fmt.Errorf("the container states what the front end does not read (flags %#x)", uint32(c.Flags&^stated))
	}
	roles := []jlm.Role{jlm.RoleAttnNorm, jlm.RoleAttnQ, jlm.RoleAttnK, jlm.RoleAttnV, jlm.RoleAttnOut,
		jlm.RoleFFNNorm, jlm.RoleFFNGate, jlm.RoleFFNUp, jlm.RoleFFNDown}
	for _, r := range roles {
		if !has(f, r, int32(li), -1) {
			return nil, fmt.Errorf("no %s", r)
		}
	}
	if err := unread(f, int32(li), roles...); err != nil {
		return nil, err
	}
	layout := graph.RopeNorm
	if c.Flags&jlm.FlagRopeNeox != 0 {
		layout = graph.RopeNeox
	}
	// The container's feed-forward is SiLU unless it states GELU.
	act := graph.ActSiLU
	if c.Flags&jlm.FlagGELU != 0 {
		act = graph.ActGELUTanh
	}
	// The score's scale is the container's where it states one, else
	// 1/sqrt of the head's width.
	scale := 1 / math.Sqrt(float64(c.HeadDimAt(li)))
	if c.AttnScale != 0 {
		scale = float64(c.AttnScale)
	}
	d := graph.Dense{Block: li, NEmbd: int(c.NEmbd), NHead: int(c.NHead), NKVHead: int(c.NKVHeadAt(li)),
		HeadDim: int(c.HeadDimAt(li)), NFFN: int(c.NFFNAt(li)), Eps: float64(c.RMSEps), NRot: int(c.NRot),
		Base: float64(c.RopeBase), Layout: layout, Act: act, Scale: scale}
	return d.Build(), nil
}

// headGraph is the final norm and the vocabulary's projection, the
// embedding's rows where the container ties them.
func headGraph(f Model) (*graph.Graph, error) {
	c := f.Config()
	out := jlm.RoleOutput
	if c.Flags&jlm.FlagTiedEmbd != 0 {
		out = jlm.RoleTokenEmbd
	}
	if !has(f, jlm.RoleOutputNorm, jlm.DenseBlock, -1) || !has(f, out, jlm.DenseBlock, -1) {
		return nil, fmt.Errorf("no output norm or projection")
	}
	if err := unread(f, jlm.DenseBlock, jlm.RoleTokenEmbd, jlm.RoleOutputNorm, out); err != nil {
		return nil, err
	}
	h := graph.Head{NEmbd: int(c.NEmbd), NVocab: int(c.NVocab), Eps: float64(c.RMSEps), Out: out}
	return h.Build(), nil
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
