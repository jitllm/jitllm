package jit_test

import (
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
)

const (
	synthEmbd   = 64
	synthFFN    = 128
	synthVocab  = 96
	synthBlocks = 3
)

// synth is a model of generic slices as the composer reads it: an embedding,
// blocks that norm, gate, up, activate, down and add, and a head.
type synth struct {
	cfg     jlm.Config
	entries []jlm.Entry
	blocks  []*graph.Graph
	embed   *graph.Graph
	head    *graph.Graph
}

var embedType = jlm.TypeF32

func newSynth(t *testing.T, typ jlm.Type) *synth {
	t.Helper()
	s := &synth{cfg: jlm.Config{NLayer: synthBlocks, NEmbd: synthEmbd, NVocab: synthVocab}}
	add := func(role jlm.Role, block int32, k, rows int) {
		e := jlm.Entry{Role: role, Block: block, Index: -1, Type: typ, NDim: 2, Dims: [4]uint64{uint64(k), uint64(rows)}}
		if rows == 0 {
			e.NDim, e.Type = 1, jlm.TypeF32
		}
		s.entries = append(s.entries, e)
	}
	add(jlm.RoleTokenEmbd, jlm.DenseBlock, synthEmbd, synthVocab)
	s.entries[0].Type = embedType
	add(jlm.RoleOutputNorm, jlm.DenseBlock, synthEmbd, 0)
	add(jlm.RoleOutput, jlm.DenseBlock, synthEmbd, synthVocab)
	s.embed = &graph.Graph{Block: -1, Values: []graph.Value{{Name: "x", Width: synthEmbd}}, Out: []graph.ValueID{0},
		Ops: []graph.Op{{Kind: graph.OpEmbed, Out: []graph.ValueID{0},
			Weight: []graph.Binding{{Role: jlm.RoleTokenEmbd, K: synthEmbd, Rows: synthVocab}}}}}
	h := graph.New()
	h.G.Block = -1
	h.Out(h.MatVec(h.Norm(h.In("x", synthEmbd), jlm.RoleOutputNorm, 1e-5), jlm.RoleOutput, synthVocab))
	s.head = h.G
	for li := 0; li < synthBlocks; li++ {
		add(jlm.RoleFFNNorm, int32(li), synthEmbd, 0)
		add(jlm.RoleFFNGate, int32(li), synthEmbd, synthFFN)
		add(jlm.RoleFFNUp, int32(li), synthEmbd, synthFFN)
		add(jlm.RoleFFNDown, int32(li), synthFFN, synthEmbd)
		s.blocks = append(s.blocks, synthBlock(li))
	}
	return s
}

func synthBlock(li int) *graph.Graph {
	b := graph.New()
	b.G.Block = li
	x := b.In("x", synthEmbd)
	n := b.Norm(x, jlm.RoleFFNNorm, 1e-5)
	gate := b.MatVec(n, jlm.RoleFFNGate, synthFFN)
	up := b.MatVec(n, jlm.RoleFFNUp, synthFFN)
	b.Out(b.Add(x, b.MatVec(b.ActMul(graph.ActSiLU, gate, up), jlm.RoleFFNDown, synthEmbd)))
	return b.G
}

func (s *synth) Config() *jlm.Config            { return &s.cfg }
func (s *synth) Entries() []jlm.Entry           { return s.entries }
func (s *synth) BlockGraph(li int) *graph.Graph { return s.blocks[li] }
func (s *synth) Blocks() int                    { return synthBlocks }
func (s *synth) SeamGraph(int) *graph.Graph     { return nil }
func (s *synth) HeadGraph() *graph.Graph        { return s.head }
func (s *synth) EmbedGraph() *graph.Graph       { return s.embed }
func (s *synth) Rotaries() []graph.Rotary       { return nil }
func (s *synth) Tower() bool                    { return false }
func (s *synth) Segment() jit.Model             { return nil }

func (s *synth) Find(role jlm.Role, block, index int32) (*jlm.Entry, bool) {
	for i := range s.entries {
		if e := &s.entries[i]; e.Role == role && e.Block == block && e.Index == index {
			return e, true
		}
	}
	return nil, false
}
