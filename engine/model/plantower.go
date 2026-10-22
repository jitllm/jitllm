package model

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/engine/run"
	"github.com/jitllm/jitllm/format/jlm"
)

// towerComposing is a vision tower as the composer reads it: its blocks'
// graphs (towergraph.go), the patches in and the projector out, each graph
// role found as the container stores it -- a block's matrices through the
// loaded block (a tower's MLP matrices are told apart by shape), its vectors
// under their vision roles.
type towerComposing struct {
	*jlm.File
	m   *Model
	t   *Tower
	cfg jlm.Config
}

// TowerComposing is the model's tower as the composer and the plan's runner
// read it, nil where the container carries none.
func (m *Model) TowerComposing() run.Source {
	t := m.tower
	if t == nil {
		return nil
	}
	cfg := *m.container.Config()
	cfg.NEmbd, cfg.NLayer, cfg.NVocab, cfg.NRot = uint32(t.Cfg.NEmbd), uint32(len(t.graphs)), 0, 0
	return towerComposing{File: m.container, m: m, t: t, cfg: cfg}
}

func (c towerComposing) Config() *jlm.Config { return &c.cfg }
func (c towerComposing) Tower() bool         { return true }
func (c towerComposing) Blocks() int         { return len(c.t.graphs) }
func (c towerComposing) ActWindow() int      { return c.m.ActWindow() }

func (c towerComposing) SeamGraph(int) *graph.Graph { return nil }
func (c towerComposing) Rotaries() []graph.Rotary   { return nil }
func (c towerComposing) Ropes() []nn.Rope           { return nil }
func (c towerComposing) BlockGraph(li int) *graph.Graph {
	if li < 0 || li >= len(c.t.graphs) {
		return nil
	}
	return c.t.graphs[li]
}

// towerVectorRoles is each block vector's vision role by the role its graph
// binds.
var towerVectorRoles = map[jlm.Role]jlm.Role{
	jlm.RoleAttnNorm: jlm.RoleVAttnNorm, jlm.RoleAttnNormBias: jlm.RoleVAttnNormBias,
	jlm.RoleFFNNorm: jlm.RoleVFFNNorm, jlm.RoleFFNNormBias: jlm.RoleVFFNNormBias,
	jlm.RoleAttnQBias: jlm.RoleVAttnQBias, jlm.RoleAttnKBias: jlm.RoleVAttnKBias,
	jlm.RoleAttnVBias: jlm.RoleVAttnVBias, jlm.RoleAttnOutBias: jlm.RoleVAttnOutBias,
	jlm.RolePostAttnNorm: jlm.RoleVPostAttnNorm, jlm.RolePostFFNNorm: jlm.RoleVPostFFNNorm,
	jlm.RoleAttnQNorm: jlm.RoleVAttnQNorm, jlm.RoleAttnKNorm: jlm.RoleVAttnKNorm, jlm.RoleVClamp: jlm.RoleVClamp,
}

// towerBiasOf is each MLP matrix's bias role, by the matrix's own.
var towerBiasOf = map[jlm.Role]jlm.Role{jlm.RoleVFC1: jlm.RoleVFC1Bias, jlm.RoleVFC2: jlm.RoleVFC2Bias}

// Find is a tensor of the tower: a block's by the role its graph binds, a
// tower-wide one (jlm.DenseBlock) as the container stores it.
func (c towerComposing) Find(role jlm.Role, block, index int32) (*jlm.Entry, bool) {
	if block == jlm.DenseBlock {
		return c.File.Find(role, block, index)
	}
	if block < 0 || int(block) >= len(c.t.graphs) {
		return nil, false
	}
	l := &c.m.layers[c.t.base+int(block)]
	mats := map[jlm.Role]*tensor{jlm.RoleAttnQ: &l.wq, jlm.RoleAttnK: &l.wk, jlm.RoleAttnV: &l.wv, jlm.RoleAttnOut: &l.wo,
		jlm.RoleFFNGate: &l.gate, jlm.RoleFFNUp: &l.up, jlm.RoleFFNDown: &l.down}
	if t, ok := mats[role]; ok {
		return t.e, t.e != nil
	}
	v, ok := towerVectorRoles[role]
	switch role {
	case jlm.RoleFFNUpBias, jlm.RoleFFNDownBias:
		m := l.up.e
		if role == jlm.RoleFFNDownBias {
			m = l.down.e
		}
		if m == nil {
			return nil, false
		}
		v, ok = towerBiasOf[m.Role]
	}
	if !ok {
		return nil, false
	}
	return c.File.Find(v, block, index)
}

// EmbedGraph is the tower's entry: its class row first where it has one,
// the patches through the patch embedding, their positions' rows and its
// pre-norm.
func (c towerComposing) EmbedGraph() *graph.Graph {
	t, tc := c.t, c.t.Cfg
	if tc.CLS && (tc.CLSLast || t.opt.towerCLSLast) || tc.PosBuckets > 0 || tc.PosSide > 0 || tc.PosAxis > 0 ||
		tc.Dynamic() || t.embdNorm != nil {
		return nil
	}
	e := graph.TowerEntry{NEmbd: tc.NEmbd, Patch: tc.PatchSz, K: t.patchW.k, PatchBias: t.patchB != nil,
		Class: tc.CLS, PreNorm: t.preLnW != nil, Eps: tc.Eps}
	if t.posW != nil {
		e.Positions = len(t.posW) / tc.NEmbd
	}
	return e.Build()
}

// HeadGraph is the tower's projector: idefics3's (its post-norm, its
// regroup and one matrix) or llava's (the class row dropped, a matrix, the
// tanh GELU llama.cpp's projector runs, a matrix).
func (c towerComposing) HeadGraph() *graph.Graph {
	t, tc := c.t, c.t.Cfg
	h := graph.TowerHead{NEmbd: tc.NEmbd, Scale: tc.Scale, ProjDim: tc.ProjDim, PostNorm: t.postLnW != nil,
		Class: tc.CLS, ProjBias: t.projB != nil, Eps: tc.Eps}
	switch {
	case tc.Kind == jlm.ProjIdefics3 && t.projW2.e == nil:
	case tc.Kind == jlm.ProjMLP && t.projW2.e != nil && tc.Scale <= 1 && (t.projB != nil) == (t.projB2 != nil):
		h.Hidden, h.Act = t.projW.rows, graph.ActGELUTanh
	default:
		return nil
	}
	return h.Build()
}

// towerRefusal names why the tower has no composed plan, nil where it has
// one.
func (c towerComposing) towerRefusal() error {
	if c.EmbedGraph() == nil || c.HeadGraph() == nil {
		return fmt.Errorf("model: the %s tower's entry or projector is not stated for the composer", c.t.Cfg.Projector)
	}
	for li, g := range c.t.graphs {
		if g == nil {
			return fmt.Errorf("model: the %s tower's block %d is not stated as a graph", c.t.Cfg.Projector, li)
		}
	}
	return nil
}
