package model

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// A vision tower's block is a transformer block over a picture's rows,
// stated in the op language a text block is: its rows attend both ways
// (graph.Attrs.Bidir), the run they attend over the step's rows'
// descriptors, as a causal block's history is. Its tensors are bound by
// the role each plays in the loaded block, a text role -- a tower's MLP
// matrices are told apart by shape at load (buildTower), not by their
// names -- and its activation is the one its flags state
// (jlm.Vision.Activation).

// towerTensors is block l's matrices by the role each plays.
func towerTensors(l *layer) [7]struct {
	r jlm.Role
	t *tensor
} {
	return [...]struct {
		r jlm.Role
		t *tensor
	}{{jlm.RoleAttnQ, &l.wq}, {jlm.RoleAttnK, &l.wk}, {jlm.RoleAttnV, &l.wv}, {jlm.RoleAttnOut, &l.wo},
		{jlm.RoleFFNGate, &l.gate}, {jlm.RoleFFNUp, &l.up}, {jlm.RoleFFNDown, &l.down}}
}

// towerVec is a tower block's vector and the role it plays.
type towerVec struct {
	Role jlm.Role
	V    []float32
}

// towerVectors is block l's vectors by the role each plays. A tower's MLP
// gate bias has no text role: it is bound under its vision role.
func towerVectors(l *layer) [15]towerVec {
	return [...]towerVec{
		{Role: jlm.RoleAttnNorm, V: l.attnNorm}, {Role: jlm.RoleAttnNormBias, V: l.attnNormB},
		{Role: jlm.RoleFFNNorm, V: l.ffnNorm}, {Role: jlm.RoleFFNNormBias, V: l.ffnNormB},
		{Role: jlm.RoleAttnQBias, V: l.bq}, {Role: jlm.RoleAttnKBias, V: l.bk},
		{Role: jlm.RoleAttnVBias, V: l.bv}, {Role: jlm.RoleAttnOutBias, V: l.bo},
		{Role: jlm.RoleFFNUpBias, V: l.upB}, {Role: jlm.RoleFFNDownBias, V: l.downB},
		{Role: jlm.RoleVFFNGateBias, V: l.gateB},
		{Role: jlm.RoleAttnQNorm, V: l.qNorm}, {Role: jlm.RoleAttnKNorm, V: l.kNorm},
		{Role: jlm.RolePostAttnNorm, V: l.postAttnNorm}, {Role: jlm.RolePostFFNNorm, V: l.postFFNNorm},
	}
}

// towerHeld is the vision roles a tower block's loaded fields are read
// from (buildTower): a tensor of another role in the block is one its graph
// would not bind.
var towerHeld = [...]jlm.Role{
	jlm.RoleVAttnQ, jlm.RoleVAttnK, jlm.RoleVAttnV, jlm.RoleVAttnOut, jlm.RoleVFC1, jlm.RoleVFC2,
	jlm.RoleVFC1Bias, jlm.RoleVFC2Bias, jlm.RoleVAttnNorm, jlm.RoleVAttnNormBias, jlm.RoleVFFNNorm,
	jlm.RoleVFFNNormBias, jlm.RoleVAttnQBias, jlm.RoleVAttnKBias, jlm.RoleVAttnVBias, jlm.RoleVAttnOutBias,
	jlm.RoleVPostAttnNorm, jlm.RoleVPostFFNNorm, jlm.RoleVAttnQNorm, jlm.RoleVAttnKNorm, jlm.RoleVClamp,
	jlm.RoleVFFNGate, jlm.RoleVFFNGateBias,
}

// towerInventory is block l's tensors by the role each plays, at the shape
// a binding states.
func towerInventory(l *layer) graph.Facts {
	inv := map[jlm.Role]graph.Shape{}
	for _, m := range towerTensors(l) {
		if m.t.e != nil {
			inv[m.r] = graph.Shape{K: m.t.k, Rows: m.t.rows}
		}
	}
	for _, v := range towerVectors(l) {
		if len(v.V) != 0 {
			inv[v.Role] = graph.Shape{K: len(v.V)}
		}
	}
	return graph.Facts{Inventory: inv}
}

// towerTensor is tower block l's matrix of role r, nil for one it does
// not hold.
func towerTensor(l *layer, r jlm.Role) *tensor {
	for _, m := range towerTensors(l) {
		if m.r == r && m.t.e != nil {
			return m.t
		}
	}
	return nil
}

// towerUncovered is why tower block li has no graph, Covered where its
// features are ones the op language states.
func (m *Model) towerUncovered(li int) (Uncovered, error) {
	t, l := m.tower, &m.layers[li]
	switch {
	case l.mn != nil:
		return UncoveredKind, nil
	case l.clamp != nil:
		return UncoveredClamp, nil
	}
	if t.Cfg.Rope {
		if _, ok := towerRopeSections(t.Cfg.towerRope()); !ok {
			return UncoveredRopeAxes, nil
		}
	}
	vi := int32(li - t.base)
	for _, e := range m.container.Entries() {
		if !e.Role.Vision() || e.Block != vi {
			continue
		}
		held := false
		for _, r := range towerHeld {
			held = held || r == e.Role
		}
		if !held {
			return UncoveredUnbound, fmt.Errorf("the container holds %s for tower block %d, which no graph binds", e.Role, vi)
		}
	}
	return Covered, nil
}

// towerRopeSections is rotary r's pairs by position axis as the op language
// states them (graph.Attrs.Sections): runs of pairs over axes in order, at
// most four. A rotary whose runs come back to an axis is not stated.
func towerRopeSections(r nn.Rope) (s [4]int, ok bool) {
	ax := -1
	for _, run := range r.Runs {
		if run.Axis <= ax || run.Axis >= len(s) {
			return s, false
		}
		ax = run.Axis
		s[run.Axis] = run.Pairs
	}
	return s, len(r.Runs) > 0
}

// towerGraph is tower block li's graph (li in the model's block space).
func (m *Model) towerGraph(li int) (*graph.Graph, Uncovered, error) {
	if why, err := m.towerUncovered(li); why != Covered {
		return nil, why, err
	}
	t, l := m.tower, &m.layers[li]
	c, tc := t.cfg, t.Cfg
	f := towerInventory(l)
	has := func(r jlm.Role) bool {
		_, ok := f.Inventory[r]
		return ok
	}
	stored := func(r jlm.Role) []jlm.Role {
		if has(r) {
			return []jlm.Role{r}
		}
		return nil
	}
	d, hd := c.NEmbd, c.HeadDim
	qdim, kvDim := c.NHead*hd, c.NKVHead*hd
	b := graph.NewBlock(li)
	norm := func(v graph.ValueID, r, bias jlm.Role) graph.ValueID {
		if !has(r) {
			return v
		}
		if c.LayerNorm {
			return b.LayerNorm(v, r, c.RMSEps, stored(bias)...)
		}
		return b.Norm(v, r, c.RMSEps)
	}
	x := b.In("x", d)
	n := norm(x, jlm.RoleAttnNorm, jlm.RoleAttnNormBias)
	q := b.MatVec(n, jlm.RoleAttnQ, qdim, stored(jlm.RoleAttnQBias)...)
	k := b.MatVec(n, jlm.RoleAttnK, kvDim, stored(jlm.RoleAttnKBias)...)
	v := b.MatVec(n, jlm.RoleAttnV, kvDim, stored(jlm.RoleAttnVBias)...)
	if has(jlm.RoleAttnQNorm) {
		q = b.HeadNorm(q, jlm.RoleAttnQNorm, hd, c.RMSEps)
	}
	if has(jlm.RoleAttnKNorm) {
		k = b.HeadNorm(k, jlm.RoleAttnKNorm, hd, c.RMSEps)
	}
	if tc.Rope {
		r := tc.towerRope()
		layout := graph.RopeNorm
		if c.RopeNeox {
			layout = graph.RopeNeox
		}
		a := graph.Attrs{HeadDim: hd, NRot: c.NRot, Base: r.Base, Layout: layout}
		a.Sections, _ = towerRopeSections(r)
		q = b.Emit(graph.OpRope, []graph.ValueID{q}, []graph.Result{{Name: "q.rope", Width: qdim}}, nil, a, nil, nil)[0]
		k = b.Emit(graph.OpRope, []graph.ValueID{k}, []graph.Result{{Name: "k.rope", Width: kvDim}}, nil, a, nil, nil)[0]
	}
	if c.VNorm {
		v = b.Emit(graph.OpHeadNorm, []graph.ValueID{v}, []graph.Result{{Name: "v.norm", Width: kvDim}}, nil,
			graph.Attrs{Eps: c.RMSEps, HeadDim: hd}, nil, nil)[0]
	}
	hist := graph.StateRef{Name: "kv", Version: b.KVAppend(k, v, 0)}
	// A windowed block's rows attend inside their window of the picture,
	// each row's from the step's window descriptors.
	att := b.AttendOf(q, hist, graph.Attrs{NHead: c.NHead, NKVHead: c.NKVHead, HeadDim: hd, Scale: c.AttnScale, Bidir: true,
		RowWindows: tc.windowedAt(li - t.base)})
	proj := b.MatVec(att, jlm.RoleAttnOut, d, stored(jlm.RoleAttnOutBias)...)
	if has(jlm.RolePostAttnNorm) {
		proj = b.Norm(proj, jlm.RolePostAttnNorm, c.RMSEps)
	}
	x2 := b.Add(x, proj)
	n2 := norm(x2, jlm.RoleFFNNorm, jlm.RoleFFNNormBias)
	var acts []graph.ValueID
	gated := has(jlm.RoleFFNGate)
	if gated {
		acts = append(acts, b.MatVec(n2, jlm.RoleFFNGate, c.NFFN, stored(jlm.RoleVFFNGateBias)...))
	}
	acts = append(acts, b.MatVec(n2, jlm.RoleFFNUp, c.NFFN, stored(jlm.RoleFFNUpBias)...))
	act, err := m.container.Vision().Activation(gated)
	if err != nil {
		return nil, UncoveredAct, err
	}
	h, err := b.Stated("act", &act, acts, c.NFFN, f)
	if err != nil {
		return nil, UncoveredAct, fmt.Errorf("the tower block's stated activation: %w", err)
	}
	ffn := b.MatVec(h, jlm.RoleFFNDown, d, stored(jlm.RoleFFNDownBias)...)
	if has(jlm.RolePostFFNNorm) {
		ffn = b.Norm(ffn, jlm.RolePostFFNNorm, c.RMSEps)
	}
	b.Out(b.Add(x2, ffn))
	return verified(b, f)
}

// towerFormat is the format tower block l holds role r in: a matrix's
// container type, a vector's f32 (read once at load and kept so).
func towerFormat(l *layer, r jlm.Role) (jlm.Type, bool) {
	if t := towerTensor(l, r); t != nil {
		return t.e.Type, true
	}
	for _, v := range towerVectors(l) {
		if v.Role == r && len(v.V) != 0 {
			return jlm.TypeF32, true
		}
	}
	return 0, false
}
