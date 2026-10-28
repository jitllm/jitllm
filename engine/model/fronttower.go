package model

import (
	"fmt"
	"slices"

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
func towerVectors(l *layer) [16]towerVec {
	return [...]towerVec{
		{Role: jlm.RoleAttnNorm, V: l.attnNorm}, {Role: jlm.RoleAttnNormBias, V: l.attnNormB},
		{Role: jlm.RoleFFNNorm, V: l.ffnNorm}, {Role: jlm.RoleFFNNormBias, V: l.ffnNormB},
		{Role: jlm.RoleAttnQBias, V: l.bq}, {Role: jlm.RoleAttnKBias, V: l.bk},
		{Role: jlm.RoleAttnVBias, V: l.bv}, {Role: jlm.RoleAttnOutBias, V: l.bo},
		{Role: jlm.RoleFFNUpBias, V: l.upB}, {Role: jlm.RoleFFNDownBias, V: l.downB},
		{Role: jlm.RoleVFFNGateBias, V: l.gateB},
		{Role: jlm.RoleAttnQNorm, V: l.qNorm}, {Role: jlm.RoleAttnKNorm, V: l.kNorm},
		{Role: jlm.RolePostAttnNorm, V: l.postAttnNorm}, {Role: jlm.RolePostFFNNorm, V: l.postFFNNorm},
		{Role: jlm.RoleVClamp, V: l.clamp},
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
	case l.clamp != nil && len(l.clamp) != 4*clampSlots:
		return UncoveredClamp, nil
	}
	if t.Cfg.Rope {
		if _, ok := towerRopeAxes(t.Cfg.towerRope()); !ok {
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

// towerRopeAxes is rotary r's pairs by position axis as the op language
// states them: runs of pairs over axes in order, at most four
// (graph.Attrs.Sections); or one pair a run, the axes repeating in a cycle
// of at most four, each pair at its cycle's first pair's frequency
// (graph.Attrs.Cycle). A rotary whose runs come back to an axis any other
// way is not stated.
func towerRopeAxes(r nn.Rope) (a graph.Attrs, ok bool) {
	if a, ok = towerRopeSections(r); ok {
		return a, true
	}
	return towerRopeCycle(r)
}

func towerRopeSections(r nn.Rope) (a graph.Attrs, ok bool) {
	ax := -1
	for _, run := range r.Runs {
		if run.Axis <= ax || run.Axis >= len(a.Sections) {
			return graph.Attrs{}, false
		}
		ax = run.Axis
		a.Sections[run.Axis] = run.Pairs
	}
	return a, len(r.Runs) > 0
}

// towerRopeCycle is rotary r's axes as a cycle (graph.Attrs.Cycle): its
// first runs' distinct axes, every run one pair, run p on the axis of run
// p%n at the frequency of pair p - p%n.
func towerRopeCycle(r nn.Rope) (a graph.Attrs, ok bool) {
	n := 0
	for n < len(r.Runs) && n < len(a.CycleAxes) {
		if slices.Contains(a.CycleAxes[:n], uint8(r.Runs[n].Axis)) {
			break
		}
		a.CycleAxes[n] = uint8(r.Runs[n].Axis)
		n++
	}
	if n < 2 || len(r.Runs)%n != 0 || len(r.Runs) != r.NRot/2 {
		return graph.Attrs{}, false
	}
	for p, run := range r.Runs {
		if run.Pairs != 1 || run.Axis < 0 || run.Axis >= len(a.CycleAxes) || uint8(run.Axis) != a.CycleAxes[p%n] || run.Freq != p-p%n {
			return graph.Attrs{}, false
		}
	}
	a.Cycle = n
	return a, true
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
	// A clamped linear's input and output are held to its slot's stored
	// bounds (jlm.RoleVClamp): the input's at 4*slot, the output's at
	// 4*slot+2.
	clamped := has(jlm.RoleVClamp)
	clamp := func(v graph.ValueID, name string, off int) graph.ValueID {
		return b.Emit(graph.OpClamp, []graph.ValueID{v}, []graph.Result{{Name: name, Width: b.G.Values[v].Width}},
			[]graph.Binding{{Role: jlm.RoleVClamp, K: 4 * clampSlots}}, graph.Attrs{Offset: off}, nil, nil)[0]
	}
	linear := func(in graph.ValueID, slot int, r jlm.Role, width int, bias ...jlm.Role) graph.ValueID {
		if clamped {
			in = clamp(in, r.String()+".in", 4*slot)
		}
		out := b.MatVec(in, r, width, bias...)
		if clamped {
			out = clamp(out, r.String()+".out", 4*slot+2)
		}
		return out
	}
	x := b.In("x", d)
	n := norm(x, jlm.RoleAttnNorm, jlm.RoleAttnNormBias)
	q := linear(n, clampQ, jlm.RoleAttnQ, qdim, stored(jlm.RoleAttnQBias)...)
	k := linear(n, clampK, jlm.RoleAttnK, kvDim, stored(jlm.RoleAttnKBias)...)
	v := linear(n, clampV, jlm.RoleAttnV, kvDim, stored(jlm.RoleAttnVBias)...)
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
		a, _ := towerRopeAxes(r)
		a.HeadDim, a.NRot, a.Base, a.Layout = hd, c.NRot, r.Base, layout
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
	proj := linear(att, clampO, jlm.RoleAttnOut, d, stored(jlm.RoleAttnOutBias)...)
	if has(jlm.RolePostAttnNorm) {
		proj = b.Norm(proj, jlm.RolePostAttnNorm, c.RMSEps)
	}
	x2 := b.Add(x, proj)
	n2 := norm(x2, jlm.RoleFFNNorm, jlm.RoleFFNNormBias)
	var acts []graph.ValueID
	gated := has(jlm.RoleFFNGate)
	if gated {
		acts = append(acts, linear(n2, clampGate, jlm.RoleFFNGate, c.NFFN, stored(jlm.RoleVFFNGateBias)...))
	}
	acts = append(acts, linear(n2, clampUp, jlm.RoleFFNUp, c.NFFN, stored(jlm.RoleFFNUpBias)...))
	act, err := m.container.Vision().Activation(gated)
	if err != nil {
		return nil, UncoveredAct, err
	}
	h, err := b.Stated("act", &act, acts, c.NFFN, f)
	if err != nil {
		return nil, UncoveredAct, fmt.Errorf("the tower block's stated activation: %w", err)
	}
	ffn := linear(h, clampDown, jlm.RoleFFNDown, d, stored(jlm.RoleFFNDownBias)...)
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

// towerProjRead is the model-wide vision tensors the tower's entry and its
// projector graphs read: a container holding another is a projector they
// do not state.
var towerProjRead = [...]jlm.Role{
	jlm.RoleVPatchEmbd, jlm.RoleVPatchBias, jlm.RoleVClassEmbd, jlm.RoleVPosEmbd, jlm.RoleVPreNorm,
	jlm.RoleVPreNormBias, jlm.RoleVPostNorm, jlm.RoleVPostNormBias, jlm.RoleVProj, jlm.RoleVProjBias,
	jlm.RoleVProj2, jlm.RoleVProj2Bias, jlm.RoleVProjNorm, jlm.RoleVProjNormBias,
}

// towerProj is the tower's projector as graphs of generic ops over the
// patch rows (behind the tower's last norm, the class row dropped): group
// regroups the rows of a projector reading several patches a row into
// consecutive rows by the request's row names (OpTakeRows), nil for one
// reading a patch a row; proj runs over the merged rows, Scale^2 patches a
// row -- their mean and its RMSNorm where the projector reads one patch's
// width (a pool), the merged row's LayerNorm where the tower holds one, the
// matrices and the activation between them. win is proj over rows in the
// window order, which holds each group already, the raster order put back
// by the request's row names after it; nil for a tower with no windows.
type towerProj struct {
	group, proj, win *graph.Graph
	pool             bool
}

// towerProjOf is the tower's projector graphs, or why it has none. corner
// is a gate's violation: the pool reads each group's first patch alone.
func (m *Model) towerProjOf(corner bool) (towerProj, Uncovered, error) {
	t := m.tower
	c := t.Cfg
	var p towerProj
	for _, e := range m.container.Entries() {
		if e.Role.Vision() && e.Block == jlm.DenseBlock && !slices.Contains(towerProjRead[:], e.Role) {
			return p, UncoveredUnbound, fmt.Errorf("the container holds %s, which the projector's graphs do not read", e.Role)
		}
	}
	switch {
	case t.projW.e == nil:
		return p, UncoveredUnbound, fmt.Errorf("the tower holds no projector matrix")
	case t.projW.kp != 0 || t.projW2.kp != 0:
		return p, UncoveredGeometry, fmt.Errorf("a projector matrix packed past its width")
	case t.projReadsPastLastBlock():
		return p, UncoveredUnbound, fmt.Errorf("the projector reads rows past its last block's")
	}
	E, S := c.NEmbd, max(c.Scale, 1)
	p.pool = S > 1 && c.projK() == E
	if len(t.projNorm) != 0 && !p.pool && !t.projNormOn() {
		return p, UncoveredUnbound, fmt.Errorf("a projector norm before the patches merge")
	}
	inv := map[jlm.Role]graph.Shape{jlm.RoleVProj: {K: t.projW.k, Rows: t.projW.rows}}
	if t.projW2.e != nil {
		inv[jlm.RoleVProj2] = graph.Shape{K: t.projW2.k, Rows: t.projW2.rows}
	}
	for _, v := range []towerVec{{jlm.RoleVProjBias, t.projB}, {jlm.RoleVProjNorm, t.projNorm},
		{jlm.RoleVProjNormBias, t.projNormB}, {jlm.RoleVProj2Bias, t.projB2}} {
		if len(v.V) != 0 {
			inv[v.Role] = graph.Shape{K: len(v.V)}
		}
	}
	f := graph.Facts{Inventory: inv}
	has := func(r jlm.Role) bool {
		_, ok := inv[r]
		return ok
	}
	stored := func(r jlm.Role) []jlm.Role {
		if has(r) {
			return []jlm.Role{r}
		}
		return nil
	}
	if S > 1 {
		b := graph.NewBlock(-1)
		x := b.In("patches", E)
		b.Out(b.Emit(graph.OpTakeRows, []graph.ValueID{x}, []graph.Result{{Name: "grouped", Width: E}}, nil, graph.Attrs{}, nil, nil)[0])
		if err := graph.Verify(b.G, f); err != nil {
			return p, UncoveredUnverified, fmt.Errorf("the projector's regroup does not verify: %w", err)
		}
		p.group = b.G
	}
	build := func(unwindow bool) (*graph.Graph, Uncovered, error) {
		b := graph.NewBlock(-1)
		x := b.In("merged", E*S*S)
		switch {
		case p.pool:
			acc := sliceRow(b, "slot", x, 0, E)
			if !corner {
				for s := 1; s < S*S; s++ {
					acc = b.Add(acc, sliceRow(b, "slot", x, s*E, E))
				}
				acc = mapRow(b, "pooled", graph.Attrs{Fn: graph.MapMul, Const: 1 / float64(S*S)}, nil, E, acc)
			}
			if has(jlm.RoleVProjNorm) {
				acc = b.Norm(acc, jlm.RoleVProjNorm, c.projEps())
			}
			x = acc
		case has(jlm.RoleVProjNorm) && t.projNormOn():
			x = b.LayerNorm(x, jlm.RoleVProjNorm, c.projEps(), stored(jlm.RoleVProjNormBias)...)
		}
		h := b.MatVec(x, jlm.RoleVProj, t.projW.rows, stored(jlm.RoleVProjBias)...)
		if t.projW2.e != nil {
			var e jlm.ExprBuilder
			if t.projAct() == nn.ActGELUErf {
				e.GELUErf(jlm.Operand(0))
			} else {
				e.GELUTanh(jlm.Operand(0))
			}
			act := jlm.Expr{Terms: e.T}
			a, err := b.Stated("proj.act", &act, []graph.ValueID{h}, t.projW.rows, f)
			if err != nil {
				return nil, UncoveredAct, fmt.Errorf("the projector's activation: %w", err)
			}
			h = b.MatVec(a, jlm.RoleVProj2, t.projW2.rows, stored(jlm.RoleVProj2Bias)...)
		}
		if unwindow {
			h = b.Emit(graph.OpTakeRows, []graph.ValueID{h}, []graph.Result{{Name: "unwindowed", Width: c.ProjDim}}, nil, graph.Attrs{}, nil, nil)[0]
		}
		b.Out(h)
		if err := graph.Verify(b.G, f); err != nil {
			return nil, UncoveredUnverified, fmt.Errorf("the projector does not verify: %w", err)
		}
		return b.G, Covered, nil
	}
	var why Uncovered
	var err error
	if p.proj, why, err = build(false); why != Covered {
		return p, why, err
	}
	if c.WinPattern > 0 {
		if p.win, why, err = build(true); why != Covered {
			return p, why, err
		}
	}
	return p, Covered, nil
}

// sliceRow is width elements of each row of v from off.
func sliceRow(b *graph.Builder, name string, v graph.ValueID, off, width int) graph.ValueID {
	return b.Emit(graph.OpSlice, []graph.ValueID{v}, []graph.Result{{Name: name, Width: width}}, nil,
		graph.Attrs{Offset: off, Width: width}, nil, nil)[0]
}

// towerEntryOf is the rows entering the tower's first block as a graph of
// generic ops over the picture (its pixels a row of each patch's width,
// [y][x][channel]): each row's patch gathered at the position the step
// names for it (OpPatchify), the patch embedding and its bias, the class
// row at row 0 where the tower holds one, each row's position-table row
// read at its position, the norm before block 0; or why there is none. A
// video's second frame (a temporal pair's second plane) is not a picture's:
// its caller declines it.
func (m *Model) towerEntryOf() (*graph.Graph, Uncovered, error) {
	t := m.tower
	c := t.Cfg
	switch {
	case c.CLSLast || t.opt.towerCLSLast:
		return nil, UncoveredEmbedRows, fmt.Errorf("a class row after the patches")
	case t.posW != nil && (c.PosSide > 0 || c.PosAxis > 0 || t.posPerGrid()):
		return nil, UncoveredEmbedRows, fmt.Errorf("a position table read past each row's own row")
	case t.embdNorm != nil || t.mn != nil:
		return nil, UncoveredEmbedRows, fmt.Errorf("a patch embedding past one matrix over the patch")
	}
	E, P := c.NEmbd, c.PatchSz
	// The patch's row is the weight's packed width, zero past the patch
	// (tensor.kp).
	kw := t.patchW.k
	if t.patchW.kp != 0 {
		kw = t.patchW.kp
	}
	inv := map[jlm.Role]graph.Shape{jlm.RoleVPatchEmbd: {K: kw, Rows: t.patchW.rows}}
	for _, v := range []towerVec{{jlm.RoleVPatchBias, t.patchB}, {jlm.RoleVClassEmbd, t.clsW},
		{jlm.RoleVPreNorm, t.preLnW}, {jlm.RoleVPreNormBias, t.preLnB}} {
		if len(v.V) != 0 {
			inv[v.Role] = graph.Shape{K: len(v.V)}
		}
	}
	if t.posW != nil {
		inv[jlm.RoleVPosEmbd] = graph.Shape{K: E, Rows: len(t.posW) / E}
	}
	f := graph.Facts{Inventory: inv}
	has := func(r jlm.Role) bool {
		_, ok := inv[r]
		return ok
	}
	stored := func(r jlm.Role) []jlm.Role {
		if has(r) {
			return []jlm.Role{r}
		}
		return nil
	}
	lead := 0
	if c.CLS {
		lead = 1
	}
	b := graph.NewBlock(-1)
	px := b.In("pixels", 3*P*P)
	x := b.Emit(graph.OpPatchify, []graph.ValueID{px}, []graph.Result{{Name: "patches", Width: kw}}, nil,
		graph.Attrs{Span: P, Offset: lead}, nil, nil)[0]
	x = b.MatVec(x, jlm.RoleVPatchEmbd, E, stored(jlm.RoleVPatchBias)...)
	if c.CLS {
		x = b.Emit(graph.OpLeadRow, []graph.ValueID{x}, []graph.Result{{Name: "class", Width: E}},
			[]graph.Binding{{Role: jlm.RoleVClassEmbd, K: E}}, graph.Attrs{Offset: 1}, nil, nil)[0]
	}
	if t.posW != nil {
		pos := b.Emit(graph.OpEmbed, nil, []graph.Result{{Name: "pos", Width: E}},
			[]graph.Binding{{Role: jlm.RoleVPosEmbd, K: E, Rows: len(t.posW) / E}}, graph.Attrs{Positional: true}, nil, nil)[0]
		x = b.Add(x, pos)
	}
	if has(jlm.RoleVPreNorm) {
		if c.RMS {
			x = b.Norm(x, jlm.RoleVPreNorm, c.Eps)
		} else {
			x = b.LayerNorm(x, jlm.RoleVPreNorm, c.Eps, stored(jlm.RoleVPreNormBias)...)
		}
	}
	b.Out(x)
	if err := graph.Verify(b.G, f); err != nil {
		return nil, UncoveredUnverified, fmt.Errorf("the tower's entry does not verify: %w", err)
	}
	return b.G, Covered, nil
}
