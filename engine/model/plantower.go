package model

import (
	"errors"
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/engine/run"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
)

// towerComposing is a vision tower as the composer reads it, lowered from
// what its container states and stores and never from its family's name:
// the vision section's numbers and flags, the conventions its converter
// stated for what the tensors do not show (jlm.VisionConv), and the tensors
// themselves -- which exist, their shapes. A part no op states is refused by
// what it is, and a stored tensor no graph binds refuses the tower.
type towerComposing struct {
	*jlm.File
	m      *Model
	cfg    jlm.Config
	entry  *graph.Graph
	head   *graph.Graph
	blocks []*graph.Graph
	seams  []*graph.Graph
	rope   *jlm.VisionRope
	hd     int
	// roles is each block's container role by the role its graph binds.
	roles []map[jlm.Role]jlm.Role
}

// towerView is the model's tower as the plan's vision segment reads it,
// lowered once, or the structural reason a part of it is not stated as ops.
// It is the one model's segment (composing.Segment), never a plan of its own.
func (m *Model) towerView() (*towerComposing, error) {
	m.towerOnce.Do(func() { m.towerSeg, m.towerErr = m.lowerTower() })
	return m.towerSeg, m.towerErr
}

func (m *Model) lowerTower() (*towerComposing, error) {
	v := m.container.Vision()
	if v == nil || m.tower == nil {
		return nil, errors.New("model: the container carries no tower")
	}
	if !v.Conv.Stated() {
		return nil, errors.New("model: the container states no conventions for its tower; reconvert or upgrade it")
	}
	tc := towerComposing{File: m.container, m: m}
	if err := tc.lower(v); err != nil {
		return nil, err
	}
	if err := tc.everyTensorBound(); err != nil {
		return nil, err
	}
	cfg := *m.container.Config()
	cfg.NEmbd, cfg.NLayer, cfg.NVocab, cfg.NRot = v.NEmbd, uint32(len(tc.blocks)), 0, 0
	tc.cfg = cfg
	return &tc, nil
}

// dense is the tower-wide tensor of role r, nil where the container has
// none.
func (c *towerComposing) dense(r jlm.Role) *jlm.Entry {
	e, ok := c.File.Find(r, jlm.DenseBlock, -1)
	if !ok {
		return nil
	}
	return e
}

// lower states the tower's graphs.
func (c *towerComposing) lower(v *jlm.Vision) error {
	conv := v.Conv
	E, H, L := int(v.NEmbd), int(v.NHead), int(v.NLayer)
	if E == 0 || H == 0 || L == 0 || E%H != 0 {
		return fmt.Errorf("model: a tower of %d blocks, %d wide, %d heads", L, E, H)
	}
	c.hd = E / H
	P, S, eps, proj := int(v.PatchSize), max(int(v.Scale), 1), float64(v.Eps), int(v.ProjDim)
	rms := conv.Norm == jlm.VisionRMSNorm
	var act graph.Act
	switch {
	case v.Flags.Has(jlm.FlagGELU) && v.Flags.Has(jlm.FlagQuickGELU):
		return errors.New("model: a tower stating GELU and quick-GELU at once")
	case v.Flags.Has(jlm.FlagGELU):
		act = graph.ActGELUTanh
	case v.Flags.Has(jlm.FlagQuickGELU):
		act = graph.ActQuickGELU
	default:
		act = graph.ActSiLU
	}
	if v.Flags.Has(jlm.FlagVisionRope) {
		r := conv.Rope
		if r == nil {
			return errors.New("model: the tower turns its heads and states no rotary")
		}
		pairs := 0
		for _, u := range r.Runs {
			pairs += int(u.Pairs)
		}
		if 2*pairs != c.hd || r.Offset != 0 {
			return fmt.Errorf("model: a tower rotary of %d pairs offset %d over heads of %d, which no op states yet", pairs, r.Offset, c.hd)
		}
		c.rope = r
	}
	for li := range L {
		g, roles, err := c.block(li, E, H, int(v.NFFN), eps, act, rms)
		if err != nil {
			return fmt.Errorf("model: tower block %d: %w", li, err)
		}
		if p := int(v.WinPattern); p > 0 && (li+1)%p != 0 {
			for i := range g.Ops {
				if g.Ops[i].Kind == graph.OpAttend {
					g.Ops[i].Attr.RowWindows = true
				}
			}
		}
		c.blocks = append(c.blocks, g)
		c.roles = append(c.roles, roles)
	}
	// The entry: the patches, the class row, the position table read at the
	// row's position or resampled to the grid, the pre-norm.
	patch := c.dense(jlm.RoleVPatchEmbd)
	if patch == nil || patch.NDim != 2 || int(patch.Dims[1]) != E || int(patch.Dims[0]) < 3*P*P {
		return fmt.Errorf("model: no flat patch embedding of patches of %d to %d", P, E)
	}
	class := c.dense(jlm.RoleVClassEmbd) != nil
	if class && conv.Class != jlm.ClassFirst {
		return errors.New("model: a class row after the patches, which no op states yet")
	}
	en := graph.TowerEntry{NEmbd: E, Patch: P, K: int(patch.Dims[0]), PatchBias: c.dense(jlm.RoleVPatchBias) != nil,
		Class: class, PreNorm: c.dense(jlm.RoleVPreNorm) != nil, Eps: eps}
	if en.PreNorm && (rms || c.dense(jlm.RoleVPreNormBias) == nil) {
		return errors.New("model: a pre-norm that is not a LayerNorm with its bias")
	}
	if pos := c.dense(jlm.RoleVPosEmbd); pos != nil {
		en.Positions = int(pos.Dims[1])
		fixed := (int(v.ImageSize) / max(P, 1)) * (int(v.ImageSize) / max(P, 1))
		if class {
			fixed++
		}
		switch {
		case conv.PosChain != 0:
			side := int(math.Sqrt(float64(en.Positions)))
			if side*side != en.Positions {
				return fmt.Errorf("model: a resampled position table of %d rows", en.Positions)
			}
			en.Resample = graph.Attrs{Span: side, Chain: uint8(conv.PosChain), Filter: uint8(conv.PosFilter)}
		case v.MaxPixels != 0 || en.Positions != fixed:
			return fmt.Errorf("model: a position table of %d rows read at a grid it does not state", en.Positions)
		}
	}
	c.entry = en.Build()
	// The deepstack taps: a merger over the residual after block b, its
	// tensors the tower's at index b.
	c.seams = make([]*graph.Graph, L+1)
	taps := 0
	for b := range L {
		fc1, ok := c.File.Find(jlm.RoleVDsFC1, jlm.DenseBlock, int32(b))
		if !ok {
			continue
		}
		fc2, ok := c.File.Find(jlm.RoleVDsFC2, jlm.DenseBlock, int32(b))
		if S < 2 || !ok || int(fc1.Dims[0]) != E*S*S || int(fc2.Dims[1]) != proj {
			return fmt.Errorf("model: a deepstack tap after block %d not of the grouped rows to %d", b, proj)
		}
		tap := graph.TowerTap{Block: b, NEmbd: E, Scale: S, Hidden: int(fc1.Dims[1]), ProjDim: proj, Eps: eps}
		c.seams[b+1] = tap.Build()
		taps++
	}
	// The projector: the post-norm, the class row dropped, the grid grouped
	// Scale x Scale (spatially, or as consecutive rows where the rows are in
	// window order, then put back in raster order), one or two matrices.
	win := v.WinPattern > 0
	h := graph.TowerHead{NEmbd: E, Scale: S, ProjDim: proj, PostNorm: c.dense(jlm.RoleVPostNorm) != nil, RMS: rms,
		Class: class, ProjBias: c.dense(jlm.RoleVProjBias) != nil, Taps: taps, Eps: eps, Fold: win, Take: win}
	if h.PostNorm && !rms && c.dense(jlm.RoleVPostNormBias) == nil {
		return errors.New("model: a LayerNorm after the blocks with no bias")
	}
	p1, p2 := c.dense(jlm.RoleVProj), c.dense(jlm.RoleVProj2)
	if p1 == nil {
		return errors.New("model: no projector matrix")
	}
	if int(p1.Dims[0]) != E*S*S {
		return fmt.Errorf("model: a projector reading %d of the grouped %d, a pooled grid no op states yet", p1.Dims[0], E*S*S)
	}
	if p2 != nil {
		if h.ProjBias != (c.dense(jlm.RoleVProj2Bias) != nil) {
			return errors.New("model: a two-matrix projector with one bias")
		}
		h.Hidden = int(p1.Dims[1])
		switch conv.ProjAct {
		case jlm.VisionGELUTanh:
			h.Act = graph.ActGELUTanh
		case jlm.VisionGELUErf:
			h.Act = graph.ActGELUErf
		default:
			return errors.New("model: a two-matrix projector with no stated activation")
		}
	}
	if int(p1.Dims[1]) != proj && p2 == nil || p2 != nil && int(p2.Dims[1]) != proj {
		return fmt.Errorf("model: a projector that does not reach the text model's %d", proj)
	}
	c.head = h.Build()
	return nil
}

// block is tower block li as a dense non-causal block over what the
// container stores for it, and its graph roles' container roles: the MLP's
// matrices told apart by shape (SmolVLM's names its expansion fc2).
func (c *towerComposing) block(li, E, H, nffn int, eps float64, act graph.Act, rms bool) (*graph.Graph, map[jlm.Role]jlm.Role, error) {
	bi := int32(li)
	has := func(r jlm.Role) bool { return c.File.Has(r, bi, -1) }
	get := func(r jlm.Role) *jlm.Entry {
		e, ok := c.File.Find(r, bi, -1)
		if !ok {
			return nil
		}
		return e
	}
	q, k, vv, o := get(jlm.RoleVAttnQ), get(jlm.RoleVAttnK), get(jlm.RoleVAttnV), get(jlm.RoleVAttnOut)
	f1, f2 := get(jlm.RoleVFC1), get(jlm.RoleVFC2)
	if q == nil || k == nil || vv == nil || o == nil || f1 == nil || f2 == nil {
		return nil, nil, errors.New("no attention and MLP matrices, a block no op states yet")
	}
	for _, m := range []*jlm.Entry{q, k, vv, o} {
		if int(m.Dims[0]) != E {
			return nil, nil, errors.New("an attention matrix stored wider than the residual")
		}
	}
	up, down := f1, f2
	if int(f2.Dims[1]) == nffn && int(f1.Dims[0]) == nffn {
		up, down = f2, f1
	}
	if int(up.Dims[1]) != nffn || int(up.Dims[0]) != E || int(down.Dims[0]) != nffn || int(down.Dims[1]) != E {
		return nil, nil, errors.New("an MLP that does not expand to its width and back")
	}
	biasOf := map[jlm.Role]jlm.Role{jlm.RoleVFC1: jlm.RoleVFC1Bias, jlm.RoleVFC2: jlm.RoleVFC2Bias}
	normB, ffnB := has(jlm.RoleVAttnNormBias), has(jlm.RoleVFFNNormBias)
	if rms && (normB || ffnB) || !rms && (!normB || !ffnB) {
		return nil, nil, errors.New("norms whose biases are not their stated kind's")
	}
	hd := E / H
	d := graph.Dense{Block: li, NEmbd: E, NHead: H, NKVHead: H, HeadDim: hd, NFFN: nffn, Eps: eps,
		Scale: 1 / math.Sqrt(float64(hd)), Act: act, LayerNorm: !rms, NonCausal: true, NoRope: c.rope == nil,
		NormBias: normB, FFNNormBias: ffnB, QKVBias: has(jlm.RoleVAttnQBias), OutBias: has(jlm.RoleVAttnOutBias),
		UpBias: has(biasOf[up.Role]), DownBias: has(biasOf[down.Role]),
		Ungated: !has(jlm.RoleVFFNGate), GateBias: has(jlm.RoleVFFNGateBias)}
	if r := c.rope; r != nil {
		d.NRot, d.Layout = hd, graph.RopeNorm
		if r.Neox {
			d.Layout = graph.RopeNeox
		}
	}
	roles := map[jlm.Role]jlm.Role{
		jlm.RoleAttnQ: jlm.RoleVAttnQ, jlm.RoleAttnK: jlm.RoleVAttnK, jlm.RoleAttnV: jlm.RoleVAttnV, jlm.RoleAttnOut: jlm.RoleVAttnOut,
		jlm.RoleAttnQBias: jlm.RoleVAttnQBias, jlm.RoleAttnKBias: jlm.RoleVAttnKBias, jlm.RoleAttnVBias: jlm.RoleVAttnVBias,
		jlm.RoleAttnOutBias: jlm.RoleVAttnOutBias, jlm.RoleAttnNorm: jlm.RoleVAttnNorm, jlm.RoleAttnNormBias: jlm.RoleVAttnNormBias,
		jlm.RoleFFNNorm: jlm.RoleVFFNNorm, jlm.RoleFFNNormBias: jlm.RoleVFFNNormBias,
		jlm.RoleFFNUp: up.Role, jlm.RoleFFNDown: down.Role, jlm.RoleFFNUpBias: biasOf[up.Role], jlm.RoleFFNDownBias: biasOf[down.Role],
		jlm.RoleFFNGate: jlm.RoleVFFNGate, jlm.RoleVFFNGateBias: jlm.RoleVFFNGateBias,
	}
	return d.Build(), roles, nil
}

// everyTensorBound refuses a tower that stores a tensor no graph binds: a
// part of the tower the lowering does not state is a refusal, never a
// tensor quietly left out.
func (c towerComposing) everyTensorBound() error {
	bound := map[[3]int32]bool{}
	add := func(g *graph.Graph, block, index int32) {
		if g == nil {
			return
		}
		for _, o := range g.Ops {
			for _, w := range o.Weight {
				if e, ok := c.Find(w.Role, block, index); ok {
					bound[[3]int32{int32(e.Role), e.Block, e.Index}] = true
				}
			}
		}
	}
	add(c.entry, jlm.DenseBlock, -1)
	add(c.head, jlm.DenseBlock, -1)
	for li, g := range c.blocks {
		add(g, int32(li), -1)
	}
	for _, g := range c.seams {
		if g != nil {
			add(g, jlm.DenseBlock, int32(g.Block))
		}
	}
	for _, e := range c.Entries() {
		// The audio tower is another segment; a video's second patch plane
		// is read only for a video.
		if !e.Role.Vision() || jlm.AudioRole(e.Role) || bound[[3]int32{int32(e.Role), e.Block, e.Index}] ||
			e.Role == jlm.RoleVPatchEmbd && e.Index > 0 {
			continue
		}
		return fmt.Errorf("model: the tower stores %v (block %d, index %d), which no op of its graphs binds", e.Role, e.Block, e.Index)
	}
	return nil
}

func (c towerComposing) Config() *jlm.Config { return &c.cfg }
func (c towerComposing) Tower() bool         { return true }
func (c towerComposing) Segment() jit.Model  { return nil }
func (c towerComposing) Blocks() int         { return len(c.blocks) }
func (c towerComposing) ActWindow() int      { return c.m.ActWindow() }

func (c towerComposing) EmbedGraph() *graph.Graph { return c.entry }
func (c towerComposing) HeadGraph() *graph.Graph  { return c.head }

func (c towerComposing) BlockGraph(li int) *graph.Graph {
	if li < 0 || li >= len(c.blocks) {
		return nil
	}
	return c.blocks[li]
}

func (c towerComposing) SeamGraph(li int) *graph.Graph {
	if li < 0 || li >= len(c.seams) {
		return nil
	}
	return c.seams[li]
}

// Rotaries is the stated rotary: each row's position a cell of the
// picture's grid.
func (c towerComposing) Rotaries() []graph.Rotary {
	if c.rope == nil {
		return nil
	}
	return []graph.Rotary{{NRot: c.hd, Base: float64(c.rope.Base), Grid: true}}
}

// Ropes is the engine's rotary for Rotaries' table, the stated runs.
func (c towerComposing) Ropes() []nn.Rope {
	r := c.rope
	if r == nil {
		return nil
	}
	rope := nn.Rope{NRot: int(r.FreqDim), Base: float64(r.Base), Neox: r.Neox}
	for _, u := range r.Runs {
		rope.Runs = append(rope.Runs, nn.RopeRun{Pairs: int(u.Pairs), Axis: int(u.Axis), Freq: int(u.Freq), Step: int(u.Step)})
	}
	return []nn.Rope{rope}
}

// Find is a tensor of the tower: a block's by the role its graph binds, as
// the block stores it; a tower-wide one as the container stores it.
func (c towerComposing) Find(role jlm.Role, block, index int32) (*jlm.Entry, bool) {
	if block == jlm.DenseBlock {
		return c.File.Find(role, block, index)
	}
	if block < 0 || int(block) >= len(c.roles) {
		return nil, false
	}
	r, ok := c.roles[block][role]
	if !ok {
		return nil, false
	}
	return c.File.Find(r, block, index)
}

// TowerPicture is picture px's step on the tower's plan, a grid gh x gw
// patches: each row's patch (window order where the tower attends inside
// windows, each merge unit's patches consecutive), each row's window, and
// the merged rows' raster order.
func (t *Tower) TowerPicture(px []float32, gh, gw int) *run.Picture {
	c := t.Cfg
	p := &run.Picture{Pixels: px, Grid: gw, Rows: gh * gw}
	if t.container.Has(jlm.RoleVClassEmbd, jlm.DenseBlock, -1) {
		p.Rows++
	}
	if c.WinPattern == 0 {
		return p
	}
	ord, segs := windowLayout(c, gh, gw)
	sc := c.Scale
	p.Pos = make([]int32, len(ord))
	p.Windows = make([]int32, 2*len(ord))
	for r, q := range ord {
		p.Pos[r] = int32(q)
	}
	for i := 0; i < len(segs); i += 2 {
		for r := segs[i]; r < segs[i+1]; r++ {
			p.Windows[2*r], p.Windows[2*r+1] = int32(segs[i]), int32(segs[i+1])
		}
	}
	mw, nt := gw/sc, len(ord)/(sc*sc)
	p.Take = make([]int32, nt)
	for i := 0; i < nt; i++ {
		q := ord[i*sc*sc]
		p.Take[(q/gw/sc)*mw+(q%gw)/sc] = int32(i)
	}
	return p
}
