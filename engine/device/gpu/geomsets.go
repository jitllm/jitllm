package gpu

import (
	"github.com/jitllm/jitllm/nn"
)

// Scratch sets by geometry. A model whose blocks attend at more than one
// geometry (nn.LayerPlan.MixedGeom) -- a head width, kv head count or rotary
// width of a block's own -- needs one scratch SET per geometry, since every
// attention kernel a scratch builds bakes one: the decode scratch, the batched
// ones by width and the widths reserved for them. A block runs in its own
// geometry's set.
// GPU.runsInto cuts a range where the geometry changes, as it cuts where the
// device changes, so every submission is one geometry and the residual
// crosses the host between them; the KV pages are per layer already
// (kvGeomOf reads the block's plan).
//
// A feed-forward width of a block's own (nn.LayerPlan.FFNWide) is a second
// axis, since the FFN kernels bake their width too, so a set is keyed by both
// (geoKey); a non-causal segment, a decision head and a cache row carrying an
// index key are the others.
//
// The set in use is swapped in place (useGeom): every path below reads g.bs and
// g.bbs as it always did. The home set is the blocks' at the model's own
// geometry, which holds the head, and every entry point that switches restores
// it before it returns.

// geoKey names a scratch set: the zero key is the home set. nkv, hd, nrot and
// local are a block's own attention geometry and rotary (nn.LayerPlan.OwnGeom),
// zero for a block at the model's; nffn is a feed-forward width of its own.
type geoKey struct {
	flags         geoFlag
	nkv, hd, nrot int32
	local         bool
	nffn          int32
}

// geoFlag is a set's kind beyond its shape.
type geoFlag uint8

const (
	geoNonCausal geoFlag = 1 << iota // a non-causal segment's blocks: a vision tower's
	// geoIndexKey is a selecting block's (msa.go, qsa.go): its cache row
	// carries the indexer's key as one more head, which every K/V writer and
	// the paged pool's geometry bake, where a dense block's row does not.
	geoIndexKey
	// geoNCHead is, beside geoNonCausal, a decision head's own non-causal
	// set (nn.LayerPlan.HeadSet): it resizes with the encoder's (visrows.go).
	geoNCHead
)

// geoOf is the set block plan p runs in. A non-causal segment (a vision
// tower) is its own set whatever else it is: its residual, heads and FFN are
// not the text model's, a call over it is one whole run, and the two never
// share a submission -- so where the hybrid rule takes a union of two block
// kinds over one residual, a segment takes a set of its own.
func geoOf(p *nn.LayerPlan) geoKey {
	if p.NonCausal {
		if p.HeadSet {
			return geoKey{flags: geoNonCausal | geoNCHead}
		}
		return geoKey{flags: geoNonCausal}
	}
	var k geoKey
	if p.MixedGeom && p.OwnGeom {
		k.nkv, k.hd, k.nrot, k.local = int32(p.NKVHead), int32(p.HeadDim), int32(p.NRot), p.LocalRope
	}
	if p.FFNWide {
		k.nffn = int32(p.NFFN)
	}
	if p.KeyHead() {
		k.flags |= geoIndexKey
	}
	return k
}

// geomSet is one geometry's scratch: what useGeom swaps.
type geomSet struct {
	bs      *blockScratch
	bbs     map[int]*blockScratch
	promptW int
	stepW   int
}

// useGeom makes set k the one g.bs and g.bbs name. Callers hold g.mu.
func (g *devTier) useGeom(k geoKey) {
	if k == g.geoCur {
		return
	}
	if g.geoSets == nil {
		g.geoSets = map[geoKey]geomSet{}
	}
	g.geoSets[g.geoCur] = geomSet{bs: g.bs, bbs: g.bbs, promptW: g.promptW, stepW: g.stepW}
	o := g.geoSets[k]
	delete(g.geoSets, k)
	g.bs, g.bbs, g.promptW, g.stepW = o.bs, o.bbs, o.promptW, o.stepW
	if g.bbs == nil {
		g.bbs = map[int]*blockScratch{}
	}
	g.geoCur = k
}

// geoKeys is the keys of the sets not in use. Callers hold g.mu.
func (g *devTier) geoKeys() []geoKey {
	ks := make([]geoKey, 0, len(g.geoSets))
	for k := range g.geoSets {
		ks = append(ks, k)
	}
	return ks
}

// graphBound is maxGraphs on every model but a split one, whose token is one
// submission per scratch-set run (and the head's own, after a run outside the
// home set), and so a recording key per run per row count a step takes: the
// E-models' four sets over a step's arms outgrew two per block and re-recorded
// every token. Eight per block bounds it; the bound only has to stop a long
// generation accumulating.
// Callers hold g.mu.
func (g *devTier) graphBound() int {
	if !g.geoUsed {
		return maxGraphs
	}
	return maxGraphs + 8*len(g.layers)
}

// restoreGeom puts back the set a caller had in use.
func (g *devTier) restoreGeom(k geoKey) {
	g.mu.Lock()
	g.useGeom(k)
	g.mu.Unlock()
}

// geomFor is the set blocks [lo, hi) run in: the range's first block's, the
// home set for an empty range (the head). Callers hold g.mu.
func (g *devTier) geomFor(lo, hi int) geoKey {
	if lo >= hi {
		return geoKey{}
	}
	if l := g.layers[lo]; l != nil {
		return l.geo
	}
	return geoKey{}
}

// enterGeomOf makes block li's scratch set the one in use and returns the set
// it left, for useGeom to restore: a check or a form of one block reads its
// own set's scratch -- a sliding block's table, a tower block's feeds --
// whichever set its caller is in. Callers hold g.mu.
func (g *devTier) enterGeomOf(li int) geoKey {
	prev := g.geoCur
	if g.geoUsed {
		g.useGeom(g.geomFor(li, li+1))
	}
	return prev
}

// scratchPlanOf is the plan a block's scratch set is built from. On a model of
// mixed geometry each set holds one geometry and one rotary: a set at the
// model's the first table, a set of blocks rotating by the second that one in
// the first slot (the caller hands it that table as cs), so none carries a
// second table. The per-block fields
// (v from k, the output scalar) are the block's, not the scratch's.
func scratchPlanOf(p *nn.LayerPlan) *nn.LayerPlan {
	if !p.MixedGeom {
		return p
	}
	q := *p
	q.SWAPeriod, q.RopeTabSWA = 0, nil
	if p.LocalRope {
		q.RopeTab = p.RopeTabSWA
	}
	q.VFromK, q.OutScale, q.KVShared, q.KVSource = false, 0, false, 0
	return &q
}

// dropOtherGeom frees every set not in use, as dropAllScratch frees the one in
// use. Callers hold g.mu.
func (g *devTier) dropOtherGeom() {
	home := g.geoCur
	for k := range g.geoSets {
		g.useGeom(k)
		for w := range g.bbs {
			g.dropBatch(w)
		}
		g.dropScratch(g.bs)
		g.bs = nil
	}
	g.useGeom(home)
	clear(g.geoSets)
}

// SetLayerInputs takes the rows' per-layer embedding inputs for the next
// call (nn.LayerInputDevice).
func (g *devTier) SetLayerInputs(in []float32) {
	g.mu.Lock()
	g.pleHost = in
	g.mu.Unlock()
}

// SetLayerInputs keeps the rows' per-layer inputs for this session's next
// call, which hands them to every device it holds (inputsTo): a session's
// inputs are its own, as its history is.
func (s *gpuSession) SetLayerInputs(in []float32) { s.ple = in }

// inputsTo hands the session's per-layer inputs to the devices enter took.
func (s *gpuSession) inputsTo(ds []*devTier) []*devTier {
	for _, d := range ds {
		s.inputsOn(d)
	}
	return ds
}

// inputsOn hands the session's inputs to d alone.
func (s *gpuSession) inputsOn(d *devTier) {
	d.setInputs(s.sid, s.ple, s.prompt)
}

// pleRows is the staged per-layer inputs, n floats: the caller's rows, then
// zeros for a grid's padding rows. Callers hold g.mu.
func (g *devTier) pleRows(n int) []float32 { return g.sideRows(g.pleHost, n) }

// sideRows is n floats of the caller's rows host for the blocks' side
// input, zeros past them, staged in pleStage.
func (g *devTier) sideRows(host []float32, n int) []float32 {
	if cap(g.pleStage) < n {
		g.pleStage = make([]float32, n)
	}
	st := g.pleStage[:n]
	m := copy(st, host)
	clear(st[m:])
	return st
}
