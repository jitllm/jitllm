package tier

import (
	"github.com/jitllm/jitllm/engine/nn"
)

// Gemma 4 on the device. Its global layers attend at another geometry than its
// sliding ones (nn.LayerPlan.GeomSplit): head_dim 512 against 256, and on the
// 31B and 26B four kv heads against sixteen. Every attention kernel a scratch
// builds bakes one geometry, so this tier keeps one scratch SET per geometry
// -- the decode scratch, the batched ones by width and the widths reserved
// for them -- and a block runs in its own geometry's set. GPU.runsInto cuts a
// range where the geometry changes, as it cuts where the device changes, so
// every submission is one geometry and the residual crosses the host between
// them; the KV pages are per layer already (kvGeomOf reads the block's plan).
//
// The E2B/E4B add a third axis: their KV-sharing layers' FFN is twice as wide
// (nn.LayerPlan.FFNWide), and the FFN kernels bake their width too, so a set
// is keyed by both (geoKey).
//
// The set in use is swapped in place (useGeom): every path below reads g.bs and
// g.bbs as it always did. The home set is the global, single-width layers',
// which holds the head, and every entry point that switches restores it before
// it returns.

// geoKey names a scratch set: the zero key is the home set.
type geoKey uint8

const (
	geoSliding   geoKey = 1 << iota // the sliding layers' attention geometry
	geoWideFFN                      // a feed-forward twice NFFN wide
	geoNonCausal                    // a non-causal segment's blocks: a vision tower's
	// geoMSA is MiniMax-M3's selecting blocks (msa.go): their cache row
	// carries the indexer's key as one more head, which every K/V writer and
	// the paged pool's geometry bake, where the dense lead's row does not.
	geoMSA
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
			return geoNonCausal | geoNCHead
		}
		return geoNonCausal
	}
	var k geoKey
	if p.GeomSplit && p.Sliding {
		k |= geoSliding
	}
	if p.FFNWide {
		k |= geoWideFFN
	}
	if p.MSA() || p.QSA() {
		k |= geoMSA
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
		return 0
	}
	if l := g.layers[lo]; l != nil {
		return l.geo
	}
	return 0
}

// scratchPlanOf is the plan a block's scratch set is built from. On a split
// model each set holds one geometry and one rotary: the global set the global
// table, the sliding set the local one in the global slot (the caller hands it
// that table as cs), so neither carries a second table. The per-block fields
// (v from k, the output scalar) are the block's, not the scratch's.
func scratchPlanOf(p *nn.LayerPlan) *nn.LayerPlan {
	if !p.GeomSplit {
		return p
	}
	q := *p
	q.SWAPeriod, q.RopeTabSWA = 0, nil
	if p.Sliding {
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

// gemma4WhyNot names a Gemma 4 shape this tier declines.
func gemma4WhyNot(p *nn.LayerPlan) string {
	switch {
	case p.GeomSplit && (p.MLA() || p.Recurrent.Conv > 0 || p.QKNormWide || p.AttnOutGate):
		return "a sliding-layer geometry with latent attention, a recurrence, a whole-projection " +
			"q/k norm or an attention out-gate"
	case (p.VNorm || p.VFromK) && !p.QKNorm:
		// v's norm runs on k's head-norm kernel.
		return "a weightless v norm with no q/k norm to share a kernel with"
	case p.VFromK && p.ClampKQV != 0:
		return "v projected by k's weights under a q/k/v clamp"
	case (p.PLEDim != 0 || p.KVShared || p.FFNWide) && !p.GeomSplit && p.AltUp == 0:
		// The per-layer embedding's add goes back through the split model's
		// copy kernel, and the scratch sets are the split model's. Gemma 3n's
		// per-layer input goes through AltUp's own kernels (altup.go) and its
		// KV sharing through each block's kvSrc, on one geometry.
		return "per-layer embeddings, KV sharing or a wide FFN on a model with one attention geometry"
	case p.DenseMoE && (p.NFFNShExp == 0 || p.ExpertWeightIn || p.MoEBias || p.ExpertSigmoid):
		return "Gemma 4's dense-and-mixture block without its dense MLP, or with another router's features"
	}
	return ""
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
		d.setInputs(s.sid, s.ple, s.ids, s.ng, s.marks)
	}
	return ds
}

// pleRows is the staged per-layer inputs, n floats: the caller's rows, then
// zeros for a grid's padding rows. Callers hold g.mu.
func (g *devTier) pleRows(n int) []float32 {
	if cap(g.pleStage) < n {
		g.pleStage = make([]float32, n)
	}
	st := g.pleStage[:n]
	m := copy(st, g.pleHost)
	clear(st[m:])
	return st
}
