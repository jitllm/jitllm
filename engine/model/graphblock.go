package model

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// The block graph (docs/design/composer.md, "Above the recipes"): each
// covered block's graph is built from what the container states, verified
// against the container's tensors and configuration, and lowered to the
// operations the block body runs, on the host (decode, prompt chunks, ragged
// steps) and on the device (the plan). A text block the graph does not cover
// runs the hand-written body, and only for a reason on the named list below
// (uncoveredReasons): a block never falls back silently, and the list is the
// work list.

// WithBlockGraph chooses how blocks are lowered. The default runs every
// covered block through the graph and every other block through the
// hand-written body for its named reason. true does the same and refuses to
// open a model with any block the graph does not cover; false runs every
// block through the hand-written body, the control arm the graph's gates
// compare against.
func WithBlockGraph(on bool) Option {
	return func(l *loadOpts) {
		if on {
			l.opt.blockGraph = graphStrict
		} else {
			l.opt.blockGraph = graphOff
		}
	}
}

// graphMode is WithBlockGraph's setting.
type graphMode uint8

const (
	graphAuto graphMode = iota
	graphStrict
	graphOff
)

// Every reason a text block runs the hand-written body. A block's reason is
// one of these, by name; TestCoveredBlocksTakeTheGraphPath asserts it.
// Shrinking this list is the graph's work list.
const (
	UncoveredMixDense      = "a mixture: every expert on every row (a dense mixture)"
	UncoveredMixRouter     = "a mixture: a router the op language does not state (sqrt-softplus, hash routing or a router norm)"
	UncoveredMixExperts    = "a mixture: experts the op language does not state (ungated, unbanked or latent)"
	UncoveredMixWeightIn   = "a mixture: the router's weight scales the experts' input"
	UncoveredUngated       = "an ungated or non-SiLU FFN"
	UncoveredLayerNorm     = "LayerNorm with biases"
	UncoveredParallel      = "a parallel residual"
	UncoveredOutGate       = "an attention output gate"
	UncoveredMLA           = "MLA"
	UncoveredValueWidth    = "a value head wider or narrower than the key head"
	UncoveredKVShared      = "KV sharing or an extra cached head"
	UncoveredIndexer       = "an attention indexer or sparse key selection"
	UncoveredStreams       = "several residual streams"
	UncoveredPLE           = "per-layer embeddings"
	UncoveredResidScale    = "a residual scale"
	UncoveredNoPE          = "a block without the rotary"
	UncoveredRopeAxes      = "a multi-axis or interleaved rotary"
	UncoveredRopeScaled    = "a scaled rotary (YaRN, linear, an attention factor)"
	UncoveredQKNormKind    = "a q/k norm after the rotary or without a weight"
	UncoveredVNorm         = "a norm over v"
	UncoveredClamp         = "clamped q, k and v"
	UncoveredTemperature   = "an attention temperature"
	UncoveredWindow        = "an attention window or chunk"
	UncoveredNonCausal     = "non-causal attention"
	UncoveredSparseAct     = "a gaussian top-k activation"
	UncoveredDoubleFFN     = "a double-width FFN"
	UncoveredCompressed    = "compressed or grouped attention output"
	UncoveredRecurrent     = "a recurrent layer"
	UncoveredUnboundTensor = "a stored tensor no op of the graph binds"
)

// UncoveredReasons is the named list, in order.
var UncoveredReasons = []string{UncoveredMixDense, UncoveredMixRouter, UncoveredMixExperts, UncoveredMixWeightIn,
	UncoveredUngated, UncoveredLayerNorm, UncoveredParallel, UncoveredOutGate, UncoveredMLA, UncoveredValueWidth,
	UncoveredKVShared, UncoveredIndexer, UncoveredStreams, UncoveredPLE, UncoveredResidScale, UncoveredNoPE,
	UncoveredRopeAxes, UncoveredRopeScaled, UncoveredQKNormKind, UncoveredVNorm, UncoveredClamp, UncoveredTemperature,
	UncoveredWindow, UncoveredNonCausal, UncoveredSparseAct, UncoveredDoubleFFN, UncoveredCompressed,
	UncoveredRecurrent, UncoveredUnboundTensor}

// uncovered is why the block graph does not cover text block li, by name, or
// "" where it does. It is decided from the block's structure alone -- the
// configuration's fields and, after it, the tensors the container stores
// (UncoveredUnboundTensor, in buildGraphs) -- never from the architecture's
// name: a block the op language states runs through the graph whatever the
// file calls its family (TestCoverageIsStructural).
func (m *Model) uncovered(li int) string {
	c := m.Cfg
	w0, an := c.AttnWindow(li, 1<<20)
	if c.MoEAt(li) {
		if _, why := m.mixtureWhy(li); why != "" {
			return why
		}
	}
	switch {
	case c.LayerKind(li).Recurrent() || c.NGram != 0:
		return UncoveredRecurrent
	case c.Act != nn.ActSiLU:
		return UncoveredUngated
	case c.LayerNorm:
		return UncoveredLayerNorm
	case c.Parallel:
		return UncoveredParallel
	case c.AttnOutGate:
		return UncoveredOutGate
	case c.MLA() || c.QLoraRank != 0:
		return UncoveredMLA
	case c.HeadDimV != 0 && c.HeadDimV != c.HeadDimAt(li):
		return UncoveredValueWidth
	case c.KVShared(li) || c.extraKVHead(li):
		return UncoveredKVShared
	case c.Indexer() || c.MSA() || c.IdxHeads != 0 || c.QSARatio != 0:
		return UncoveredIndexer
	case c.streamHead():
		return UncoveredStreams
	case c.PLEDim != 0:
		return UncoveredPLE
	case c.ResidualScale != 1:
		return UncoveredResidScale
	case !c.RopeAt(li):
		return UncoveredNoPE
	case c.RopeInterleaved || c.RopeXD || len(m.rope.Runs) > 0:
		return UncoveredRopeAxes
	case c.YarnFactor > 0 || (c.RopeLinear != 0 && c.RopeLinear != 1) || (c.AttnFactor != 0 && c.AttnFactor != 1):
		return UncoveredRopeScaled
	case c.QKNormPost || c.QKL2Norm:
		return UncoveredQKNormKind
	case c.VNorm:
		return UncoveredVNorm
	case c.ClampKQV != 0:
		return UncoveredClamp
	case c.AttnTemp(li, 1) != 1:
		return UncoveredTemperature
	case w0 != 0 || an != 1<<20+1:
		return UncoveredWindow
	case c.NonCausal || c.BidirSWA:
		return UncoveredNonCausal
	case c.NSparse != 0:
		return UncoveredSparseAct
	case c.DoubleFFN:
		return UncoveredDoubleFFN
	case len(c.CompKinds) > 0 || c.CompRateCSA != 0 || c.CompRateHCA != 0 || c.OGroups != 0 || c.OLoraRank != 0:
		return UncoveredCompressed
	}
	return ""
}

// mixLowered is whether block li's FFN is a mixture the block graph lowers: a
// routed mixture the op language states (mixtureFacts). A dense block beside
// mixtures (a dense lead, ERNIE's interleave) is a dense FFN, not this.
func (m *Model) mixLowered(li int) bool {
	if !m.Cfg.MoEAt(li) {
		return false
	}
	_, why := m.mixtureWhy(li)
	return why == ""
}

// blockFacts is what the container states about text block li: the dense
// configuration and the tensors stored for the block by role.
func (m *Model) blockFacts(li int) (graph.Facts, error) {
	c := m.Cfg
	if why := m.uncovered(li); why != "" {
		return graph.Facts{}, fmt.Errorf("model: block %d of %s is not covered by the block graph: %s", li, c.Arch, why)
	}
	layout := graph.RopeNorm
	if c.RopeNeox {
		layout = graph.RopeNeox
	}
	d := graph.Dense{Block: li, NEmbd: c.NEmbd, NHead: c.NHead, NKVHead: c.NKVHeadAt(li), HeadDim: c.HeadDimAt(li),
		NFFN: c.NFFNAt(li), Eps: c.RMSEps, NRot: c.NRot, Base: c.RopeBase, Layout: layout, RopeFreqs: m.rope.Freqs != nil,
		Scale: c.AttnScale, Softcap: float64(c.AttnSoftcap), Act: graph.ActSiLU,
		QKNorm: c.QKNormAt(li) && !c.QKNormWide, QKNormWhole: c.QKNormAt(li) && c.QKNormWide}
	inv := map[jlm.Role]graph.Shape{}
	for i := range m.container.Entries() {
		e := &m.container.Entries()[i]
		if int(e.Block) != li {
			continue
		}
		s := graph.Shape{K: int(e.Dims[0])}
		if e.NDim >= 2 {
			s.Rows = int(e.Dims[1])
		}
		if e.NDim >= 3 {
			s.Sheets = int(e.Dims[2])
		}
		inv[e.Role] = s
	}
	_, d.QKVBias = inv[jlm.RoleAttnQBias]
	f := graph.Facts{Config: d, Inventory: inv}
	if c.MoEAt(li) {
		mf, ok := m.mixtureFacts(li)
		if !ok {
			return graph.Facts{}, fmt.Errorf("model: block %d: a mixture the op language does not state", li)
		}
		f.Config.Mixture, f.Mixture = mf.Mixture, mf.Mixture
	}
	return f, nil
}

// denseLowering is a verified graph lowered for the host decode: the dense
// block's op sequence matched against the operations the hand-written body
// runs. qk says the matched sequence holds the per-head q/k norms, which
// attnPrep applies with the rotary and the cache write.
type denseLowering struct {
	g  *graph.Graph
	qk bool
	// mix says the block's FFN is a routed mixture (graph.Dense.Mixture),
	// which the host runs through the mixture's operations (moe, moeBatch).
	mix bool
	// offered is every rewrite matched on the block (nn.LayerPlan.Rewrites),
	// which the device asks at its own Target, and rewrites every one's
	// target-free decision, for the plan explanation.
	offered  []graph.Offered
	rewrites []graph.Decision
}

// lowerDense matches g's ops, in order, against the dense block's host
// operations, and refuses anything else: an op the lowering does not know is
// an op with no host lowering, never one skipped.
func lowerDense(g *graph.Graph) (*denseLowering, error) {
	ops, i := g.Ops, 0
	fail := func(want string) error {
		if i < len(ops) {
			return fmt.Errorf("model: block %d graph op %d is %s; the host lowering wants %s", g.Block, i, ops[i].Kind, want)
		}
		return fmt.Errorf("model: block %d graph ends where the host lowering wants %s", g.Block, want)
	}
	take := func(k graph.OpKind, role jlm.Role) bool {
		if i < len(ops) && ops[i].Kind == k && (role == 0 || (len(ops[i].Weight) > 0 && ops[i].Weight[0].Role == role)) {
			i++
			return true
		}
		return false
	}
	seq := []struct {
		k    graph.OpKind
		role jlm.Role
	}{{graph.OpNorm, jlm.RoleAttnNorm}, {graph.OpMatVec, jlm.RoleAttnQ}, {graph.OpMatVec, jlm.RoleAttnK}, {graph.OpMatVec, jlm.RoleAttnV}}
	for _, s := range seq {
		if !take(s.k, s.role) {
			return nil, fail(fmt.Sprintf("%s of %v", s.k, s.role))
		}
	}
	lw := &denseLowering{g: g}
	// The q/k norms, per head (qwen3) or over each whole projection
	// (OLMoE): attnPrep applies either, by QKNormShapeAt.
	for _, k := range []graph.OpKind{graph.OpHeadNorm, graph.OpNorm} {
		if !lw.qk && take(k, jlm.RoleAttnQNorm) {
			if !take(k, jlm.RoleAttnKNorm) {
				return nil, fail(k.String() + " of attn_k_norm")
			}
			lw.qk = true
		}
	}
	type step struct {
		k    graph.OpKind
		role jlm.Role
	}
	run := func(seq []step) error {
		for _, s := range seq {
			if !take(s.k, s.role) {
				return fail(fmt.Sprintf("%s of %v", s.k, s.role))
			}
		}
		return nil
	}
	if err := run([]step{{graph.OpRope, 0}, {graph.OpRope, 0}, {graph.OpKVAppend, 0}, {graph.OpAttend, 0},
		{graph.OpMatVec, jlm.RoleAttnOut}, {graph.OpAdd, 0}, {graph.OpNorm, jlm.RoleFFNNorm}}); err != nil {
		return nil, err
	}
	if i < len(ops) && ops[i].Kind == graph.OpRoute {
		// The routed mixture, as moe() computes it: route, gather, the
		// experts' gate, up, act_mul and down, the combine, then the shared
		// expert [gate scalar] added, then the residual add.
		lw.mix = true
		if err := run([]step{{graph.OpRoute, jlm.RoleRouter}, {graph.OpGather, 0},
			{graph.OpExpertMatVec, jlm.RoleExpGateBank}, {graph.OpExpertMatVec, jlm.RoleExpUpBank},
			{graph.OpActMul, 0}, {graph.OpExpertMatVec, jlm.RoleExpDownBank}, {graph.OpCombine, 0}}); err != nil {
			return nil, err
		}
		if take(graph.OpMatVec, jlm.RoleShExpGate) {
			if err := run([]step{{graph.OpMatVec, jlm.RoleShExpUp}, {graph.OpActMul, 0},
				{graph.OpMatVec, jlm.RoleShExpDown}}); err != nil {
				return nil, err
			}
			take(graph.OpGateScalar, jlm.RoleShRouter)
			if err := run([]step{{graph.OpAdd, 0}}); err != nil {
				return nil, err
			}
		}
		if err := run([]step{{graph.OpAdd, 0}}); err != nil {
			return nil, err
		}
	} else if err := run([]step{{graph.OpMatVec, jlm.RoleFFNGate}, {graph.OpMatVec, jlm.RoleFFNUp}, {graph.OpActMul, 0},
		{graph.OpMatVec, jlm.RoleFFNDown}, {graph.OpAdd, 0}}); err != nil {
		return nil, err
	}
	if i != len(ops) {
		return nil, fail("the end of the block")
	}
	return lw, nil
}

// buildGraphs builds, verifies and lowers every covered text block's graph,
// and records every other block's named reason (m.graphWhy). A covered block
// whose graph fails to verify or lower refuses the model: that is a bug, never
// a fallback. Under WithBlockGraph(true) an uncovered block refuses it too.
func (m *Model) buildGraphs() error {
	m.graphs = make([]*denseLowering, m.Cfg.NLayer)
	m.graphWhy = make([]string, m.Cfg.NLayer)
	for li := range m.graphs {
		refuse := func(why string) error {
			if m.opt.blockGraph == graphStrict {
				return fmt.Errorf("model: block %d of %s is not covered by the block graph: %s", li, m.Cfg.Arch, why)
			}
			m.graphWhy[li] = why
			return nil
		}
		if why := m.uncovered(li); why != "" {
			if err := refuse(why); err != nil {
				return err
			}
			continue
		}
		f, err := m.blockFacts(li)
		if err != nil {
			return err
		}
		g := f.Config.Build()
		// The container holds a tensor the stated block does not read: the
		// block computes something the op language does not state.
		if u := graph.Unbound(g, f); len(u) > 0 {
			if err := refuse(UncoveredUnboundTensor); err != nil {
				return fmt.Errorf("%w (%v)", err, u)
			}
			continue
		}
		if err := graph.Verify(g, f); err != nil {
			return err
		}
		for i, o := range g.Ops {
			if _, ok := hostLowering[o.Kind]; !ok {
				return fmt.Errorf("model: block %d op %d (%s) has no host lowering", li, i, o.Kind)
			}
		}
		lw, err := lowerDense(g)
		if err != nil {
			return err
		}
		if lw.qk != m.Cfg.QKNormAt(li) {
			return fmt.Errorf("model: block %d graph's q/k norm is %v and attnPrep's %v", li, lw.qk, m.Cfg.QKNormAt(li))
		}
		lw.rewrites = graph.Apply(g, graph.Rewrites...)
		lw.offered = graph.Match(g, graph.Rewrites...)
		m.graphs[li] = lw
	}
	return nil
}

// GraphUncovered is why text block li runs the hand-written body: one of
// UncoveredReasons, "control" under WithBlockGraph(false), or "" where the
// graph covers it.
func (m *Model) GraphUncovered(li int) string {
	if m.opt.blockGraph == graphOff {
		return "control"
	}
	if li < 0 || li >= len(m.graphWhy) {
		return ""
	}
	return m.graphWhy[li]
}

// BlockRewrites is every rewrite's decision on text block li's graph, legal
// or refused with its reason (nil without WithBlockGraph).
func (m *Model) BlockRewrites(li int) []graph.Decision {
	if li < 0 || li >= len(m.graphs) || m.graphs[li] == nil {
		return nil
	}
	return m.graphs[li].rewrites
}

// BlockGraph is text block li's verified graph, or nil where the model was
// opened without WithBlockGraph.
func (m *Model) BlockGraph(li int) *graph.Graph {
	if li < 0 || li >= len(m.graphs) || m.graphs[li] == nil {
		return nil
	}
	return m.graphs[li].g
}

// GraphBlocks counts the blocks a State ran through the block graph's
// lowering: the forced path's evidence (RULE 10).
func (s *State) GraphBlocks() int { return s.graphRuns }

// GraphChunks counts the block-chunks a State ran through the block graph's
// prompt-row lowering (graphRows): a prompt's evidence, as GraphBlocks is a
// decode's.
func (s *State) GraphChunks() int { return s.graphChunkRuns }

// GraphRaggedBlocks counts the blocks a State ran through the block graph's
// ragged-step lowering (graphRagged): a batch step's evidence.
func (s *State) GraphRaggedBlocks() int { return s.graphRaggedRuns }

// BlockPath is which path ran text block li in a State, counted per run of
// the block (a decoded token, a prompt chunk, a ragged step): Graph through
// the block graph's host lowering, Hand through the hand-written body, and
// Device on a device, whose plan is lowered from the graph exactly where the
// model holds one (Model.BlockGraph). Why is the block's named reason for the
// hand-written body (Model.GraphUncovered), "" where the graph covers it.
type BlockPath struct {
	Graph, Hand, Device int
	DeviceGraphed       bool
	Why                 string
}

// BlockPaths is every text block's path record in the State.
func (s *State) BlockPaths() []BlockPath {
	out := make([]BlockPath, s.m.Cfg.NLayer)
	for li := range out {
		if li < len(s.paths) {
			out[li] = s.paths[li]
		}
		out[li].DeviceGraphed = out[li].Device > 0 && s.m.BlockGraph(li) != nil
		out[li].Why = s.m.GraphUncovered(li)
	}
	return out
}

// pathOf is text block li's path record; a block past the text blocks (the
// vision tower's) shares a scratch record nobody reads.
func (s *State) pathOf(li int) *BlockPath {
	if s.paths == nil {
		s.paths = make([]BlockPath, s.m.Cfg.NLayer+1)
	}
	if li < 0 || li >= s.m.Cfg.NLayer {
		return &s.paths[s.m.Cfg.NLayer]
	}
	return &s.paths[li]
}

// handRan records that text block li ran through the hand-written body.
func (s *State) handRan(li int) { s.pathOf(li).Hand++ }

// devRan records that blocks lo..hi-1 ran on a device.
func (s *State) devRan(lo, hi int) {
	for li := lo; li < hi; li++ {
		s.pathOf(li).Device++
	}
}

// graphBlock runs block li's host decode through its lowering: the calls the
// hand-written body makes for a dense block, in its order, so the kernels and
// their order are the control's.
func (s *State) graphBlock(li int, l *layer, lw *denseLowering) error {
	c, m := s.c, s.m
	s.graphRuns++
	s.pathOf(li).Graph++
	// norm(x) -> attn_in
	s.norm(s.h, s.x, l.attnNorm, l.attnNormB)
	s.jit.NewInput()
	m.trace(li, "attn_norm", s.h)
	// matvec q, k, v (one region inside a fused run, as the body groups them)
	kvd := c.KVDimAt(li)
	s.aheadOf(li, 0, [nn.MaxGroup]*tensor{&l.wo})
	grouped := s.groupOf(s.h, [nn.MaxGroup]*tensor{&l.wq, &l.wk, &l.wv}, [nn.MaxGroup][]float32{s.q, s.kc, s.vc}, 3)
	if grouped {
		s.addBiasRows(s.q, l.bq, 1)
	} else {
		if _, err := s.projectQ(l, s.q, s.ogate, s.h, 1, false); err != nil {
			return err
		}
		if err := s.mv(s.kc, l.wk, s.h); err != nil {
			return err
		}
	}
	s.addBias(s.kc, l.bk)
	if !grouped {
		if err := s.mv(s.vc, l.wv, s.h); err != nil {
			return err
		}
	}
	s.addBias(s.vc, l.bv)
	// [head_norm q, head_norm k,] rope q, rope k, kv_append: attnPrep
	s.attnPrep(l, s.q, s.kc[:kvd], s.vc[:kvd], s.ropeTable(li, s.cs, s.csSWA, 0), li, 0, s.pos)
	m.trace(li, "k", s.kc[:kvd])
	// attend
	s.attendRow(li, l)
	// matvec o, add into the residual (folded into o's rows where the run
	// fuses it)
	qd := c.QDimAt(li)
	src := s.xb
	plain := s.residFold && l.bo == nil
	if plain {
		s.grp.Resid, s.grp.Alpha = s.x, float32(c.ResidualScale)
	}
	s.aheadOf(li, 1, [nn.MaxGroup]*tensor{&l.gate, &l.up})
	attnResid := plain && s.groupOf(src[:qd], [nn.MaxGroup]*tensor{&l.wo}, [nn.MaxGroup][]float32{s.h}, 1)
	s.grp.Resid = nil
	if !attnResid && !s.groupOf(src[:qd], [nn.MaxGroup]*tensor{&l.wo}, [nn.MaxGroup][]float32{s.h}, 1) {
		if err := s.mv(s.h, l.wo, src[:qd]); err != nil {
			return err
		}
	}
	s.addBias(s.h, l.bo)
	if !attnResid {
		s.addInto(s.x, s.h)
	}
	m.trace(li, "attn_resid", s.x)
	// norm -> gate, up -> act_mul -> down, add into the residual
	s.norm(s.h, s.x, l.ffnNorm, l.ffnNormB)
	s.jit.NewInput()
	if lw.mix {
		// route -> gather -> experts -> combine [+ shared], added into the
		// residual by the mixture itself, as the hand-written body does.
		if err := s.moe(li, l, s.h, s.h, s.x); err != nil {
			return err
		}
		s.layerOutScale(l, s.x)
		m.trace(li, "ffn_resid", s.x)
		return nil
	}
	ff := c.NFFNAt(li)
	gate, up, act := s.gate[:ff], s.up[:ff], c.Act
	s.aheadOf(li, 2, [nn.MaxGroup]*tensor{&l.down})
	if !s.groupOf(s.h, [nn.MaxGroup]*tensor{&l.gate, &l.up}, [nn.MaxGroup][]float32{gate, up}, 2) {
		if err := s.mv(gate, l.gate, s.h); err != nil {
			return err
		}
		if err := s.mv(up, l.up, s.h); err != nil {
			return err
		}
	}
	s.gaussTopK(li, gate, ff)
	s.grp.ActUp, s.grp.Act = up, act
	if s.residFold {
		s.grp.Resid, s.grp.Alpha = s.x, float32(c.ResidualScale)
	}
	s.aheadNext(li)
	grouped = s.groupOf(gate, [nn.MaxGroup]*tensor{&l.down}, [nn.MaxGroup][]float32{s.h}, 1)
	ffnResid := grouped && s.grp.Resid != nil
	s.grp.ActUp, s.grp.Resid = nil, nil
	if !grouped {
		s.actmulAll(gate, up, act)
		s.jit.NewInput()
		if err := s.mv(s.h, l.down, gate); err != nil {
			return err
		}
	}
	if !ffnResid {
		s.addInto(s.x, s.h)
	}
	s.layerOutScale(l, s.x)
	m.trace(li, "ffn_resid", s.x)
	return nil
}

// graphRows runs block li over a host prompt chunk -- rows p0..p0+n-1 of slot,
// in s.bx -- through the graph's lowering: the dense subset of hostRows'
// calls, in its order, so the kernels and their order are the hand-written
// chunk's. The graph is the one the decode lowering matched (lowerDense):
// norm, q/k/v, [the per-head q/k norms,] the rotary and the cache write for
// every row before any row attends, attention, o, the residual add, then the
// FFN norm, gate and up, act_mul, down and the second add.
func (s *State) graphRows(li int, l *layer, lw *denseLowering, slot, p0, n int, bidir bool) error {
	c := s.c
	s.graphChunkRuns++
	s.pathOf(li).Graph++
	ne := c.NEmbd
	// norm(x) -> attn_in, every row
	tn := s.tick()
	s.batchNormB(s.bh, s.bx, l.attnNorm, l.attnNormB, ne, n)
	s.tock(opRMSNorm, tn)
	// matvec q, k, v over the chunk
	kvDim, qDim := c.KVDimAt(li), c.QDimAt(li)
	if _, err := s.projectQ(l, s.bq, s.ogate, s.bh[:n*ne], n, true); err != nil {
		return err
	}
	if err := s.mm(s.bk, l.wk, s.bh[:n*ne], n); err != nil {
		return err
	}
	s.addBiasRows(s.bk, l.bk, n)
	if err := s.mm(s.bv, l.wv, s.bh[:n*ne], n); err != nil {
		return err
	}
	s.addBiasRows(s.bv, l.bv, n)
	// [head_norm q, head_norm k,] rope q, rope k, kv_append: every row's,
	// before any row attends (a row attends to the chunk's earlier rows).
	for i := 0; i < n; i++ {
		s.attnPrep(l, s.bq[i*qDim:(i+1)*qDim], s.bk[i*kvDim:(i+1)*kvDim], s.bv[i*kvDim:(i+1)*kvDim],
			s.ropeTable(li, s.bcs, s.bcsSWA, i), li, slot, p0+i)
	}
	// attend
	s.attendRows(li, l, slot, p0, n, qDim, c.AttnScale, bidir, false)
	// matvec o, add into the residual
	if err := s.mm(s.bh, l.wo, s.bxb[:n*qDim], n); err != nil {
		return err
	}
	s.addBiasRows(s.bh, l.bo, n)
	tr := s.tick()
	s.addInto(s.bx[:n*ne], s.bh[:n*ne])
	s.tock(opResid, tr)
	// norm -> gate, up -> act_mul -> down, add into the residual
	tn = s.tick()
	s.batchNormB(s.bh, s.bx, l.ffnNorm, l.ffnNormB, ne, n)
	s.tock(opRMSNorm, tn)
	if lw.mix {
		// The mixture over the chunk, expert-major (moeBatch), into the
		// residual.
		if err := s.moeBatch(li, l, s.bh, s.bx, n); err != nil {
			return err
		}
		s.layerOutScale(l, s.bx[:n*ne])
		return nil
	}
	ff := c.NFFNAt(li)
	if err := s.mm(s.bgate, l.gate, s.bh[:n*ne], n); err != nil {
		return err
	}
	if err := s.mm(s.bup, l.up, s.bh[:n*ne], n); err != nil {
		return err
	}
	gate, up := s.bgate[:n*ff], s.bup[:n*ff]
	tac := s.tick()
	s.actmulAll(gate, up, c.Act)
	s.tock(opAct, tac)
	if err := s.mm(s.bh, l.down, gate, n); err != nil {
		return err
	}
	tr = s.tick()
	s.addInto(s.bx[:n*ne], s.bh[:n*ne])
	s.tock(opResid, tr)
	s.layerOutScale(l, s.bx[:n*ne])
	return nil
}

// graphRagged runs block li over a ragged step's rows -- each row its own
// sequence's slot and position, in s.bx -- through the graph's lowering: the
// dense subset of rowsHost's block body, in its order, so the kernels and
// their order are the hand-written step's. Every row's rotary and cache write
// lands before any row attends, as in graphRows.
func (s *State) graphRagged(li int, l *layer, lw *denseLowering, n int, seq, pos []int, cs, csSWA []float32, unitHeads, units, astride int, scale float64) error {
	c := s.c
	s.graphRaggedRuns++
	s.pathOf(li).Graph++
	ne := c.NEmbd
	// norm(x) -> attn_in, every row
	t0 := s.tick()
	s.batchNormB(s.bh, s.bx, l.attnNorm, l.attnNormB, ne, n)
	s.tock(opRMSNorm, t0)
	// matvec q, k, v over the step's rows
	kvDim, qDim := c.KVDimAt(li), c.QDimAt(li)
	if _, err := s.projectQ(l, s.bq, s.ogate, s.bh, n, true); err != nil {
		return err
	}
	if err := s.mm(s.bk, l.wk, s.bh, n); err != nil {
		return err
	}
	s.addBiasRows(s.bk, l.bk, n)
	if err := s.mm(s.bv, l.wv, s.bh, n); err != nil {
		return err
	}
	s.addBiasRows(s.bv, l.bv, n)
	// [head_norm q, head_norm k,] rope q, rope k, kv_append: each row at its
	// owner's slot and position
	for i := 0; i < n; i++ {
		o, slot := s.rowOwner(i, seq)
		o.attnPrep(l, s.bq[i*qDim:(i+1)*qDim], s.bk[i*kvDim:(i+1)*kvDim], s.bv[i*kvDim:(i+1)*kvDim],
			s.ropeTable(li, cs, csSWA, i), li, slot, pos[i])
	}
	// attend
	s.rowsAttend(li, l, n, seq, pos, unitHeads, units, astride, scale)
	// matvec o, add into the residual
	if err := s.mm(s.bh, l.wo, s.bxb[:n*qDim], n); err != nil {
		return err
	}
	s.addBiasRows(s.bh, l.bo, n)
	t0 = s.tick()
	s.addInto(s.bx[:n*ne], s.bh[:n*ne])
	s.tock(opResid, t0)
	// norm -> gate, up -> act_mul -> down, add into the residual
	t0 = s.tick()
	s.batchNormB(s.bh, s.bx, l.ffnNorm, l.ffnNormB, ne, n)
	s.tock(opRMSNorm, t0)
	if lw.mix {
		// Each row its own sequence's experts (moeBatch), into the residual.
		if err := s.moeBatch(li, l, s.bh, s.bx, n); err != nil {
			return err
		}
		s.layerOutScale(l, s.bx[:n*ne])
		return nil
	}
	if err := s.mm(s.bgate, l.gate, s.bh, n); err != nil {
		return err
	}
	if err := s.mm(s.bup, l.up, s.bh, n); err != nil {
		return err
	}
	t0 = s.tick()
	ff := c.NFFNAt(li)
	gate, up := s.bgate[:n*ff], s.bup[:n*ff]
	s.gaussTopK(li, gate, ff)
	s.actmulAll(gate, up, c.Act)
	s.jit.NewInput()
	s.tock(opAct, t0)
	if err := s.mm(s.bh, l.down, s.bgate, n); err != nil {
		return err
	}
	t0 = s.tick()
	s.addInto(s.bx[:n*ne], s.bh[:n*ne])
	s.tock(opResid, t0)
	s.layerOutScale(l, s.bx[:n*ne])
	return nil
}

// lowerPlan writes what the block's graph states into a device plan: the
// residual and head widths, the norm epsilon, the rotary (rotated width, base,
// pair layout), the attention scale and softcap, the per-head q/k norm, the
// activation and the FFN width -- each read from the op that carries it, so
// the device runs the graph's block rather than the configuration's. The
// plan's other fields (paging, placement, the model-wide head) are not
// block operations and keep planFor's values.
func (lw *denseLowering) lowerPlan(p *nn.LayerPlan) {
	g := lw.g
	p.NEmbd = g.Values[g.In[0]].Width
	p.QKNorm = false
	_, producer := graphProducers(g)
	for _, o := range g.Ops {
		switch o.Kind {
		case graph.OpNorm:
			p.RMSEps = o.Attr.Eps
			if len(o.Weight) == 1 && o.Weight[0].Role == jlm.RoleAttnQNorm {
				p.QKNorm = true
			}
		case graph.OpHeadNorm:
			p.QKNorm = true
		case graph.OpRope:
			p.NRot, p.RopeBase, p.RopeNeox = o.Attr.NRot, o.Attr.Base, o.Attr.Layout == graph.RopeNeox
		case graph.OpAttend:
			p.NHead, p.NKVHead, p.HeadDim = o.Attr.NHead, o.Attr.NKVHead, o.Attr.HeadDim
			p.AttnScale, p.AttnSoftcap = o.Attr.Scale, float32(o.Attr.Softcap)
		case graph.OpActMul:
			// A mixture's act_muls (the experts' pairs, the shared expert)
			// are not the dense FFN's width: the plan keeps planFor's.
			if pr := producer[o.In[0]]; lw.mix || pr < 0 || g.Ops[pr].Kind != graph.OpMatVec ||
				g.Ops[pr].Weight[0].Role != jlm.RoleFFNGate {
				break
			}
			p.NFFN = g.Values[o.Out[0]].Width
			switch o.Attr.Act {
			case graph.ActSiLU:
				p.Act = nn.ActSiLU
			case graph.ActGELUTanh:
				p.Act = nn.ActGELU
			}
		}
	}
}

// graphProducers is, per value of g, the op that writes it (-1 for a block
// input), and how many ops read it.
func graphProducers(g *graph.Graph) (readers []int, producer []int) {
	readers, producer = make([]int, len(g.Values)), make([]int, len(g.Values))
	for i := range producer {
		producer[i] = -1
	}
	for i, o := range g.Ops {
		for _, v := range o.In {
			readers[v]++
		}
		for _, v := range o.Out {
			producer[v] = i
		}
	}
	return readers, producer
}

// OpLowering is how one op of a block's graph is lowered on the host: the
// routine that runs it and whether that routine is composed from the op (a
// generated kernel chosen by the op's shape and the target) or monolithic (a
// hand-written routine that runs this op together with its neighbours, the
// composer's work list in docs/design/composer.md). On a device the whole
// block is one plan (nn.LayerPlan) the tier lowers, which Device says.
type OpLowering struct {
	Op       int
	Kind     graph.OpKind
	Host     string
	Composed bool
	Device   string
}

// hostLowering is the host routine per op kind. It is data, not a branch per
// family: a new op kind with no entry is unlowered and refused at build
// (TestEveryGraphOpHasALowering).
var hostLowering = map[graph.OpKind]struct {
	routine  string
	composed bool
}{
	graph.OpNorm:         {"nn RMSNorm (generated, by width)", true},
	graph.OpMatVec:       {"nn matvec (generated by quant and shape; grouped where a run fuses)", true},
	graph.OpAdd:          {"residual add (generated, or folded into the matvec's rows)", true},
	graph.OpActMul:       {"nn activation-multiply (generated, or fused into down's matvec)", true},
	graph.OpHeadNorm:     {"attnPrep: q/k norm, rotary and cache append as one routine", false},
	graph.OpRope:         {"attnPrep: q/k norm, rotary and cache append as one routine", false},
	graph.OpKVAppend:     {"attnPrep: q/k norm, rotary and cache append as one routine", false},
	graph.OpAttend:       {"attendRow / the chunk's attention: one routine per row class", false},
	graph.OpRoute:        {"moe / moeBatch: the routed mixture as one routine", false},
	graph.OpGather:       {"moe / moeBatch: the routed mixture as one routine", false},
	graph.OpExpertMatVec: {"moe / moeBatch: the routed mixture as one routine", false},
	graph.OpCombine:      {"moe / moeBatch: the routed mixture as one routine", false},
	graph.OpGateScalar:   {"moe / moeBatch: the routed mixture as one routine", false},
}

// BlockLowering is every op of text block li's graph with its lowering, nil
// where the block is not on the graph.
func (m *Model) BlockLowering(li int) []OpLowering {
	g := m.BlockGraph(li)
	if g == nil {
		return nil
	}
	out := make([]OpLowering, len(g.Ops))
	for i, o := range g.Ops {
		h := hostLowering[o.Kind]
		out[i] = OpLowering{Op: i, Kind: o.Kind, Host: h.routine, Composed: h.composed,
			Device: "the block's plan, lowered from the graph (lowerPlan) and run by the tier as one block"}
	}
	return out
}

// String is the lowering as the manifest prints it.
func (l OpLowering) String() string {
	how := "monolithic"
	if l.Composed {
		how = "composed"
	}
	return fmt.Sprintf("op %d %s: host %s [%s]", l.Op, l.Kind, l.Host, how)
}
