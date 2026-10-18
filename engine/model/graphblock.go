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
	UncoveredUngated       = "an activation the op language does not state"
	UncoveredMLA           = "a q latent without the latent KV cache"
	UncoveredValueWidth    = "a value head wider or narrower than the key head"
	UncoveredKVShared      = "KV sharing or an extra cached head"
	UncoveredIndexer       = "an attention indexer or sparse key selection"
	UncoveredStreams       = "several residual streams"
	UncoveredPLE           = "per-layer embeddings"
	UncoveredRopeAxes      = "a multi-axis or interleaved rotary"
	UncoveredNonCausal     = "non-causal attention"
	UncoveredSparseAct     = "a gaussian top-k activation"
	UncoveredDoubleFFN     = "a double-width FFN"
	UncoveredCompressed    = "compressed or grouped attention output"
	UncoveredRecurrent     = "a recurrent layer"
	UncoveredUnboundTensor = "a stored tensor no op of the graph binds"
)

// UncoveredReasons is the named list, in order.
var UncoveredReasons = []string{UncoveredMixDense, UncoveredMixRouter, UncoveredMixExperts,
	UncoveredUngated, UncoveredMLA, UncoveredValueWidth,
	UncoveredKVShared, UncoveredIndexer, UncoveredStreams, UncoveredPLE,
	UncoveredRopeAxes,
	UncoveredNonCausal, UncoveredSparseAct, UncoveredDoubleFFN, UncoveredCompressed,
	UncoveredRecurrent, UncoveredUnboundTensor}

// uncovered is why the block graph does not cover text block li, by name, or
// "" where it does. It is decided from the block's structure alone -- the
// configuration's fields and, after it, the tensors the container stores
// (UncoveredUnboundTensor, in buildGraphs) -- never from the architecture's
// name: a block the op language states runs through the graph whatever the
// file calls its family (TestCoverageIsStructural).
func (m *Model) uncovered(li int) string {
	c := m.Cfg
	roles := m.blockRoles(li)
	if c.MoEAt(li) {
		if _, why := m.mixtureWhy(li); why != "" {
			return why
		}
	}
	switch {
	case c.NGram != 0 || c.LayerKind(li).Recurrent() && c.LayerKind(li).Attends():
		return UncoveredRecurrent
	case m.blockAct(roles) == 0:
		return UncoveredUngated
	case !c.MLA() && c.QLoraRank != 0:
		return UncoveredMLA
	// A value head of its own width: HeadDimV against the model's head
	// width (a sliding layer's narrower heads narrow k and v alike, KVDimAt).
	case !c.MLA() && c.HeadDimV != 0 && c.HeadDimV != c.HeadDim:
		return UncoveredValueWidth
	case c.KVShared(li) || c.extraKVHead(li) || c.LayerKind(li).Attends() && !c.MLA() && !hasRole(roles, jlm.RoleAttnV):
		return UncoveredKVShared
	case c.Indexer() || c.MSA() || c.IdxHeads != 0 || c.QSARatio != 0:
		return UncoveredIndexer
	case c.streamHead():
		return UncoveredStreams
	case c.PLEDim != 0:
		return UncoveredPLE
	case c.RopeXD:
		return UncoveredRopeAxes
	case c.NonCausal:
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

// blockRoles is every tensor the container stores for text block li, by
// role, with its shape.
func (m *Model) blockRoles(li int) map[jlm.Role]graph.Shape {
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
	return inv
}

func hasRole(roles map[jlm.Role]graph.Shape, r jlm.Role) bool {
	_, ok := roles[r]
	return ok
}

// blockAct is the op language's name for the block's FFN activation: the
// stored xIELU numbers where the block carries them, else the configuration's
// kind; 0 where the op language does not state it.
func (m *Model) blockAct(roles map[jlm.Role]graph.Shape) graph.Act {
	if hasRole(roles, jlm.RoleXIELU) {
		return graph.ActXIELU
	}
	return graphAct(m.Cfg.Act)
}

// graphAct is the op language's name for a generated kernel's activation, 0
// where it states none.
func graphAct(a nn.ActKind) graph.Act {
	for g, n := range nnAct {
		if n == a {
			return g
		}
	}
	return 0
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
	inv := m.blockRoles(li)
	has := func(r jlm.Role) bool { return hasRole(inv, r) }
	d := graph.Dense{Block: li, NEmbd: c.NEmbd, NHead: c.NHead, NKVHead: c.NKVHeadAt(li), HeadDim: c.HeadDimAt(li),
		NFFN: c.NFFNAt(li), Eps: c.RMSEps, NRot: c.NRot, Base: c.RopeBase, Layout: layout, RopeFreqs: m.rope.Freqs != nil,
		Scale: c.AttnScale, Softcap: float64(c.AttnSoftcap), Act: m.blockAct(inv),
		QKNorm: c.QKNormAt(li) && !c.QKNormWide, QKNormWhole: c.QKNormAt(li) && c.QKNormWide,
		LayerNorm: c.LayerNorm, NormBias: has(jlm.RoleAttnNormBias), FFNNormBias: has(jlm.RoleFFNNormBias),
		Parallel: c.Parallel, SharedNorm: c.Parallel && !has(jlm.RoleFFNNorm),
		OutBias: has(jlm.RoleAttnOutBias), UpBias: has(jlm.RoleFFNUpBias), DownBias: has(jlm.RoleFFNDownBias),
		Ungated: !c.MoEAt(li) && !has(jlm.RoleFFNGate), Clamp: float64(c.ClampKQV),
		PostNorm: has(jlm.RolePostAttnNorm), NoPreNorm: !has(jlm.RoleAttnNorm), NoRope: !c.RopeAt(li)}
	d.QKVBias, d.Sinks = has(jlm.RoleAttnQBias), has(jlm.RoleAttnSinks)
	d.OutGate, d.VNorm = c.AttnOutGate, c.VNorm
	d.BidirOff = c.BidirSWA && !c.SWA(li)
	d.OutScale = has(jlm.RoleLayerOutScale)
	if c.ResidualScale != 1 {
		d.ResidScale = c.ResidualScale
	}
	d.QKNormPost, d.QKL2Norm = c.QKNormPost, c.QKL2Norm && c.RopeAt(li)
	// A recurrent mixer in place of the attention, binding the mixer's
	// tensors the block stores.
	if k := c.LayerKind(li); k.Recurrent() {
		d.Mixer, d.NoFFN = uint8(k), m.layers[li].noFFN
		for _, r := range graph.MixerRoles {
			if s, ok := inv[r]; ok {
				d.MixerW = append(d.MixerW, graph.Binding{Role: r, K: s.K, Rows: s.Rows, Sheets: s.Sheets})
			}
		}
	}
	// The query temperature, where AttnTemp applies it to this block.
	if c.AttnTempScale != 0 && !(c.NoPEGlobal && c.RopeAt(li)) {
		d.TempScale, d.TempFloor, d.TempOffs = c.AttnTempScale, c.AttnTempFloor, c.AttnTempOffs
	}
	// A sliding block attends over its window and turns by the local
	// rotary where the model carries one (gemma3's second base); either
	// table's scaling (linear, YaRN, LongRoPE) is its per-pair factors and
	// its cos/sin magnitude.
	// The widths and bases are the configuration's, which the tables are
	// built from.
	tab := &m.rope
	if c.SWA(li) && c.SWAWindow > 0 {
		d.Window, d.Chunked = c.SWAWindow, c.SWAChunked
		if m.ropeSWA != nil {
			tab = m.ropeSWA
			d.NRot = c.NRotSWA
			if c.RopeBaseSWA != 0 {
				d.Base = c.RopeBaseSWA
			}
		}
	}
	d.RopeFreqs = tab.Freqs != nil
	if len(m.rope.Runs) > 0 {
		d.RopeSections, d.RopeInterleaved = c.RopeSections, c.RopeInterleaved
	}
	if tab.Scale != 1 {
		d.Mscale = tab.Scale
	}
	if c.MLA() {
		d.MLA = &graph.MLA{NHead: c.NHead, HeadDim: c.HeadDimAt(li), NRot: d.NRot, KVLoraRank: c.KVLoraRank,
			QLoraRank: c.QLoraRank, HeadDimV: c.HeadDimV, LatentEps: float64(c.LatentNormEps),
			NoRope: c.NoPosEnc || !c.RopeAt(li)}
	}
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
	// qkPost: q's norm after the rotary (Hunyuan's); qkL2: weightless head
	// norms on q and k after it (Llama 4's). attnPrep applies both.
	qkPost, qkL2 bool
	// prep is the block's attention prologue, its ops from the projections
	// to the attention lowered one by one (lowerPrep).
	prep []prepStep
	// mix says the block's FFN is a routed mixture (graph.Dense.Mixture),
	// which the host runs through the mixture's operations (moe, moeBatch).
	mix bool
	// The classic block's switches, as the matched ops state them: par, the
	// FFN reads the block input (normed before attention, by its own norm
	// or, with shared, the attention's); ungated, an FFN of up, act and
	// down; postAttn and postFFN, an RMSNorm over each sublayer's output
	// before its residual add.
	par, shared, ungated, postAttn, postFFN bool
	// mla says the block's attention is latent (graph.Dense.MLA), run
	// absorbed (graph.MLAAbsorb) through mlaProject, the attention over the
	// latent rows and the un-absorb.
	mla bool
	// outScale: the block's output times its stored scalar (layerOutScale).
	outScale bool
	// outGate: q's projection carries each head's output gate (projectQ
	// splits it; applyOutGate after the attention).
	outGate bool
	// mixer: a recurrent mixer (linearAttn) in place of the attention;
	// noFFN: the block ends at the mixer's residual add.
	mixer, noFFN bool
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
	// preNorm takes a sublayer's input norm: an RMSNorm or a LayerNorm, or
	// none (OLMo 2's block reads its input as it is).
	preNorm := func(role jlm.Role) bool { return take(graph.OpNorm, role) || take(graph.OpLayerNorm, role) }
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
	// last is the value the op just taken writes.
	last := func() graph.ValueID { return ops[i-1].Out[0] }
	lw := &denseLowering{g: g}
	attnIn, ffnPre := g.In[0], graph.ValueID(-1)
	if preNorm(jlm.RoleAttnNorm) {
		attnIn = last()
	}
	// A parallel block's FFN norm, over the block input, before attention.
	if preNorm(jlm.RoleFFNNorm) {
		ffnPre = last()
	}
	qa := false
	if take(graph.OpRecurrent, 0) {
		// A recurrent mixer in place of the attention (linearAttn), added
		// into the residual; a mixer-only layer ends there.
		lw.mixer = true
		if err := run([]step{{graph.OpAdd, 0}}); err != nil {
			return nil, err
		}
		if i == len(ops) {
			lw.noFFN = true
			return lw, nil
		}
	} else if qa = take(graph.OpMatVec, jlm.RoleAttnQA); qa || (i+1 < len(ops) && ops[i].Kind == graph.OpMatVec && ops[i+1].Kind == graph.OpSlice) {
		// The latent attention, as mlaProject computes it.
		lw.mla = true
		if qa {
			if err := run([]step{{graph.OpNorm, jlm.RoleAttnQANorm}, {graph.OpMatVec, jlm.RoleAttnQB}}); err != nil {
				return nil, err
			}
		} else if err := run([]step{{graph.OpMatVec, jlm.RoleAttnQ}}); err != nil {
			return nil, err
		}
		if err := run([]step{{graph.OpSlice, 0}, {graph.OpSlice, 0}}); err != nil {
			return nil, err
		}
		rot := take(graph.OpRope, 0)
		if err := run([]step{{graph.OpMatVec, jlm.RoleAttnKVA}, {graph.OpSlice, 0}, {graph.OpNorm, jlm.RoleAttnKVANorm},
			{graph.OpSlice, 0}}); err != nil {
			return nil, err
		}
		if rot != take(graph.OpRope, 0) {
			return nil, fail("both rotary parts turned, or neither")
		}
		if err := run([]step{{graph.OpLatentAppend, 0}, {graph.OpAttendLatent, 0}}); err != nil {
			return nil, err
		}
	} else {
		if err := run([]step{{graph.OpMatVec, jlm.RoleAttnQ}, {graph.OpMatVec, jlm.RoleAttnK}, {graph.OpMatVec, jlm.RoleAttnV}}); err != nil {
			return nil, err
		}
		qkv := [3]graph.ValueID{ops[i-3].Out[0], ops[i-2].Out[0], ops[i-1].Out[0]}
		// The output gate: q's heads and the gate sliced apart (projectQ).
		if take(graph.OpSlice, 0) {
			if err := run([]step{{graph.OpSlice, 0}}); err != nil {
				return nil, err
			}
			lw.outGate, qkv[0] = true, ops[i-2].Out[0]
		}
		prep0 := i
		// DBRX's clamps, first.
		if take(graph.OpClamp, 0) {
			if err := run([]step{{graph.OpClamp, 0}, {graph.OpClamp, 0}}); err != nil {
				return nil, err
			}
		}
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
		// Hunyuan's post-rotary form: k's norm before the rotary, q's after.
		if !lw.qk && take(graph.OpHeadNorm, jlm.RoleAttnKNorm) {
			lw.qk, lw.qkPost = true, true
		}
		// The rotary, absent on a block without one; Llama 4's weightless
		// norms after it.
		if take(graph.OpRope, 0) {
			if err := run([]step{{graph.OpRope, 0}}); err != nil {
				return nil, err
			}
			// Llama 4's weightless norm reads the rotated q (Gemma 4's over v
			// reads v, below).
			if i < len(ops) && ops[i].Kind == graph.OpHeadNorm && len(ops[i].Weight) == 0 && ops[i].In[0] == ops[i-2].Out[0] {
				i++
				if i >= len(ops) || ops[i].Kind != graph.OpHeadNorm || len(ops[i].Weight) != 0 {
					return nil, fail("head_norm of k with no weight")
				}
				i++
				lw.qkL2 = true
			}
		}
		if lw.qkPost && !take(graph.OpHeadNorm, jlm.RoleAttnQNorm) {
			return nil, fail("head_norm of attn_q_norm after the rotary")
		}
		// Gemma 4's weightless norm over v.
		if i < len(ops) && ops[i].Kind == graph.OpHeadNorm && len(ops[i].Weight) == 0 {
			i++
		}
		if err := run([]step{{graph.OpKVAppend, 0}}); err != nil {
			return nil, err
		}
		prep, err := lowerPrep(g, prep0, i, qkv)
		if err != nil {
			return nil, err
		}
		lw.prep = prep
		if err := run([]step{{graph.OpAttend, 0}}); err != nil {
			return nil, err
		}
		if lw.outGate && !take(graph.OpActMul, 0) {
			return nil, fail("act_mul of the output gate")
		}
	}
	if !lw.mixer {
		if err := run([]step{{graph.OpMatVec, jlm.RoleAttnOut}}); err != nil {
			return nil, err
		}
		lw.postAttn = take(graph.OpNorm, jlm.RolePostAttnNorm)
		if err := run([]step{{graph.OpAdd, 0}}); err != nil {
			return nil, err
		}
	}
	// The FFN's input, by the value its first op reads: its own norm over
	// the residual (or the residual itself, with no pre-norm), or a parallel
	// block's -- its own norm of the block input, or the attention's.
	ffnIn := last()
	if ffnPre < 0 && preNorm(jlm.RoleFFNNorm) {
		ffnIn = last()
	}
	if i >= len(ops) || len(ops[i].In) == 0 {
		return nil, fail("the FFN")
	}
	switch ops[i].In[0] {
	case ffnIn:
	case ffnPre:
		lw.par = true
	case attnIn:
		lw.par, lw.shared = true, true
	default:
		return nil, fail("an FFN over the residual, its norm or a parallel block's input")
	}
	if ffnPre >= 0 && !lw.par {
		return nil, fail("an FFN that reads the norm taken before attention")
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
	} else if take(graph.OpMatVec, jlm.RoleFFNUp) {
		// The ungated FFN: up, its activation alone, down.
		lw.ungated = true
		if err := run([]step{{graph.OpAct, 0}, {graph.OpMatVec, jlm.RoleFFNDown}}); err != nil {
			return nil, err
		}
	} else if err := run([]step{{graph.OpMatVec, jlm.RoleFFNGate}, {graph.OpMatVec, jlm.RoleFFNUp}, {graph.OpActMul, 0},
		{graph.OpMatVec, jlm.RoleFFNDown}}); err != nil {
		return nil, err
	}
	lw.postFFN = take(graph.OpNorm, jlm.RolePostFFNNorm)
	if lw.mix && lw.postFFN {
		return nil, fail("a mixture with no norm after it (moe adds into the residual itself)")
	}
	if err := run([]step{{graph.OpAdd, 0}}); err != nil {
		return nil, err
	}
	// Gemma 4's output scalar (layerOutScale).
	lw.outScale = take(graph.OpScale, jlm.RoleLayerOutScale)
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
			if l := &m.layers[li]; !m.Cfg.NonCausal && l.gate.e == nil && l.router.e == nil && !l.noFFN {
				// Only the graph's lowering runs an ungated FFN (an encoder
				// runs its own body, encoder.go).
				return fmt.Errorf("model: block %d of %s has an ungated FFN and is not covered by the block graph (%s); "+
					"the hand-written body has no ungated FFN", li, m.Cfg.Arch, why)
			}
			if m.Cfg.Parallel {
				// Only the graph's lowering runs a parallel residual.
				return fmt.Errorf("model: block %d of %s is parallel and not covered by the block graph (%s); "+
					"the hand-written body has no parallel residual", li, m.Cfg.Arch, why)
			}
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
		if u := graph.Unbound(g, f); len(u) > 0 && m.laxUnbound {
			// A violation's graph without an op: its tensors stay stored,
			// unread.
			for _, r := range u {
				delete(f.Inventory, r)
			}
		} else if len(u) > 0 {
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
		// What the host routines read from the configuration, held to the
		// graph's ops: a chunked window (the KV layer's) and a weight-in
		// mixture (moe's).
		for _, o := range g.Ops {
			if o.Kind == graph.OpAttend && o.Attr.Chunked != (m.Cfg.SWA(li) && m.Cfg.SWAChunked) {
				return fmt.Errorf("model: block %d graph's chunked window is %v, the KV layer's is not", li, o.Attr.Chunked)
			}
			if o.Kind == graph.OpAttend && o.Attr.BidirOff != (m.Cfg.BidirSWA && !m.Cfg.SWA(li)) {
				return fmt.Errorf("model: block %d graph's bidirectional runs off is %v, the attention's is not", li, o.Attr.BidirOff)
			}
			if o.Kind == graph.OpGather && o.Attr.WeightIn != m.Cfg.ExpertWeightIn {
				return fmt.Errorf("model: block %d graph's weight-in is %v and moe's %v", li, o.Attr.WeightIn, m.Cfg.ExpertWeightIn)
			}
		}
		if lw.outScale != (m.layers[li].outScale != 0) {
			return fmt.Errorf("model: block %d graph's output scale is %v, the layer's is not", li, lw.outScale)
		}
		if lw.qk != m.Cfg.QKNormAt(li) || lw.qkPost != (lw.qk && m.Cfg.QKNormPost) ||
			lw.qkL2 != (m.Cfg.QKL2Norm && m.Cfg.RopeAt(li)) {
			return fmt.Errorf("model: block %d graph's q/k norms (%v, post %v, weightless %v) are not attnPrep's",
				li, lw.qk, lw.qkPost, lw.qkL2)
		}
		lw.rewrites = graph.Apply(g, graph.Rewrites...)
		if lw.mla && !rewriteLegal(lw.rewrites, graph.MLAAbsorb.Name) {
			return fmt.Errorf("model: block %d: the latent attention is not absorbed (%s), and its reference form has no lowering", li, graph.MLAAbsorb.Name)
		}
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
	// norm(x) -> attn_in [, a parallel block's FFN input]
	s.norm(s.h, s.x, l.attnNorm, l.attnNormB)
	if lw.par {
		if lw.shared {
			copy(s.hf, s.h)
		} else {
			s.norm(s.hf, s.x, l.ffnNorm, l.ffnNormB)
		}
	}
	s.jit.NewInput()
	m.trace(li, "attn_norm", s.h)
	if lw.mixer {
		// The recurrent mixer in place of the attention, added into the
		// residual; a mixer-only layer ends there.
		if err := s.linearAttn(li, l, 0, s.h, s.h); err != nil {
			return err
		}
		m.trace(li, "ssm_out", s.h)
		s.addInto(s.x, s.h)
		m.trace(li, "attn_resid", s.x)
		if lw.noFFN {
			return nil
		}
		return s.graphFFN(li, l, lw)
	}
	attnResid := false
	if lw.mla {
		// [matvec q_a, norm,] matvec q, the slices, [rope,] matvec kv_a, the
		// latent's norm, [rope,] latent_append: mlaProject; attend_latent,
		// absorbed (graph.MLAAbsorb): attendRow over the latent rows, then
		// mlaOutProject's un-absorb and o.
		if err := s.mlaProject(li, l, s.h, s.ropeTable(li, s.cs, s.csSWA, 0), 0, s.pos); err != nil {
			return err
		}
		m.trace(li, "k", s.kc[:c.KVDim()])
		s.attendRow(li, l)
		if err := s.mlaOutProject(s.h, l); err != nil {
			return err
		}
		s.addBias(s.h, l.bo)
	} else {
		// matvec q, k, v (one region inside a fused run, as the body groups them)
		kvd := c.KVDimAt(li)
		s.aheadOf(li, 0, [nn.MaxGroup]*tensor{&l.wo})
		grouped := !lw.outGate && s.groupOf(s.h, [nn.MaxGroup]*tensor{&l.wq, &l.wk, &l.wv}, [nn.MaxGroup][]float32{s.q, s.kc, s.vc}, 3)
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
		s.runPrep(l, lw.prep, s.q[:c.QDimAt(li)], s.kc[:kvd], s.vc[:kvd], s.ropeTable(li, s.cs, s.csSWA, 0), li, 0, s.pos)
		m.trace(li, "k", s.kc[:kvd])
		// attend
		s.attendRow(li, l)
		// matvec o, add into the residual (folded into o's rows where the run
		// fuses it)
		qd := c.QDimAt(li)
		src := s.xb
		if lw.outGate {
			s.applyOutGate(s.ogate, s.xb, 1)
			s.jit.NewInput()
			src = s.ogate
		}
		plain := s.residFold && l.bo == nil && !lw.postAttn
		if plain {
			s.grp.Resid, s.grp.Alpha = s.x, float32(c.ResidualScale)
		}
		s.aheadOf(li, 1, [nn.MaxGroup]*tensor{&l.gate, &l.up})
		attnResid = plain && s.groupOf(src[:qd], [nn.MaxGroup]*tensor{&l.wo}, [nn.MaxGroup][]float32{s.h}, 1)
		s.grp.Resid = nil
		if !attnResid && !s.groupOf(src[:qd], [nn.MaxGroup]*tensor{&l.wo}, [nn.MaxGroup][]float32{s.h}, 1) {
			if err := s.mv(s.h, l.wo, src[:qd]); err != nil {
				return err
			}
		}
		s.addBias(s.h, l.bo)
	}
	// [norm(o)] before the residual add
	if lw.postAttn {
		s.rmsnorm(s.h, s.h, l.postAttnNorm, c.RMSEps)
	}
	if !attnResid {
		s.addInto(s.x, s.h)
	}
	m.trace(li, "attn_resid", s.x)
	return s.graphFFN(li, l, lw)
}

// graphFFN is the second half of block li's host decode through its
// lowering, from the residual the first half added into: the FFN's norm (or a
// parallel block's input), the mixture, the ungated or the gated FFN, [its
// norm,] and the residual add.
func (s *State) graphFFN(li int, l *layer, lw *denseLowering) error {
	c, m := s.c, s.m
	// norm -> gate, up -> act_mul -> down, add into the residual; a
	// parallel block's FFN reads the input normed above.
	ffnIn := s.h
	if lw.par {
		ffnIn = s.hf
	} else {
		s.norm(s.h, s.x, l.ffnNorm, l.ffnNormB)
	}
	s.jit.NewInput()
	if lw.mix {
		// The mixture's ops, each its own routine, added into the residual
		// as the hand-written body's moe adds them: route (the router's
		// matvec and the route kernel); gather, the experts' gate, up,
		// act_mul and down, and combine (one grouped region); [the shared
		// expert's matvecs, act_mul and gate_scalar, and its add].
		if err := s.moeRoute(li, l, ffnIn, ffnIn); err != nil {
			return err
		}
		if err := s.moeExperts(li, l, ffnIn, s.x); err != nil {
			return err
		}
		if err := s.sharedExpert(l, ffnIn, s.x); err != nil {
			return err
		}
		s.layerOutScale(l, s.x)
		m.trace(li, "ffn_resid", s.x)
		return nil
	}
	if lw.ungated {
		// up [+ bias] -> act -> down [+ bias]
		if err := s.mv(s.up, l.up, ffnIn); err != nil {
			return err
		}
		s.addBias(s.up, l.upB)
		s.ungatedAct(l, s.up)
		s.jit.NewInput()
		if err := s.mv(s.h, l.down, s.up); err != nil {
			return err
		}
		s.addBias(s.h, l.downB)
		if lw.postFFN {
			s.rmsnorm(s.h, s.h, l.postFFNNorm, c.RMSEps)
		}
		s.addInto(s.x, s.h)
		s.layerOutScale(l, s.x)
		m.trace(li, "ffn_resid", s.x)
		return nil
	}
	ff := c.NFFNAt(li)
	gate, up, act := s.gate[:ff], s.up[:ff], c.Act
	s.aheadOf(li, 2, [nn.MaxGroup]*tensor{&l.down})
	if !s.groupOf(ffnIn, [nn.MaxGroup]*tensor{&l.gate, &l.up}, [nn.MaxGroup][]float32{gate, up}, 2) {
		if err := s.mv(gate, l.gate, ffnIn); err != nil {
			return err
		}
		if err := s.mv(up, l.up, ffnIn); err != nil {
			return err
		}
	}
	s.gaussTopK(li, gate, ff)
	s.grp.ActUp, s.grp.Act = up, act
	if s.residFold && !lw.postFFN {
		s.grp.Resid, s.grp.Alpha = s.x, float32(c.ResidualScale)
	}
	s.aheadNext(li)
	grouped := s.groupOf(gate, [nn.MaxGroup]*tensor{&l.down}, [nn.MaxGroup][]float32{s.h}, 1)
	ffnResid := grouped && s.grp.Resid != nil
	s.grp.ActUp, s.grp.Resid = nil, nil
	if !grouped {
		s.actmulAll(gate, up, act)
		s.jit.NewInput()
		if err := s.mv(s.h, l.down, gate); err != nil {
			return err
		}
	}
	if lw.postFFN {
		s.rmsnorm(s.h, s.h, l.postFFNNorm, c.RMSEps)
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
func (s *State) graphRows(li int, l *layer, lw *denseLowering, slot, base, p0, n int, bidir bool) error {
	c := s.c
	s.graphChunkRuns++
	s.pathOf(li).Graph++
	ne := c.NEmbd
	// norm(x) -> attn_in, every row
	tn := s.tick()
	s.batchNormB(s.bh, s.bx, l.attnNorm, l.attnNormB, ne, n)
	s.parallelIn(l, lw, n)
	s.tock(opRMSNorm, tn)
	if lw.mixer {
		// The recurrent mixer over the chunk, row by row, into bh.
		if err := s.mixerChunk(li, l, slot, base, s.bh, n); err != nil {
			return err
		}
		return s.rowsResidFFN(li, l, lw, n, false)
	}
	if lw.mla {
		// The latent attention over the chunk: every row's projections,
		// rotary and latent append (mlaProjectRows) before any row attends,
		// then the absorbed attention and mlaOutRows' un-absorb and o.
		if err := s.mlaProjectRows(li, l, n, s.bcs, s.bcsSWA,
			func(i int) (int, int) { return slot, p0 + i }); err != nil {
			return err
		}
		s.attendRows(li, l, slot, p0, n, c.QDimAt(li), c.AttnScale, bidir, true)
		if err := s.mlaOutRows(l, n); err != nil {
			return err
		}
		s.addBiasRows(s.bh, l.bo, n)
		return s.rowsResidFFN(li, l, lw, n, false)
	}
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
		s.runPrep(l, lw.prep, s.bq[i*qDim:(i+1)*qDim], s.bk[i*kvDim:(i+1)*kvDim], s.bv[i*kvDim:(i+1)*kvDim],
			s.ropeTable(li, s.bcs, s.bcsSWA, i), li, slot, p0+i)
	}
	// attend
	s.attendRows(li, l, slot, p0, n, qDim, c.AttnScale, bidir, false)
	// [the output gate,] matvec o, add into the residual
	asrc := s.bxb
	if lw.outGate {
		s.applyOutGate(s.ogate, s.bxb, n)
		s.jit.NewInput()
		asrc = s.ogate
	}
	if err := s.mm(s.bh, l.wo, asrc[:n*qDim], n); err != nil {
		return err
	}
	s.addBiasRows(s.bh, l.bo, n)
	return s.rowsResidFFN(li, l, lw, n, false)
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
	s.parallelIn(l, lw, n)
	s.tock(opRMSNorm, t0)
	if lw.mixer {
		// Each row steps its own sequence's state, in position order.
		if err := s.mixerRagged(li, l, seq, s.rowCausal[:n], s.bh, n); err != nil {
			return err
		}
		return s.rowsResidFFN(li, l, lw, n, true)
	}
	if lw.mla {
		// The latent attention over the step: each row its own sequence's
		// slot and position (mlaProjectRows), the absorbed attention, then
		// mlaOutRows' un-absorb and o.
		if err := s.mlaProjectRows(li, l, n, cs, csSWA,
			func(i int) (int, int) { return seq[i], pos[i] }); err != nil {
			return err
		}
		s.rowsAttend(li, l, n, seq, pos, unitHeads, units, astride, scale)
		if err := s.mlaOutRows(l, n); err != nil {
			return err
		}
		s.addBiasRows(s.bh, l.bo, n)
		return s.rowsResidFFN(li, l, lw, n, true)
	}
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
		o.runPrep(l, lw.prep, s.bq[i*qDim:(i+1)*qDim], s.bk[i*kvDim:(i+1)*kvDim], s.bv[i*kvDim:(i+1)*kvDim],
			s.ropeTable(li, cs, csSWA, i), li, slot, pos[i])
	}
	// attend
	s.rowsAttend(li, l, n, seq, pos, unitHeads, units, astride, scale)
	// [the output gate,] matvec o, add into the residual
	asrc := s.bxb
	if lw.outGate {
		s.applyOutGate(s.ogate, s.bxb, n)
		s.jit.NewInput()
		asrc = s.ogate
	}
	if err := s.mm(s.bh, l.wo, asrc[:n*qDim], n); err != nil {
		return err
	}
	s.addBiasRows(s.bh, l.bo, n)
	return s.rowsResidFFN(li, l, lw, n, true)
}

// rowsResidFFN is the second half of a block over n rows in s.bx (a prompt
// chunk, or with ragged a batch step's rows), from the attention output in
// s.bh: [its norm,] the residual add, then the FFN -- its norm or a parallel
// block's input, the mixture, the ungated or the gated FFN, [its norm,] and
// the second add -- as hostRows and rowsHost run them.
func (s *State) rowsResidFFN(li int, l *layer, lw *denseLowering, n int, ragged bool) error {
	c := s.c
	ne := c.NEmbd
	if lw.postAttn {
		tn := s.tick()
		s.batchNorm(s.bh, s.bh, l.postAttnNorm, ne, n)
		s.tock(opRMSNorm, tn)
	}
	tr := s.tick()
	s.addInto(s.bx[:n*ne], s.bh[:n*ne])
	s.tock(opResid, tr)
	// A mixer-only layer ends at the first residual.
	if lw.noFFN {
		return nil
	}
	// norm -> gate, up -> act_mul -> down, add into the residual
	tn := s.tick()
	ffnIn := s.bh
	if lw.par {
		ffnIn = s.bhf
		s.jit.NewInput()
	} else {
		s.batchNormB(s.bh, s.bx, l.ffnNorm, l.ffnNormB, ne, n)
	}
	s.tock(opRMSNorm, tn)
	if lw.mix {
		// The mixture over the rows, op by op as moeBatch runs them: every
		// row's route, then the experts expert-major, then the shared
		// expert per row, into the residual; on a ragged step each row its
		// own sequence's experts.
		if err := s.moeRouteRows(li, l, ffnIn, n); err != nil {
			return err
		}
		if err := s.moeExpertsRows(li, l, ffnIn, s.bx, n); err != nil {
			return err
		}
		if err := s.sharedExpertRows(l, ffnIn, s.bx, n); err != nil {
			return err
		}
		s.layerOutScale(l, s.bx[:n*ne])
		return nil
	}
	if lw.ungated {
		// up [+ bias] -> act -> down [+ bias]
		if err := s.mm(s.bup, l.up, ffnIn, n); err != nil {
			return err
		}
		s.addBiasRows(s.bup, l.upB, n)
		tac := s.tick()
		s.ungatedAct(l, s.bup[:n*c.NFFN])
		s.jit.NewInput()
		s.tock(opAct, tac)
		if err := s.mm(s.bh, l.down, s.bup, n); err != nil {
			return err
		}
		s.addBiasRows(s.bh, l.downB, n)
	} else {
		ff := c.NFFNAt(li)
		if err := s.mm(s.bgate, l.gate, ffnIn[:n*ne], n); err != nil {
			return err
		}
		if err := s.mm(s.bup, l.up, ffnIn[:n*ne], n); err != nil {
			return err
		}
		gate, up := s.bgate[:n*ff], s.bup[:n*ff]
		tac := s.tick()
		s.gaussTopK(li, gate, ff)
		s.actmulAll(gate, up, c.Act)
		if ragged {
			s.jit.NewInput()
		}
		s.tock(opAct, tac)
		if err := s.mm(s.bh, l.down, gate, n); err != nil {
			return err
		}
	}
	if lw.postFFN {
		tn := s.tick()
		s.batchNorm(s.bh, s.bh, l.postFFNNorm, ne, n)
		s.tock(opRMSNorm, tn)
	}
	tr = s.tick()
	s.addInto(s.bx[:n*ne], s.bh[:n*ne])
	s.tock(opResid, tr)
	s.layerOutScale(l, s.bx[:n*ne])
	return nil
}

// parallelIn is a parallel block's FFN input over n rows, taken from the
// block input before attention adds into it: its own norm, or a copy of the
// attention's.
func (s *State) parallelIn(l *layer, lw *denseLowering, n int) {
	if !lw.par {
		return
	}
	ne := s.c.NEmbd
	if lw.shared {
		copy(s.bhf[:n*ne], s.bh[:n*ne])
		return
	}
	s.batchNormB(s.bhf, s.bx, l.ffnNorm, l.ffnNormB, ne, n)
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
	p.QKNorm, p.QKNormPost, p.LayerNorm, p.Parallel = false, false, false, lw.par
	_, producer := graphProducers(g)
	windowed := false
	for _, o := range g.Ops {
		windowed = windowed || o.Kind == graph.OpAttend && o.Attr.Window != 0
	}
	for _, o := range g.Ops {
		switch o.Kind {
		case graph.OpLayerNorm:
			p.RMSEps, p.LayerNorm = o.Attr.Eps, true
		case graph.OpAct:
			p.NFFN = g.Values[o.Out[0]].Width
			if a, ok := nnAct[o.Attr.Act]; ok && o.Attr.Act != graph.ActXIELU {
				p.Act = a
			}
		case graph.OpNorm:
			// A latent attention's q and kv latent norms carry their own
			// epsilon, which the plan states apart (planFor's).
			if r := o.Weight[0].Role; r == jlm.RoleAttnQANorm || r == jlm.RoleAttnKVANorm {
				break
			}
			p.RMSEps = o.Attr.Eps
			if len(o.Weight) == 1 && o.Weight[0].Role == jlm.RoleAttnQNorm {
				p.QKNorm = true
			}
		case graph.OpHeadNorm:
			// A weightless head norm is Llama 4's, which the plan carries
			// model-wide (QKL2Norm); a weighted one is the q/k norm.
			if len(o.Weight) == 1 {
				p.QKNorm = true
				p.QKNormPost = lw.qkPost
			}
		case graph.OpRope:
			p.RopeNeox = o.Attr.Layout == graph.RopeNeox
			// A sliding block's rotary is the local table the plan carries
			// model-wide (RopeTabSWA); the plan's own is the global's.
			if !windowed {
				p.NRot, p.RopeBase = o.Attr.NRot, o.Attr.Base
			}
		case graph.OpAttendLatent:
			p.NHead, p.HeadDim, p.AttnScale = o.Attr.NHead, o.Attr.HeadDim, o.Attr.Scale
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
			if a, ok := nnAct[o.Attr.Act]; ok {
				p.Act = a
			}
		}
	}
}

// nnAct is the generated kernels' activation per op-language activation.
var nnAct = map[graph.Act]nn.ActKind{graph.ActSiLU: nn.ActSiLU, graph.ActGELUTanh: nn.ActGELU,
	graph.ActGELUErf: nn.ActGELUErf, graph.ActReLU2: nn.ActReLU2, graph.ActReLU: nn.ActReLU,
	graph.ActSwiGLUOAI: nn.ActSwiGLUOAI}

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
	graph.OpHeadNorm:     {"nn RMSNorm per head or over the projection (generated, by width; runPrep)", true},
	graph.OpRope:         {"nn rotary (generated, by layout; runPrep)", true},
	graph.OpKVAppend:     {"KV page store (the cache layout's writer, by precision; runPrep)", true},
	graph.OpAttend:       {"attendRow / the chunk's attention: one routine per row class", false},
	graph.OpRoute:        {"moeRoute / moeRouteRows: the router's matvec (generated) and the route kernel (generated, by experts, top-k, router and groups)", true},
	graph.OpGather:       {moeExpertsRoutine, false},
	graph.OpExpertMatVec: {moeExpertsRoutine, false},
	graph.OpCombine:      {moeExpertsRoutine, false},
	graph.OpGateScalar:   {"sharedExpert: the gate's matvec and sigmoid-multiply (generated)", true},
	graph.OpLayerNorm:    {"nn LayerNorm (generated, by width)", true},
	graph.OpAct:          {"nn activation (generated, by kind)", true},
	graph.OpClamp:        {"nn clamp (generated; runPrep)", true},
	graph.OpSlice:        {mlaProjectRoutine, false},
	graph.OpLatentAppend: {mlaProjectRoutine, false},
	graph.OpAttendLatent: {"absorbed (mla-absorb): attendRow / the chunk's attention over the latent rows, then mlaOutProject's un-absorb", false},
	graph.OpScale:        {"the generated scale kernel (layerOutScale)", true},
	graph.OpRecurrent:    {"linearAttn: the recurrent mixer as one routine (mixerChunk, mixerRagged over rows)", false},
}

// moeExpertsRoutine is the host routine a routed mixture's gather, expert
// matvecs, their act_mul and the combine run in: moeExperts (one grouped
// region per stage) and moeExpertsRows (expert-major over the rows).
const moeExpertsRoutine = "moeExperts / moeExpertsRows: gather, the experts' matvecs and act_mul, and combine as one grouped region"

// mlaProjectRoutine is the host routine every op of a latent attention runs
// in before the attention: the projections, the latent norms, the rotary,
// the absorb into the query and the latent append (mlaProject,
// mlaProjectRows).
const mlaProjectRoutine = "mlaProject: the q and kv projections, latent norms, rotary, absorb and latent append as one routine"

// BlockLowering is every op of text block li's graph with its lowering, nil
// where the block is not on the graph.
func (m *Model) BlockLowering(li int) []OpLowering {
	g := m.BlockGraph(li)
	if g == nil {
		return nil
	}
	out := make([]OpLowering, len(g.Ops))
	// A latent attention's projections and norms before its attention run
	// inside mlaProject, not as the generated op their kind names.
	mlaFrom, mlaTo := -1, -1
	_, producer := graphProducers(g)
	for i, o := range g.Ops {
		if o.Kind == graph.OpMatVec && (o.Weight[0].Role == jlm.RoleAttnQA || o.Weight[0].Role == jlm.RoleAttnQ) && mlaFrom < 0 {
			mlaFrom = i
		}
		if o.Kind == graph.OpAttendLatent {
			mlaTo = i
		}
	}
	for i, o := range g.Ops {
		h := hostLowering[o.Kind]
		if mlaTo >= 0 && i >= mlaFrom && i < mlaTo {
			h.routine, h.composed = mlaProjectRoutine, false
		}
		// The experts' act_mul runs inside their grouped region.
		if o.Kind == graph.OpActMul && producer[o.In[0]] >= 0 && g.Ops[producer[o.In[0]]].Kind == graph.OpExpertMatVec {
			h.routine, h.composed = moeExpertsRoutine, false
		}
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

// rewriteLegal is whether decisions holds a legal decision of the rewrite
// named name.
func rewriteLegal(decisions []graph.Decision, name string) bool {
	for _, d := range decisions {
		if d.Rewrite == name && d.Err == nil {
			return true
		}
	}
	return false
}

// handless is what the model has that the hand-written body no longer runs
// (a parallel residual, an ungated FFN), "" where it has neither: such a
// model runs through the block graph alone.
func (m *Model) handless() string {
	if m.Cfg.Parallel {
		return "a parallel residual"
	}
	if m.Cfg.NonCausal {
		// An encoder runs its own body (encoder.go).
		return ""
	}
	for li := range m.Cfg.NLayer {
		if l := &m.layers[li]; l.gate.e == nil && l.router.e == nil && !l.noFFN {
			return "an ungated FFN"
		}
	}
	return ""
}
