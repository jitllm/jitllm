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
// steps) and on the device (the plan). The graph is every text block's only
// body: a block the op language does not state refuses the model at Open, by
// a reason on the named list below.

// Every reason a text block is refused: what the op language does not state,
// by name.
// Shrinking this list is the graph's work list.
const (
	UncoveredMixRouter     = "a mixture: a router the op language does not state (sqrt-softplus, hash routing or a router norm)"
	UncoveredMixExperts    = "a mixture: experts the op language does not state (ungated, unbanked or latent)"
	UncoveredUngated       = "an activation the op language does not state"
	UncoveredMLA           = "a q latent without the latent KV cache"
	UncoveredValueWidth    = "a value head wider or narrower than the key head"
	UncoveredKVShared      = "KV sharing or an extra cached head"
	UncoveredIndexer       = "an attention indexer or sparse key selection"
	UncoveredStreams       = "several residual streams"
	UncoveredRopeAxes      = "a multi-axis or interleaved rotary"
	UncoveredNonCausal     = "non-causal attention"
	UncoveredSparseAct     = "a gaussian top-k activation"
	UncoveredCompressed    = "compressed or grouped attention output"
	UncoveredRecurrent     = "a recurrent layer"
	UncoveredUnboundTensor = "a stored tensor no op of the graph binds"
)

// UncoveredReasons is the named list, in order.
var UncoveredReasons = []string{UncoveredMixRouter, UncoveredMixExperts,
	UncoveredUngated, UncoveredMLA, UncoveredValueWidth,
	UncoveredKVShared, UncoveredIndexer, UncoveredStreams,
	UncoveredRopeAxes,
	UncoveredNonCausal, UncoveredSparseAct, UncoveredCompressed,
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
	case m.blockAct(roles) == 0:
		return UncoveredUngated
	case !c.MLA() && c.QLoraRank != 0 && !c.DSV4():
		return UncoveredMLA
	// A value head of its own width: HeadDimV against the model's head
	// width (a sliding layer's narrower heads narrow k and v alike, KVDimAt).
	case !c.MLA() && c.HeadDimV != 0 && c.HeadDimV != c.HeadDim:
		return UncoveredValueWidth
	// A KV-sharing block and Gemma 4's v-is-k blocks are the graph's
	// (Dense.KVShared, VFromK); an absent v anywhere else is not.
	case c.LayerKind(li).Attends() && !c.MLA() && !c.DSV4() && !c.KVShared(li) && !hasRole(roles, jlm.RoleAttnV) &&
		!(c.VNorm && hasRole(roles, jlm.RoleAttnK)):
		return UncoveredKVShared
	case c.IdxHeads != 0 && !c.MSA() && !c.Q4E() && !c.DSV4() && !(c.MLA() && c.Indexer()):
		return UncoveredIndexer
	case c.streamHead() && !c.Q4E() && c.AltUp == 0 && !c.DSV4() && !c.ResAttn():
		return UncoveredStreams
	case !c.DSV4() && (len(c.CompKinds) > 0 || c.CompRateCSA != 0 || c.CompRateHCA != 0 || c.OGroups != 0 || c.OLoraRank != 0):
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
		if int(e.Block) != li || e.Role.Vision() {
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

// blockFacts is what the container states about text block li: the dense
// configuration and the tensors stored for the block by role.
func (m *Model) blockFacts(li int) (graph.Facts, error) {
	c := m.Cfg
	if why := m.uncovered(li); why != "" {
		return graph.Facts{}, fmt.Errorf("model: block %d of %s is not covered by the block graph: %s", li, c.Arch, why)
	}
	layout := graph.RopeNorm
	switch {
	case c.RopeXD:
		layout = graph.RopeXD
	case c.RopeNeox:
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
		PostNorm: has(jlm.RolePostAttnNorm), NoPreNorm: !has(jlm.RoleAttnNorm),
		NoFFNNorm: !has(jlm.RoleFFNNorm), NoRope: !c.RopeAt(li)}
	d.QKVBias, d.Sinks = has(jlm.RoleAttnQBias), has(jlm.RoleAttnSinks)
	d.OutGate, d.VNorm = c.AttnOutGate, c.VNorm
	d.VFromK = c.VNorm && has(jlm.RoleAttnK) && !has(jlm.RoleAttnV)
	if c.KVShared(li) {
		d.KVShared, d.KVFrom = true, c.KVSource(li)
	}
	d.BidirOff = c.BidirSWA && !c.SWA(li)
	d.OutScale = has(jlm.RoleLayerOutScale)
	d.DenseMix = c.DenseMoE && c.MoEAt(li)
	if has(jlm.RolePLEGate) {
		d.PLEDim = c.PLEDim
	}
	if c.ResidualScale != 1 {
		d.ResidScale = c.ResidualScale
	}
	d.QKNormPost, d.QKL2Norm = c.QKNormPost, c.QKL2Norm && c.RopeAt(li)
	// A recurrent mixer in place of the attention, binding the mixer's
	// tensors the block stores.
	if k := c.LayerKind(li); k.Recurrent() {
		d.Mixer, d.NoFFN, d.MixerAttends = uint8(k), m.layers[li].noFFN, k.Attends()
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
		if c.Indexer() {
			d.MLA.Index = &graph.Index{Heads: c.IdxHeads, HeadDim: c.IdxHeadDim, TopK: c.IdxTopK}
		}
		d.MLA.Gate = has(jlm.RoleAttnGate)
	}
	d.NonCausal = c.NonCausal
	d.SubNorm = has(jlm.RoleAttnSubNorm)
	if li >= c.NLayer {
		for _, r := range []jlm.Role{jlm.RoleNextnEHProj, jlm.RoleNextnENorm, jlm.RoleNextnHNorm, jlm.RoleNextnHeadNorm,
			jlm.RoleNextnHead, jlm.RoleNextnEmbd, jlm.RoleHCHeadNorm, jlm.RoleHCHeadDown, jlm.RoleHCHeadUp} {
			if sh, ok := inv[r]; ok {
				d.PredictW = append(d.PredictW, graph.Binding{Role: r, K: sh.K, Rows: sh.Rows, Sheets: sh.Sheets})
			}
		}
	}
	if c.MSAAt(li) {
		d.Select = &graph.Select{Heads: c.IdxHeads, HeadDim: c.IdxHeadDim, TopK: c.IdxTopK, Block: c.IdxBlock, Local: c.IdxLocal}
	}
	if c.QSAAt(li) {
		d.Select = &graph.Select{Heads: c.IdxHeads, HeadDim: c.IdxHeadDim, TopK: c.IdxTopK, Block: c.QSARatio, Pooled: true}
	}
	if c.AltUp != 0 {
		d.AltUp = &graph.AltUp{N: c.Streams(), LaurelRank: inv[jlm.RoleLaurelL].Rows}
	}
	if li < c.NSparse {
		d.SparseStd = float64(c.SparseStd)
	}
	if c.DSV4() {
		bind := func(rs ...jlm.Role) []graph.Binding {
			var ws []graph.Binding
			for _, r := range rs {
				if sh, ok := inv[r]; ok {
					ws = append(ws, graph.Binding{Role: r, K: sh.K, Rows: sh.Rows, Sheets: sh.Sheets})
				}
			}
			return ws
		}
		d.Streams = &graph.Streams{N: c.HCMult, Sinkhorn: true, Iters: c.HCIters,
			AttnW: bind(jlm.RoleHCAttnFn, jlm.RoleHCAttnBase, jlm.RoleHCAttnScale),
			FFNW:  bind(jlm.RoleHCFFNFn, jlm.RoleHCFFNBase, jlm.RoleHCFFNScale)}
		d.Compressed = &graph.Compressed{Kind: uint8(c.CompAt(li)), Rate: c.CompRateAt(li), OGroups: c.OGroups,
			W: bind(jlm.RoleAttnQA, jlm.RoleAttnQANorm, jlm.RoleAttnQB, jlm.RoleAttnK, jlm.RoleAttnKVA, jlm.RoleAttnKVANorm,
				jlm.RoleAttnSinks, jlm.RoleAttnOutA, jlm.RoleAttnOut, jlm.RoleCompKV, jlm.RoleCompGate, jlm.RoleCompAPE,
				jlm.RoleCompNorm, jlm.RoleIdxQB, jlm.RoleIdxProj, jlm.RoleIdxCompKV, jlm.RoleIdxCompGate,
				jlm.RoleIdxCompAPE, jlm.RoleIdxCompNorm)}
	}
	if c.ResAttn() {
		k, _ := c.resCheckpoint(li)
		d.Streams = &graph.Streams{N: c.Streams(), Bank: true, BankAttn: c.resBank(li, false), BankFFN: c.resBank(li, true),
			Checkpoint: k}
	}
	if c.Q4E() {
		d.Streams = &graph.Streams{N: c.HCMult, LowRank: c.HCLowRank}
		if c.NGramAt(li) {
			d.NGram = true
			for _, r := range []jlm.Role{jlm.RoleNGramKey, jlm.RoleNGramValue, jlm.RoleNGramNormKey,
				jlm.RoleNGramNormQuery, jlm.RoleNGramNormConv, jlm.RoleNGramConv} {
				sh := inv[r]
				d.NGramW = append(d.NGramW, graph.Binding{Role: r, K: sh.K, Rows: sh.Rows, Sheets: sh.Sheets})
			}
		}
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

// denseLowering is a verified graph lowered for the host: the block's op
// sequence matched against the host operations each op runs in. qk says the matched sequence holds the per-head q/k norms, which
// runPrep applies with the rotary and the cache write.
type denseLowering struct {
	g  *graph.Graph
	qk bool
	// qkPost: q's norm after the rotary (Hunyuan's); qkL2: weightless head
	// norms on q and k after it (Llama 4's). runPrep applies both.
	qkPost, qkL2 bool
	// prep is the block's attention prologue, its ops from the projections
	// to the attention lowered one by one (lowerPrep).
	prep []prepStep
	// mix says the block's FFN is a routed mixture (graph.Dense.Mixture),
	// which the host runs through the mixture's operations (moe, moeExpertsRows).
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
	// vFromK: v is a copy of k's projection (graph.Dense.VFromK);
	// kvShared: no k or v, the attention reads another block's history.
	vFromK, kvShared bool
	// ple: a per-layer embedding after the FFN's add (pleApply).
	ple bool
	// denseMix: Gemma 4's mixture block (denseMoE), its post-FFN norm
	// inside it.
	denseMix bool
	// mixAttn: a recurrent mixer beside the attention over the same normed
	// row (Falcon-H1's), its output (in hf/bhf) added to the attention's.
	mixAttn bool
	// index: an attention indexer selects the positions attended (OpIndex);
	// msa: over a GQA row, its key one more cached head (msaKey, msaQuery,
	// and attendRow's block mask, msaMask).
	// pooled: QSA's selection (qsaProject, qsaMask) in place of MSA's.
	index, msa, pooled bool
	// ngram: the n-gram embedding added into the streams (ngramApply);
	// hc: the residual is several streams, each sublayer reading its mix
	// (hcMix) and adding through its placement (hcPlace).
	ngram, hc bool
	// alt: AltUp's streams (altPredict, altCorrect); laurel: LAuReL beside
	// the attention; sparse: the FFN gate's gaussian top-k (gaussTopK).
	alt, laurel, sparse bool
	// ds4: the compressed attention over Sinkhorn-mixed streams; the block
	// runs as ds4Block does (hcPre, ds4Attn, hcPost, ds4FFN).
	ds4 bool
	// bank: the residual attention's checkpoint streams (k3AttnIn,
	// k3Resid, k3FFNIn); latent: the latent mixture (k3MoE).
	bank, latent bool
	// mlaGate: Kimi-K3's MLA output gate.
	mlaGate bool
	// predict: a prediction block, its prologue the draft's (draftRows).
	predict bool
	// subNorm: BitNet's norms before wo and before down.
	subNorm bool
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
	// A prediction block's prologue, which the draft runs before the step
	// (draftRows): the block's input is its output.
	lw.predict = take(graph.OpPredict, 0)
	// The n-gram embedding into the streams, then the attention's read of
	// them (ngramApply, hcMix).
	lw.ngram = take(graph.OpNGram, 0)
	lw.hc = take(graph.OpStreamMix, jlm.RoleHCAttnNorm) || take(graph.OpStreamMix, jlm.RoleHCAttnFn)
	// The residual attention's read of the bank (k3AttnIn).
	lw.bank = take(graph.OpStreamMix, jlm.RoleAttnResScore)
	// AltUp's prediction: the block runs on the active stream (altPredict).
	lw.alt = take(graph.OpAltPredict, jlm.RoleAltUpRouterNorm)
	// resid takes a sublayer's residual add: an add, or with streams the
	// placement (hcPlace).
	resid := func() error {
		if lw.hc || lw.bank {
			return run([]step{{graph.OpStreamPlace, 0}})
		}
		return run([]step{{graph.OpAdd, 0}})
	}
	attnIn, ffnPre := g.In[0], graph.ValueID(-1)
	if lw.predict {
		attnIn = ops[0].Out[0]
	}
	switch {
	case lw.hc, lw.bank:
		attnIn = last()
	case lw.alt:
		attnIn = ops[i-1].Out[1]
	}
	if preNorm(jlm.RoleAttnNorm) {
		attnIn = last()
	}
	// LAuReL's branch of the normed row (laurel), joined after the
	// attention's add (laurelJoin).
	lw.laurel = take(graph.OpLaurel, jlm.RoleLaurelL)
	// A parallel block's FFN norm, over the block input, before attention.
	if preNorm(jlm.RoleFFNNorm) {
		ffnPre = last()
	}
	qa := false
	// A recurrent mixer beside the attention (Falcon-H1's): the mixer, then
	// the attention below, the two summed before the residual add.
	if i+1 < len(ops) && ops[i].Kind == graph.OpRecurrent && ops[i+1].Kind != graph.OpAdd &&
		ops[i+1].Kind != graph.OpStreamPlace {
		i++
		lw.mixAttn = true
	}
	if take(graph.OpCompressedAttend, 0) {
		// The compressed attention and its output projection (ds4Attn),
		// placed into the streams.
		lw.ds4 = true
		if err := resid(); err != nil {
			return nil, err
		}
	} else if take(graph.OpRecurrent, 0) {
		// A recurrent mixer in place of the attention (linearAttn), added
		// into the residual; a mixer-only layer ends there.
		lw.mixer = true
		if err := resid(); err != nil {
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
		if err := run([]step{{graph.OpLatentAppend, 0}}); err != nil {
			return nil, err
		}
		// DeepSeek V3.2's indexer: mlaProject computes its key and query,
		// attendRow masks the scores to its selection.
		lw.index = take(graph.OpIndex, jlm.RoleIdxQB)
		if err := run([]step{{graph.OpAttendLatent, 0}}); err != nil {
			return nil, err
		}
		// Kimi-K3's output gate (mlaProject takes it, mlaOutProject applies
		// it).
		if lw.mlaGate = take(graph.OpMatVec, jlm.RoleAttnGate); lw.mlaGate && !take(graph.OpActMul, 0) {
			return nil, fail("act_mul of the MLA output gate")
		}
	} else {
		// q, then k and v; k alone where v is k's projection (VFromK);
		// neither on a KV-sharing block, which attends to another block's
		// history.
		if err := run([]step{{graph.OpMatVec, jlm.RoleAttnQ}}); err != nil {
			return nil, err
		}
		qkv := [3]graph.ValueID{ops[i-1].Out[0], -1, -1}
		if take(graph.OpMatVec, jlm.RoleAttnK) {
			qkv[1], qkv[2] = ops[i-1].Out[0], ops[i-1].Out[0]
			if take(graph.OpMatVec, jlm.RoleAttnV) {
				qkv[2] = ops[i-1].Out[0]
			} else {
				lw.vFromK = true
			}
		} else {
			lw.kvShared = true
		}
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
		// (OLMoE): runPrep applies either, by QKNormShapeAt.
		for _, k := range []graph.OpKind{graph.OpHeadNorm, graph.OpNorm} {
			if !lw.qk && take(k, jlm.RoleAttnQNorm) {
				if !lw.kvShared && !take(k, jlm.RoleAttnKNorm) {
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
			// k's after q's; a KV-sharing block turns q alone, the history's
			// keys turned by their own block.
			if !lw.kvShared {
				if err := run([]step{{graph.OpRope, 0}}); err != nil {
					return nil, err
				}
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
		switch {
		case lw.qkPost && !take(graph.OpHeadNorm, jlm.RoleAttnQNorm):
			return nil, fail("head_norm of attn_q_norm after the rotary")
		case !lw.qk && take(graph.OpHeadNorm, jlm.RoleAttnQNorm):
			// XD-RoPE's: both norms after the rotary.
			if !lw.kvShared && !take(graph.OpHeadNorm, jlm.RoleAttnKNorm) {
				return nil, fail("head_norm of attn_k_norm after the rotary")
			}
			lw.qk, lw.qkPost = true, true
		}
		// Gemma 4's weightless norm over v.
		if i < len(ops) && ops[i].Kind == graph.OpHeadNorm && len(ops[i].Weight) == 0 {
			i++
		}
		if !lw.kvShared {
			if err := run([]step{{graph.OpKVAppend, 0}}); err != nil {
				return nil, err
			}
		}
		prep, err := lowerPrep(g, prep0, i, qkv)
		if err != nil {
			return nil, err
		}
		lw.prep = prep
		if lw.msa = take(graph.OpIndex, jlm.RoleIdxQ); lw.msa {
			lw.pooled = ops[i-1].Attr.Pooled
		}
		if err := run([]step{{graph.OpAttend, 0}}); err != nil {
			return nil, err
		}
		if lw.outGate && !take(graph.OpActMul, 0) {
			return nil, fail("act_mul of the output gate")
		}
	}
	if !lw.mixer && !lw.ds4 {
		// BitNet's norm over the attention output, before wo.
		lw.subNorm = take(graph.OpNorm, jlm.RoleAttnSubNorm)
		if err := run([]step{{graph.OpMatVec, jlm.RoleAttnOut}}); err != nil {
			return nil, err
		}
		if lw.mixAttn {
			if err := run([]step{{graph.OpAdd, 0}}); err != nil {
				return nil, err
			}
		}
		lw.postAttn = take(graph.OpNorm, jlm.RolePostAttnNorm)
		if err := resid(); err != nil {
			return nil, err
		}
		if lw.laurel && !take(graph.OpAdd, 0) {
			return nil, fail("add of LAuReL's join")
		}
	}
	// The FFN's input, by the value its first op reads: its own norm over
	// the residual (or the residual itself, with no pre-norm), or a parallel
	// block's -- its own norm of the block input, or the attention's.
	ffnIn := last()
	switch {
	case lw.bank:
		if !take(graph.OpStreamMix, jlm.RoleFFNResScore) || !take(graph.OpNorm, jlm.RoleFFNNorm) {
			return nil, fail("the FFN's read of the bank and its norm")
		}
		ffnIn = last()
	case lw.hc:
		if !take(graph.OpStreamMix, jlm.RoleHCFFNNorm) && !take(graph.OpStreamMix, jlm.RoleHCFFNFn) {
			return nil, fail("the FFN's stream mix")
		}
		ffnIn = last()
		// The Sinkhorn form's norm after the mix (hcPre).
		if ops[i-1].Attr.Sinkhorn {
			if !take(graph.OpNorm, jlm.RoleFFNNorm) {
				return nil, fail("norm of the FFN's mixed row")
			}
			ffnIn = last()
		}
	case ffnPre < 0 && preNorm(jlm.RoleFFNNorm):
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
	if take(graph.OpMatVec, jlm.RoleShExpGate) {
		// Gemma 4's mixture block, as denseMoE computes it: the dense MLP
		// and its post-norm, the router's and the experts' norms of the
		// residual, the mixture, its post-norm, and the two summed.
		lw.denseMix = true
		if err := run([]step{{graph.OpMatVec, jlm.RoleShExpUp}, {graph.OpActMul, 0}, {graph.OpMatVec, jlm.RoleShExpDown},
			{graph.OpNorm, jlm.RolePostFFNNorm1}, {graph.OpNorm, jlm.RoleRouterNorm}, {graph.OpNorm, jlm.RoleFFNNorm2},
			{graph.OpRoute, jlm.RoleRouter}, {graph.OpGather, 0},
			{graph.OpExpertMatVec, jlm.RoleExpGateBank}, {graph.OpExpertMatVec, jlm.RoleExpUpBank},
			{graph.OpActMul, 0}, {graph.OpExpertMatVec, jlm.RoleExpDownBank}, {graph.OpCombine, 0},
			{graph.OpNorm, jlm.RolePostFFNNorm2}, {graph.OpAdd, 0}}); err != nil {
			return nil, err
		}
		if !take(graph.OpNorm, jlm.RolePostFFNNorm) {
			return nil, fail("norm of the dense mixture's sum")
		}
	} else if i < len(ops) && ops[i].Kind == graph.OpRoute {
		// The routed mixture, as moe() computes it: route, gather, the
		// experts' gate, up, act_mul and down, the combine, then the shared
		// expert [gate scalar] added, then the residual add.
		// Ungated experts (Nemotron 3's): up, act, down, the shared
		// expert's alike.
		lw.mix = true
		if err := run([]step{{graph.OpRoute, jlm.RoleRouter}}); err != nil {
			return nil, err
		}
		// A latent mixture's routed_down (k3MoE).
		lw.latent = take(graph.OpMatVec, jlm.RoleFFNRoutedDown)
		if err := run([]step{{graph.OpGather, 0}}); err != nil {
			return nil, err
		}
		ffn := []step{{graph.OpExpertMatVec, jlm.RoleExpGateBank}, {graph.OpExpertMatVec, jlm.RoleExpUpBank}, {graph.OpActMul, 0}}
		if !take(graph.OpExpertMatVec, jlm.RoleExpGateBank) {
			ffn = []step{{graph.OpExpertMatVec, jlm.RoleExpUpBank}, {graph.OpAct, 0}}
		} else {
			ffn = ffn[1:]
		}
		if err := run(append(ffn, step{graph.OpExpertMatVec, jlm.RoleExpDownBank}, step{graph.OpCombine, 0})); err != nil {
			return nil, err
		}
		if lw.latent {
			take(graph.OpNorm, jlm.RoleFFNRoutedNorm)
			if err := run([]step{{graph.OpMatVec, jlm.RoleFFNRoutedUp}}); err != nil {
				return nil, err
			}
		}
		sh := take(graph.OpMatVec, jlm.RoleShExpGate)
		if sh {
			if err := run([]step{{graph.OpMatVec, jlm.RoleShExpUp}, {graph.OpActMul, 0}}); err != nil {
				return nil, err
			}
		} else if sh = take(graph.OpMatVec, jlm.RoleShExpUp); sh {
			if err := run([]step{{graph.OpAct, 0}}); err != nil {
				return nil, err
			}
		}
		if sh {
			if err := run([]step{{graph.OpMatVec, jlm.RoleShExpDown}}); err != nil {
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
	} else {
		// gate, up, [gauss_top_k of the gate (gaussTopK),] act_mul, down
		if err := run([]step{{graph.OpMatVec, jlm.RoleFFNGate}, {graph.OpMatVec, jlm.RoleFFNUp}}); err != nil {
			return nil, err
		}
		lw.sparse = take(graph.OpGaussTopK, 0)
		if err := run([]step{{graph.OpActMul, 0}}); err != nil {
			return nil, err
		}
		if lw.subNorm != take(graph.OpNorm, jlm.RoleFFNSubNorm) {
			return nil, fail("BitNet's two sub-norms, or neither")
		}
		if err := run([]step{{graph.OpMatVec, jlm.RoleFFNDown}}); err != nil {
			return nil, err
		}
	}
	lw.postFFN = take(graph.OpNorm, jlm.RolePostFFNNorm)
	if lw.mix && lw.postFFN {
		return nil, fail("a mixture with no norm after it (moe adds into the residual itself)")
	}
	if err := resid(); err != nil {
		return nil, err
	}
	// AltUp's correction, the per-layer input inside it (altCorrect).
	if lw.alt {
		if len(g.In) != 2 || !take(graph.OpAltCorrect, jlm.RoleAltUpRouterNorm) {
			return nil, fail("alt_correct of the block's output and the per-layer input")
		}
	}
	// Gemma 4's per-layer embedding (pleApply): gate, act_mul with the
	// block's second input, projection, norm, add.
	if take(graph.OpMatVec, jlm.RolePLEGate) {
		if len(g.In) != 2 || ops[i].Kind != graph.OpActMul || ops[i].In[1] != g.In[1] {
			return nil, fail("act_mul of the per-layer input")
		}
		if err := run([]step{{graph.OpActMul, 0}, {graph.OpMatVec, jlm.RolePLEProj}, {graph.OpNorm, jlm.RolePLEPostNorm},
			{graph.OpAdd, 0}}); err != nil {
			return nil, err
		}
		if lw.mix {
			return nil, fail("a per-layer embedding after a mixture (moe adds into the residual itself)")
		}
		lw.ple = true
	}
	// Gemma 4's output scalar (layerOutScale).
	lw.outScale = take(graph.OpScale, jlm.RoleLayerOutScale)
	if i != len(ops) {
		return nil, fail("the end of the block")
	}
	return lw, nil
}

// buildGraphs builds, verifies and lowers every text block's graph, the
// trunk's and the prediction blocks'. A block the op language does not state
// refuses the model by its named reason, and a graph that fails to verify or
// lower refuses it as the bug it is: the graph is the only body.
func (m *Model) buildGraphs() error {
	// The trunk's blocks and the prediction blocks after it.
	m.graphs = make([]*denseLowering, m.Cfg.NLayer+m.Cfg.NMTP)
	for li := range m.graphs {
		if why := m.uncovered(li); why != "" {
			return fmt.Errorf("model: block %d of %s is not covered by the block graph: %s", li, m.Cfg.Arch, why)
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
			return fmt.Errorf("model: block %d of %s is not covered by the block graph: %s (%v)", li, m.Cfg.Arch,
				UncoveredUnboundTensor, u)
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
			return fmt.Errorf("model: block %d graph's q/k norms (%v, post %v, weightless %v) are not the configuration's",
				li, lw.qk, lw.qkPost, lw.qkL2)
		}
		if lw.index != (lw.mla && m.Cfg.Indexer()) {
			return fmt.Errorf("model: block %d graph's indexer is %v, the configuration's is not", li, lw.index)
		}
		if lw.alt != (m.Cfg.AltUp != 0) || lw.laurel != lw.alt || lw.sparse != (li < m.Cfg.NSparse) {
			return fmt.Errorf("model: block %d graph's AltUp (%v, LAuReL %v, sparse %v) is not the configuration's",
				li, lw.alt, lw.laurel, lw.sparse)
		}
		if lw.bank != m.Cfg.ResAttn() || lw.latent != (lw.mix && m.Cfg.ExpertLatent != 0) {
			return fmt.Errorf("model: block %d graph's residual attention (%v) or latent mixture (%v) is not the configuration's",
				li, lw.bank, lw.latent)
		}
		if lw.predict != (li >= m.Cfg.NLayer) {
			return fmt.Errorf("model: block %d graph's prediction prologue is %v, the block's place is not", li, lw.predict)
		}
		if lw.mlaGate != (m.layers[li].mlaGate.rows != 0) {
			return fmt.Errorf("model: block %d graph's MLA output gate is %v, the layer's is not", li, lw.mlaGate)
		}
		if lw.ds4 != m.Cfg.DSV4() {
			return fmt.Errorf("model: block %d graph's compressed attention is %v, the configuration's is not", li, lw.ds4)
		}
		if lw.hc != (m.Cfg.Q4E() || m.Cfg.DSV4()) || lw.ngram != m.Cfg.NGramAt(li) {
			return fmt.Errorf("model: block %d graph's streams (%v, n-gram %v) are not the configuration's", li, lw.hc, lw.ngram)
		}
		if lw.msa != (m.Cfg.MSAAt(li) || m.Cfg.QSAAt(li)) {
			return fmt.Errorf("model: block %d graph's block selection is %v, the configuration's is not", li, lw.msa)
		}
		if lw.ple != (m.Cfg.PLEDim != 0 && m.Cfg.AltUp == 0) {
			return fmt.Errorf("model: block %d graph's per-layer embedding is %v, the configuration's is not", li, lw.ple)
		}
		if !lw.mixer && !lw.mla && (lw.vFromK != m.layers[li].vFromK || lw.kvShared != m.Cfg.KVShared(li)) {
			return fmt.Errorf("model: block %d graph's v-from-k %v and KV sharing %v are not the layer's", li, lw.vFromK, lw.kvShared)
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

// BlockRewrites is every rewrite's decision on text block li's graph, legal
// or refused with its reason.
func (m *Model) BlockRewrites(li int) []graph.Decision {
	if li < 0 || li >= len(m.graphs) || m.graphs[li] == nil {
		return nil
	}
	return m.graphs[li].rewrites
}

// BlockGraph is text block li's verified graph (nil past the text blocks).
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
// the block graph's host lowering, Device on a device, whose plan is lowered
// from the graph (DeviceGraphed).
type BlockPath struct {
	Graph, Device int
	DeviceGraphed bool
}

// BlockPaths is every text block's path record in the State.
func (s *State) BlockPaths() []BlockPath {
	out := make([]BlockPath, s.m.Cfg.NLayer)
	for li := range out {
		if li < len(s.paths) {
			out[li] = s.paths[li]
		}
		out[li].DeviceGraphed = out[li].Device > 0 && s.m.BlockGraph(li) != nil
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

// devRan records that blocks lo..hi-1 ran on a device.
func (s *State) devRan(lo, hi int) {
	for li := lo; li < hi; li++ {
		s.pathOf(li).Device++
	}
}

// graphBlock runs block li's host decode through its lowering: each op's
// host routine, in the graph's order.
func (s *State) graphBlock(li int, l *layer, lw *denseLowering) error {
	c, m := s.c, s.m
	s.graphRuns++
	s.pathOf(li).Graph++
	if lw.ds4 {
		// stream_mix, norm, compressed_attend, stream_place, then the FFN's
		// stream_mix, norm, the mixture and stream_place: ds4Block.
		s.ds.rslot[0], s.ds.rpos[0], s.ds.rid[0] = 0, s.pos, s.tok
		if err := s.ds4Block(li, l, s.xa, 1); err != nil {
			return err
		}
		m.trace(li, "ffn_resid", s.x)
		return nil
	}
	// [ngram,] [stream_mix,] norm(x) -> attn_in [, a parallel block's FFN
	// input]
	attnIn := s.x
	if lw.ngram {
		tok := s.tok
		if err := s.ngramApply(li, l, s.xa, 1, nil, func(int) (*State, int) { return s, 0 },
			func(int) int32 { return tok }); err != nil {
			return err
		}
	}
	if lw.hc {
		if err := s.hcMix(s.xa, 1, &l.q4e.attn); err != nil {
			return err
		}
		attnIn = s.q4.mix[:c.NEmbd]
	}
	// stream_mix over the bank (k3AttnIn): the checkpoints banked so far
	// and the running residual, a checkpoint block banking its input.
	if lw.bank {
		in, err := s.k3AttnIn(li, l, s.xa, 1)
		if err != nil {
			return err
		}
		attnIn = in
	}
	// alt_predict: the active stream into stream 0, s.x.
	if lw.alt {
		if err := s.altPredict(l, s.xa, s.altPred, 1); err != nil {
			return err
		}
	}
	s.norm(s.h, attnIn, l.attnNorm, l.attnNormB)
	if lw.laurel {
		if err := s.laurel(l, s.h, s.altLaur); err != nil {
			return err
		}
	}
	if lw.par {
		if lw.shared {
			copy(s.hf, s.h)
		} else {
			s.norm(s.hf, s.x, l.ffnNorm, l.ffnNormB)
		}
	}
	s.jit.NewInput()
	m.trace(li, "attn_norm", s.h)
	if lw.mixAttn {
		if err := s.linearAttn(li, l, 0, s.h, s.hf); err != nil {
			return err
		}
		m.trace(li, "ssm_out", s.hf)
	}
	if lw.mixer {
		// The recurrent mixer in place of the attention, added into the
		// residual; a mixer-only layer ends there.
		if err := s.linearAttn(li, l, 0, s.h, s.h); err != nil {
			return err
		}
		m.trace(li, "ssm_out", s.h)
		s.graphResid(lw, li, s.h, true)
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
		grouped := !lw.outGate && !lw.vFromK && !lw.kvShared &&
			s.groupOf(s.h, [nn.MaxGroup]*tensor{&l.wq, &l.wk, &l.wv}, [nn.MaxGroup][]float32{s.q, s.kc, s.vc}, 3)
		if grouped {
			s.addBiasRows(s.q, l.bq, 1)
		} else {
			if _, err := s.projectQ(l, s.q, s.ogate, s.h, 1, false); err != nil {
				return err
			}
			if !lw.kvShared {
				if err := s.mv(s.kc, l.wk, s.h); err != nil {
					return err
				}
			}
		}
		if !lw.kvShared {
			s.addBias(s.kc, l.bk)
			switch {
			case lw.vFromK:
				copy(s.vc[:kvd], s.kc[:kvd])
			case !grouped:
				if err := s.mv(s.vc, l.wv, s.h); err != nil {
					return err
				}
			}
			s.addBias(s.vc, l.bv)
		}
		// index's key and query (MiniMax Sparse Attention's, or QSA's raw
		// key padded to a head): the key rides the appended row as one more
		// kv head, its value zero (msa.go, qsa.go).
		kvr := kvd
		if lw.msa {
			kvr = c.KVRowAt(li)
			cs := s.ropeTable(li, s.cs, s.csSWA, 0)
			if lw.pooled {
				if err := s.qsaProject(s.kc[kvd:kvr], s.qsa().q, l, s.h, cs); err != nil {
					return err
				}
			} else {
				if err := s.msaKey(s.kc[kvd:kvr], l, s.h, cs); err != nil {
					return err
				}
				if err := s.msaQuery(s.msaQ, l, s.h, cs); err != nil {
					return err
				}
			}
			clear(s.vc[kvd:kvr])
		}
		// [head_norm q, head_norm k,] rope q, rope k, kv_append: runPrep
		s.runPrep(l, lw.prep, s.q[:c.QDimAt(li)], s.kc[:kvr], s.vc[:kvr], s.ropeTable(li, s.cs, s.csSWA, 0), li, 0, s.pos)
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
		// BitNet's norm over the attention output, before wo.
		if lw.subNorm && l.attnSubNorm != nil {
			s.rmsnorm(src[:qd], src[:qd], l.attnSubNorm, c.RMSEps)
			s.jit.NewInput()
		}
		plain := s.residFold && l.bo == nil && !lw.postAttn && !lw.mixAttn && !lw.hc && !lw.laurel
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
		if lw.mixAttn {
			s.addInto(s.h, s.hf)
		}
	}
	// [norm(o)] before the residual add
	if lw.postAttn {
		s.rmsnorm(s.h, s.h, l.postAttnNorm, c.RMSEps)
	}
	if !attnResid {
		s.graphResid(lw, li, s.h, true)
	}
	if lw.laurel {
		s.laurelJoin(s.x, s.altLaur)
	}
	m.trace(li, "attn_resid", s.x)
	return s.graphFFN(li, l, lw)
}

// selectRows projects every row's indexer key and query for a selecting
// block: MSA's (msaProjectRows) or QSA's (qsaProjectRows).
func (s *State) selectRows(lw *denseLowering, l *layer, n int, cs, csSWA []float32, li int) error {
	switch {
	case !lw.msa:
		return nil
	case lw.pooled:
		return s.qsaProjectRows(l, n, cs, csSWA, li)
	}
	return s.msaProjectRows(l, n, cs, csSWA, li)
}

// selectRowOf is row i's cache vectors for a selecting block, its indexer
// key one more head (msaRowOf, qsaRowOf).
func (s *State) selectRowOf(lw *denseLowering, li, i int, k, v []float32) ([]float32, []float32) {
	if lw.pooled {
		return s.qsaRowOf(li, i, k, v)
	}
	return s.msaRowOf(li, i, k, v)
}

// altPredictRows is AltUp's prologue over n rows in s.bx: the active stream
// into stream 0, where the block runs (altPredict).
func (s *State) altPredictRows(l *layer, lw *denseLowering, n int) error {
	if !lw.alt {
		return nil
	}
	s.growAltUpRows(n)
	return s.altPredict(l, s.bx[:n*s.c.ResidW()], s.baltPred, n)
}

// laurelRows is LAuReL's branch of each of n rows' normed input in s.bh,
// into baltLaur (laurel).
func (s *State) laurelRows(l *layer, lw *denseLowering, n int) error {
	if !lw.laurel {
		return nil
	}
	ne := s.c.NEmbd
	for r := 0; r < n; r++ {
		if err := s.laurel(l, s.bh[r*ne:(r+1)*ne], s.baltLaur[r*ne:(r+1)*ne]); err != nil {
			return err
		}
	}
	return nil
}

// rowsResid adds n rows' sublayer output y into the residual rows in s.bx:
// an add, or with streams the placement (hcPlace).
func (s *State) rowsResid(lw *denseLowering, li int, y []float32, n int, attn bool) {
	if lw.bank && attn {
		// A checkpoint block restarts the running residual (k3Resid).
		s.k3Resid(li, s.bx, y, n)
		return
	}
	if lw.hc && !lw.bank {
		s.hcPlace(s.bx[:n*s.c.ResidW()], n, y)
		return
	}
	ne := s.c.NEmbd
	s.addInto(s.bx[:n*ne], y[:n*ne])
}

// mixOp is the index of the block's first stream mix.
func (lw *denseLowering) mixOp() int {
	for i, o := range lw.g.Ops {
		if o.Kind == graph.OpStreamMix || o.Kind == graph.OpAltPredict {
			return i
		}
	}
	return -1
}

// graphResid adds a decode sublayer's output y into the residual: an add,
// or with streams the placement at the weights its mix left (hcPlace).
func (s *State) graphResid(lw *denseLowering, li int, y []float32, attn bool) {
	if lw.bank && attn {
		// A checkpoint block restarts the running residual (k3Resid).
		s.k3Resid(li, s.x, y, 1)
		return
	}
	if lw.hc && !lw.bank {
		s.hcPlace(s.xa, 1, y)
		return
	}
	s.addInto(s.x, y)
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
	switch {
	case lw.par:
		ffnIn = s.hf
	case lw.bank:
		// stream_mix over the bank, its own checkpoint included, then the
		// norm (k3FFNIn).
		in, err := s.k3FFNIn(li, l, s.xa, 1)
		if err != nil {
			return err
		}
		s.norm(s.h, in, l.ffnNorm, l.ffnNormB)
	case lw.hc:
		// stream_mix: the FFN reads the mix with no norm.
		if err := s.hcMix(s.xa, 1, &l.q4e.ffn); err != nil {
			return err
		}
		copy(s.h, s.q4.mix[:c.NEmbd])
	default:
		s.norm(s.h, s.x, l.ffnNorm, l.ffnNormB)
	}
	s.jit.NewInput()
	if lw.denseMix {
		// dense MLP, router and expert norms, the mixture, the norms and
		// the sum, added into the residual: denseMoE.
		if err := s.denseMoE(li, l, s.h, s.x); err != nil {
			return err
		}
		if lw.ple {
			if err := s.pleApply(li, l, s.x, s.ple); err != nil {
				return err
			}
		}
		s.layerOutScale(l, s.x)
		m.trace(li, "ffn_resid", s.x)
		return nil
	}
	if lw.mix {
		// The mixture's ops, each its own routine, added into the residual
		// as moe adds them: route (the router's
		// matvec and the route kernel); gather, the experts' gate, up,
		// act_mul and down, and combine (one grouped region); [the shared
		// expert's matvecs, act_mul and gate_scalar, and its add].
		if lw.latent {
			// The latent mixture: routed_down, the experts in the latent,
			// the norm and routed_up, the shared experts (k3MoE).
			if err := s.k3MoE(li, l, ffnIn, s.x); err != nil {
				return err
			}
			s.layerOutScale(l, s.x)
			m.trace(li, "ffn_resid", s.x)
			return nil
		}
		// With streams the mixture's sum is placed (stream_place).
		dst := s.x
		if lw.hc && !lw.bank {
			dst = s.q4.out[:c.NEmbd]
			clear(dst)
		}
		if err := s.moeRoute(li, l, ffnIn, ffnIn); err != nil {
			return err
		}
		// gather: the selected experts' pages; then the experts' matvecs,
		// act_mul (fused into down's quantize) and combine.
		if err := s.moeGather(li, l); err != nil {
			return err
		}
		err := s.moeRun(l, ffnIn, dst)
		s.expHold.release()
		if err != nil {
			return err
		}
		if err := s.sharedExpert(l, ffnIn, dst); err != nil {
			return err
		}
		if lw.hc && !lw.bank {
			s.hcPlace(s.xa, 1, dst)
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
		s.graphResid(lw, li, s.h, false)
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
	if s.residFold && !lw.postFFN && !lw.hc {
		s.grp.Resid, s.grp.Alpha = s.x, float32(c.ResidualScale)
	}
	s.aheadNext(li)
	// BitNet's norm over the product sits between act_mul and down, which a
	// group would fuse.
	grouped := !lw.subNorm && s.groupOf(gate, [nn.MaxGroup]*tensor{&l.down}, [nn.MaxGroup][]float32{s.h}, 1)
	ffnResid := grouped && s.grp.Resid != nil
	s.grp.ActUp, s.grp.Resid = nil, nil
	if !grouped {
		s.actmulAll(gate, up, act)
		if lw.subNorm && l.ffnSubNorm != nil {
			s.rmsnorm(gate, gate, l.ffnSubNorm, c.RMSEps)
		}
		s.jit.NewInput()
		if err := s.mv(s.h, l.down, gate); err != nil {
			return err
		}
	}
	if lw.postFFN {
		s.rmsnorm(s.h, s.h, l.postFFNNorm, c.RMSEps)
	}
	if !ffnResid {
		s.graphResid(lw, li, s.h, false)
	}
	if lw.ple {
		if err := s.pleApply(li, l, s.x, s.ple); err != nil {
			return err
		}
	}
	// alt_correct, the per-layer input inside it.
	if lw.alt {
		if err := s.altCorrect(li, l, s.xa, s.altPred, s.ple, 1); err != nil {
			return err
		}
	}
	s.layerOutScale(l, s.x)
	m.trace(li, "ffn_resid", s.x)
	return nil
}

// graphRows runs block li over a host prompt chunk -- rows p0..p0+n-1 of slot,
// in s.bx -- through the graph's lowering. The graph is the one the decode
// lowering matched (lowerDense):
// norm, q/k/v, [the per-head q/k norms,] the rotary and the cache write for
// every row before any row attends, attention, o, the residual add, then the
// FFN norm, gate and up, act_mul, down and the second add.
func (s *State) graphRows(li int, l *layer, lw *denseLowering, slot, base, p0, n int, bidir bool) error {
	c := s.c
	s.graphChunkRuns++
	s.pathOf(li).Graph++
	ne := c.NEmbd
	if lw.ds4 {
		// ds4Block over the chunk, each row its slot and position.
		for i := 0; i < n; i++ {
			s.ds.rslot[i], s.ds.rpos[i] = slot, p0+i
		}
		return s.ds4Block(li, l, s.bx[:n*c.ResidW()], n)
	}
	// [ngram,] [stream_mix,] norm(x) -> attn_in, every row
	tn := s.tick()
	attnIn := s.bx
	if lw.hc {
		x := s.bx[:n*c.ResidW()]
		if lw.ngram {
			if s.rows != nil {
				s.rows.ngBase = base
			}
			if err := s.ngramApply(li, l, x, n, nil, func(int) (*State, int) { return s, slot },
				func(i int) int32 { return s.q4.rid[i] }); err != nil {
				return err
			}
		}
		if err := s.hcMix(x, n, &l.q4e.attn); err != nil {
			return err
		}
		attnIn = s.q4.mix
	}
	if lw.bank {
		in, err := s.k3AttnIn(li, l, s.bx[:n*c.ResidW()], n)
		if err != nil {
			return err
		}
		attnIn = in
	}
	if err := s.altPredictRows(l, lw, n); err != nil {
		return err
	}
	s.batchNormB(s.bh, attnIn, l.attnNorm, l.attnNormB, ne, n)
	if err := s.laurelRows(l, lw, n); err != nil {
		return err
	}
	s.parallelIn(l, lw, n)
	s.tock(opRMSNorm, tn)
	if lw.mixer {
		// The recurrent mixer over the chunk, row by row, into bh.
		if err := s.mixerChunk(li, l, slot, base, s.bh, n); err != nil {
			return err
		}
		return s.rowsResidFFN(li, l, lw, n, false)
	}
	if lw.mixAttn {
		// The mixer beside the attention, into bhf; rowsResidFFN adds it.
		if err := s.mixerChunk(li, l, slot, base, s.bhf, n); err != nil {
			return err
		}
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
	if err := s.kvRows(l, lw, s.bh[:n*ne], n, kvDim); err != nil {
		return err
	}
	// index's keys and queries, every row's (msa.go, qsa.go).
	if err := s.selectRows(lw, l, n, s.bcs, s.bcsSWA, li); err != nil {
		return err
	}
	// [head_norm q, head_norm k,] rope q, rope k, kv_append: every row's,
	// before any row attends (a row attends to the chunk's earlier rows).
	for i := 0; i < n; i++ {
		k, v := s.bk[i*kvDim:(i+1)*kvDim], s.bv[i*kvDim:(i+1)*kvDim]
		if lw.msa {
			k, v = s.selectRowOf(lw, li, i, k, v)
		}
		s.runPrep(l, lw.prep, s.bq[i*qDim:(i+1)*qDim], k, v, s.ropeTable(li, s.bcs, s.bcsSWA, i), li, slot, p0+i)
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
	if lw.subNorm && l.attnSubNorm != nil {
		s.batchNorm(asrc, asrc, l.attnSubNorm, qDim, n)
		s.jit.NewInput()
	}
	if err := s.mm(s.bh, l.wo, asrc[:n*qDim], n); err != nil {
		return err
	}
	s.addBiasRows(s.bh, l.bo, n)
	return s.rowsResidFFN(li, l, lw, n, false)
}

// graphRagged runs block li over a ragged step's rows -- each row its own
// sequence's slot and position, in s.bx -- through the graph's lowering. Every
// row's rotary and cache write
// lands before any row attends, as in graphRows.
func (s *State) graphRagged(li int, l *layer, lw *denseLowering, n int, tokens []int32, seq, pos []int, cs, csSWA []float32, unitHeads, units, astride int, scale float64) error {
	c := s.c
	s.graphRaggedRuns++
	s.pathOf(li).Graph++
	ne := c.NEmbd
	if lw.ds4 {
		// ds4Block over the step, each row its sequence's slot, position
		// and token.
		s.growDS4(n)
		for i := 0; i < n; i++ {
			s.ds.rslot[i], s.ds.rpos[i], s.ds.rid[i] = seq[i], pos[i], tokens[i]
		}
		return s.ds4Block(li, l, s.bx[:n*c.ResidW()], n)
	}
	// [ngram,] [stream_mix,] norm(x) -> attn_in, every row: the n-gram in
	// each sequence's position order.
	t0 := s.tick()
	attnIn := s.bx
	if lw.hc {
		x := s.bx[:n*c.ResidW()]
		if lw.ngram {
			if err := s.ngramApply(li, l, x, n, s.rowCausal[:n], func(i int) (*State, int) { return s.rowOwner(i, seq) },
				func(i int) int32 { return tokens[i] }); err != nil {
				return err
			}
		}
		if err := s.hcMix(x, n, &l.q4e.attn); err != nil {
			return err
		}
		attnIn = s.q4.mix
	}
	if lw.bank {
		in, err := s.k3AttnIn(li, l, s.bx[:n*c.ResidW()], n)
		if err != nil {
			return err
		}
		attnIn = in
	}
	if err := s.altPredictRows(l, lw, n); err != nil {
		return err
	}
	s.batchNormB(s.bh, attnIn, l.attnNorm, l.attnNormB, ne, n)
	if err := s.laurelRows(l, lw, n); err != nil {
		return err
	}
	s.parallelIn(l, lw, n)
	s.tock(opRMSNorm, t0)
	if lw.mixer {
		// Each row steps its own sequence's state, in position order.
		if err := s.mixerRagged(li, l, seq, s.rowCausal[:n], s.bh, n); err != nil {
			return err
		}
		return s.rowsResidFFN(li, l, lw, n, true)
	}
	if lw.mixAttn {
		if err := s.mixerRagged(li, l, seq, s.rowCausal[:n], s.bhf, n); err != nil {
			return err
		}
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
	if err := s.kvRows(l, lw, s.bh, n, kvDim); err != nil {
		return err
	}
	// index's keys and queries, every row's (msa.go, qsa.go).
	if err := s.selectRows(lw, l, n, cs, csSWA, li); err != nil {
		return err
	}
	// [head_norm q, head_norm k,] rope q, rope k, kv_append: each row at its
	// owner's slot and position
	for i := 0; i < n; i++ {
		k, v := s.bk[i*kvDim:(i+1)*kvDim], s.bv[i*kvDim:(i+1)*kvDim]
		if lw.msa {
			k, v = s.selectRowOf(lw, li, i, k, v)
		}
		o, slot := s.rowOwner(i, seq)
		o.runPrep(l, lw.prep, s.bq[i*qDim:(i+1)*qDim], k, v, s.ropeTable(li, cs, csSWA, i), li, slot, pos[i])
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
	if lw.subNorm && l.attnSubNorm != nil {
		s.batchNorm(asrc, asrc, l.attnSubNorm, qDim, n)
		s.jit.NewInput()
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
	if lw.mixAttn {
		s.addInto(s.bh[:n*ne], s.bhf[:n*ne])
	}
	if lw.postAttn {
		tn := s.tick()
		s.batchNorm(s.bh, s.bh, l.postAttnNorm, ne, n)
		s.tock(opRMSNorm, tn)
	}
	tr := s.tick()
	s.rowsResid(lw, li, s.bh, n, true)
	if lw.laurel {
		s.laurelJoin(s.bx[:n*ne], s.baltLaur[:n*ne])
	}
	s.tock(opResid, tr)
	// A mixer-only layer ends at the first residual.
	if lw.noFFN {
		return nil
	}
	// norm -> gate, up -> act_mul -> down, add into the residual
	tn := s.tick()
	ffnIn := s.bh
	switch {
	case lw.par:
		ffnIn = s.bhf
		s.jit.NewInput()
	case lw.bank:
		in, err := s.k3FFNIn(li, l, s.bx[:n*c.ResidW()], n)
		if err != nil {
			return err
		}
		s.batchNormB(s.bh, in, l.ffnNorm, l.ffnNormB, ne, n)
	case lw.hc:
		// stream_mix: the FFN reads the mix with no norm.
		if err := s.hcMix(s.bx[:n*c.ResidW()], n, &l.q4e.ffn); err != nil {
			return err
		}
		copy(s.bh[:n*ne], s.q4.mix[:n*ne])
	default:
		s.batchNormB(s.bh, s.bx, l.ffnNorm, l.ffnNormB, ne, n)
	}
	s.tock(opRMSNorm, tn)
	if lw.denseMix {
		if err := s.denseMoERows(li, l, s.bh, s.bx, n); err != nil {
			return err
		}
		if lw.ple {
			if err := s.pleRows(li, l, s.bx, n); err != nil {
				return err
			}
		}
		s.layerOutScale(l, s.bx[:n*ne])
		return nil
	}
	if lw.mix {
		// The mixture over the rows, op by op as moeExpertsRows runs them: every
		// row's route, then the experts expert-major, then the shared
		// expert per row, into the residual; on a ragged step each row its
		// own sequence's experts.
		if lw.latent {
			if err := s.k3MoEBatch(li, l, ffnIn, s.bx, n); err != nil {
				return err
			}
			s.layerOutScale(l, s.bx[:n*ne])
			return nil
		}
		// With streams the mixture's sum is placed (stream_place).
		dst := s.bx
		if lw.hc && !lw.bank {
			dst = s.q4.out[:n*ne]
			clear(dst)
		}
		if err := s.moeRouteRows(li, l, ffnIn, n); err != nil {
			return err
		}
		if err := s.moeExpertsRows(li, l, ffnIn, dst, n); err != nil {
			return err
		}
		if err := s.sharedExpertRows(l, ffnIn, dst, n); err != nil {
			return err
		}
		if lw.hc && !lw.bank {
			s.hcPlace(s.bx[:n*c.ResidW()], n, dst)
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
		if lw.subNorm && l.ffnSubNorm != nil {
			s.batchNorm(gate, gate, l.ffnSubNorm, ff, n)
			s.jit.NewInput()
		}
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
	s.rowsResid(lw, li, s.bh, n, false)
	s.tock(opResid, tr)
	if lw.ple {
		if err := s.pleRows(li, l, s.bx, n); err != nil {
			return err
		}
	}
	if lw.alt {
		if err := s.altCorrect(li, l, s.bx[:n*c.ResidW()], s.baltPred, s.bple, n); err != nil {
			return err
		}
	}
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
	// The block's row: the input, or a stream's of a residual of several.
	p.NEmbd = g.Values[g.In[0]].Width
	if lw.hc || lw.alt || lw.bank {
		p.NEmbd /= lw.g.Ops[lw.mixOp()].Attr.Streams
	}
	// A recurrent block runs no q/k norm, but the tier bakes one shape per
	// model, so a mixer block keeps planFor's: the attention blocks' shape.
	if !lw.mixer {
		p.QKNorm, p.QKNormPost = false, false
	}
	p.LayerNorm, p.Parallel = false, lw.par
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
			// An ungated mixture's experts' (and shared expert's) act is not
			// the dense FFN's width: the plan keeps planFor's.
			if lw.mix {
				break
			}
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
	graph.ActSwiGLUOAI: nn.ActSwiGLUOAI, graph.ActSwiGLUClamp: nn.ActSwiGLUClamp, graph.ActSitu: nn.ActSitu}

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
	// Opaque is an op the language states whole (graph.Opaque): its inner
	// operations are its routine's, not the graph's, until they are stated.
	Opaque bool
	Device string
}

// hostLowering is the host routine per op kind. It is data, not a branch per
// family: a new op kind with no entry is unlowered and refused at build
// (TestEveryGraphOpHasALowering).
var hostLowering = map[graph.OpKind]struct {
	routine  string
	composed bool
}{
	graph.OpNorm:             {"nn RMSNorm (generated, by width)", true},
	graph.OpMatVec:           {"nn matvec (generated by quant and shape; grouped where a run fuses)", true},
	graph.OpAdd:              {"residual add (generated, or folded into the matvec's rows)", true},
	graph.OpActMul:           {"nn activation-multiply (generated, or fused into down's matvec)", true},
	graph.OpHeadNorm:         {"nn RMSNorm per head or over the projection (generated, by width; runPrep)", true},
	graph.OpRope:             {"nn rotary (generated, by layout; runPrep)", true},
	graph.OpKVAppend:         {"KV page store (the cache layout's writer, by precision; runPrep)", true},
	graph.OpAttend:           {"the generated attention kernels (scores, softmax, accumulate), chosen by row class: attendRow, attendRows, rowsAttend", true},
	graph.OpRoute:            {"moeRoute / moeRouteRows: the router's matvec (generated) and the route kernel (generated, by experts, top-k, router and groups)", true},
	graph.OpGather:           {"moeGather: the selected experts' pages read (no arithmetic); the rows' gather inside moeExpertsRows", true},
	graph.OpExpertMatVec:     {"the grouped generated matvec over the k experts (MatVecPackedMulti for gate and up, MatVecPackedGather for down)", true},
	graph.OpCombine:          {"moeReduce: the weighted sum in ascending expert id (generated axpy)", true},
	graph.OpGateScalar:       {"sharedExpert: the gate's matvec and sigmoid-multiply (generated)", true},
	graph.OpLayerNorm:        {"nn LayerNorm (generated, by width)", true},
	graph.OpAct:              {"nn activation (generated, by kind)", true},
	graph.OpClamp:            {"nn clamp (generated; runPrep)", true},
	graph.OpSlice:            {"a view of the projection's heads (no arithmetic); compacted where the next kernel reads it whole", true},
	graph.OpLatentAppend:     {"KV page store of the latent row (the cache layout's writer)", true},
	graph.OpAttendLatent:     {"absorbed (the mla-absorb rewrite): the generated attention kernels over the latent rows, then W_v's un-absorb (mlaOutProject's gathered matvec)", true},
	graph.OpScale:            {"the generated scale kernel (layerOutScale)", true},
	graph.OpIndex:            {"mlaProject's indexer key and query (idxKey, idxQuery), then attendRow's top-k mask over the scores (idxMask)", false},
	graph.OpStreamMix:        {"hcMix: the streams' norm, the low-rank sigmoid gate, the mean and the placement weights (generated kernels)", false},
	graph.OpStreamPlace:      {"hcPlace: the output added into every stream at its weight (generated axpy)", true},
	graph.OpNGram:            {"ngramApply: the hashed rows read from the table, the projections, the gate and the convolution", false},
	graph.OpAltPredict:       {"altPredict: the router over the active stream's norm and the predictions (generated kernels)", false},
	graph.OpAltCorrect:       {"altCorrect: the correction and the per-layer input into the streams (generated kernels)", false},
	graph.OpLaurel:           {"laurel: the low-rank pair, its norm and the add (generated)", false},
	graph.OpGaussTopK:        {"gaussTopK: the generated gaussian top-k over each gate row", true},
	graph.OpPredict:          {"draftRows (spec.go): enorm, hnorm and eh_proj before the step; the block's own head after it", false},
	graph.OpCompressedAttend: {"ds4Attn: the projections, the compressor and indexer, the attention and the grouped output", false},
	graph.OpRecurrent:        {"linearAttn: the recurrent mixer as one routine (mixerChunk, mixerRagged over rows)", false},
}

// moeActFused is the experts' act_mul's lowering: fused into down's quantize
// (MatVecPackedGatherAct), a fusion decided at compile time -- its own
// generated kernel while a tracer reads the activation between.
const moeActFused = "fused into down's quantize (MatVecPackedGatherAct); its own generated kernel under a tracer"

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
	ds4 := false
	for _, o := range g.Ops {
		ds4 = ds4 || o.Kind == graph.OpCompressedAttend
	}
	for i, o := range g.Ops {
		h := hostLowering[o.Kind]
		if ds4 {
			h.routine, h.composed = "ds4Block: hcPre, ds4Attn, hcPost and ds4FFN (the mixture through moe)", false
		}
		if o.Kind == graph.OpIndex && mlaTo < 0 {
			h.routine = "msaKey and msaQuery (the key one more head of the appended row), then attendRow's block mask (msaMask)"
		}
		if mlaTo >= 0 && i >= mlaFrom && i < mlaTo && o.Kind != graph.OpIndex {
			// Each op's own generated kernel, called in the graph's order
			// by mlaProject (mlaProjectRows over a chunk's rows).
			h.routine += " (in mlaProject's sequence)"
		}
		// The experts' act_mul runs inside their grouped region.
		if (o.Kind == graph.OpActMul || o.Kind == graph.OpAct) && producer[o.In[0]] >= 0 && g.Ops[producer[o.In[0]]].Kind == graph.OpExpertMatVec {
			h.routine, h.composed = moeActFused, true
		}
		out[i] = OpLowering{Op: i, Kind: o.Kind, Host: h.routine, Composed: h.composed, Opaque: graph.Opaque(o.Kind),
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

// kvRows projects a prompt's k and v rows into bk and bv as the lowering
// states them: both, v a copy of k (vFromK), or neither (kvShared).
func (s *State) kvRows(l *layer, lw *denseLowering, h []float32, n, kvDim int) error {
	if lw.kvShared {
		return nil
	}
	if err := s.mm(s.bk, l.wk, h, n); err != nil {
		return err
	}
	s.addBiasRows(s.bk, l.bk, n)
	if lw.vFromK {
		copy(s.bv[:n*kvDim], s.bk[:n*kvDim])
		return nil
	}
	if err := s.mm(s.bv, l.wv, h, n); err != nil {
		return err
	}
	s.addBiasRows(s.bv, l.bv, n)
	return nil
}
