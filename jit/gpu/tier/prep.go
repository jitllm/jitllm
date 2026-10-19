package tier

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// PrepLayer is prepLayerFor for the zero session: a caller that never attached.
func (g *devTier) PrepLayer(li int, p *nn.LayerPlan, w *nn.LayerWeights) bool {
	return g.prepLayerFor(0, li, p, w)
}

// prepLayerFor uploads one block and compiles the kernels it needs, taking it only
// if it fits.
func (g *devTier) prepLayerFor(sid uint64, li int, p *nn.LayerPlan, w *nn.LayerWeights) bool {
	return g.prepLayer(sid, li, p, w, false)
}

// declineReason names the graph feature this tier cannot run, or "" when it can
// run the block's shape. A string rather than a bool so the reason survives: a
// full card, a missing kernel and an unknown graph want opposite responses.
func declineReason(p *nn.LayerPlan) string {
	// The router's gating shape, declined by name: a kernel gate on ExpertRank
	// proves the kernel, not that the tier asked for the right gating function.
	if p.NExpert > 0 {
		if r := routerWhyNot(p); r != "" {
			return r
		}
	}
	// MLA runs; what is left is a shape check (a tier that does not implement
	// MLA still declines on KVLoraRank alone).
	if r := mlaWhyNot(p); r != "" {
		return r
	}
	if r := llama4WhyNot(p); r != "" {
		return r
	}
	if r := gemma4WhyNot(p); r != "" {
		return r
	}
	if r := msaWhyNot(p); r != "" {
		return r
	}
	if r := ds4WhyNot(p); r != "" {
		return r
	}
	if r := k3WhyNot(p); r != "" {
		return r
	}
	if r := q4eWhyNot(p); r != "" {
		return r
	}
	switch {
	case p.Recurrent.Centred && (!p.NonCausal || !p.Recurrent.ShortConv || p.Recurrent.Conv != 3 || !p.Windowed):
		// The centred convolution reads its neighbours through the call's row
		// windows (kernels.Conv3Centred): a non-causal call is the whole
		// sequence, and the windows are where the taps stop.
		return "a centred convolution outside a windowed non-causal short convolution of three taps"
	case p.NonCausal && (p.Parallel || p.Recurrent.Conv > 0 && !p.Recurrent.Centred || p.MLA() || p.NExpert > 0 ||
		p.AttnSinks):
		// A non-causal call is one submission of the block body's plain
		// attention over its runs; these features are built only on the causal
		// paths, and no non-causal model has them.
		return "a non-causal block with a parallel residual, a recurrence, latent attention, experts or sinks"
	case p.Windowed && !p.NonCausal:
		return "windowed key runs on a causal block"
	case p.PostResidNorm && (!p.NonCausal || !p.NoPreNorm || !p.LayerNorm):
		// The norms after the residual adds are wired into the non-causal
		// block (BERT's encoder): its attention reads the residual through
		// the no-pre-norm copy, and its norms are LayerNorms.
		return "post-residual norms on a block other than a non-causal LayerNorm one with no pre-norm"
	case p.Clamps && (!p.NonCausal || p.ClampKQV != 0 || p.AttnOutGate || ungatedFFN(p)):
		// The clipped linears are wired into the non-causal block's plain
		// q/k/v, its output projection and a gated MLP: Gemma 4's tower, the
		// one block that has them.
		return "clipped linears on a block other than a non-causal one with a gated MLP"
	case p.ClampKQV != 0 && (p.AttnOutGate || p.MLA()):
		// The clamp is wired into the plain q/k/v path; an out-gate splits the
		// double-width q first and MLA projects through its own sequence. No
		// architecture combines them.
		return "a q/k/v clamp with an attention out-gate or latent attention"
	case p.Recurrent.Conv > 0 && p.Recurrent.KHeads == 0 && !p.Recurrent.ShortConv:
		// The gated delta rule runs (recPair gives every kernel an out-of-place
		// state); what is left is a shape check.
		return "a linear-attention block with no key heads"
	case p.Recurrent.Mamba1 && p.Recurrent.Rank <= 0:
		return fmt.Sprintf("a Mamba-1 dt rank of %d", p.Recurrent.Rank)
	case p.Recurrent.WithAttn && (!p.Recurrent.SSD || p.MLA() || p.NoFFN):
		// The sum is built for Falcon-H1's shape alone: Mamba-2 beside plain
		// attention, then an FFN.
		return "attention beside a recurrent mixer in a shape no architecture has"
	case p.Recurrent.SSD && (p.Recurrent.ChanDecay || p.Recurrent.KeyTiled):
		// Mamba-2's update has one decay a head and grouped keys; neither
		// switch exists on any Mamba-2, and the fused form refuses both.
		return "Mamba-2's update with a per-channel decay or tiled keys"
	case p.Recurrent.SSD && !p.Recurrent.NoNorm && (p.Recurrent.NormGroups <= 0 ||
		(p.Recurrent.VHeads*p.Recurrent.VDim)%p.Recurrent.NormGroups != 0):
		return fmt.Sprintf("a Mamba-2 gated norm of %d groups over %d channels",
			p.Recurrent.NormGroups, p.Recurrent.VHeads*p.Recurrent.VDim)
	case !actHasKernel(p):
		// Every activation the host bakes has a device kernel (quick-GELU
		// for CLIP towers, swiglu-oai for gpt-oss); what is left is a kind in a
		// shape it does not exist in -- a gated kind in an ungated block.
		return "the " + p.Act.String() + " activation in a " + map[bool]string{true: "ungated", false: "gated"}[ungatedFFN(p)] + " block"
	}
	return ""
}

// residScaled reports a plan whose block outputs are scaled before their
// residual adds (Granite's residual_scale). Such a block takes neither the
// projection's fused residual (mvres: the bias slot has no scale) nor the
// add-and-norm launches: the output lands in bs.mvOut as for a post-norm, and
// bs.addS adds s times it. A mixture's routed weights carry no part of it on
// the device, unlike the host's routed scale (model.Config.routedScale): the
// scaled add covers the routed sum and the shared expert at once.
func residScaled(p *nn.LayerPlan) bool { return p.ResidualScale != 0 && p.ResidualScale != 1 }

// scoreScale is the attention score multiplier every kernel of p's attention
// bakes: the plan's (nn.LayerPlan.Scale), which Granite's attention.scale and
// YaRN's magnitude correction replace, or 1/sqrt(HeadDim) under
// ScaleFaultAttn.
func (g *devTier) scoreScale(p *nn.LayerPlan) float32 {
	if g.kb.scaleFault == ScaleFaultAttn {
		return float32(1 / math.Sqrt(float64(p.HeadDim)))
	}
	return float32(p.Scale())
}

// llama4WhyNot names the Llama 4 attention or mixture feature this tier cannot
// run, or "". Each is fluent when skipped (a sliding mask is exact inside
// the first chunk, a rotated NoPE layer at position 0), so the block goes home
// by name rather than running wrong (RULE 8a). What remains are combinations no
// model ships: an input-weighted mixture with expert biases, and a chunk on an
// MLA or vision block. A temperature on every layer (mistral3's form) runs:
// the rotated layers scale q into qt before the rotary.
func llama4WhyNot(p *nn.LayerPlan) string {
	switch {
	case p.AttnTempScale != 0 && (p.AttnTempFloor < 1 || p.AttnTempOffset < 0 ||
		p.AttnTempFloor != float32(int(p.AttnTempFloor)) || p.AttnTempOffset != float32(int(p.AttnTempOffset))):
		return fmt.Sprintf("an attention temperature floor %g / offset %g that is not a whole count",
			p.AttnTempFloor, p.AttnTempOffset)
	case p.ExpertWeightIn && p.MoEBias:
		return "a routed weight on the input of a biased expert"
	case p.ExpertWeightIn && p.NExpert > 0 && p.Act != kernels.ActSiLU && p.Act != kernels.ActGELU:
		return "a routed weight on the input of a " + p.Act.String() + " expert"
	case (p.SWAChunked || p.NoPEGlobal || p.QKL2Norm) && (p.MLA() || p.NonCausal):
		return "Llama 4's attention on a latent or non-causal block"
	}
	return ""
}

// l4TempTab is Llama 4's attention temperature as a table over the integer
// k = floor((pos+off)/floor): entry k is 1 + s*ln(1+k). It covers every
// position a plan of maxSeq can reach, so the kernel's clamp never bites.
func l4TempTab(p *nn.LayerPlan) []float32 {
	fl, off := int(p.AttnTempFloor), int(p.AttnTempOffset)
	n := (max(p.MaxSeq, 1)-1+off)/fl + 1
	out := make([]float32, n)
	for k := range out {
		out[k] = float32(1 + float64(p.AttnTempScale)*math.Log(1+float64(k)))
	}
	return out
}

// windowArg is the window a windowed score kernel is built with: the sliding
// width, or kernels.ChunkWindow of it where the layers attend within a chunk.
func windowArg(p *nn.LayerPlan) int {
	if p.SWAChunked {
		return kernels.ChunkWindow(p.SWAWindow)
	}
	return p.SWAWindow
}

// routerWhyNot names the mixture gating this tier cannot express, or "". The
// answer comes from the kernels (kernels.MoERouteWhyNot), so a missing gating
// feature is a device kernel owed, not a refused capability.
func routerWhyNot(p *nn.LayerPlan) string {
	if p.NExpert == 0 {
		return ""
	}
	return kernels.MoERouteWhyNot(routeOf(p))
}

// routeOf is the plan's router as the kernels describe one: one translation,
// so the decline and the kernel cannot disagree.
func routeOf(p *nn.LayerPlan) kernels.MoERoute {
	// DeepSeek V4 selects through a per-row plane on every block (ds4.go):
	// its selection bias, or a hash block's table.
	ds4 := p.DS4 != nil
	return kernels.MoERoute{
		NExpert: p.NExpert, K: p.NExpertUsed,
		Sigmoid: p.ExpertSigmoid, Bias: p.ExpertSelBias || ds4,
		SqrtSoftplus: ds4 && p.DS4.ExpertSqrtSoftplus, RowBias: ds4,
		NGroup: p.ExpertGroups, NGroupUsed: p.ExpertGroupsUsed,
		Norm: !p.NoExpertNorm, Scale: float32(p.ExpertScale),
		SparseMixer: p.ExpertSparseMixer,
	}
}

// subFor is the activation sub-block a matvec of this format and width needs,
// or 0 for the format's own. It is float-only: a quantized format's scales are
// laid out against its sub-block (see kernels.MatVecShape.Sub), while a float
// weight has no scale plane. What reaches this is MLA's absorbed banks, whose k
// can be below 32 on a small fixture, where the sub-block narrows to the width
// itself, and a ragged width past 32 (KDA's output at two heads of 40), where
// it narrows to the widest of 16, 8 and 4 that divides k.
func subFor(q kernels.Quant, k int) int { return kernels.FloatSub(q, k) }

// mlaWhyNot names the MLA shape this tier cannot express, or "". Each is a
// width, so each is a kernel owed rather than a refusal (RULE 8): the
// 32-element granule of kernels.Quantize and of the norm's partial reduction.
// The batched arm is refused separately, at submission (see layersOnce), so a
// correct decode stays on the device.
func mlaWhyNot(p *nn.LayerPlan) string {
	if !p.MLA() {
		return ""
	}
	if p.IdxHeads != 0 && p.NoPosEnc {
		return "DeepSeek Sparse Attention's lightning indexer on a model with no rotary"
	}
	nope := p.HeadDim - p.NRot
	if nope <= 0 {
		return fmt.Sprintf("multi-head latent attention with %d rotary dimensions of a %d-wide "+
			"key head, which leaves no nope half to absorb W_k into", p.NRot, p.HeadDim)
	}
	if p.NRot%2 != 0 || p.NRot == 0 {
		return fmt.Sprintf("multi-head latent attention with %d rotary dimensions, which the "+
			"rotation needs to be an even non-zero count", p.NRot)
	}
	for _, v := range []struct {
		n    int
		what string
	}{
		{p.NHead * nope, "the compacted nope half of the query"},
		{p.NHead * p.KVLoraRank, "the per-head attention result"},
		{p.NHead * p.HeadDimV, "the un-absorbed attention output"},
		{p.QLoraRank, "the query latent"},
	} {
		// QLoraRank is 0 on the one-step query, which is not a shape at all.
		if v.n != 0 && v.n%32 != 0 {
			return fmt.Sprintf("multi-head latent attention where %s is %d elements: the "+
				"activation quantizer bakes a multiple of 32 and this tier has no narrower one",
				v.what, v.n)
		}
	}
	return ""
}

// lnBlock reports whether the block's norms are LayerNorms: a vision tower's,
// or a classic text block's (C6, nn.LayerPlan.LayerNorm). Every other block
// is RMSNorm, whose fused norm+quantize kernels a LayerNorm cannot use.
func lnBlock(p *nn.LayerPlan) bool { return p.LayerNorm }

// ungatedFFN reports whether the block's FFN has no gate matrix: up, the
// activation alone, down -- a vision tower's MLP, or a classic text block's
// (C6: phi-2, starcoder, falcon, nemotron).
func ungatedFFN(p *nn.LayerPlan) bool { return p.UngatedFFN }

// actHasKernel reports whether kernels.Act (an ungated FFN) or kernels.ActMul
// (every gated FFN) bakes the plan's activation.
func actHasKernel(p *nn.LayerPlan) bool {
	// A block with no FFN runs no activation.
	if p.NoFFN {
		return true
	}
	set := kernels.Gated[:]
	if ungatedFFN(p) {
		// xIELU reads its numbers from a buffer (kernels.XIELU), so it is in
		// no baked set.
		if p.Act == kernels.ActXIELU {
			return true
		}
		set = kernels.Ungated[:]
	}
	for _, k := range set {
		if k == p.Act {
			return true
		}
	}
	return false
}

// declineWeights names a weight this tier cannot express, or "" when every one
// of them is expressible. declineReason is its twin for the plan: the plan says
// what graph the block runs, and this says whether its buffers can be uploaded
// as they are.
func declineWeights(w *nn.LayerWeights) string {
	for _, x := range []struct {
		name string
		w    nn.Weight
	}{{"q", w.Wq}, {"k", w.Wk}, {"v", w.Wv}, {"o", w.Wo},
		{"gate", w.Gate}, {"up", w.Up}, {"down", w.Down},
		// MLA's five. The upload reads QS/D/SC whole and has no row base, so an
		// offset weight would be indexed from row 0 and answer fluently; this
		// guard checks rather than assumes none is.
		{"q_a", w.Wqa}, {"q_b", w.Wqb}, {"kv_a", w.Wkva},
		{"k_b", w.Wkb}, {"v_b", w.Wvb}} {
		p := x.w.Packed
		if p == nil || len(x.w.Data) == 0 {
			continue
		}
		// Stride 0 means "the whole tensor", which is the ordinary case.
		if p.Row != 0 || (p.Stride != 0 && p.Stride != x.w.Rows) {
			return fmt.Sprintf("the %s projection is rows [%d,%d) of a packed tensor of %d "+
				"(a fused qkv, phi3-style); the upload has no row base",
				x.name, p.Row, p.Row+x.w.Rows, p.Stride)
		}
	}
	return ""
}

// prepLayer is PrepLayer with the pager allowed. mayPage lets a block that does
// not fit evict another one rather than be declined. It is a second pass, not a
// mode: GPU.PrepLayer offers every device with mayPage false first, so a second
// card with room still gets the block and only when no device has room does
// anybody swap.
func (g *devTier) prepLayer(sid uint64, li int, p *nn.LayerPlan, w *nn.LayerWeights, mayPage bool) (placed bool) {
	// The host side of a hybrid block is the offering session's: a placed
	// block is shared by every session that attaches, and each runs its experts
	// on its own State (runHost looks it up by the call's session). Recorded
	// before anything can decline, so a block another session placed still
	// learns this one's.
	if w != nil && w.HostExperts != nil {
		g.mu.Lock()
		if g.hostFns == nil {
			g.hostFns = map[uint64]map[int]hostSide{}
		}
		if g.hostFns[sid] == nil {
			g.hostFns[sid] = map[int]hostSide{}
		}
		g.hostFns[sid][li] = hostSide{one: w.HostExperts, rows: w.HostExpertsRows, uses: w.ExpertUses}
		g.mu.Unlock()
	}
	// Shapes this tier cannot express are declined first and by name (RULE 8a):
	// each would run wrong if skipped, so the block goes to the host whole. They
	// are checked before any allocation so a decline takes no VRAM from the next
	// block.
	if p.Conv != nil {
		g.mu.Lock()
		defer g.mu.Unlock()
		return g.prepConv(li, p, w, mayPage)
	}
	if r := declineReason(p); r != "" {
		g.LastErr = fmt.Sprintf("block %d: %s", li, r)
		return false
	}
	// A row range of a packed tensor cannot be expressed: the device layout
	// interleaves rows, resident() uploads QS/D/SC whole, and MatVecShape has no
	// row base, so the block runs on the host instead of reading the wrong rows.
	if r := declineWeights(w); r != "" {
		g.LastErr = fmt.Sprintf("block %d: %s", li, r)
		return false
	}
	// The q/k norm is launched when the block carries its weights; the plan's
	// flag builds the kernels. The model offers the weights by the same
	// derivation that sets the flag (jlm.QKNormOn), so a block where the two
	// disagree is a contract violation, refused rather than run without its norm.
	if attn := !p.NonCausal && (p.Recurrent.Conv == 0 || p.Recurrent.WithAttn); attn && p.QKNorm != (w.QNorm != nil) {
		g.LastErr = fmt.Sprintf("block %d: the plan's q/k norm is %v and the block offers %d norm "+
			"weights", li, p.QKNorm, len(w.QNorm))
		return false
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	// A tower block the device could not take leaves nothing behind: the
	// first one built its set's scratch before the budget said no, and with
	// no tower block on the device that scratch is a ceiling nobody runs.
	if p.NonCausal {
		defer func() {
			if placed {
				return
			}
			g.dropIdleTower()
			// And on a device it was the first block of, the staging its
			// build sized goes too, as the last block out takes it.
			if len(g.layers) == 0 && (g.bs == nil || g.bs.head == nil) {
				g.dropGraph()
				g.dropAllScratch()
			}
		}()
	}
	// The block's own geometry's scratch set, and the home set again on the
	// way out (gemma4.go): Gemma 4's sliding layers, and a non-causal
	// segment's blocks -- a vision tower's, whose residual, heads and FFN are
	// not the text model's. sp is the plan that set is built from.
	sp := scratchPlanOf(p)
	if k := geoOf(p); k != 0 {
		g.geoUsed = true
		g.useGeom(k)
		defer g.useGeom(0)
	}

	// The KV ceiling is the largest context any session asked for, not the
	// first's, or a longer second session is refused at its next position.
	if !p.NonCausal && g.kvCap != 0 && p.MaxSeq > g.maxSeqAsked {
		g.maxSeqAsked = p.MaxSeq
	}
	if r := planConflict(g.bs, sp); r != "" {
		g.LastErr = fmt.Sprintf("block %d: %s", li, r)
		return false
	}
	if r := nonCausalConflict(g.bs, sp); r != "" {
		g.LastErr = fmt.Sprintf("block %d: %s", li, r)
		return false
	}

	// A block already on this device is shared and only the history is new: the
	// weights, norms, biases and router are the model's. Rebuilding would
	// re-upload them and replace the layer the first session is running against.
	if l := g.layers[li]; l != nil && l.ok {
		// The new session's plan still has to reach the scratch: a longer session
		// may need a sliding window the first did not (model.windowFor), so take
		// the union as the fresh-block path does.
		if !l.nonCausal && g.bs != nil {
			if want := g.bs.p; planUnion(&want, sp) {
				g.dropGraph()
				if !g.rebuildScratch(&want) {
					return false
				}
			}
		}
		// A linear block's history is a recurrent pair, not a KV cache; a
		// block that attends as well keeps both.
		if l.linear {
			if l.recOf(sid) == nil && !g.addSessionRec(sid, l, p) {
				return false
			}
			if !l.withAttn {
				return true
			}
		}
		if l.kvOf(sid) != nil || l.kvSrc != li {
			return true // this session already has a history here, or reads its source's
		}
		if !g.addSessionKV(sid, l, p) {
			return false
		}
		return true
	}
	g.dropGraph()
	// The KV capacity is decided before the scratch is built: bs.p is the capped
	// plan, and the scratch's score buffers and MigrateKV's staging are sized
	// from its MaxSeq, so building first would size them for the whole context.
	g.initKVCap(p)
	// A non-causal set is built for the rows of the pictures it runs, not for
	// the tower's largest grid (visrows.go): the first block sets where it
	// starts, ReserveRows moves it.
	if p.NonCausal {
		g.visMax = max(g.visMax, p.MaxSeq)
		if g.visCap == 0 {
			g.visCap = visRowsFor(min(p.MaxSeq, visStart), p.MaxSeq)
		}
	}
	if g.bs == nil && p.NonCausal {
		// The softmax's subgroup width is a property of the device, which the
		// one-row scratch probes and every batched one inherits (pickSoftmax).
		// A device whose first block is a tower's has built no one-row
		// scratch, and a batched scratch built unprobed takes the
		// one-thread-per-head softmax: SmolVLM's tower ran 2.6x slower on CUDA
		// for it. So the probe runs here, on a one-row scratch of this plan
		// that is dropped again.
		if g.SoftmaxLanes == 0 {
			if probe := g.initScratch(sp, 1); probe != nil {
				g.dropScratch(probe)
			}
		}
		// A non-causal call is one whole run, so its set's scratch is built
		// for the run's rows -- the rows the set is built for (visCap), which
		// capped hands the plan as its MaxSeq: there is no decode width.
		if !g.buildTowerScratch(sp) {
			return false
		}
	} else if g.bs == nil {
		g.shapeChanged()
		if g.bs = g.initScratch(sp, 1); g.bs == nil {
			return false
		}
	} else if want := g.bs.p; p.NonCausal && (sp.NFFN > want.NFFN || sp.Recurrent.Conv > 0 && want.Recurrent.Conv == 0 ||
		sp.QKNorm && !want.QKNorm) {
		// A non-causal block with a wider MLP than the set was built for (a
		// decision head after its encoder): the set grows to it, and every
		// block runs its own width inside it (layer.nffn). An LFM2 encoder's
		// convolution blocks and attention blocks share a set too: it carries
		// the convolution's kernels and the q/k norm's, whichever came first.
		want.NFFN, want.MaxSeq = max(want.NFFN, sp.NFFN), g.visMax
		if want.Recurrent.Conv == 0 {
			want.Recurrent = sp.Recurrent
		}
		want.QKNorm = want.QKNorm || sp.QKNorm
		if !g.towerAt(&want, g.visCap) {
			return false
		}
	} else if want := g.bs.p; !p.NonCausal && planUnion(&want, sp) {
		// g.bs is built from whichever block arrives first, and on a hybrid that
		// may be a linear block with no out-gate kernels. A hybrid's two kinds
		// alternate over one residual inside one submission, so they share one
		// scratch carrying both kernel sets, built from the union.
		if !g.rebuildScratch(&want) {
			return false
		}
	}
	if g.layers == nil {
		g.layers = map[int]*layer{}
	}
	l := &layer{nonCausal: p.NonCausal, windowed: p.Windowed, geo: geoOf(p), vFromK: p.VFromK, graphed: p.Graphed, rewrites: p.Rewrites}
	if p.NonCausal {
		l.nffn, l.ungated, l.act = p.NFFN, ungatedFFN(p), p.Act
		l.noRope = p.NRot == 0 || p.NoPosEnc
		if !g.ownActKern(l) {
			return false
		}
	}
	mkw := func(slot int, x nn.Weight) *resident {
		q, ok := quantOf(x.T)
		// A float format's block is the activation granule, not a file structure,
		// so a k below it is a width (see subFor); the k-quants keep the check.
		if !ok || x.Rows <= 0 || len(x.Data) == 0 {
			return nil
		}
		if x.K%q.Elems() != 0 && subFor(q, x.K) == 0 {
			return nil
		}
		return g.resident(li, slot, q, x.Data, x.Rows, x.K, x.Packed)
	}
	build := func(x nn.Weight, split int, bias bool) (mv, bool) {
		q, ok := quantOf(x.T)
		if !ok {
			return mv{}, false
		}
		// A narrowed float sub-block (subFor) walks k/sub steps, which the
		// split, chosen against k/32, must divide.
		if sub := subFor(q, x.K); sub > 0 && (x.K/sub)%split != 0 {
			split = 1
		}
		// The decode row tile: its rows are adjacent, so a lane loads Rowt
		// contiguous u32 instead of one. It composes with the split (the tile buys
		// bytes per lane, the split buys threads). The row count must divide.
		rowt := 1
		if g.DecodeRowt > 1 && x.Rows%g.DecodeRowt == 0 {
			rowt = g.DecodeRowt
		}
		// Counted, so "it made no difference" and "it never ran" can be told
		// apart.
		if rowt > 1 {
			g.RowtTiles++
			if split > 1 {
				g.RowtSplitTiles++
			}
		} else {
			g.RowtPlain++
		}
		// The in-group split reduction, where tuneSplit measured it faster. It
		// writes the final row, so no reduction kernel and no partial plane; see
		// kernels.MatVecShape.GroupSplit.
		grp := !g.NoGroupSplit && split > 1 && rowt == 1 &&
			(g.ForceGroupSplit || g.groupSplit[splitKey{q, x.Rows, x.K}]) &&
			kernels.GroupSplitOK(x.Rows, split)
		kern := g.kernelMode(q, x.K, x.Rows, split, rowt, bias, grp)
		if kern == nil {
			// A refused tile is not a refused weight: fall back to one row.
			rowt, grp = 1, false
			if kern = g.kernel(q, x.K, x.Rows, split, 1, bias); kern == nil {
				return mv{}, false
			}
		}
		var red backend.Kernel
		if split > 1 && !grp {
			if red = g.reduceKernel(x.Rows, split); red == nil {
				split = 1
				if kern = g.kernel(q, x.K, x.Rows, 1, rowt, bias); kern == nil {
					return mv{}, false
				}
			}
		}
		if !grp && !g.sizePart(x.Rows*split) {
			return mv{}, false
		}
		return mv{kern: kern, red: red, rows: x.Rows, split: split, q: q, k: x.K,
			rowt: rowt, group: grp, pad: p.KPad > 0 && x.K == p.KPad}, true
	}
	// bias is threaded into every kernel: a 6-buffer kernel launched with 7 is
	// refused on Vulkan and silently drops the bias on CUDA, which does not
	// check arity.
	mkk := func(x nn.Weight, r *resident, bias bool) (mv, bool) {
		q, ok := quantOf(x.T)
		if !ok {
			return mv{}, false
		}
		m, ok := build(x, g.tuneSplit(q, x.Rows, x.K, r), bias)
		if !ok {
			return mv{}, false
		}
		if tbl := chooseSplit(x.Rows, x.K/32, g.kb.split); tbl != m.split {
			if a, ok := build(x, tbl, bias); ok {
				m.alt = &a
			}
		}
		return m, true
	}
	// mkkID is mkk with the expert fields. A dense weight passes experts=1 and
	// gets exactly the kernel it always got -- MatVec is byte-identical there,
	// which TestDenseKernelsUnchanged holds it to.
	mkkID := func(x nn.Weight, r *resident, experts, slots int, slotAct, bias bool) (mv, bool) {
		if experts <= 1 {
			return mkk(x, r, bias)
		}
		q, ok := quantOf(x.T)
		if !ok {
			return mv{}, false
		}
		// The split is chosen from Rows*Slots, not Rows: an indexed launch is
		// rows*slots*split threads and MatVec lays its partials out at stride
		// Rows*Slots, so Rows alone would over-split and reduce the wrong stride.
		// It comes from the table, not the tuner, which has no selection to pass;
		// WithSplit still pins it.
		nout := x.Rows * slots
		mk := func(split int) (mv, bool) {
			// ForceIndexedGroup selects the in-group reduction here without
			// depending on what retuneIndexed happens to time fastest.
			grp := g.ForceIndexedGroup && !g.NoGroupSplit && split > 1 && !bias &&
				kernels.GroupSplitOK(x.Rows, split)
			c, e := g.idKernel(kernels.MatVecShape{Center: g.center, T: q, K: x.K, Rows: x.Rows,
				Split: split, Experts: experts, Slots: slots, SlotAct: slotAct, Bias: bias,
				Sub: subFor(q, x.K), GroupSplit: grp})
			if e != nil {
				g.LastErr = e.Error()
				return mv{}, false
			}
			m := mv{kern: c, rows: x.Rows, slots: slots, split: split, q: q, k: x.K, group: grp,
				experts: experts, slotAct: slotAct}
			if grp {
				g.IndexedGroup++
			}
			if split > 1 && !grp {
				if m.red = g.reduceKernel(nout, split); m.red == nil {
					return mv{}, false
				}
				if !g.sizePart(nout * split) {
					return mv{}, false
				}
			}
			return m, true
		}
		// A narrowed sub-block takes no split: chooseSplit is handed K/32, zero
		// below 32 elements, which MatVec would refuse.
		if subFor(q, x.K) == 0 {
			if m, ok := mk(chooseSplit(nout, x.K/32, g.kb.split)); ok {
				return m, true
			}
		}
		return mk(1)
	}
	// An expert bank is one matrix of NExpert*Rows rows: a GGUF 3-D expert tensor
	// is expert-major and contiguous, so it packs, uploads and indexes as one
	// tall matrix, and the kernel does the row offset from Experts and Slots.
	moe := w.Router.Data != nil
	bank := 1
	slots := 0
	if moe {
		if p.NExpert < 1 || p.NExpertUsed < 1 || p.NExpertUsed > p.NExpert {
			return false
		}
		bank, slots = p.NExpert, p.NExpertUsed
	}
	// One list: layer.tensors() derives the free walk and pageSizeWhy the price
	// from this order, so a matrix uploaded outside it would be neither freed
	// nor charged. A linear block leaves 1..3 empty (Wq carries its fused q|k|v),
	// an attention block leaves 10..12 empty, and pageSizeWhy skips zero-length
	// weights.
	l.bound = boundRoles(p.Graph)
	ws := weightList(make([]nn.Weight, 0, numPositional+len(l.bound)), w, l.bound)
	// A streamed bank is priced at its compact size by scaling the three bank
	// rows from NExpert sheets to NExpertUsed; every size derives from
	// Weight.Rows, so the charge and the refund cannot disagree. slots > 1 is a
	// kernel constraint: at experts <= 1 mkkID returns the plain matvec, which
	// has no pSel parameter.
	mode, auto := g.autoStream[li]
	// hostExp says the block's experts will run on the host (hybrid): the
	// placement's choice, else the tier's defaults. Such a block keeps no bank
	// on the card at all -- no compact bank, no expert kernels, no packed
	// layout needed -- and a top-1 router is no obstacle, since no indexed
	// matvec runs.
	hostExp := w.HostExperts != nil && (auto && mode == expHost ||
		g.HybridExperts && !(auto && mode == expCard) || auto && mode == expAuto && !g.NoHybrid)
	stream := moe && (g.StreamExperts || auto) && slots < bank && (slots > 1 || hostExp)
	// A hybrid block that can hold an expert cache (Config.NoHybridCache)
	// is placed as a streamed one -- the compact bank and its kernels on the
	// card -- marked hybrid, so the cache sized after placement has kernels
	// to run its experts with (streamBank.split). One whose experts carry
	// biases or a dense scale keeps no bank, as cacheSlotsFor refuses them,
	// and so does a float bank, which has no packed sheets to send.
	cacheable := !g.NoHybridCache && g.StreamCacheSlots >= 0 && slots > 1 && !p.DenseMoE &&
		w.ExpGateB == nil && w.ExpUpB == nil && w.ExpDownB == nil &&
		w.Up.Packed != nil && w.Down.Packed != nil && (w.Gate.Packed != nil || len(w.Gate.Data) == 0)
	hybrid := stream && hostExp && !cacheable
	// cslots is the streamed bank's size in sheets, decided at the first bank
	// matrix (cacheSlotsFor): slots without an expert cache, more with one.
	cslots := -1
	// price is ws with the three banks removed for a streamed block: the
	// compact bank is one buffer for the whole device (sharedCompact), so a
	// block costs its base and nothing else.
	price := ws
	if stream {
		for i := 4; i <= 6; i++ {
			if ws[i].Rows%bank != 0 {
				g.LastErr = fmt.Sprintf("block %d: %s has %d rows, not a multiple of %d experts",
					li, weightRole(i), ws[i].Rows, bank)
				return false
			}
			ws[i].Rows = ws[i].Rows / bank * slots
		}
		price = append([]nn.Weight(nil), ws...)
		for i := 4; i <= 6; i++ {
			price[i].Data, price[i].Packed = nil, nil
		}
	}
	// Decide on the whole block before uploading any of it: declining halfway
	// leaves earlier tensors resident, counted and unreachable. Every decline
	// sets LastErr.
	full, why, ok := g.pageSizeWhy(price)
	if !ok {
		g.LastErr = fmt.Sprintf("block %d: %s", li, why)
		return false
	}
	// The slot width is the widest block, not this one; see slots().
	if full > g.widest {
		g.widest = full
	}
	need, _ := g.blockBytes(price)
	// Include what admitting it charges for as long as it is resident: its KV
	// cache (or recurrent pair) and norms. Leaving them out admitted blocks that
	// pushed the card over budget and made trim() page every token.
	perm := g.permBytes(p, w)
	need += perm
	// Before the block's weights: the prompt's and the step's scratch are
	// budgeted first, with room left for this block, so a card that cannot
	// hold both takes a block fewer -- or a narrower chunk -- rather than
	// refusing the prompt later (scratch.go).
	g.reserveScratch(sp, need)
	// The unpack staging never costs a block. On a device that cannot page the
	// staging would never be reused, so it is freed when the block does not
	// otherwise fit; admission is then what it was without staging, and the
	// block's covered tensors take the host packer (NoUnpackRoom). On a paging
	// device it stays, because without it every page-in is a slow host repack;
	// reclaim() and slots() both price it.
	// The KV pools' free pages go first: they hold nothing.
	g.kvCompact(need, nil)
	if !g.room(need) && !g.stream[li] {
		g.freeRaw()
	}
	if !g.room(need) {
		// A block that does not fit pages another one out rather than falling to
		// the host. Below minSlots there is nowhere to swap through, so it
		// declines and LastErr says which case it is.
		// The slots are counted after this block's own permanent charge: a
		// streamed block whose norms and history take the last slot leaves
		// nothing to swap through, and the next page-in fails the device.
		if n := g.slotsIfPerm(perm); !mayPage || !g.stream[li] || n < minSlots {
			// Only the final pass counts as a decline: GPU.PrepLayer asks every
			// device with mayPage false first.
			if mayPage {
				g.Declined++
			}
			// Say which of the three it is: another device may still have room,
			// the block is placed resident, or the budget is too small to swap
			// through.
			why := "another device is asked first"
			if mayPage {
				why = "it is placed resident and does not fit, so it runs on the host " +
					"(a placement that streams it -- model.Place.Stream, -placement N=DEV~ -- " +
					"swaps it through the device instead)"
				if g.stream[li] {
					why = fmt.Sprintf("the budget holds %d block slot(s), below the %d a swap needs", n, minSlots)
				}
			}
			g.LastErr = fmt.Sprintf("block %d needs %d bytes, %d of %d used (pool %s: %d of %d); %s",
				li, need, g.used, g.limit, g.pool.Name(), g.pool.Used(), g.pool.Limit(), why)
			// Recorded on the final pass only, like Declined.
			if mayPage && g.DeclineWhy == "" {
				g.DeclineWhy = why
			}
			return false
		}
		// At load the scan ascends from here, so the victim is the block just
		// behind this one and a resident prefix remains: the set the run-time
		// policy settles on, so the first token starts warm. See victim().
		if !g.reclaim(need, li, 0, li+1) {
			g.Declined++
			g.LastErr = fmt.Sprintf("block %d needs %d bytes and nothing can be paged out (%d of %d used)",
				li, need, g.used, g.limit)
			if g.DeclineWhy == "" {
				g.DeclineWhy = "nothing can be paged out"
			}
			return false
		}
	}
	// The staging is sized after the block is admitted, from what is left
	// (need is what the block still has to spend). If that is not enough it is
	// not allocated and the covered tensors are packed on the host. It is
	// allocated once and survives into later uploads and page-ins.
	//
	// The bytes are fetched here, after the decision and before the first read:
	// nn.LayerWeights.Ensure makes the contents valid (a paged mixture sets it),
	// and calling it earlier would read a bank for every block the budget then
	// refuses.
	if w.Ensure != nil {
		if err := w.Ensure(w); err != nil {
			g.LastErr = fmt.Sprintf("block %d: %v", li, err)
			return false
		}
	}
	g.ensureRaw(ws, need)
	// The block's matrices past the positional list: each uploaded and paged
	// with the block, and bound by role (layer.roles).
	l.extra = make([]*resident, len(ws)-numPositional)
	l.mvExtra = make([]mv, len(ws)-numPositional)
	rs := l.tensors()
	ms := []*mv{&l.mvq, &l.mvk, &l.mvv, &l.mvo, &l.mvg, &l.mvu, &l.mvd,
		&l.mvsg, &l.mvsu, &l.mvsd, &l.mvSG, &l.mvBA, &l.mvSO,
		// MLA's five, in weightList/tensors() order.
		&l.mvQA, &l.mvQB, &l.mvKVA, &l.mvKB, &l.mvVB,
		// KDA's four, in weightList/tensors() order.
		&l.mvFA, &l.mvFB, &l.mvGA, &l.mvGB,
		// Gemma 4's per-layer embedding gate and projection.
		&l.mvpg, &l.mvpp,
		// DeepSeek V3.2's indexer, Gemma 3n's LAuReL, then Falcon-H1's mixer
		// projection, last.
		&l.mvIQ, &l.mvIK, &l.mvIW,
		&l.mvLL, &l.mvLR, &l.mvSI,
		// MiniMax-M3's indexer query (its key is slot 25's).
		&l.mvMQ,
		// DeepSeek V4's grouped output and compressors.
		&l.mvWoA, &l.mvCKV, &l.mvCG, &l.mvICKV, &l.mvICG,
		// Kimi-K3's MLA output gate and latent mixture projections.
		&l.mvMG, &l.mvRD, &l.mvRU,
		// Qwen3.8-Flash-Next's hyper-connections and n-gram projections.
		&l.mvHAD, &l.mvHAU, &l.mvHAI, &l.mvHFD, &l.mvHFU, &l.mvHFI, &l.mvNGK, &l.mvNGV}
	for j := range l.mvExtra {
		ms = append(ms, &l.mvExtra[j])
	}
	// Same order as rs/ms. Attention carries biases on every architecture that
	// has them; a vision block also on up and down. For every text model the
	// FFN entries stay nil, so their kernels are byte-identical to before
	// (TestDenseKernelsUnchanged).
	bias := [][]float32{w.Bq, w.Bk, w.Bv, w.Bo, w.BGate, w.BUp, w.BDown,
		nil, nil, nil, nil, nil, nil,
		// MLA's five carry no bias on any architecture that has them, and
		// neither do KDA's four, the per-layer embedding's two, the indexer's
		// three, LAuReL's two, the mixer's projection or MiniMax-M3's query.
		nil, nil, nil, nil, nil,
		nil, nil, nil, nil, nil, nil,
		nil, nil, nil, nil, nil, nil,
		nil,
		nil, nil, nil, nil, nil,
		nil, nil, nil,
		nil, nil, nil, nil, nil, nil, nil, nil}
	// The latent block's own slots: 0 (q) is the one-step query and is empty
	// where 13/14 are populated; 1 and 2 are always empty (MLA has no k or v
	// projection); 15..17 are always there. Each is an absent weight, not a
	// failed upload.
	l.mla = p.MLA()
	mlaSkip := func(i int) bool {
		if !l.mla {
			// No latent block, no latent matrices -- except Mamba-1's four,
			// which ride KDA's slots 18..21 (nn.SSMWeights).
			return i >= 13 && !(i >= 18 && p.Recurrent.Mamba1)
		}
		// A linear block of an MLA model (Kimi-Linear) has none of them either:
		// l.mla is a property of the model, not the block.
		if p.Recurrent.Conv > 0 && i >= 13 && i < 18 {
			return true
		}
		switch i {
		case 0:
			// A linear block's q is its fused q|k|v whatever the model's
			// query rank (Kimi-K3 compresses its MLA query; Kimi-Linear
			// does not).
			return p.QLoraRank != 0 && p.Recurrent.Conv == 0
		case 1, 2:
			return true
		case 13, 14:
			return p.QLoraRank == 0
		}
		return false
	}
	for i, x := range ws {
		if i >= numPositional {
			// A matrix only the device program binds: one expert, no bias,
			// its own activation, uploaded as any other.
			r := mkw(i, x)
			if r == nil || !r.ok {
				if g.LastErr == "" {
					g.LastErr = fmt.Sprintf("block %d: %v (%s %dx%d) not resident", li, x.Role, x.T, x.Rows, x.K)
				}
				return false
			}
			*rs[i] = r
			k, ok := mkkID(x, r, 1, 0, false, false)
			if !ok {
				return false
			}
			*ms[i] = k
			continue
		}
		// A DeepSeek V4 block names its own slots (ds4Slot): the generic
		// skips below are for every other architecture.
		if p.DS4 != nil {
			want := ds4Slot(p, i)
			if want && len(x.Data) == 0 {
				g.LastErr = fmt.Sprintf("block %d: a DeepSeek V4 block and %s is empty", li, weightRole(i))
				return false
			}
			if !want {
				continue
			}
		} else if i >= 31 && i <= 35 {
			continue // DeepSeek V4's slots (31..35), absent everywhere else
		}
		// Qwen3.8-Flash-Next's (39..46), on the blocks the plan says carry
		// them and nowhere else.
		if i >= 39 {
			if !q4eSlot(p, i) {
				continue
			}
			if len(x.Data) == 0 {
				g.LastErr = fmt.Sprintf("block %d: a Qwen3.8-Flash-Next block and %s is empty", li, weightRole(i))
				return false
			}
		}
		// Kimi-K3's MLA output gate (36) and latent projections (37, 38), on
		// the blocks the plan says carry them.
		if i >= 36 && i <= 38 {
			want := i == 36 && p.MLA() && p.Recurrent.Conv == 0 && len(x.Data) != 0 ||
				i >= 37 && p.K3 != nil && p.K3.Latent != 0
			if !want {
				continue
			}
			if len(x.Data) == 0 {
				g.LastErr = fmt.Sprintf("block %d: the plan has a latent mixture and %s is empty",
					li, weightRole(i))
				return false
			}
		}
		// The per-layer embedding's two (22, 23), the indexer's three (24..26),
		// Gemma 3n's LAuReL (27, 28) and Falcon-H1's mixer projection (29),
		// past every other range.
		extra := i >= 22
		if i >= 24 && i <= 28 && p.DS4 == nil {
			// MiniMax-M3's indexer has a key (25) and its own query (30),
			// not DeepSeek V3.2's query latent (24) or weights (26).
			if (i < 27 && (p.IdxHeads == 0 || (p.MSA() || p.QSA()) && i != 25)) || (i >= 27 && p.AltUp == 0) {
				continue
			}
			if len(x.Data) == 0 {
				g.LastErr = fmt.Sprintf("block %d: the plan has an indexer or LAuReL and %s is empty",
					li, weightRole(i))
				return false
			}
		}
		ple := i == 22 || i == 23
		if ple {
			if p.PLEDim == 0 {
				continue
			}
			if len(x.Data) == 0 {
				g.LastErr = fmt.Sprintf("block %d: the plan has per-layer embeddings and %s is empty",
					li, weightRole(i))
				return false
			}
		}
		// The mixer's own projection beside the attention's four (Falcon-H1, 29),
		// absent by construction everywhere else.
		if i == 29 {
			if !p.Recurrent.WithAttn {
				continue
			}
			if len(x.Data) == 0 {
				g.LastErr = fmt.Sprintf("block %d: the plan runs attention beside a Mamba-2 mixer "+
					"and %s is empty", li, weightRole(i))
				return false
			}
		}
		// MiniMax-M3's indexer query (30), absent by construction elsewhere.
		if i == 30 {
			if !p.MSA() && !p.QSA() {
				continue
			}
			if len(x.Data) == 0 {
				g.LastErr = fmt.Sprintf("block %d: the plan selects key blocks and %s is empty",
					li, weightRole(i))
				return false
			}
		}
		// KDA's four low-rank gate matrices, checked before mlaSkip because they
		// sit past MLA's range and are not MLA's.
		if i >= 18 && !extra && p.DS4 == nil {
			switch {
			case !p.Recurrent.ChanDecay && !p.Recurrent.Mamba1, p.Recurrent.Conv == 0:
				// Not a KDA model, or an attention block of one: absent by
				// construction rather than by a missing tensor.
				continue
			case (i == 20 || i == 21) && p.Recurrent.ChanDecay && len(w.SSM.Gate.Data) != 0:
				// Kimi-K3's full-rank output gate is slot 10's, and there is no
				// low-rank pair.
				continue
			case len(x.Data) == 0:
				g.LastErr = fmt.Sprintf("block %d: the plan is a Kimi Delta Attention "+
					"block and %s is empty", li, weightRole(i))
				return false
			}
			// A present weight falls through to the upload; a continue here would
			// leave l.ssmGA nil and crash inside the recorded session.
		}
		if p.DS4 == nil && !extra && mlaSkip(i) {
			continue
		}
		if i >= 13 && !extra {
			if len(x.Data) == 0 {
				g.LastErr = fmt.Sprintf("block %d: the plan is a latent attention block "+
					"(KVLoraRank=%d) and %s is empty", li, p.KVLoraRank, weightRole(i))
				return false
			}
		}
		// An ungated block has no gate matrix: slot 4 is empty by construction.
		if ungatedFFN(p) && i == 4 {
			continue
		}
		// A KV-sharing block has no k or v matrix (slots 1, 2): it attends to
		// its source's history.
		if p.KVShared && (i == 1 || i == 2) {
			continue
		}
		// Nor has a block whose v is k's projection (Gemma 4's global layers)
		// a v matrix: slot 2. The emit reads k's projection for v.
		if p.VFromK && i == 2 {
			if len(x.Data) != 0 {
				g.LastErr = fmt.Sprintf("block %d: v is k's projection and the block offers a v matrix", li)
				return false
			}
			continue
		}
		// A block that is its mixer alone has no FFN at all.
		if p.NoFFN && i >= 4 && i <= 6 {
			continue
		}
		// A linear block has no k, v or o matrix: its one projection is slot 0
		// (Wq carries the fused q|k|v|gate), and offerRange clears the other three.
		if p.Recurrent.Conv > 0 && !p.Recurrent.WithAttn && i >= 1 && i <= 3 {
			continue
		}
		if i >= 10 && i < 13 {
			// A linear block's three; an attention block leaves them empty.
			if p.Recurrent.Conv == 0 {
				continue
			}
			// Kimi-Linear's KDA has no ssm_gate: its output gate is the
			// low-rank ssm_g_a/ssm_g_b pair. Kimi-K3's is one full-rank matrix,
			// here.
			if i == 10 && p.Recurrent.ChanDecay && len(w.SSM.GA.Data) != 0 {
				continue
			}
			// Mamba-1 has no ssm_ba: dt comes off its x_proj (slot 18).
			if i == 11 && p.Recurrent.Mamba1 {
				continue
			}
			if len(x.Data) == 0 {
				g.LastErr = fmt.Sprintf("block %d: the plan is a linear block and %s is empty",
					li, weightRole(i))
				return false
			}
		}
		if i >= 7 && i < 10 {
			if p.NFFNShExp == 0 {
				continue // no shared expert in this architecture at all
			}
			// An ungated shared expert (Nemotron 3) has no gate matrix.
			if i == 7 && ungatedFFN(p) {
				continue
			}
			// A plan with a shared expert and an empty weight is a decline: a
			// skip would run the FFN on an unwritten buffer.
			if len(x.Data) == 0 {
				g.LastErr = fmt.Sprintf("block %d: the plan has a shared expert "+
					"(NFFNShExp=%d) and %s is empty", li, p.NFFNShExp,
					[...]string{"ShGate", "ShUp", "ShDown"}[i-7])
				return false
			}
		}
		e, sl := 1, 0
		// MLA's two absorb banks are one sheet per head, and the "expert" indexed
		// is the head: every head is selected on every token, so the selection is
		// the identity 0..NHead-1 and one indexed matvec replaces NHead, each slot
		// reading its own activation like the mixture's down projection.
		if l.mla && (i == 16 || i == 17) {
			e, sl = p.NHead, p.NHead
		}
		// DeepSeek V4's grouped output is one sheet per group, every group
		// selected: the identity, as MLA's heads are.
		if p.DS4 != nil && i == 31 {
			e, sl = p.DS4.OGroups, p.DS4.OGroups
		}
		// 4..6 only: the shared expert's matrices at 7..9 are dense.
		if moe && i >= 4 && i <= 6 { // gate, up and down are the banks
			e, sl = bank, slots
			if stream {
				// The kernel is told the bank is k experts and indexes it with
				// 0..k-1; backend.TestIndexedMatVecMatchesACompactBank is where
				// "the same answer, bit for bit" comes from. With an expert
				// cache it is told the cache's size and indexed by slot.
				if cslots < 0 {
					// An auto-streamed block is placed on the plain bank;
					// its cache comes after placement (sizeAutoCaches).
					cslots = g.cacheSlotsFor(p, ws, g.StreamCacheSlots, slots, bank,
						w.ExpGateB != nil || w.ExpUpB != nil || w.ExpDownB != nil)
				}
				e = cslots
			}
		}
		// Weight.Rows already counts the whole bank (model.get multiplies every
		// trailing dimension), which PackWeights and the upload want; the kernel is
		// built for one expert's rows and does the offset, so it is divided below.
		var r *resident
		if hybrid && i >= 4 && i <= 6 {
			// Nothing of the bank goes on the card; the host runs it.
			if l.stream == nil {
				l.stream = &streamBank{sel: make([]uint32, slots),
					ensure: w.Ensure, sel2: w.EnsureExperts,
					pre: w.PrefetchExperts, spec: w.SpeculateExperts, host: w.HostExperts, w: *w, hybrid: true, nExpert: bank, li: li}
				g.StreamBlocks++
			}
			continue
		}
		if stream && i >= 4 && i <= 6 {
			q, qok := quantOf(x.T)
			if !qok {
				g.LastErr = fmt.Sprintf("block %d: %s is %s, which has no device kernel",
					li, weightRole(i), x.T)
				return false
			}
			if x.Packed == nil {
				// A streamed bank needs the packed layout, since fill() sends a
				// sheet at a time. A float bank (an F32 or F16 file) has none, so
				// such a mixture streams nothing and is declined here by name.
				g.LastErr = fmt.Sprintf("block %d: %s is not pre-packed, so its bank cannot be streamed",
					li, weightRole(i))
				return false
			}
			var sh [3]int
			var ok bool
			if cslots > slots {
				// The block's own bank of cslots sheets, kept across tokens:
				// it is the expert cache, so it cannot be the device's one
				// shared compact bank. Its charge is the block's.
				r, sh, ok = g.residentCompact(q, x.Rows/slots, x.K, cslots)
			} else {
				r, sh, ok = g.sharedCompact(i, q, x.Rows/slots, x.K, slots)
			}
			if !ok {
				return false
			}
			if l.stream == nil {
				// w is seeded here so a host side that does not page (Ensure nil)
				// still has valid spans; otherwise fill() sends nothing and the FFN
				// contributes zeros.
				l.stream = &streamBank{sel: make([]uint32, slots),
					ensure: w.Ensure, sel2: w.EnsureExperts,
					pre: w.PrefetchExperts, spec: w.SpeculateExperts, host: w.HostExperts, w: *w, li: li}
				l.stream.nExpert = bank
				// An auto-streamed block runs its experts on the host unless
				// told otherwise: on Kimi-K3 over the V100s that decoded 1.43x
				// the host's rate where sending the sheets lost to it
				// (placement.md 16c-2).
				switch {
				case l.stream.host == nil || auto && mode == expCard:
				case auto && mode == expHost, g.HybridExperts:
					l.stream.hybrid = true
				default:
					l.stream.hybrid = auto && !g.NoHybrid
				}
				l.stream.initCache(cslots, bank)
				g.StreamBlocks++
			}
			l.stream.sh[i-4] = sh
		} else {
			r = mkw(i, x)
		}
		if r == nil || !r.ok {
			if g.LastErr == "" {
				g.LastErr = fmt.Sprintf("block %d tensor %d (%s %dx%d) not resident",
					li, i, x.T, x.Rows, x.K)
			}
			return false
		}
		*rs[i] = r
		if stream && i >= 4 && i <= 6 {
			// A streamed bank's rows were scaled to the selection (slots
			// sheets) above, whatever size its cache is.
			x.Rows /= slots
		} else if e > 1 {
			// e, not bank: they differ for a streamed bank, whose kernel is built for
			// k experts. This is rows per expert either way.
			if x.Rows%e != 0 {
				return false
			}
			x.Rows /= e
		}
		// slotAct: each slot reads its own activation vector. The mixture's
		// down projection (6) does, and so do MLA's two banks -- head h absorbs
		// its own nope half and un-absorbs its own attention result.
		k, ok := mkkID(x, r, e, sl, i == 6 || i == 16 || i == 17 || i == 31, bias[i] != nil)
		if !ok {
			return false
		}
		if (i == 3 || i == 6) && e <= 1 && bias[i] == nil && !p.NonCausal && k.kern != nil {
			k.res = g.kernelMode(k.q, k.k, k.rows, k.split, max(k.rowt, 1), true, k.group)
		}
		if i == 5 && e <= 1 && bias[i] == nil && !ungatedFFN(p) && k.kern != nil && k.rowt <= 1 &&
			(p.Act == kernels.ActSiLU || p.Act == kernels.ActGELU) {
			k.gated = g.gatedKernel(k.q, k.k, k.rows, k.split, k.group, p.Act)
			k.gate2 = g.gate2Kernel(k.q, k.k, k.rows, k.split, k.group, p.Act)
			// nil at a partial-writing split; the retune builds it once the
			// split it picks has a final-row form.
			k.wantGated = true
		}
		*ms[i] = k
	}
	if moe {
		// The router is unquantized, so it does not go through resident() at
		// all -- it is uploaded as the row-major floats it arrives as.
		if w.Router.Rows != p.NExpert || w.Router.K != p.NEmbd {
			return false
		}
		// F32 or F16, with the bytes counted against the type (Mixtral's router
		// is F16; reading it as F32 runs past the buffer). Any other type is
		// refused by name. Data may run past the matrix, so it is sliced to it.
		es := 0
		switch w.Router.T {
		case quant.F32:
			es = 4
		case quant.F16:
			es = 2
		}
		n := p.NExpert * p.NEmbd
		if es == 0 || len(w.Router.Data) < es*n {
			g.LastErr = fmt.Sprintf("block %d: the router is %v in %d bytes, and the "+
				"device reads an F32 or F16 router of %dx%d", li, w.Router.T,
				len(w.Router.Data), p.NExpert, p.NEmbd)
			return false
		}
		rk, e := kernels.RouterMatVecOf(p.NExpert, p.NEmbd, es == 2)
		if e != nil {
			g.LastErr = e.Error()
			return false
		}
		if l.mvRouter, e = g.dev.Compile(rk); e != nil {
			g.LastErr = e.Error()
			return false
		}
		// Padded to a whole word: the f16 kernel loads the word an element
		// sits in, and an odd element count's last word runs past the matrix.
		rb := w.Router.Data[:es*n]
		if len(rb)%4 != 0 {
			rb = append(append([]byte(nil), rb...), make([]byte, 4-len(rb)%4)...)
		}
		var b backend.Buf
		if b, e = g.dev.Alloc(len(rb)); e != nil {
			return false
		}
		if e = b.Write(rb); e != nil {
			return false
		}
		l.router, l.routerF16 = b, es == 2
	}
	var err error
	// Norms and biases arrive as f32, which is what the device reads.
	// upped is every host vector uploaded below and its buffer, so a vector
	// the program binds by role (layer.vecs) reuses the upload.
	upped := make([]upload, 0, 32)
	up := func(dst *backend.Buf, v []float32) {
		// A KV-sharing block's k norm is absent: it caches nothing.
		if err != nil || len(v) == 0 {
			return
		}
		f := v
		var b backend.Buf
		if b, err = g.dev.Alloc(len(f) * 4); err != nil {
			return
		}
		err = b.Write(f32b(f))
		*dst = b
		upped = append(upped, upload{&v[0], len(v), b})
	}
	if p.NFFNShExp > 0 && w.ShRouter != nil {
		// The shared gate is a vector producing one logit, uploaded as plain
		// floats like a norm. A nil gate is an ungated shared expert (DeepSeek's,
		// weight 1), matching the host's `if l.shRouter != nil`; a present gate of
		// the wrong width is an error.
		if len(w.ShRouter) != p.NEmbd {
			g.LastErr = fmt.Sprintf("block %d: the shared expert's gate is %d wide, want %d",
				li, len(w.ShRouter), p.NEmbd)
			return false
		}
		up(&l.shRouter, w.ShRouter)
	}
	// A post-norm block (OLMo 2, EXAONE 4) has neither pre-norm: the emit
	// copies the residual where the norm would have written (bs.copyRows).
	noPre := w.AttnNorm == nil && w.PostAttnNorm != nil && w.PostFFNNorm != nil
	// Qwen3.8-Flash-Next's sublayers read their hyper-connection's mix, whose
	// grouped norm stands where the pre-norms would (q4e.go).
	noPre = noPre || p.Q4E != nil && w.AttnNorm == nil && w.FFNNorm == nil
	// ModernBERT's first block has no attention norm and keeps its FFN norm:
	// the attention reads the residual as it is, the same copy.
	noAttnNorm := w.AttnNorm == nil && w.FFNNorm != nil && p.NoPreNorm && p.NonCausal
	if noPre && !p.NoPreNorm && p.Q4E == nil {
		g.LastErr = fmt.Sprintf("block %d has no pre-norm on a plan that does not say so", li)
		return false
	}
	if w.AttnNorm != nil {
		up(&l.nAttn, w.AttnNorm)
	} else if !noPre && !noAttnNorm {
		g.LastErr = fmt.Sprintf("block %d has no attention norm", li)
		return false
	}
	// A parallel block sharing one norm has no FFN norm (phi-2) and the FFN reads
	// the attention's normed row; every other block needs one.
	switch {
	case w.FFNNorm != nil:
		up(&l.nFFN, w.FFNNorm)
	case noPre, p.NoFFN:
	case !p.Parallel:
		g.LastErr = fmt.Sprintf("block %d has no FFN norm and is not a parallel block", li)
		return false
	}
	// Lengths are checked because a wrong one runs: a short sink array is read
	// past its end, and an expert's bias slice would land on its neighbour's rows.
	for _, v := range []struct {
		name     string
		on       bool
		have     []float32
		want     int
		dst      *backend.Buf
		optional bool
	}{
		{"attention sinks", p.AttnSinks, w.Sinks, p.NHead, &l.sinks, false},
		{"router bias", p.MoEBias, w.RouterB, p.NExpert, &l.routerB, true},
		{"expert gate bias", p.MoEBias, w.ExpGateB, p.NExpert * p.NFFNExp, &l.expGateB, true},
		{"expert up bias", p.MoEBias, w.ExpUpB, p.NExpert * p.NFFNExp, &l.expUpB, true},
		{"expert down bias", p.MoEBias, w.ExpDownB, p.NExpert * p.NEmbd, &l.expDownB, true},
		// The block, not just the model, must be a mixture: DeepSeek's leading
		// dense blocks carry no exp_probs_b.
		{"expert selection bias", p.ExpertSelBias && p.NExpert > 0 && !p.HashExperts, w.ExpSelB, p.NExpert, &l.expSelB, false},
	} {
		if !v.on || (v.optional && v.have == nil) {
			continue
		}
		if len(v.have) != v.want {
			g.LastErr = fmt.Sprintf("block %d: %s is %d wide, want %d", li, v.name, len(v.have), v.want)
			return false
		}
		up(v.dst, v.have)
	}
	if w.PostAttnNorm != nil {
		up(&l.nPostAttn, w.PostAttnNorm)
	}
	// BitNet's inner norms: both, at their widths, on every such block.
	if p.SubNorm {
		if len(w.AttnSubNorm) != p.NHead*p.HeadDim || len(w.FFNSubNorm) != p.NFFN {
			g.LastErr = fmt.Sprintf("block %d: inner norms %d and %d wide, want %d and %d", li,
				len(w.AttnSubNorm), len(w.FFNSubNorm), p.NHead*p.HeadDim, p.NFFN)
			return false
		}
		up(&l.nSubAttn, w.AttnSubNorm)
		up(&l.nSubFFN, w.FFNSubNorm)
	} else if w.AttnSubNorm != nil || w.FFNSubNorm != nil {
		g.LastErr = fmt.Sprintf("block %d carries inner norms the plan does not name", li)
		return false
	}
	if w.PostFFNNorm != nil {
		up(&l.nPostFFN, w.PostFFNNorm)
	}
	if p.Clamps {
		if len(w.Clamp) != 28 {
			g.LastErr = fmt.Sprintf("block %d: clipped-linear bounds are %d floats, want 28", li, len(w.Clamp))
			return false
		}
		up(&l.clampB, w.Clamp)
	}
	// Gemma 4's mixture block: four norms and the expert factors, charged
	// with the norms.
	if p.DenseMoE {
		for _, v := range []struct {
			name string
			have []float32
			want int
			dst  *backend.Buf
		}{
			{"experts' pre-norm", w.FFNNorm2, p.NEmbd, &l.nFFN2},
			{"dense MLP's post-norm", w.PostFFNNorm1, p.NEmbd, &l.nPost1},
			{"experts' post-norm", w.PostFFNNorm2, p.NEmbd, &l.nPost2},
			{"router input's norm", w.RouterNorm, p.NEmbd, &l.nRouter},
			{"per-expert scale", w.ExpScale, p.NExpert, &l.expScale},
		} {
			if len(v.have) != v.want {
				g.LastErr = fmt.Sprintf("block %d: %s is %d wide, want %d", li, v.name, len(v.have), v.want)
				return false
			}
			up(v.dst, v.have)
			l.auxBytes += uint64(4 * v.want)
		}
	}
	// Gemma 4's per-layer embedding norm and the block's slice offset.
	if p.PLEDim != 0 {
		if len(w.PLEPost) != p.NEmbd {
			g.LastErr = fmt.Sprintf("block %d: the per-layer embedding norm is %d wide, want %d",
				li, len(w.PLEPost), p.NEmbd)
			return false
		}
		up(&l.nPLE, w.PLEPost)
		if err == nil {
			if l.pleOff, err = g.dev.Alloc(4); err == nil {
				err = l.pleOff.Write(g.u32b(uint32(li * p.PLEDim)))
			}
		}
		l.auxBytes += uint64(4*p.NEmbd + 4)
	}
	if why := g.prepAltUp(l, li, p, w, up); why != "" {
		g.LastErr = why
		return false
	}
	if why := g.prepDS4(l, li, p, w, up); why != "" {
		g.LastErr = why
		return false
	}
	if why := g.prepK3(l, li, p, w, up); why != "" {
		g.LastErr = why
		return false
	}
	if why := g.prepQ4E(l, li, p, w, up); why != "" {
		g.LastErr = why
		return false
	}
	// Gemma 4's output scalar, one float, charged with the norms.
	if p.OutScale != 0 {
		up(&l.outScale, []float32{p.OutScale})
		l.auxBytes += 4
	}
	// Apertus's xIELU numbers, four floats, the block's own.
	if p.Act == kernels.ActXIELU {
		if len(w.XIELU) != 4 {
			g.LastErr = fmt.Sprintf("block %d: xIELU wants 4 numbers, the block carries %d", li, len(w.XIELU))
			return false
		}
		up(&l.xielu, w.XIELU)
		l.auxBytes += 16
	}
	if w.AttnNormB != nil {
		up(&l.nAttnB, w.AttnNormB)
	}
	// BERT's norms after the residual adds: the attention reads the block
	// input (no attention norm), FFNNorm normalises x + attention and the
	// residual takes it, ResidNorm normalises x + mlp.
	if p.PostResidNorm {
		if w.AttnNorm != nil || w.FFNNorm == nil || len(w.ResidNorm) != p.NEmbd ||
			(w.ResidNormB != nil && len(w.ResidNormB) != p.NEmbd) {
			g.LastErr = fmt.Sprintf("block %d: a post-norm block wants no attention norm and two "+
				"norms of %d after its adds", li, p.NEmbd)
			return false
		}
		up(&l.nResid, w.ResidNorm)
		if w.ResidNormB != nil {
			up(&l.nResidB, w.ResidNormB)
		}
	}
	if w.FFNNormB != nil {
		up(&l.nFFNB, w.FFNNormB)
	}
	// The attention biases, uploaded as float32 like every other aux vector.
	// The kernel was already generated with pBias for exactly these slots.
	for _, bp := range []struct {
		v []float32
		m *mv
	}{{w.Bq, &l.mvq}, {w.Bk, &l.mvk}, {w.Bv, &l.mvv}, {w.Bo, &l.mvo},
		// A vision block's MLP is biased too; skipping these runs and is wrong
		// in exactly the way the attention biases are. A gated one's gate as
		// well (Qwen2.5-VL); nil on every text block.
		{w.BGate, &l.mvg}, {w.BUp, &l.mvu}, {w.BDown, &l.mvd}} {
		if bp.v == nil {
			continue
		}
		up(&bp.m.bias, bp.v)
		if bp.m.alt != nil {
			bp.m.alt.bias = bp.m.bias
		}
		l.biasBytes += uint64(len(bp.v) * 4)
	}
	// After the biases: the fused kernel bakes which segments take one, and the
	// launch appends a buffer for every bias it finds. CUDA does not check the
	// count, so a mismatch shifts every later buffer.
	g.fuseQKV(l)
	if w.QNorm != nil {
		up(&l.nQ, w.QNorm)
		up(&l.nK, w.KNorm)
	}
	// MiniMax-M3's indexer norms, each one head wide.
	l.qsa = p.QSA()
	if l.msa = p.MSA(); l.msa || l.qsa {
		if len(w.IdxQNorm) != p.IdxHeadDim || len(w.IdxKNorm) != p.IdxHeadDim {
			g.LastErr = fmt.Sprintf("block %d: the indexer norms are %d/%d wide, want %d",
				li, len(w.IdxQNorm), len(w.IdxKNorm), p.IdxHeadDim)
			return false
		}
		up(&l.nIdxQ, w.IdxQNorm)
		up(&l.nIdxK, w.IdxKNorm)
		l.auxBytes += uint64(2 * p.IdxHeadDim * 4)
	}
	// A linear block of an MLA model has neither latent norm (see mlaSkip).
	if l.mla && p.Recurrent.Conv == 0 {
		// Widths are checked because a wrong one runs: KVANorm covers the latent
		// only, and a whole-row weight would norm the rotary key too.
		if len(w.KVANorm) != p.KVLoraRank {
			g.LastErr = fmt.Sprintf("block %d: the kv latent norm is %d wide, want %d",
				li, len(w.KVANorm), p.KVLoraRank)
			return false
		}
		up(&l.nKVA, w.KVANorm)
		l.auxBytes += uint64(len(w.KVANorm) * 4)
		if p.QLoraRank != 0 {
			if len(w.QANorm) != p.QLoraRank {
				g.LastErr = fmt.Sprintf("block %d: the query latent norm is %d wide, want %d",
					li, len(w.QANorm), p.QLoraRank)
				return false
			}
			up(&l.nQA, w.QANorm)
			l.auxBytes += uint64(len(w.QANorm) * 4)
		}
		if p.IdxHeads != 0 {
			if len(w.IdxKNorm) != p.IdxHeadDim || len(w.IdxKNormB) != p.IdxHeadDim {
				g.LastErr = fmt.Sprintf("block %d: the indexer key norm is %d/%d wide, want %d",
					li, len(w.IdxKNorm), len(w.IdxKNormB), p.IdxHeadDim)
				return false
			}
			up(&l.nIdxK, w.IdxKNorm)
			up(&l.nIdxKB, w.IdxKNormB)
			l.auxBytes += uint64(2 * p.IdxHeadDim * 4)
		}
	}
	if r := p.Recurrent; r.Conv > 0 {
		// The convolution weight stays plane-major, W[u][c] with the host's stride,
		// so migrating the state is a memcpy rather than a shuffle.
		if len(w.SSM.Conv1d) != r.Conv*r.Chans {
			g.LastErr = fmt.Sprintf("block %d: conv1d is %d floats, want %d x %d",
				li, len(w.SSM.Conv1d), r.Conv, r.Chans)
			return false
		}
		up(&l.ssmConv1d, w.SSM.Conv1d)
		l.auxBytes += uint64(len(w.SSM.Conv1d) * 4)
	}
	// A short convolution (LFM2) has its window and nothing else.
	if r := p.Recurrent; r.Conv > 0 && !r.ShortConv {
		up(&l.ssmA, w.SSM.A)
		up(&l.ssmDt, w.SSM.DtBias)
		if !r.NoNorm {
			up(&l.ssmNorm, w.SSM.Norm)
		}
		if r.Mamba1 {
			// Mamba-1's D and convolution bias, one a channel, and its three
			// norms where the block has them; lengths checked because a short
			// one is read past its end.
			if len(w.SSM.D) != r.VHeads || len(w.SSM.ConvBias) != r.Chans || len(w.SSM.A) != r.VHeads*r.KDim ||
				len(w.SSM.DtBias) != r.VHeads {
				g.LastErr = fmt.Sprintf("block %d: Mamba-1's D is %d, its conv bias %d, A %d and dt bias %d wide, "+
					"want %d, %d, %d and %d", li, len(w.SSM.D), len(w.SSM.ConvBias), len(w.SSM.A),
					len(w.SSM.DtBias), r.VHeads, r.Chans, r.VHeads*r.KDim, r.VHeads)
				return false
			}
			up(&l.ssmD, w.SSM.D)
			up(&l.ssmCB, w.SSM.ConvBias)
			l.auxBytes += uint64((len(w.SSM.D) + len(w.SSM.ConvBias)) * 4)
			if w.SSM.DtNorm != nil {
				if len(w.SSM.DtNorm) != r.Rank || len(w.SSM.BNorm) != r.KDim || len(w.SSM.CNorm) != r.KDim {
					g.LastErr = fmt.Sprintf("block %d: Mamba-1's dt/B/C norms are %d, %d and %d wide, want %d, %d and %d",
						li, len(w.SSM.DtNorm), len(w.SSM.BNorm), len(w.SSM.CNorm), r.Rank, r.KDim, r.KDim)
					return false
				}
				up(&l.m1DtN, w.SSM.DtNorm)
				up(&l.m1BN, w.SSM.BNorm)
				up(&l.m1CN, w.SSM.CNorm)
				l.auxBytes += uint64((r.Rank + 2*r.KDim) * 4)
			}
		}
		if r.SSD {
			// Mamba-2's two vectors more; lengths checked because a short one
			// is read past its end.
			if len(w.SSM.D) != r.VHeads || len(w.SSM.ConvBias) != r.Chans ||
				(!r.NoNorm && len(w.SSM.Norm) != r.VHeads*r.VDim) {
				g.LastErr = fmt.Sprintf("block %d: Mamba-2's D is %d, its conv bias %d and its norm %d wide, "+
					"want %d, %d and %d", li, len(w.SSM.D), len(w.SSM.ConvBias), len(w.SSM.Norm),
					r.VHeads, r.Chans, r.VHeads*r.VDim)
				return false
			}
			up(&l.ssmD, w.SSM.D)
			up(&l.ssmCB, w.SSM.ConvBias)
			l.auxBytes += uint64((len(w.SSM.D) + len(w.SSM.ConvBias)) * 4)
		}
		// A unit weight: the per-key-head L2 norm is an RMSNorm with a weight of
		// ones (as engine/model/delta.go builds it), so no second reduction kernel.
		ones := make([]float32, r.KDim)
		for i := range ones {
			ones[i] = 1
		}
		up(&l.ssmOnes, ones)
		l.auxBytes += uint64((len(w.SSM.A) + len(w.SSM.DtBias) + len(w.SSM.Norm) + len(ones)) * 4)
	}
	// One extra KV position is a trash slot for the batched prefill: a ragged
	// last chunk runs the full grid width with surplus rows zeroed, and those rows
	// still write K and V. A chunk-width of slack costs too much VRAM and a real
	// row's slot would be clobbered, so they all write here and nothing reads it.
	//
	// A linear block takes no KV cache at all: it keeps a running summary, and
	// the plan's NKVHead/HeadDim are the model-wide attention shape.
	linear := p.Recurrent.Conv > 0
	l.linear = linear
	l.withAttn = p.Recurrent.WithAttn
	l.noFFN = p.NoFFN
	l.centred = p.Recurrent.Centred
	if linear && !l.centred {
		if !g.addSessionRec(sid, l, p) {
			return false
		}
	}
	// A block that runs attention keeps a history, beside a recurrent state
	// where it runs a mixer too.
	attends := !linear || l.withAttn
	var cache, vcache int
	var kvp *kvPair
	l.kvSrc = li
	if p.KVShared {
		// No history of its own: it reads its source's, which must be on
		// this device already (blocks are offered in order).
		src := g.layers[p.KVSource]
		if src == nil || !src.ok || p.KVSource >= li {
			g.LastErr = fmt.Sprintf("block %d attends to block %d's history, which is not on this device",
				li, p.KVSource)
			return false
		}
		l.kvSrc = p.KVSource
	}
	if p.KVShared {
		// Nothing to allocate.
	} else if attends && g.pagedPlan(p) {
		err = g.pagedLayer(sid, l, li, p)
	} else if attends {
		g.initKVCap(p)
		cp := g.capped(p)
		// The V cache is half the size when packed (KVF16), which can decide
		// whether a model fits on the card at all. Under MLA it is zero: the value
		// is the key row's own prefix (kvPair.value).
		cache, vcache = g.kvCacheBytes(p, cp.MaxSeq)
		// One cache per session. PrepLayer runs for a session (sid), so
		// this allocates for whichever sequence is placing blocks; a second
		// State placing the same block reuses the shared weights above and gets
		// its own history. A non-causal block takes the device's one shared
		// pair instead (transientkv.go), and charges nothing of its own.
		kvp = &kvPair{}
		if p.NonCausal {
			if sh := g.transientKV(p); sh != nil {
				kvp, cache, vcache = sh, 0, 0
			}
		}
		for i, dst := range []*backend.Buf{&kvp.kc, &kvp.vc} {
			if kvp.shared {
				break
			}
			n := cache
			if i == 1 {
				n = vcache
			}
			if err == nil && n > 0 {
				*dst, err = g.dev.Alloc(n)
			}
		}
		if l.kv == nil {
			l.kv = map[uint64]*kvPair{}
		}
		if old := l.kv[sid]; old != nil && old.shared {
			g.transientKVPut()
		} else if old != nil {
			// Re-preparing a block this session already holds: its previous
			// history goes, or it leaks (see dropLayerLocal).
			if old.kc != nil {
				old.kc.Free()
			}
			if old.vc != nil {
				old.vc.Free()
			}
		}
		l.kv[sid] = kvp
	}
	if err != nil {
		g.LastErr = err.Error()
		return false
	}
	// WithPoisonKV fills the cache with NaN so an attention kernel reading a
	// position nothing wrote fails a gate instead of reading a lucky zero.
	if g.kb.poisonKV && kvp != nil && !kvp.shared {
		nan := make([]byte, cache)
		for i := 0; i+3 < len(nan); i += 4 {
			nan[i], nan[i+1], nan[i+2], nan[i+3] = 0x01, 0x00, 0xC0, 0x7F
		}
		kvp.kc.Write(nan)
		if kvp.vc != nil {
			kvp.vc.Write(nan[:vcache])
		}
	}
	// cache + vcache, not 2*cache: with KVF16 the V half is half the size.
	l.kvBytes = uint64(cache + vcache)
	if kvp != nil && !kvp.shared {
		kvp.bytes = l.kvBytes
	}
	// += because a linear block already charged its five vectors above, and
	// auxBytes is what ReleaseLayers refunds.
	l.auxBytes += uint64((2*len(w.AttnNorm)+len(w.PostAttnNorm)+len(w.PostFFNNorm)+len(w.Clamp))*4) + l.biasBytes
	g.charge(l.auxBytes + l.kvBytes)
	g.KVBytes += l.kvBytes
	l.ok = true
	// The page record is built for every block, including on a model that fits,
	// so a later demotion is expressible; the budget decides whether anything
	// moves. Its size is what the residents actually charged, so reclaim() asks
	// for exactly what a page-in will spend.
	var pb, pr uint64
	for _, rp := range l.tensors() {
		r := *rp
		pb += r.bytes()
		if n := r.bytes(); n > 0 {
			if f, err := g.planesFootprint(r.t, r.nrows, r.k); err == nil && f > n {
				pr += f - n
			}
		}
	}
	l.roles = rolesOf(ws, w.All)
	// The vectors the block's program binds, by role: the buffer an upload
	// above made from the same host floats, else one of their own.
	l.vecs, l.vecOwned = l.vecs[:0], l.vecOwned[:0]
	for _, r := range boundVecRoles(p.Graph) {
		var v []float32
		for _, x := range w.Vecs {
			if x.Role == r {
				v = x.V
				break
			}
		}
		if len(v) == 0 {
			continue
		}
		var b backend.Buf
		for _, u := range upped {
			if u.p == &v[0] && u.n == len(v) {
				b = u.b
				break
			}
		}
		if b == nil {
			if b, err = g.dev.Alloc(len(v) * 4); err == nil {
				err = b.Write(f32b(v))
			}
			if err != nil {
				g.LastErr = fmt.Sprintf("block %d: %v: %v", li, r, err)
				return false
			}
			// Counted with the block's other vectors, which its release
			// refunds; charged here, past where prep charged the rest.
			l.auxBytes += uint64(len(v) * 4)
			g.charge(uint64(len(v) * 4))
			l.vecOwned = append(l.vecOwned, b)
		}
		l.vecs = append(l.vecs, roleBuf{r, b})
	}
	l.rs, l.ms = rs, ms
	if stream {
		// A streamed block is pinned (pg == nil, which every pager path reads as
		// "resident, never a victim"): its base is small and its bank is not
		// resident, so there is nothing to relocate. Paging it would also be
		// wrong, since pageIn would refresh the bank at NExpert sheets against an
		// admitted shape of k.
		l.pg = nil
	} else {
		l.pg = &page{ws: append([]nn.Weight(nil), ws...), bytes: pb, round: pr, in: true,
			// Kept so a page-in can ask again: ws points into a reused host frame.
			ensure: w.Ensure}
		g.pagesIn++
		// Inside the else because ReleaseLayers refunds pageBytes only when
		// l.pg.in was true.
		g.pageBytes += pb + pr
	}
	// Re-preparing a block frees the layer it replaces. Only the per-layer
	// buffers go: the weights live in g.res keyed by their bytes and may be
	// shared with the layer being installed.
	if old := g.layers[li]; old != nil && old != l {
		g.dropLayerLocal(old)
	}
	l.prog, l.progWhy = buildDevProgram(p, l)
	g.layers[li] = l
	g.layerGen++
	// A non-causal block's batched twins are built here, not lazily: every
	// call over it is batched, and a Compile from inside layersOnce would post
	// to the backend goroutine that is blocked running the session -- a
	// deadlock. prepBatch walks the causal layers eagerly for the same reason.
	if p.NonCausal && g.bs != nil {
		for _, m := range []*mv{&l.mvq, &l.mvk, &l.mvv, &l.mvo, &l.mvg, &l.mvu, &l.mvd, &l.mvSG, &l.mvBA, &l.mvSO} {
			if m.kern == nil {
				continue
			}
			if _, ok := g.batchMV(*m, g.bs.rows, g.bs.tok); !ok {
				g.LastErr = fmt.Sprintf("non-causal block %d: no batched twin at %d rows",
					li, g.bs.rows)
				return false
			}
		}
	}
	// The reserved widths' batched twins are built here, against this block,
	// as the non-causal twins are above. Their shared buffers -- the binary16
	// activation and the k-split partials of sm_70's tensor-core twin
	// (voltaMV) -- are sized by the block's matrices, which reserveScratch,
	// running before the first block is installed, cannot see. Left to the
	// first prompt or step, they were asked of a card the blocks had filled:
	// 12 MB for Llama-3.2-1B on a Volta-class card, a driver out-of-memory on a
	// full card and the last swap slot on a streaming one.
	if !p.NonCausal {
		for _, w := range []int{g.promptW, g.stepW} {
			if w > 0 {
				g.prepBatch(w)
			}
		}
	}
	// Admitting a block charges its KV cache forever, so the slot count just
	// fell; give back whatever no longer fits under it.
	g.trim()
	return true
}

// poisonBuf fills a freshly allocated device buffer with NaN when the knob is
// on. It is the one place the two poison sites agree on the pattern.
func (g *devTier) poisonBuf(b backend.Buf, n int) {
	if g.kb.poisonScratch && b != nil && n > 0 {
		b.Write(nanFill(n))
	}
}

// partsFor is how many partial sums a norm over n elements reduces through:
// initScratch's own 64, halved until it divides.
// It is a function because MLA norms other widths (KVLoraRank, QLoraRank)
// that 64 may not divide, and NormPart refuses those.
func partsFor(n int) int {
	parts := 64
	for parts > 1 && (n <= 0 || n%parts != 0) {
		parts /= 2
	}
	return parts
}

// nanFill is n bytes of float32 NaN, for poisoning a scratch allocation.
func nanFill(n int) []byte {
	b := make([]byte, n)
	for i := 0; i+4 <= n; i += 4 {
		// 0x7FC00001, a quiet NaN, in little-endian.
		b[i], b[i+1], b[i+2], b[i+3] = 0x01, 0x00, 0xC0, 0x7F
	}
	return b
}
