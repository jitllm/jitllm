package tier

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// devKVPage is the granule the device's capacity grows in. It matches the
// host's page so a block's cache has the same shape on both sides of the seam
// and MigrateKV stays a straight copy of a position range.
const devKVPage = 256

// capped is the plan with MaxSeq replaced by what this device is actually built
// for. Every kernel build, every allocation and every bound goes through it. It
// also caps the score scratch (rows*NHead*sstride*4 twice), which at a long
// context is larger than the KV cache.
func (g *devTier) capped(p *nn.LayerPlan) nn.LayerPlan {
	q := *p
	// A vision plan's MaxSeq is the largest grid its tower takes, which a
	// dynamic-resolution tower puts far past any picture (65536 patches on
	// HunyuanOCR): the set is built for the rows ReserveRows last sized it to
	// (visrows.go), and a call is one submission of at most that many.
	if q.NonCausal {
		if g.visCap > 0 {
			q.MaxSeq = min(q.MaxSeq, max(g.visCap, g.visKeys))
		}
		return q
	}
	// The device's capacity, not the smaller of the two: every stride and slot
	// is kvCap wide, so a session with a shorter context still needs a cache
	// that wide.
	if g.kvCap > 0 {
		q.MaxSeq = g.kvCap
	}
	return q
}

// ensureKVCap makes the scratch cover the longest context a session has asked
// for (prepLayer raises maxSeqAsked). The history has no capacity -- it is
// paged -- so only the scratch's bound moves, and rebuilding it drops the
// graph, which cannot happen inside a Session (a CUDA deadlock): Layers calls
// this before entering one. Callers must NOT hold g.mu.
func (g *devTier) ensureKVCap(pos int) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	// Per lane: a lane built before the growth still has the old bound,
	// whichever lane grew first.
	if !g.paged || g.bs == nil || g.kvCap >= g.maxSeqAsked && g.bs.p.MaxSeq >= g.maxSeqAsked {
		return true
	}
	g.kvCap = g.maxSeqAsked
	// The recordings made in this lane name the scratch about to go; no
	// other lane's changes.
	g.dropLaneGraph()
	// Every scratch set (gemma4.go) is built at the capacity, not only the
	// one in use: a set left at the old bound refuses the next position.
	home := g.geoCur
	defer g.useGeom(home)
	for _, k := range append([]geoKey{home}, g.geoKeys()...) {
		g.useGeom(k)
		// A non-causal set is built for its run, not for a context.
		if g.bs == nil || g.bs.p.NonCausal {
			continue
		}
		plan := g.bs.p
		plan.MaxSeq = g.maxSeqAsked
		if !g.rebuildScratch(&plan) {
			return false
		}
	}
	return true
}

// rebuildScratch replaces every scratch with one built for plan, carrying the
// output projection across.
//
// The head is what a rebuild would otherwise lose: its matvec and its
// hNorm/logits buffers do not depend on the KV capacity or the block shape, but
// freeScratch frees them and a fresh scratch forgets the resident head, leaving
// the model believing the head is placed.
func (g *devTier) rebuildScratch(plan *nn.LayerPlan) bool {
	prev := g.bs
	// Same shape at the same bound is nothing a rebuild would change.
	if prev != nil && sameShape(prev.p, g.capped(plan)) && prev.p.MaxSeq == g.kvCap {
		return true
	}
	fresh := g.initScratch(plan, 1)
	if fresh == nil {
		return false
	}
	moveHead(fresh, prev)
	g.dropScratch(prev)
	for w := range g.bbs {
		g.dropBatch(w)
	}
	g.bs = fresh
	g.shapeChanged()
	// The reserved widths went with the old scratches: built again at the new
	// shape, here, because a capacity growth rebuilds outside any placement.
	g.reserveScratch(plan, 0)
	return true
}

// planUnion widens a to cover b, and reports whether it had to.
//
// Sessions and block kinds can offer one tier different plans that one scratch
// must serve: a hybrid's linear blocks (Recurrent set, AttnOutGate cleared by
// model.offerRange), a session needing a sliding window, a dense lead block
// before mixtures. Each widened field only adds kernels and buffers that the
// narrower blocks never launch.
func planUnion(a *nn.LayerPlan, b *nn.LayerPlan) bool {
	grew := false
	// Sinks and mixture biases only add work a block without them skips (a
	// sinkless block passes a -inf sink; an unbiased one skips the adds), so a
	// scratch built for them serves every block.
	if b.AttnSinks && !a.AttnSinks {
		a.AttnSinks, grew = true, true
	}
	if b.MoEBias && !a.MoEBias {
		a.MoEBias, grew = true, true
	}
	if b.AttnOutGate && !a.AttnOutGate {
		a.AttnOutGate, grew = true, true
	}
	if b.NoPreNorm && !a.NoPreNorm {
		a.NoPreNorm, grew = true, true
	}
	if b.Recurrent.Conv > 0 && a.Recurrent.Conv == 0 {
		a.Recurrent, grew = b.Recurrent, true
	}
	// A window is a mask: a session that never reaches it computes the same
	// thing through the windowed kernels.
	if b.SWAWindow > 0 && a.SWAWindow == 0 {
		a.SWAWindow, a.SWALocal, grew = b.SWAWindow, b.SWALocal, true
	}
	// The mixture: DeepSeek's first blocks are dense, so the first block this
	// tier sees may allocate no MoE buffers. A mixture scratch serves a dense
	// block too, since the two use different buffers; the one non-additive
	// kernel is the activation (baked element count), hence bs.actMulE.
	if b.NExpert > 0 && a.NExpert == 0 {
		a.NExpert, a.NExpertUsed, a.NFFNExp = b.NExpert, b.NExpertUsed, b.NFFNExp
		a.ExpertSigmoid, a.ExpertSelBias = b.ExpertSigmoid, b.ExpertSelBias
		a.ExpertGroups, a.ExpertGroupsUsed = b.ExpertGroups, b.ExpertGroupsUsed
		a.ExpertScale, a.NoExpertNorm = b.ExpertScale, b.NoExpertNorm
		a.ExpertSparseMixer = b.ExpertSparseMixer
		grew = true
	}
	// The shared expert and the dense FFN are widths, so the union is the max:
	// a DeepSeek mixture block carries no dense FFN at all and the dense lead
	// carries no experts, so neither alone sizes the scratch.
	if b.NFFNShExp > a.NFFNShExp {
		a.NFFNShExp, grew = b.NFFNShExp, true
	}
	if b.NFFN > a.NFFN {
		a.NFFN, grew = b.NFFN, true
	}
	// Kimi-K3: the streams, the banks and the checkpoints are per block, and
	// the scratch builds a mix for every bank and a placement for every
	// checkpoint (k3.go); the latent width is the mixture blocks', which the
	// dense lead does not carry.
	if b.K3 != nil && (a.K3 == nil || b.K3.Latent > a.K3.Latent) {
		k := *b.K3
		if a.K3 != nil {
			k = *a.K3
			k.Latent = b.K3.Latent
		}
		a.K3, grew = &k, true
	}
	// Gemma 3n's sparse lead: the gaussian's kernels serve the blocks that ask.
	if b.Sparse && !a.Sparse {
		a.Sparse, grew = true, true
	}
	// A block that is its mixer alone says nothing about the FFN the others
	// run, gated or not.
	if a.NoFFN && !b.NoFFN {
		a.NoFFN, a.UngatedFFN, grew = false, b.UngatedFFN, true
	}
	return grew
}

// planConflict names a plan this tier's scratch cannot serve alongside the one
// it holds, or "". Two different windows are the case: the scratch bakes one,
// and taking either would run the other session's local layers wrong.
func planConflict(bs *blockScratch, p *nn.LayerPlan) string {
	if bs == nil || p.NonCausal {
		return ""
	}
	if a, b := bs.p.SWAWindow, p.SWAWindow; a > 0 && b > 0 && a != b {
		return fmt.Sprintf("a sliding window of %d keys on a tier built for %d", b, a)
	}
	// Llama 4's attention is baked the same way (chunk, temperature table, norm,
	// input weighting), and planUnion widens none of it.
	if a, b := bs.p, *p; a.SWAChunked != b.SWAChunked || a.NoPEGlobal != b.NoPEGlobal ||
		a.QKL2Norm != b.QKL2Norm || a.ExpertWeightIn != b.ExpertWeightIn ||
		a.AttnTempScale != b.AttnTempScale || a.AttnTempFloor != b.AttnTempFloor ||
		a.AttnTempOffset != b.AttnTempOffset {
		return "Llama 4 attention of a different shape from the one this tier was built for"
	}
	// The activation kernels bake whether the FFN has a gate.
	if a, b := bs.p, *p; !a.NoFFN && !b.NoFFN && ungatedFFN(&a) != ungatedFFN(&b) {
		return "an ungated FFN beside a gated one on one tier"
	}
	// Where q's norm runs is baked into the scratch's launch sequence.
	if a, b := bs.p, *p; a.QKNorm != b.QKNorm || a.QKNormPost != b.QKNormPost {
		return fmt.Sprintf("a q/k norm (%v, after the rotary %v) on a tier built for (%v, %v)",
			b.QKNorm, b.QKNormPost, a.QKNorm, a.QKNormPost)
	}
	// The latent geometry is baked into every kernel of the scratch. No
	// architecture mixes them, which is why it is named rather than assumed.
	if a, b := bs.p, *p; a.KVLoraRank != b.KVLoraRank || a.QLoraRank != b.QLoraRank ||
		a.HeadDimV != b.HeadDimV {
		if a.MLA() || b.MLA() {
			return fmt.Sprintf("a latent attention geometry of (%d, %d, %d) on a tier "+
				"built for (%d, %d, %d)", b.KVLoraRank, b.QLoraRank, b.HeadDimV,
				a.KVLoraRank, a.QLoraRank, a.HeadDimV)
		}
	}
	// The classic block's switches (C6) are baked too: norm kernels, gating,
	// and the parallel residual read from the scratch's plan.
	if a, b := bs.p, *p; a.LayerNorm != b.LayerNorm || a.Parallel != b.Parallel ||
		a.UngatedFFN != b.UngatedFFN {
		return fmt.Sprintf("a classic block (LayerNorm %v, parallel %v, ungated %v) on a tier "+
			"built for (%v, %v, %v)", b.LayerNorm, b.Parallel, b.UngatedFFN,
			a.LayerNorm, a.Parallel, a.UngatedFFN)
	}
	// The score scale is baked into the attention kernels and the residual
	// scale into the scratch's one-element buffer: a model of another scale on
	// this scratch would run with this one's.
	if a, b := bs.p.Scale(), p.Scale(); a != b {
		return fmt.Sprintf("an attention score scale of %g on a tier built for %g", b, a)
	}
	if a, b := bs.p.ResidualScale, p.ResidualScale; residScaled(&bs.p) || residScaled(p) {
		if a != b {
			return fmt.Sprintf("a residual scale of %g on a tier built for %g", b, a)
		}
	}
	// The clamp's bound is baked into its kernel, and a scratch built without
	// one has neither the kernels nor the buffers.
	if bs.p.ClampKQV != p.ClampKQV {
		return fmt.Sprintf("a q/k/v clamp of %g on a tier built for %g", p.ClampKQV, bs.p.ClampKQV)
	}
	// The delta-rule kernel bakes one key-head pairing; a scratch built for
	// grouped heads runs a tiled block's value heads against the wrong keys,
	// and every shape still matches.
	if a, b := bs.p.Recurrent, p.Recurrent; a.Conv > 0 && b.Conv > 0 && a.KeyTiled != b.KeyTiled {
		return fmt.Sprintf("a delta rule with key heads tiled=%v on a tier built for tiled=%v",
			b.KeyTiled, a.KeyTiled)
	}
	return ""
}

// nonCausalConflict names a non-causal block the non-causal set, built from
// the first such block's plan, cannot run, or "". planConflict passes every
// non-causal plan, since a tower's blocks are one shape; an encoder with a
// decision head after it (Laya) is two. A block's own MLP width, gating and
// activation, and whether it rotates, are its own (layer.nffn, the set grown
// to the widest MLP); what the set's kernels bake for every block is the
// heads and their width, the norm kind, the rotary's width and pairing, a
// gate buffer and the copy where a norm is missing, so a block differing in
// one of those would run on the first one's kernels and is declined by name.
// A convolutional tower block runs its own program and is not compared.
func nonCausalConflict(bs *blockScratch, p *nn.LayerPlan) string {
	if bs == nil || !p.NonCausal || p.Conv != nil || bs.p.Conv != nil {
		return ""
	}
	a := &bs.p
	gated := map[bool]string{true: "ungated", false: "gated"}
	switch {
	case a.NEmbd != p.NEmbd || a.NHead != p.NHead || a.NKVHead != p.NKVHead || a.HeadDim != p.HeadDim:
		return fmt.Sprintf("a non-causal block of %d heads of %d (%d kv) over %d on a set built for %d of %d (%d kv) over %d",
			p.NHead, p.HeadDim, p.NKVHead, p.NEmbd, a.NHead, a.HeadDim, a.NKVHead, a.NEmbd)
	case !ungatedFFN(p) && ungatedFFN(a):
		// The set of an ungated MLP has no gate buffer; a block's own width,
		// gating and activation are otherwise its own (layer.nffn).
		return fmt.Sprintf("a non-causal block's MLP (%s %s, %d wide) on a set built for (%s %s, %d wide)",
			gated[ungatedFFN(p)], p.Act, p.NFFN, gated[ungatedFFN(a)], a.Act, a.NFFN)
	case p.LayerNorm != a.LayerNorm:
		return fmt.Sprintf("a non-causal block with LayerNorm %v on a set built with %v", p.LayerNorm, a.LayerNorm)
	case p.NRot > 0 && !p.NoPosEnc && (a.NRot == 0 || a.NoPosEnc):
		// A set that rotates nothing builds no rotary kernels; a block that
		// rotates nothing runs on one that does (layer.noRope).
		return fmt.Sprintf("a non-causal block rotating %d of each head on a set built to rotate none", p.NRot)
	case p.NRot > 0 && a.NRot > 0 && (p.NRot != a.NRot || p.RopeNeox != a.RopeNeox):
		return fmt.Sprintf("a non-causal block rotating %d (NEOX %v) on a set built for %d (NEOX %v)",
			p.NRot, p.RopeNeox, a.NRot, a.RopeNeox)
	case p.NoPreNorm && !a.NoPreNorm:
		return "a non-causal block with no pre-norm on a set built for blocks that have one"
	}
	return ""
}

// kvGrow is one session's cache on one text block, and the buffers a growth
// replaces it with.
type kvGrow struct {
	li     int
	kvp    *kvPair
	nk, nv backend.Buf
}

// kvCacheBytes is one session's K and V cache for one block at capacity c.
func (g *devTier) kvCacheBytes(p *nn.LayerPlan, c int) (cache, vcache int) {
	cache = (c + 1) * p.KVRow() * 4
	// MLA has no V buffer: one row per position, the value its prefix.
	if p.MLA() {
		return cache, 0
	}
	vcache = cache
	if g.KVF16 {
		vcache = cache / 2
	}
	return cache, vcache
}

// copyKVAcrossStride moves the live history from a cache of oldCap positions
// into one of newCap, on the device, with oldkernels.Restride.
//
// K's transpose makes a growth a reshape (at stride S row e lives at e*S), a
// strided copy a generated kernel does without leaving the card. trans is
// g.kStride's answer rather than an assumption: under WithNoKT or MLA the cache
// is row-major, and a transposing copy would scramble the history.
//
// Positions past the live prefix are left as the fresh buffer holds them:
// attention is bounded by the position count, so nothing reads them.
func (g *devTier) copyKVAcrossStride(kvp *kvPair, nk, nv backend.Buf, kvDim, oldCap, newCap int, trans bool) bool {
	// A shrink keeps the positions the smaller cache has room for; TrimKV only
	// asks for one when none past it are live.
	keep := min(oldCap, newCap) + 1
	if kvp.kc != nil {
		rows, cols, ss, ds := 1, keep*kvDim, keep*kvDim, keep*kvDim
		if trans {
			rows, cols, ss, ds = kvDim, keep, oldCap+1, newCap+1
		}
		if !g.restride(kvp.kc, nk, rows, cols, ss, ds) {
			return false
		}
	}
	// Under MLA there is no V buffer; the value travelled with K.
	if nv == nil || kvp.vc == nil {
		return true
	}
	// V is row-major, so its live prefix is one contiguous run of words.
	words := keep * kvDim
	if g.KVF16 {
		words /= 2
	}
	return g.restride(kvp.vc, nv, 1, words, words, words)
}

// restride runs one oldkernels.Restride and waits for it. Every block of a growth
// has the same shape, so the kernel is compiled once per shape. One contiguous
// run is a device copy instead: no kernel compiled for every size a buffer
// grows to, and no grid to outgrow (an integrated GPU may dispatch at most
// 65535 groups, 33.5 MB of floats at 128 threads a group).
func (g *devTier) restride(src, dst backend.Buf, rows, cols, srcStride, dstStride int) bool {
	if rows == 1 || cols == srcStride && cols == dstStride {
		if err := g.dev.Copy(dst, 0, src, 0, rows*cols*4); err != nil {
			g.LastErr = err.Error()
			return false
		}
		return true
	}
	key := [4]int{rows, cols, srcStride, dstStride}
	k := g.restrides[key]
	if k == nil {
		ker, err := oldkernels.Restride(rows, cols, srcStride, dstStride)
		if err != nil {
			g.LastErr = err.Error()
			return false
		}
		if k, err = g.dev.Compile(ker); err != nil {
			g.LastErr = err.Error()
			return false
		}
		if g.restrides == nil {
			g.restrides = map[[4]int]backend.Kernel{}
		}
		g.restrides[key] = k
	}
	if err := k.Launch((rows*cols+127)/128, 128, src, dst); err != nil {
		g.LastErr = err.Error()
		return false
	}
	return true
}

// initKVCap sets the starting capacity the first time this device sees a plan:
// one page, or the whole context when that is smaller.
func (g *devTier) initKVCap(p *nn.LayerPlan) {
	if g.kvCap != 0 || p.NonCausal {
		// A tower carries no growing history, so it must not be what sets the
		// ceiling the text model grows toward.
		return
	}
	g.maxSeqAsked = p.MaxSeq
	// Paged history has no capacity to grow: no kernel bakes one, so the
	// scratch is built for the whole context once and every session shares it.
	g.paged, g.kvCap = true, p.MaxSeq
}

// devKVReserve is how many positions of history a device reserves up front,
// before growing (see initKVCap).
const devKVReserve = 2048

// DeclinePlan answers nn.PlanDecliner: would this tier refuse a block of this
// shape, before anyone reads its weights? Empty means it would not.
//
// It is declineReason itself, so the cheap check and the one after the upload
// cannot disagree.
func (g *GPU) DeclinePlan(p *nn.LayerPlan) string { return declineReason(p) }

// partialRotary reports that this block rotates only part of each head, so the
// untouched tail has to be copied into the rope's destination before the rope
// runs. qwen3next rotates 64 of a 256-wide head; every other model rotates the
// whole of it.
func partialRotary(p *nn.LayerPlan) bool { return p.NRot > 0 && p.NRot != p.HeadDim }
