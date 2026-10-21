package cpu

import (
	"fmt"
	"math"
)

// moeShapeErr is what every tier's emitter answers for a shape outside the
// contract. n and k come from the container's config, so reaching it means the
// config is wrong, and nn turns it into an error naming the router.
func moeShapeErr(n, k int) error {
	return fmt.Errorf("jit: moe top-k: %d experts, k = %d (want 1 <= k <= n)", n, k)
}

// MoETopKConsts is the constant block and the scratch layout every MoE-router
// top-k kernel reads. One copy for all three tiers, in an untagged file,
// because the layout is the ABI between the emitters and nn.
//
//	bytes  0..31  int32 0..7   the initial index vector. The AVX2 kernel loads
//	                           all eight; the SSE and NEON kernels load the
//	                           first four. Byte 4 is also the integer 1, the
//	                           tail loop's index step.
//	byte  32      int32 8      the AVX2 vector loop's index step
//	byte  36      int32 0x7FFFFFFF   INT_MAX: the "this lane has no candidate"
//	                           sentinel, and the index a non-tying lane
//	                           contributes to the min fold
//	byte  40      -Inf         the running maximum's initial value
//	byte  44      int32 4      the SSE and NEON loops' index step
//	byte  48      int32 -1     the ascending pass's initial "previous id",
//	                           below every real expert id
//	byte  52      NaN          the tail's filler for the lanes past the last
//	                           element. A NaN is never eligible (every branch
//	                           of the predicate is an ordered compare), so the
//	                           tail can run the vector body.
//	byte  56      1.0          the divisor when the architecture does not
//	                           renormalise; x/1 is x, bit for bit, so that arm
//	                           needs no branch.
//	byte  60      1e-20        the renormalising divisor's floor, DeepSeek's
//	                           `sum + 1e-20` rather than llama.cpp's clamp at
//	                           the smallest normal f16: the two agree at zero,
//	                           and this does not distort 0 < sum < 6.1e-5. See
//	                           oracle.MoEDivergences.
func MoETopKConsts() []float32 {
	c := make([]float32, 16)
	for i := 0; i < 8; i++ {
		c[i] = math.Float32frombits(uint32(i))
	}
	c[8] = math.Float32frombits(8)
	c[9] = math.Float32frombits(0x7FFFFFFF)
	c[10] = float32(math.Inf(-1))
	c[11] = math.Float32frombits(4)
	c[12] = math.Float32frombits(0xFFFFFFFF)
	c[13] = math.Float32frombits(0x7FC00000)
	c[14] = 1
	c[15] = moeSumFloorConst
	return c
}

// MoETopKConstsFor is MoETopKConsts with one appended word: the architecture's
// routed_scaling_factor, which the kernel multiplies every weight by after the
// renormalisation.
//
// It is appended, not inserted: every emitter addresses this block by byte
// offset, and engine/nn/sample.go reads the same block. It is a per-model block
// rather than an immediate because materialising an arbitrary f32 costs a GPR
// round trip on two of the three tiers, where a broadcast from the block is
// one instruction. nn caches one block per kernel key beside the code.
func MoETopKConstsFor(scale float32) []float32 {
	return append(MoETopKConsts(), scale)
}

// The byte offsets into that block, named so an emitter cannot transpose two.
const (
	moeIdxOff    = 0
	moeOneIOff   = 4
	moeStep8Off  = 32
	moeMaxIOff   = 36
	moeNegInfOff = 40
	moeStep4Off  = 44
	moeMinusOne  = 48
	moeNaNOff    = 52
	moeOneFOff   = 56
	moeSumFloor  = 60
	moeScaleOff  = 64
)

// moeSumFloorConst is the value at moeSumFloor. It is exported for the gate
// that asserts the three tiers, the device IR and the oracle all guard the
// divide with the same number.
const moeSumFloorConst = 1e-20

// MoESumFloor is moeSumFloorConst, for that gate.
func MoESumFloor() float32 { return moeSumFloorConst }

// MoEGate is the router's configuration beyond (n, k). Every field is a
// property of the container and is baked: a model asks for exactly one kernel
// for its whole life.
//
// The gating function is not here: sigmoid (V3, R1, Kimi-K2, GLM-4-MoE) and
// softmax are separate generated kernels that run over the logits first, and
// nn.MoERouteJIT picks the one the architecture asks for. The bias is a flag
// rather than a separate entry point: an absent bias is this router with a
// plane it does not read.
type MoEGate struct {
	// Norm renormalises the selected weights by their sum. llama.cpp's norm_w,
	// HF's norm_topk_prob.
	Norm bool
	// Bias says an e_score_correction_bias plane is present in Args.Q2. It is
	// added to the scores for selection only; the returned weights come from
	// the unbiased scores.
	Bias bool
	// NGroup is how many contiguous groups the experts are split into and
	// NGroupUsed how many survive. 0 or 1 group is ungrouped and emits no group
	// code, so Kimi-K2's n_group=1/topk_group=1 takes the plain path.
	NGroup, NGroupUsed int
	// Scale multiplies every weight after the renormalisation, by the block's
	// word at moeScaleOff. It is a flag because llama.cpp emits no scaling at
	// w_scale == 1, and x*1 would be a no-op the token pays for.
	Scale bool
	// ExpScale multiplies each selected weight, last, by its expert's own
	// scale: the plane in Args.AScale2, one float per expert (Gemma 4's
	// router.per_expert_scale, llama.cpp's ffn_down_exps.scale). It is
	// gathered by id, so the plane needs no padding.
	ExpScale bool
}

// Grouped reports whether the group phase is emitted.
func (g MoEGate) Grouped() bool { return g.NGroup > 1 }

// needsBiasedPlane reports whether the kernel must materialise a second plane
// to select over. It does when a bias is added to the scores, and it does when
// grouping masks part of them to -Inf -- both write, and the scores themselves
// are the caller's buffer and the source of the returned weights.
func (g MoEGate) needsBiasedPlane() bool { return g.Bias || g.Grouped() }

// moeRouteCheck validates a shape. Everything it refuses is a config error
// rather than a poor host, so nn turns it into an error naming the router.
func moeRouteCheck(n, k int, g MoEGate) error {
	if n <= 0 || k <= 0 || k > n {
		return moeShapeErr(n, k)
	}
	if !g.Grouped() {
		// One group or none is ungrouped; NGroupUsed is then ignored, which is
		// what makes n_group=1/topk_group=1 fall through to the plain path.
		return nil
	}
	if g.NGroupUsed <= 0 || g.NGroupUsed > g.NGroup {
		return fmt.Errorf("jit: moe router: %d of %d expert group(s) used (want 1 <= used <= groups)",
			g.NGroupUsed, g.NGroup)
	}
	if n%g.NGroup != 0 {
		return fmt.Errorf("jit: moe router: %d experts do not divide into %d group(s)", n, g.NGroup)
	}
	return nil
}

// moePlanes is the scratch layout, in float32 words. All three tiers derive
// their offsets from this one function. The first two planes are the pre-V3
// kernel's at the same offsets, so a route allocated for the plain router is
// byte-compatible.
//
//	id    [pad]    the selection's ids, INT_MAX-filled
//	wt    [pad]    their weights
//	bi    [nPad]   scores + bias, with the dropped groups at -Inf   (if needed)
//	gs    [gPad]   one score per expert group                       (if grouped)
//	keep  [gPad]   one all-ones / all-zeros mask per group          (if grouped)
type moeLayout struct {
	pad, nPad, gPad      int
	id, wt, bi, gs, keep int
	total                int
}

func moePlanes(n, k int, g MoEGate) moeLayout {
	l := moeLayout{pad: MoETopKPad(k)}
	l.id, l.wt = 0, l.pad
	l.total = 2 * l.pad
	if g.needsBiasedPlane() {
		l.nPad = (n + 7) &^ 7
		l.bi = l.total
		l.total += l.nPad
	}
	if g.Grouped() {
		l.gPad = (g.NGroup + 7) &^ 7
		l.gs = l.total
		l.total += l.gPad
		l.keep = l.total
		l.total += l.gPad
	}
	return l
}

// MoERouteScratch is how many float32 words of scratch this configuration
// needs. The caller owns the buffer and it is not shared between concurrent
// calls: the kernel writes the selection, the biased plane and the group
// bookkeeping through it.
func MoERouteScratch(n, k int, g MoEGate) int { return moePlanes(n, k, g).total }

// MoETopKScratch is how many float32 words of scratch a kernel for k selected
// experts needs. The caller owns it and it is not shared between concurrent
// calls.
//
// The pad is eight lanes whatever the tier, so one buffer serves all three:
// the ascending-order epilogue reads the pad as whole vectors, and the lanes
// past k hold INT_MAX, which is never the minimum while a real id is left.
//
// It is [selection ids | renormalised weights], each MoETopKPad(k) words.
func MoETopKScratch(k int) int { return 2 * MoETopKPad(k) }

// MoETopKPad rounds k up to a whole eight-lane vector.
func MoETopKPad(k int) int { return (k + 7) &^ 7 }
