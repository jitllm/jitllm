package kernels

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/jit/ir"
)

// The mixture-of-experts router, on the device.
//
// It has to be on the device: the selection is an input to the expert
// matvecs, so a host router would read the logits back and write the
// selection out per layer per token, a synchronisation inside the block that
// nn.LayerDevice exists to avoid. It is branch-free: ir.OpSelect is data flow,
// so the clamp invariant (RULE 13) is untouched.

// MoERoute is the router's shape on the device: the device's copy of
// nn.MoEGate, which is the host's.
//
// The V3 fields (sigmoid gating, a selection-only correction bias, grouped
// top-k, renormalise then scale) are one capability: DeepSeek-V3, R1, Kimi-K2
// and GLM-4-MoE have all of them. On the plain softmax route Gated() is false
// and ExpertRank and ExpertWeights emit exactly their original IR, which the
// lowertest goldens check.
type MoERoute struct {
	NExpert, K         int
	Sigmoid            bool
	Bias               bool
	NGroup, NGroupUsed int
	Norm               bool
	Scale              float32
	// SqrtSoftplus gates with sqrt(softplus()) instead (DeepSeek V4); the
	// weights are then the renormalised scores, as the sigmoid's are.
	SqrtSoftplus bool
	// RowBias reads the selection bias per row, row r's at r*NExpert, rather
	// than one shared: DeepSeek V4 hands every block a plane of its own, the
	// table's experts for a block that routes by token id.
	RowBias bool
	// Rows routes that many tokens in one launch (a batched mixture); 0 and 1
	// are one token. Row r reads its logits at r*NExpert and writes its
	// selection at r*(K+1) and its weights at r*K. A grouped route's two
	// planes are per row too: group scores at r*NGroup, the masked selection
	// scores at r*NExpert. The selection bias is per expert and shared.
	Rows int
	// SparseMixer, when non-zero, is Phi-3.5-MoE's router and its jitter_eps:
	// ExpertRank's plain logit rank keeps the top two, and ExpertWeights forms
	// each weight as a softmax over the logits within a relative
	// 2*SparseMixer of it, not renormalised (expertWeightsSparseMixer).
	SparseMixer float32
}

// rows is Rows with 0 read as 1.
func (r MoERoute) rows() int { return max(r.Rows, 1) }

// rowAt returns idx offset by row*stride, or idx itself for one row -- so a
// single-token kernel emits exactly what it always did (lowertest's goldens).
func (r MoERoute) rowAt(b *ir.Builder, row ir.Value, stride int) func(ir.Value) ir.Value {
	if r.rows() == 1 || stride == 0 {
		return func(i ir.Value) ir.Value { return i }
	}
	base := b.Mul(ir.U32, row, b.Const(ir.U32, int64(stride)))
	return func(i ir.Value) ir.Value { return b.Add(ir.U32, base, i) }
}

// Grouped reports whether the selection keeps whole groups of experts.
func (r MoERoute) Grouped() bool { return r.NGroup > 1 }

// Gated reports whether anything beyond the plain softmax top-k applies.
// It keys on Sigmoid alone because groups only arrive with sigmoid gating,
// which check enforces; a grouped softmax router would need the full softmax
// the ungated path avoids.
func (r MoERoute) Gated() bool { return r.Sigmoid || r.SqrtSoftplus }

// softBias reports a softmax router with a selection bias (ERNIE 4.5): the
// bias is added to the softmax PROBABILITY, not the logit, so the rank key is
// exp(l-m)/Z + b and needs the full softmax first. A bias on the logit would
// be a different selection, since the softmax is not additive.
func (r MoERoute) softBias() bool { return !r.Gated() && r.Bias }

// scaleOf is the multiply the weights kernel applies: 0 and 1 both mean none.
func (r MoERoute) scaleOf() float32 {
	if r.Scale == 0 || r.Scale == 1 {
		return 1
	}
	return r.Scale
}

func (r MoERoute) check(who string) error {
	if r.K < 1 || r.NExpert < 1 || r.K > r.NExpert {
		return fmt.Errorf("kernels: %s: k=%d of %d experts", who, r.K, r.NExpert)
	}
	if r.Grouped() {
		if r.NExpert%r.NGroup != 0 {
			return fmt.Errorf("kernels: %s: %d experts do not divide into %d groups",
				who, r.NExpert, r.NGroup)
		}
		if r.NGroupUsed < 1 || r.NGroupUsed > r.NGroup {
			return fmt.Errorf("kernels: %s: %d of %d groups kept", who, r.NGroupUsed, r.NGroup)
		}
	}
	if r.Sigmoid && r.SqrtSoftplus {
		return fmt.Errorf("kernels: %s: a sigmoid and a sqrt-softplus gate at once", who)
	}
	if r.RowBias && (!r.Bias || r.Grouped()) {
		return fmt.Errorf("kernels: %s: a per-row selection bias needs a bias and no groups", who)
	}
	if !r.Gated() && r.Grouped() {
		return fmt.Errorf("kernels: %s: a softmax router with expert groups needs the full "+
			"softmax formed before the group scores, which these kernels do not do", who)
	}
	if r.SparseMixer != 0 && (r.K != 2 || r.NExpert < 2 || r.Gated() || r.Bias || r.Grouped() || r.Norm ||
		r.scaleOf() != 1) {
		return fmt.Errorf("kernels: %s: sparsemixer is a plain top-2 over the logits "+
			"(k=%d of %d, gated %v, bias %v, %d groups, renormalised %v, scale %g)",
			who, r.K, r.NExpert, r.Gated(), r.Bias, r.NGroup, r.Norm, r.scaleOf())
	}
	return nil
}

// MoERouteWhyNot names a router shape these kernels cannot express, or "".
func MoERouteWhyNot(r MoERoute) string {
	if err := r.check("router"); err != nil {
		return err.Error()
	}
	return ""
}

// negBig masks a rejected group. It is -MaxFloat32 rather than -Inf because
// only the ordering matters and an infinity is not known to fold identically
// in all three lowerings. Every masked expert ties at the same value, so the
// lexicographic rule sends the lowest index first, as the oracle's -Inf does.
const negBig = -math.MaxFloat32

// gateScore is sigma(l) for a sigmoid router and l itself otherwise. It is a
// function so the three kernels below cannot drift in how they compute it.
func gateScore(b *ir.Builder, r MoERoute, l ir.Value) ir.Value {
	if r.SqrtSoftplus {
		return b.Sqrt(softplus(b, l))
	}
	if !r.Sigmoid {
		return l
	}
	one := b.ConstF32(1)
	return b.Div(ir.F32, one, b.Add(ir.F32, one,
		b.Exp(b.Mul(ir.F32, l, b.ConstF32(-1)))))
}

// ExpertGroupScore writes each group's score: the sum of the two highest
// biased scores in it, which is what DeepSeek's group_scores computes.
//
// It is a kernel of its own because the IR has no nested loops: ranking an
// expert needs its group's rank, which needs every group's score, so the group
// scores and then the masked selection plane are materialised as flat passes.
// The top two are tracked with two running maxima; the tie rule cannot change
// their sum, so no indices are carried.
func ExpertGroupScore(r MoERoute) (*ir.Kernel, error) {
	if err := r.check("ExpertGroupScore"); err != nil {
		return nil, err
	}
	if !r.Grouped() {
		return nil, fmt.Errorf("kernels: ExpertGroupScore: %d group(s) is ungrouped", r.NGroup)
	}
	sz := r.NExpert / r.NGroup
	b := ir.New("expertgroupscore", [3]int{128, 1, 1})
	pL := b.Param("pL", ir.F32)   // nExpert router logits
	pGS := b.Param("pGS", ir.F32) // nGroup group scores
	var pB ir.Value
	if r.Bias {
		pB = b.Param("pB", ir.F32) // nExpert selection corrections
	}

	zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
	// One thread per (row, group) at Rows > 1; rowAt is the identity at one
	// row, so the single-token kernel is unchanged.
	var row ir.Value
	g := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(r.rows()*r.NGroup-1)))
	if r.rows() > 1 {
		row = b.Div(ir.U32, g, b.Const(ir.U32, int64(r.NGroup)))
		g = b.Rem(ir.U32, g, b.Const(ir.U32, int64(r.NGroup)))
	}
	atL, atG := r.rowAt(b, row, r.NExpert), r.rowAt(b, row, r.NGroup)
	base := b.Mul(ir.U32, g, b.Const(ir.U32, int64(sz)))
	low := b.ConstF32(negBig)

	b.Loop(int64(sz))
	m1 := b.Phi(ir.F32, low)
	m2 := b.Phi(ir.F32, low)
	i := b.Phi(ir.U32, zero)
	idx := b.Add(ir.U32, base, i)
	v := gateScore(b, r, b.Load(ir.F32, pL, atL(idx), 0))
	if r.Bias {
		v = b.Add(ir.F32, v, b.Load(ir.F32, pB, idx, 0))
	}
	// A new maximum pushes the old one down; otherwise the candidate competes
	// for second place. Both reads see the loop head's values, so the order of
	// these two SetPhi calls does not matter.
	b.SetPhi(m2, b.Select(ir.F32, b.Lt(ir.F32, m1, v), m1, b.Max(ir.F32, m2, v)))
	b.SetPhi(m1, b.Max(ir.F32, m1, v))
	b.SetPhi(i, b.Add(ir.U32, i, one))
	b.EndLoop()

	gs := m1
	if sz > 1 {
		gs = b.Add(ir.F32, m1, m2)
	}
	b.Store(pGS, atG(g), gs, 0)
	return b.Done()
}

// ExpertGroupMask writes the selection plane: each expert's biased score, or
// negBig where its group was not kept.
//
// A group is kept when its lexicographic rank among the group scores is below
// NGroupUsed -- the same counting ExpertRank does over experts, for the same
// reason (no mutable state, ties to the lower index).
func ExpertGroupMask(r MoERoute) (*ir.Kernel, error) {
	if err := r.check("ExpertGroupMask"); err != nil {
		return nil, err
	}
	if !r.Grouped() {
		return nil, fmt.Errorf("kernels: ExpertGroupMask: %d group(s) is ungrouped", r.NGroup)
	}
	sz := r.NExpert / r.NGroup
	b := ir.New("expertgroupmask", [3]int{128, 1, 1})
	pL := b.Param("pL", ir.F32)   // nExpert router logits
	pGS := b.Param("pGS", ir.F32) // nGroup group scores
	pBM := b.Param("pBM", ir.F32) // nExpert masked selection scores
	var pB ir.Value
	if r.Bias {
		pB = b.Param("pB", ir.F32)
	}

	zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
	var row ir.Value
	e := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(r.rows()*r.NExpert-1)))
	if r.rows() > 1 {
		row = b.Div(ir.U32, e, b.Const(ir.U32, int64(r.NExpert)))
		e = b.Rem(ir.U32, e, b.Const(ir.U32, int64(r.NExpert)))
	}
	atL, atG := r.rowAt(b, row, r.NExpert), r.rowAt(b, row, r.NGroup)
	g := b.Div(ir.U32, e, b.Const(ir.U32, int64(sz)))
	mine := b.Load(ir.F32, pGS, atG(g), 0)

	b.Loop(int64(r.NGroup))
	rank := b.Phi(ir.U32, zero)
	h := b.Phi(ir.U32, zero)
	gh := b.Load(ir.F32, pGS, atG(h), 0)
	gt := b.Select(ir.U32, b.Lt(ir.F32, mine, gh), one, zero)
	lt := b.Select(ir.U32, b.Lt(ir.F32, gh, mine), one, zero)
	eq := b.Sub(ir.U32, one, b.Add(ir.U32, gt, lt))
	lower := b.Select(ir.U32, b.Lt(ir.U32, h, g), one, zero)
	b.SetPhi(rank, b.Add(ir.U32, rank, b.Add(ir.U32, gt, b.Mul(ir.U32, eq, lower))))
	b.SetPhi(h, b.Add(ir.U32, h, one))
	b.EndLoop()

	score := gateScore(b, r, b.Load(ir.F32, pL, atL(e), 0))
	if r.Bias {
		score = b.Add(ir.F32, score, b.Load(ir.F32, pB, e, 0))
	}
	keep := b.Lt(ir.U32, rank, b.Const(ir.U32, int64(r.NGroupUsed)))
	b.Store(pBM, atL(e), b.Select(ir.F32, keep, score, b.ConstF32(negBig)), 0)
	return b.Done()
}

// ExpertRank writes the ids of the k highest router logits, best first, and the
// logits themselves.
//
// The rank is counted, not searched, so there is no mutable state (a serial
// top-k would read and write one buffer, which ir.Validate refuses). Thread e
// computes
//
//	rank(e) = #{f : l_f > l_e} + #{f < e : l_f == l_e}
//
// and expert e is selected iff rank(e) < k, at slot rank(e). The second term
// is the tie rule: equal keys go to the lower index, as ggml_argsort and the
// host router (jit/cpu/topk.go) do, and it makes ranks distinct so stores never
// collide.
//
// There is no conditional store, so unselected threads store into a spare slot
// k: pSel and pTop are k+1 long and slot k is a bin nothing reads.
func ExpertRank(r MoERoute) (*ir.Kernel, error) {
	if err := r.check("ExpertRank"); err != nil {
		return nil, err
	}
	nExpert, k := r.NExpert, r.K
	b := ir.New("expertrank", [3]int{128, 1, 1})
	pL := b.Param("pL", ir.F32)     // nExpert router logits
	pSel := b.Param("pSel", ir.U32) // k+1 expert ids, best first
	pTop := b.Param("pTop", ir.F32) // k+1 logits, matching
	// The gated path's extra plane is declared here and nowhere else, so
	// parameter order stays a property of the route: backends bind buffers
	// positionally and CUDA does not check arity.
	var pBM, pB ir.Value
	switch {
	case r.Grouped():
		pBM = b.Param("pBM", ir.F32) // nExpert masked selection scores
	case r.Bias:
		pB = b.Param("pB", ir.F32) // nExpert selection corrections
	}

	zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
	flat := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	var row ir.Value
	if r.rows() > 1 {
		flat = b.Min(ir.U32, flat, b.Const(ir.U32, int64(r.rows()*nExpert-1)))
		row = b.Div(ir.U32, flat, b.Const(ir.U32, int64(nExpert)))
		flat = b.Rem(ir.U32, flat, b.Const(ir.U32, int64(nExpert)))
	}
	atL, atS := r.rowAt(b, row, nExpert), r.rowAt(b, row, k+1)
	e := b.Min(ir.U32, flat, b.Const(ir.U32, int64(nExpert-1)))
	// A softmax router with a selection bias ranks on the probability plus
	// the bias, so every thread forms the softmax's max and sum first, over
	// the row's logits in the same order: each thread's keys are then the same
	// numbers, and the ranks a permutation. Flat loops, one after another: the
	// IR has no nesting, and every Phi seed is hoisted.
	var sm, sz ir.Value
	if r.softBias() {
		seed := b.Load(ir.F32, pL, atL(zero), 0)
		zeroF := b.ConstF32(0)
		b.Loop(int64(nExpert))
		m := b.Phi(ir.F32, seed)
		i := b.Phi(ir.U32, zero)
		b.SetPhi(m, b.Max(ir.F32, m, b.Load(ir.F32, pL, atL(i), 0)))
		b.SetPhi(i, b.Add(ir.U32, i, one))
		b.EndLoop()
		b.Loop(int64(nExpert))
		z := b.Phi(ir.F32, zeroF)
		j := b.Phi(ir.U32, zero)
		b.SetPhi(z, b.Add(ir.F32, z, b.Exp(b.Sub(ir.F32, b.Load(ir.F32, pL, atL(j), 0), m))))
		b.SetPhi(j, b.Add(ir.U32, j, one))
		b.EndLoop()
		sm, sz = m, z
	}
	// plane is the key the rank compares: the raw logit on the ungated path
	// (the same single Load as always), the biased score for a sigmoid router,
	// the biased probability for a softmax one, and the pre-masked plane once
	// groups are involved.
	plane := func(i ir.Value) ir.Value {
		switch {
		case r.Grouped():
			return b.Load(ir.F32, pBM, atL(i), 0)
		case r.Gated():
			v := gateScore(b, r, b.Load(ir.F32, pL, atL(i), 0))
			if r.Bias {
				bi := i
				if r.RowBias {
					bi = atL(i)
				}
				v = b.Add(ir.F32, v, b.Load(ir.F32, pB, bi, 0))
			}
			return v
		case r.softBias():
			p := b.Div(ir.F32, b.Exp(b.Sub(ir.F32, b.Load(ir.F32, pL, atL(i), 0), sm)), sz)
			return b.Add(ir.F32, p, b.Load(ir.F32, pB, i, 0))
		}
		return b.Load(ir.F32, pL, atL(i), 0)
	}
	le := plane(e)

	b.Loop(int64(nExpert))
	rank := b.Phi(ir.U32, zero)
	f := b.Phi(ir.U32, zero)
	lf := plane(f)
	// Predicates become 0/1 and combine arithmetically: there is no Eq op and
	// no portable AND of Preds, so equality is "neither less nor greater" and
	// the conjunction is a multiply.
	gt := b.Select(ir.U32, b.Lt(ir.F32, le, lf), one, zero)
	lt := b.Select(ir.U32, b.Lt(ir.F32, lf, le), one, zero)
	eq := b.Sub(ir.U32, one, b.Add(ir.U32, gt, lt))
	lower := b.Select(ir.U32, b.Lt(ir.U32, f, e), one, zero)
	b.SetPhi(rank, b.Add(ir.U32, rank, b.Add(ir.U32, gt, b.Mul(ir.U32, eq, lower))))
	b.SetPhi(f, b.Add(ir.U32, f, one))
	b.EndLoop()

	slot := atS(b.Min(ir.U32, rank, b.Const(ir.U32, int64(k)))) // k is the bin
	b.Store(pSel, slot, e, 0)
	// pTop carries the unbiased score, not the rank key: the correction bias
	// moves the selection only and must never reach a weight, and a grouped
	// key is masked to negBig for most experts.
	// A softmax router's slot carries the raw logit, which ExpertWeights
	// softmaxes over the selected k (or reads pL for the full one).
	top := le
	if r.Gated() || r.softBias() {
		top = gateScore(b, r, b.Load(ir.F32, pL, atL(e), 0))
	}
	b.Store(pTop, slot, top, 0)
	return b.Done()
}

// ExpertWeights turns the k selected logits into the mixture weights.
//
// With renormalisation it is a softmax over the selected k, which is exactly
// what llama.cpp computes (softmax all, take top k, divide by their sum):
//
//	p_e / SUM_sel p_f  =  exp(l_e) / SUM_sel exp(l_f)
//
// since Z cancels. Without renormalisation (olmoe) the weight is
// exp(l_e)/Z over all nExpert logits, which is the only reason pL is a
// parameter.
//
// One thread per token: k is 2 to 8 and these are short loops.
func ExpertWeights(r MoERoute) (*ir.Kernel, error) {
	if err := r.check("ExpertWeights"); err != nil {
		return nil, err
	}
	if r.Gated() {
		return expertWeightsSigmoid(r)
	}
	if r.SparseMixer != 0 {
		return expertWeightsSparseMixer(r)
	}
	k, nExpert, norm := r.K, r.NExpert, r.Norm
	// The reduction runs over the selected k when the weights are renormalised
	// and over every expert when they are not.
	red := k
	if !norm {
		red = nExpert
	}
	b := ir.New("expertweights", [3]int{r.weightsGroup(), 1, 1})
	pTop := b.Param("pTop", ir.F32) // k+1 logits from ExpertRank
	pW := b.Param("pW", ir.F32)     // k mixture weights
	// Declared only when read: an unused parameter would still change the
	// renormalised kernel's emitted code.
	var pL ir.Value
	if !norm {
		pL = b.Param("pL", ir.F32) // nExpert router logits
	}

	zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
	row := r.weightsRow(b)
	atTop, atW := r.rowAt(b, row, k+1), r.rowAt(b, row, k)
	atL := r.rowAt(b, row, nExpert)
	// Every Phi seed is hoisted, including the loads: ir.Validate refuses a Phi
	// whose init is emitted inside its own loop. src is what the max and the
	// sum walk; pTop's slot k is the bin, so the loops stop at k.
	src, at := pTop, atTop
	if !norm {
		src, at = pL, atL
	}
	seed := b.Load(ir.F32, src, at(zero), 0)
	zeroF := b.ConstF32(0)
	// max, because exp of a raw logit overflows f32 well inside the range a
	// router produces.
	b.Loop(int64(red))
	m := b.Phi(ir.F32, seed)
	i := b.Phi(ir.U32, zero)
	b.SetPhi(m, b.Max(ir.F32, m, b.Load(ir.F32, src, at(i), 0)))
	b.SetPhi(i, b.Add(ir.U32, i, one))
	b.EndLoop()

	b.Loop(int64(red))
	sum := b.Phi(ir.F32, zeroF)
	j := b.Phi(ir.U32, zero)
	b.SetPhi(sum, b.Add(ir.F32, sum, b.Exp(b.Sub(ir.F32, b.Load(ir.F32, src, at(j), 0), m))))
	b.SetPhi(j, b.Add(ir.U32, j, one))
	b.EndLoop()

	// The routed scale after the divide (DeepSeek-V2's routed_scaling_factor is
	// 16). At scale 1 no Mul is emitted.
	sc := r.scaleOf()
	b.Loop(int64(k))
	n := b.Phi(ir.U32, zero)
	w := b.Div(ir.F32, b.Exp(b.Sub(ir.F32, b.Load(ir.F32, pTop, atTop(n), 0), m)), sum)
	if sc != 1 {
		w = b.Mul(ir.F32, w, b.ConstF32(sc))
	}
	b.Store(pW, atW(n), w, 0)
	b.SetPhi(n, b.Add(ir.U32, n, one))
	b.EndLoop()
	return b.Done()
}

// MoESumFloor is added to the renormalising divisor so an all-underflowed
// selection yields 0 rather than NaN. It is DeepSeek's own `sum + 1e-20`
// rather than llama.cpp's clamp; see its use.
const MoESumFloor = 1e-20

// expertWeightsSigmoid is the V3 family's mixture weights. sigma() already ran
// in ExpertRank and pTop holds the selected scores, so there is no softmax: a
// weight is its score, renormalised then scaled. The scale comes after the norm
// (scaling first divides it back out), and the sum is in selection order, one
// addition at a time, which the host paths are held bit-identical against.
func expertWeightsSigmoid(r MoERoute) (*ir.Kernel, error) {
	b := ir.New("expertweightssigmoid", [3]int{r.weightsGroup(), 1, 1})
	pTop := b.Param("pTop", ir.F32) // k+1 scores from ExpertRank
	pW := b.Param("pW", ir.F32)     // k mixture weights

	zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
	row := r.weightsRow(b)
	atTop, atW := r.rowAt(b, row, r.K+1), r.rowAt(b, row, r.K)
	var sum ir.Value
	if r.Norm {
		// Every Phi seed is hoisted: ir.Validate refuses one emitted inside its
		// own loop.
		zeroF := b.ConstF32(0)
		b.Loop(int64(r.K))
		acc := b.Phi(ir.F32, zeroF)
		j := b.Phi(ir.U32, zero)
		b.SetPhi(acc, b.Add(ir.F32, acc, b.Load(ir.F32, pTop, atTop(j), 0)))
		b.SetPhi(j, b.Add(ir.U32, j, one))
		b.EndLoop()
		// The divisor's floor. DeepSeek's `sum + 1e-20` and llama.cpp's
		// `ggml_clamp(sum, 6.1e-5, +Inf)` agree at sum ~= 0 but differ for
		// 0 < sum < 6.1e-5, where the clamp distorts; 1e-20 is below an f32 ulp
		// of any sum above ~1e-13, so it only ever prevents 0/0. See
		// oracle.MoEDivergences.
		sum = b.Add(ir.F32, acc, b.ConstF32(MoESumFloor))
	}
	sc := r.scaleOf()

	b.Loop(int64(r.K))
	n := b.Phi(ir.U32, zero)
	w := b.Load(ir.F32, pTop, atTop(n), 0)
	if r.Norm {
		w = b.Div(ir.F32, w, sum)
	}
	if sc != 1 {
		w = b.Mul(ir.F32, w, b.ConstF32(sc))
	}
	b.Store(pW, atW(n), w, 0)
	b.SetPhi(n, b.Add(ir.U32, n, one))
	b.EndLoop()
	return b.Done()
}

// weightsGroup is the weights kernels' group: one thread for one token, as it
// always was, and 64 when a launch routes many.
func (r MoERoute) weightsGroup() int {
	if r.rows() == 1 {
		return 1
	}
	return 64
}

// weightsRow is the token a weights thread serves, clamped; nothing for one.
func (r MoERoute) weightsRow(b *ir.Builder) ir.Value {
	if r.rows() == 1 {
		return 0
	}
	flat := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	return b.Min(ir.U32, flat, b.Const(ir.U32, int64(r.rows()-1)))
}

// ExpertRoute is ExpertRank and ExpertWeights in one launch for the softmax
// router.
//
// One workgroup of ExpertRouteWidth(NExpert) threads, thread e for expert e.
// Each computes its rank exactly as ExpertRank does and, in the same pass over
// the logits, the maximum; a second pass forms the softmax denominator in
// ascending expert order -- ExpertWeights' order, so every value is the same
// float it was. The renormalised route needs the SELECTED logits, which live
// in other threads: each thread parks its logit at its slot in threadgroup
// memory, and after a barrier the denominator is summed from there in slot
// order, as ExpertWeights sums pTop. pSel, pTop and pW are k+1 long: slot k is
// the bin every unselected expert writes into (see ExpertRank).
//
// Params: pL, pSel, pTop, pW. Only the plain softmax route; the sigmoid, bias
// and group routes keep their kernels.
func ExpertRoute(r MoERoute) (*ir.Kernel, error) {
	if err := r.check("ExpertRoute"); err != nil {
		return nil, err
	}
	if r.Gated() || r.Bias || r.Grouped() || r.rows() > 1 {
		return nil, fmt.Errorf("kernels: ExpertRoute: only the plain softmax route, one token")
	}
	w := ExpertRouteWidth(r.NExpert)
	if w == 0 {
		return nil, fmt.Errorf("kernels: ExpertRoute: %d experts exceed one workgroup", r.NExpert)
	}
	nExpert, k := r.NExpert, r.K
	b := ir.New("expertroute", [3]int{w, 1, 1})
	pL := b.Param("pL", ir.F32)
	pSel := b.Param("pSel", ir.U32)
	pTop := b.Param("pTop", ir.F32)
	pW := b.Param("pW", ir.F32)
	sh := b.Shared("top", ir.F32, k+1)
	// The logits are read once and the exponentials formed once, in
	// threadgroup memory. exp(l_f - m) is the same deterministic operation
	// whichever thread takes it, so the sums keep their order and bits.
	shL := b.Shared("logit", ir.F32, nExpert)
	var shE ir.Value
	if !r.Norm {
		shE = b.Shared("expd", ir.F32, nExpert)
	}

	zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
	zeroF := b.ConstF32(0)
	e := b.Min(ir.U32, b.TID(), b.Const(ir.U32, int64(nExpert-1)))
	le := b.Load(ir.F32, pL, e, 0)
	b.Store(shL, e, le, 0)
	b.Barrier()
	seed := b.Load(ir.F32, shL, zero, 0)

	b.Loop(int64(nExpert))
	rank := b.Phi(ir.U32, zero)
	m := b.Phi(ir.F32, seed)
	f := b.Phi(ir.U32, zero)
	lf := b.Load(ir.F32, shL, f, 0)
	gt := b.Select(ir.U32, b.Lt(ir.F32, le, lf), one, zero)
	lt := b.Select(ir.U32, b.Lt(ir.F32, lf, le), one, zero)
	eq := b.Sub(ir.U32, one, b.Add(ir.U32, gt, lt))
	lower := b.Select(ir.U32, b.Lt(ir.U32, f, e), one, zero)
	b.SetPhi(rank, b.Add(ir.U32, rank, b.Add(ir.U32, gt, b.Mul(ir.U32, eq, lower))))
	b.SetPhi(m, b.Max(ir.F32, m, lf))
	b.SetPhi(f, b.Add(ir.U32, f, one))
	b.EndLoop()

	slot := b.Min(ir.U32, rank, b.Const(ir.U32, int64(k)))
	b.Store(pSel, slot, e, 0)
	b.Store(pTop, slot, le, 0)
	var wv ir.Value
	if !r.Norm {
		// Z over every expert, in ascending order, from the exponentials each
		// thread formed for its own expert.
		ee := b.Exp(b.Sub(ir.F32, le, m))
		b.Store(shE, e, ee, 0)
		b.Barrier()
		b.Loop(int64(nExpert))
		sum := b.Phi(ir.F32, zeroF)
		j := b.Phi(ir.U32, zero)
		b.SetPhi(sum, b.Add(ir.F32, sum, b.Load(ir.F32, shE, j, 0)))
		b.SetPhi(j, b.Add(ir.U32, j, one))
		b.EndLoop()
		wv = b.Div(ir.F32, ee, sum)
	} else {
		// The selected logits, best first, in threadgroup memory. The bin
		// slot k is written by every unselected thread and read by nobody.
		b.Store(sh, slot, le, 0)
		b.Barrier()
		ms := b.Load(ir.F32, sh, zero, 0)
		b.Loop(int64(k))
		mk := b.Phi(ir.F32, ms)
		i := b.Phi(ir.U32, zero)
		b.SetPhi(mk, b.Max(ir.F32, mk, b.Load(ir.F32, sh, i, 0)))
		b.SetPhi(i, b.Add(ir.U32, i, one))
		b.EndLoop()
		b.Loop(int64(k))
		sum := b.Phi(ir.F32, zeroF)
		j := b.Phi(ir.U32, zero)
		b.SetPhi(sum, b.Add(ir.F32, sum, b.Exp(b.Sub(ir.F32, b.Load(ir.F32, sh, j, 0), mk))))
		b.SetPhi(j, b.Add(ir.U32, j, one))
		b.EndLoop()
		wv = b.Div(ir.F32, b.Exp(b.Sub(ir.F32, le, mk)), sum)
	}
	// The routed scale, as in ExpertWeights.
	if sc := r.scaleOf(); sc != 1 {
		wv = b.Mul(ir.F32, wv, b.ConstF32(sc))
	}
	b.Store(pW, slot, wv, 0)
	return b.Done()
}

// ExpertRouteWarp is ExpertRoute for the renormalised softmax route on one
// 32-lane subgroup, launched as a single workgroup of 32 threads.
//
// ExpertRoute is a thread per expert and each thread walks every logit, which
// is O(E^2) behind a barrier and slow at hundreds of experts. This is
// llama.cpp's topk_moe shape: each lane holds its E/32 logits in registers
// (expert lane + 32j), and the k winners are k rounds of an argmax (a local
// pass, a butterfly across lanes, the winner struck out).
//
// It is bit-for-bit ExpertRank + ExpertWeights: ties go to the lower expert id,
// and the weights are exp(l - top1) over the selected logits in slot order.
// The un-renormalised route sums over every expert in ascending id, an order a
// butterfly cannot keep, so it stays on ExpertRoute.
//
// Params: pL, pSel, pTop, pW, as ExpertRoute's; slots 0..k-1 are written and
// the bin slot k is not.
func ExpertRouteWarp(r MoERoute) (*ir.Kernel, error) {
	if err := r.check("ExpertRouteWarp"); err != nil {
		return nil, err
	}
	if r.Gated() || r.Bias || r.Grouped() || r.rows() > 1 || !r.Norm {
		return nil, fmt.Errorf("kernels: ExpertRouteWarp: only the renormalised softmax route, one token")
	}
	const lanes = ir.SubgroupLanes
	nExpert, k := r.NExpert, r.K
	per := (nExpert + lanes - 1) / lanes
	if per > 32 || k > 64 {
		return nil, fmt.Errorf("kernels: ExpertRouteWarp: %d experts, k=%d is past its registers", nExpert, k)
	}
	b := ir.New("expertroutewarp", [3]int{lanes, 1, 1})
	pL := b.Param("pL", ir.F32)
	pSel := b.Param("pSel", ir.U32)
	pTop := b.Param("pTop", ir.F32)
	pW := b.Param("pW", ir.F32)

	zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
	negInf := b.ConstF32(float32(math.Inf(-1)))
	lane := b.And(ir.U32, b.TID(), b.Const(ir.U32, lanes-1))
	nE := b.Const(ir.U32, int64(nExpert))
	v := make([]ir.Value, per)
	id := make([]ir.Value, per)
	for j := 0; j < per; j++ {
		id[j] = b.Add(ir.U32, lane, b.Const(ir.U32, int64(j*lanes)))
		if (j+1)*lanes <= nExpert {
			v[j] = b.Load(ir.F32, pL, lane, int64(j*lanes))
			continue
		}
		// The ragged last column: a clamped load, and -inf past the end so a
		// padding lane can never be selected (k <= E).
		ld := b.Load(ir.F32, pL, b.Min(ir.U32, id[j], b.Const(ir.U32, int64(nExpert-1))), 0)
		v[j] = b.Select(ir.F32, b.Lt(ir.U32, id[j], nE), ld, negInf)
	}
	flag := func(p ir.Value) ir.Value { return b.Select(ir.U32, p, one, zero) }
	// eqU is 1 where x == y and 0 elsewhere, from Lt alone.
	eqU := func(t ir.Type, x, y ir.Value) ir.Value {
		return b.Sub(ir.U32, one, b.Add(ir.U32, flag(b.Lt(t, x, y)), flag(b.Lt(t, y, x))))
	}
	// The k rounds are a loop, not unrolled: unrolled, the kernel was ~48 KB
	// of code and met a cold instruction cache on every call inside a model.
	// One round's body stays resident across all k.
	sh := b.Shared("sel", ir.F32, k)
	pv := make([]ir.Value, per)
	b.Loop(int64(k))
	n := b.Phi(ir.U32, zero)
	for j := range pv {
		pv[j] = b.Phi(ir.F32, v[j])
	}
	// The lane's own best, ties to the lower id: its ids ascend with j, so only
	// a strictly greater value replaces the incumbent.
	best, bi := pv[0], id[0]
	for j := 1; j < per; j++ {
		gt := b.Lt(ir.F32, best, pv[j])
		best = b.Select(ir.F32, gt, pv[j], best)
		bi = b.Select(ir.U32, gt, id[j], bi)
	}
	for m := lanes / 2; m >= 1; m /= 2 {
		ov := b.ShuffleXor(ir.F32, best, int64(m))
		oi := b.ShuffleXor(ir.U32, bi, int64(m))
		// take the other when it is greater, or equal with a lower id
		take := b.Add(ir.U32, flag(b.Lt(ir.F32, best, ov)),
			b.Mul(ir.U32, eqU(ir.F32, best, ov), flag(b.Lt(ir.U32, oi, bi))))
		t := b.Lt(ir.U32, zero, take)
		best = b.Select(ir.F32, t, ov, best)
		bi = b.Select(ir.U32, t, oi, bi)
	}
	// Every lane stores the same bits to the same word.
	b.Store(pSel, n, bi, 0)
	b.Store(pTop, n, best, 0)
	b.Store(sh, n, best, 0)
	for j := range pv {
		b.SetPhi(pv[j], b.Select(ir.F32, b.Lt(ir.U32, zero, eqU(ir.U32, id[j], bi)), negInf, pv[j]))
	}
	b.SetPhi(n, b.Add(ir.U32, n, one))
	b.EndLoop()
	b.Barrier()

	// ExpertWeights' renormalised arithmetic, loop for loop: the maximum of the
	// selected logits, their exponentials summed in slot order, then each
	// divided by that sum and scaled.
	seed := b.Load(ir.F32, sh, zero, 0)
	zeroF := b.ConstF32(0)
	b.Loop(int64(k))
	mk := b.Phi(ir.F32, seed)
	i := b.Phi(ir.U32, zero)
	b.SetPhi(mk, b.Max(ir.F32, mk, b.Load(ir.F32, sh, i, 0)))
	b.SetPhi(i, b.Add(ir.U32, i, one))
	b.EndLoop()
	b.Loop(int64(k))
	sum := b.Phi(ir.F32, zeroF)
	jj := b.Phi(ir.U32, zero)
	b.SetPhi(sum, b.Add(ir.F32, sum, b.Exp(b.Sub(ir.F32, b.Load(ir.F32, sh, jj, 0), mk))))
	b.SetPhi(jj, b.Add(ir.U32, jj, one))
	b.EndLoop()
	sc := r.scaleOf()
	b.Loop(int64(k))
	o := b.Phi(ir.U32, zero)
	w := b.Div(ir.F32, b.Exp(b.Sub(ir.F32, b.Load(ir.F32, sh, o, 0), mk)), sum)
	if sc != 1 {
		w = b.Mul(ir.F32, w, b.ConstF32(sc))
	}
	b.Store(pW, o, w, 0)
	b.SetPhi(o, b.Add(ir.U32, o, one))
	b.EndLoop()
	return b.Done()
}

// ExpertRouteWidth is ExpertRoute's workgroup: the expert count rounded up to
// a warp, or 0 when it will not fit one workgroup.
func ExpertRouteWidth(nExpert int) int {
	w := (nExpert + 31) / 32 * 32
	if w > 1024 {
		return 0
	}
	return w
}

// ExpertCombine sums the k expert outputs, weighted: out[i] = SUM_j w[j]*e[j*n+i].
func ExpertCombine(n, slots int) (*ir.Kernel, error) {
	if n < 1 || slots < 1 {
		return nil, fmt.Errorf("kernels: ExpertCombine: n=%d slots=%d", n, slots)
	}
	b := ir.New("expertcombine", [3]int{128, 1, 1})
	pE := b.Param("pE", ir.F32) // slots x n expert outputs
	pW := b.Param("pW", ir.F32) // slots weights
	pOut := b.Param("pOut", ir.F32)

	zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(n-1)))

	zeroF := b.ConstF32(0)
	stride := b.Const(ir.U32, int64(n))
	b.Loop(int64(slots))
	acc := b.Phi(ir.F32, zeroF)
	j := b.Phi(ir.U32, zero)
	idx := b.Phi(ir.U32, i)
	b.SetPhi(acc, b.Fma(b.Load(ir.F32, pW, j, 0), b.Load(ir.F32, pE, idx, 0), acc))
	b.SetPhi(j, b.Add(ir.U32, j, one))
	b.SetPhi(idx, b.Add(ir.U32, idx, stride))
	b.EndLoop()
	b.Store(pOut, i, acc, 0)
	return b.Done()
}

// RouterMatVecOf is out[r] = SUM_i W[r*k+i] * x[i] over float32 weights: the
// router, the one matrix in a MoE block that is not quantized
// (ffn_gate_inp). Without it the block would fall back to the host. Launch it
// with RouterThreads(rows) threads.
//
// With f16 set it reads binary16 weights (some
// files, Mixtral's among them, store the router as F16; reading it as f32
// walks past the buffer). The f16 form reads the weight as a u32 word and
// decodes the half the element sits in; the F32 kernel is unchanged.
//
// The buffer must be padded to a whole word: an odd rows*k reads the last
// element's word, whose other half is past the matrix.
func RouterMatVecOf(rows, k int, f16 bool) (*ir.Kernel, error) { return routerMatVec(rows, k, 1, f16) }

// RouterMatVecTok is RouterMatVecOf over ntok tokens: token t reads x at t*k
// and writes its rows logits at t*rows. Launch RouterThreads(rows*ntok) threads.
func RouterMatVecTok(rows, k, ntok int, f16 bool) (*ir.Kernel, error) {
	if ntok < 1 {
		return nil, fmt.Errorf("kernels: RouterMatVecTok: ntok=%d", ntok)
	}
	return routerMatVec(rows, k, ntok, f16)
}

func routerMatVec(rows, k, ntok int, f16 bool) (*ir.Kernel, error) {
	if rows < 1 || k < 1 {
		return nil, fmt.Errorf("kernels: RouterMatVec: rows=%d k=%d", rows, k)
	}
	name, wt := "routermatvec", ir.F32
	if f16 {
		name, wt = "routermatvec_f16", ir.U32
	}
	b := ir.New(name, [3]int{routerGroup, 1, 1})
	pW := b.Param("pW", wt)
	pX := b.Param("pX", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	part := b.Shared("part", ir.F32, routerGroup)

	// A workgroup per row, not a thread per row: the group's threads stride k
	// together so every load is one coalesced line, and the partials meet in
	// shared memory. A thread per row left most of the card idle. One token
	// emits the original kernel; more carry the token in the workgroup index,
	// row fastest.
	r := b.Min(ir.U32, b.CTAID(), b.Const(ir.U32, int64(rows*ntok-1)))
	out, xAt := r, func(i ir.Value) ir.Value { return i }
	if ntok > 1 {
		tok := b.Div(ir.U32, r, b.Const(ir.U32, int64(rows)))
		r = b.Rem(ir.U32, r, b.Const(ir.U32, int64(rows)))
		xBase := b.Mul(ir.U32, tok, b.Const(ir.U32, int64(k)))
		xAt = func(i ir.Value) ir.Value { return b.Add(ir.U32, xBase, i) }
	}
	t := b.TID()
	rowBase := b.Mul(ir.U32, r, b.Const(ir.U32, int64(k)))
	last := b.Const(ir.U32, int64(k-1))
	kk := b.Const(ir.U32, int64(k))
	zeroF := b.ConstF32(0)
	zeroU, step := b.Const(ir.U32, 0), b.Const(ir.U32, routerGroup)
	var oneU ir.Value // only the f16 unpack reads it; an unused constant would move the f32 kernel's bytes
	if f16 {
		oneU = b.Const(ir.U32, 1)
	}

	// A ragged k has no conditional to skip the tail (RULE 13): the index is
	// clamped so the load stays in bounds, and the product is selected away.
	b.Loop(int64((k + routerGroup - 1) / routerGroup))
	acc := b.Phi(ir.F32, zeroF)
	i := b.Phi(ir.U32, t)
	idx := b.Min(ir.U32, i, last)
	var w ir.Value
	if e := b.Add(ir.U32, rowBase, idx); f16 {
		word := b.Load(ir.U32, pW, b.Shr(ir.U32, e, oneU), 0)
		sh := b.Shl(ir.U32, b.And(ir.U32, e, oneU), b.Const(ir.U32, 4))
		w = b.CvtF16H(b.Shr(ir.U32, word, sh))
	} else {
		w = b.Load(ir.F32, pW, e, 0)
	}
	prod := b.Mul(ir.F32, w, b.Load(ir.F32, pX, xAt(idx), 0))
	b.SetPhi(acc, b.Add(ir.F32, acc, b.Select(ir.F32, b.Lt(ir.U32, i, kk), prod, zeroF)))
	b.SetPhi(i, b.Add(ir.U32, i, step))
	b.EndLoop()

	b.Store(part, t, acc, 0)
	b.Barrier()
	// A tree, not a serial sum: at each step every thread adds its partner's
	// partial, clamped into the group; threads above the live range compute
	// sums nobody reads, and a barrier separates every read from the next
	// write (no conditional needed).
	top := b.Const(ir.U32, routerGroup-1)
	for s := routerGroup / 2; s >= 1; s /= 2 {
		mine := b.Load(ir.F32, part, t, 0)
		other := b.Load(ir.F32, part, b.Min(ir.U32, b.Add(ir.U32, t, b.Const(ir.U32, int64(s))), top), 0)
		sum := b.Add(ir.F32, mine, other)
		b.Barrier()
		b.Store(part, t, sum, 0)
		b.Barrier()
	}
	// Every thread stores the same bits to the same word -- the benign shape
	// MatVec's GroupSplit uses.
	b.Store(pOut, out, b.Load(ir.F32, part, zeroU, 0), 0)
	return b.Done()
}

// routerGroup is RouterMatVec's workgroup: one per row.
const routerGroup = 128

// RouterThreads is how many threads a RouterMatVec of rows launches.
func RouterThreads(rows int) int { return rows * routerGroup }

// SharedRouterLogits is the shared expert's gate: one dot product per row
// against a vector of model-dimension weights. Launch SharedRouterThreads(rows)
// threads.
//
// It is not RouterMatVec with rows=1, whose rows count experts; it is
// RouterMatVecTok with one expert and the chunk's rows as tokens, so token t
// reads x at t*k and writes logit t. Reusing the expert axis would apply the
// first row's gate to every row.
func SharedRouterLogits(rows, k int) (*ir.Kernel, error) {
	if rows < 1 || k < 1 {
		return nil, fmt.Errorf("kernels: SharedRouterLogits: rows=%d k=%d", rows, k)
	}
	return routerMatVec(1, k, rows, false)
}

// SharedRouterThreads is how many threads a SharedRouterLogits of rows launches.
func SharedRouterThreads(rows int) int { return RouterThreads(rows) }

// SharedExpertAdd folds the always-on expert into the routed sum:
//
//	dst[i] = acc[i] + sigma(logit[0]) * v[i]
//
// The gate is one scalar computed on the device from the block's input, so
// reading it back would synchronise mid-block. It is added after the routed
// sum, not into it: the routed weights may be renormalised by their own sum,
// which must not rescale the shared contribution (see engine/model/moe.go).
//
// Out of place (RULE 13), so clamped surplus threads rewrite rather than add
// twice.
func SharedExpertAdd(n, rows int) (*ir.Kernel, error) {
	if n < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: SharedExpertAdd: n=%d rows=%d", n, rows)
	}
	b := ir.New("sharedexpertadd", [3]int{128, 1, 1})
	pAcc := b.Param("pAcc", ir.F32)
	pV := b.Param("pV", ir.F32)
	pLogit := b.Param("pLogit", ir.F32)
	pDst := b.Param("pDst", ir.F32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(rows*n-1)))
	one := b.ConstF32(1)
	// The logit is per row: a batched chunk has one gate per token.
	g := b.Load(ir.F32, pLogit, b.Div(ir.U32, i, b.Const(ir.U32, int64(n))), 0)
	sig := b.Div(ir.F32, one, b.Add(ir.F32, one, b.Exp(b.Sub(ir.F32, b.ConstF32(0), g))))
	b.Store(pDst, i, b.Add(ir.F32, b.Load(ir.F32, pAcc, i, 0),
		b.Mul(ir.F32, sig, b.Load(ir.F32, pV, i, 0))), 0)
	return b.Done()
}

// ExpertWeightScale multiplies each routed weight by its expert's own factor,
// out[r*k+j] = w[r*k+j] * scale[sel[r*(k+1)+j]] over rows of k (Gemma 4's
// router.per_expert_scale, applied after the renormalisation). The selection
// has ExpertRank's k+1 stride, the weights ExpertWeights' k. Indexed by the
// true expert id, as IndexedBiasAdd is, and out of place for RULE 13.
func ExpertWeightScale(rows, k int) (*ir.Kernel, error) {
	if rows < 1 || k < 1 {
		return nil, fmt.Errorf("kernels: ExpertWeightScale: rows=%d k=%d", rows, k)
	}
	b := ir.New("expertweightscale", [3]int{128, 1, 1})
	pW := b.Param("pW", ir.F32)
	pSel := b.Param("pSel", ir.U32)
	pScale := b.Param("pScale", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(rows*k-1)))
	nk := b.Const(ir.U32, int64(k))
	si := b.Add(ir.U32, i, b.Div(ir.U32, i, nk)) // r*(k+1)+j = i + r
	e := b.Load(ir.U32, pSel, si, 0)
	b.Store(pOut, i, b.Mul(ir.F32, b.Load(ir.F32, pW, i, 0), b.Load(ir.F32, pScale, e, 0)), 0)
	return b.Done()
}

// IndexedBiasAdd adds each slot's expert bias to its rows:
// out[j*rows + r] = in[j*rows + r] + bias[sel[j]*rows + r], for k slots.
//
// Indexed by the true expert id: the bias belongs to the expert, not the
// slot, so it is read through the selection even where the weights come from
// a compacted bank indexed 0..k-1. Out of place for RULE 13.
func IndexedBiasAdd(k, rows int) (*ir.Kernel, error) {
	if k < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: IndexedBiasAdd: k=%d rows=%d", k, rows)
	}
	b := ir.New("indexedbiasadd", [3]int{128, 1, 1})
	pIn := b.Param("pIn", ir.F32)
	pSel := b.Param("pSel", ir.U32)
	pBias := b.Param("pBias", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(k*rows-1)))
	nr := b.Const(ir.U32, int64(rows))
	j, r := b.Div(ir.U32, i, nr), b.Rem(ir.U32, i, nr)
	e := b.Load(ir.U32, pSel, j, 0)
	bi := b.Add(ir.U32, b.Mul(ir.U32, e, nr), r)
	b.Store(pOut, i, b.Add(ir.F32, b.Load(ir.F32, pIn, i, 0), b.Load(ir.F32, pBias, bi, 0)), 0)
	return b.Done()
}
