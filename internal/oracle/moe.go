package oracle

import "math"

// The mixture-of-experts router's selection, in plain Go: the reference
// cpu.EmitMoERoute and nn.MoERouteJIT are gated against on every tier.
//
// It covers two routers, the second a superset of the first. The softmax top-k
// router (llama, qwen3moe, qwen3next, olmoe, gpt-oss) is this one with no bias,
// no groups and a scale of 1. DeepSeek-V3's, shared by R1, Kimi-K2 and
// GLM-4-MoE, is:
//
//	scores   = sigmoid(logits)                       (not softmax)
//	biased   = scores + e_score_correction_bias      SELECTION ONLY
//	groups   = the best topk_group of n_group groups, scored by the sum of
//	           their top TWO biased values; the rest are masked to -Inf
//	sel      = the top k of what is left, over the BIASED scores
//	weights  = the UNBIASED scores of those k
//	          / their sum if norm_topk_prob, then * routed_scaling_factor
//
// The bias applied to the returned weights as well as the selection gives the
// right experts and plausible text from the wrong model; MoEFaultBiasedWeights
// is that mistake. Where the references disagree, see MoEDivergences().

// MoEGate is a router's configuration. Everything here is a property of the
// container, fixed before the first token.
type MoEGate struct {
	// Bias is e_score_correction_bias, one value per expert, or nil. It is
	// added to the scores for selection and never to the returned weights.
	Bias []float32
	// NGroup is how many contiguous groups the experts are split into and
	// NGroupUsed how many survive. 0 or 1 group is ungrouped -- with one group
	// every expert is kept whatever the scores are, so Kimi-K2's
	// n_group=1/topk_group=1 is the ungrouped path and not a special case.
	NGroup, NGroupUsed int
	// Norm renormalises the selected weights by their sum (norm_topk_prob).
	Norm bool
	// Scale is routed_scaling_factor, applied after the renormalisation. 1
	// means no scaling and is the value every pre-V3 architecture uses.
	Scale float32
	// ExpScale is one factor per expert, multiplying its selected weight
	// after everything else (Gemma 4's router.per_expert_scale), or nil.
	ExpScale []float32
}

// Grouped reports whether the group phase runs at all.
func (g MoEGate) Grouped() bool { return g.NGroup > 1 }

// MoERoute is one router call's outputs, in the two orders the engine reads.
// Sel/Wt are in selection order (descending biased score) and Ord/OWt are the
// same k experts by ascending id. Sum is the divisor, before Scale.
type MoERoute struct {
	Sel, Ord []int32
	Wt, OWt  []float32
	Sum      float32
}

// MoEFault is a deliberate defect for a gate to fire on; every one of these
// leaves the model fluent.
type MoEFault int

const (
	MoEFaultNone MoEFault = iota
	// The fold keeps the last of a tie instead of the first.
	MoEFaultLastOfTie
	// The divisor is summed in ascending id order instead of selection order:
	// the same k terms, a different float32 parenthesisation.
	MoEFaultAscendingSum
	// ord is left in selection order.
	MoEFaultOrdInSelectionOrder
	// The returned weights come from the biased scores. The right experts,
	// every weight off by that expert's bias.
	MoEFaultBiasedWeights
	// A group is scored by its top-1 biased value instead of its top two. It
	// reorders groups only when the best members disagree with the runners-up,
	// which on a real router is most tokens and on a flat one is none.
	MoEFaultGroupTop1
	// The groups are masked after the global top-k rather than before, i.e.
	// the selection ignores the grouping entirely. Every selected expert is
	// still a real expert with its real weight.
	MoEFaultMaskAfterTopK
	// routed_scaling_factor is applied before the renormalisation, where the
	// division then cancels it exactly. Invisible whenever Norm is set.
	MoEFaultScaleBeforeNorm
	// The per-expert scale is taken by selection RANK instead of expert id:
	// s[i] for the i-th winner rather than s[sel[i]].
	MoEFaultExpScaleByRank
	// The per-expert scale is not applied.
	MoEFaultNoExpScale
)

func (f MoEFault) String() string {
	switch f {
	case MoEFaultNone:
		return "none"
	case MoEFaultLastOfTie:
		return "a fold that keeps the LAST of a tie"
	case MoEFaultAscendingSum:
		return "the divisor summed in ASCENDING id order"
	case MoEFaultOrdInSelectionOrder:
		return "ord left in SELECTION order"
	case MoEFaultBiasedWeights:
		return "the weights taken from the BIASED scores"
	case MoEFaultGroupTop1:
		return "a group scored by its top-1 instead of its top-2"
	case MoEFaultMaskAfterTopK:
		return "the groups masked AFTER the global top-k"
	case MoEFaultScaleBeforeNorm:
		return "the scaling factor applied BEFORE the renormalisation"
	case MoEFaultExpScaleByRank:
		return "the per-expert scale taken by selection RANK, not expert id"
	case MoEFaultNoExpScale:
		return "the per-expert scale dropped"
	}
	return "?"
}

// MoEFaults() is every defect, so a gate can enumerate them rather than restate
// them and a ninth cannot be added on one side only.
func MoEFaults() []MoEFault {
	return []MoEFault{
		MoEFaultLastOfTie,
		MoEFaultAscendingSum,
		MoEFaultOrdInSelectionOrder,
		MoEFaultBiasedWeights,
		MoEFaultGroupTop1,
		MoEFaultMaskAfterTopK,
		MoEFaultScaleBeforeNorm,
		MoEFaultExpScaleByRank,
		MoEFaultNoExpScale,
	}
}

// moeSumFloor guards the renormalising divide. It is kernels.MoESumFloor and
// the two are asserted equal by TestRouterFloorMatchesTheKernel: an oracle that
// guards differently from the kernel it gates is not an oracle.
const moeSumFloor = 1e-20

// MoESumFloorForTest exposes that constant so a gate in another package can
// assert it equals the kernel's. See nn.TestRouterFloorMatchesTheKernel.
const MoESumFloorForTest = moeSumFloor

// MoEDivergences() is where llama.cpp and DeepSeek's own modeling_deepseek.py
// disagree about this router, and which one jitllm matches (AGENTS.md RULE 7m:
// know which one you are matching, and keep the list a reviewed artefact).
//
// Each entry is {what, llama.cpp, modeling_deepseek.py, jitllm}.
func MoEDivergences() [][4]string {
	return [][4]string{
		{
			"the renormalising divisor's floor",
			"ggml_clamp(sum, 6.103515625e-5, +Inf) -- the smallest normal f16",
			"sum + 1e-20",
			"modeling_deepseek.py: sum + 1e-20, DECIDED. It used to be " +
				"NEITHER -- an all-underflowed selection divided 0 by 0 and took " +
				"the token non-finite. The two guards are identical where the sum " +
				"is zero and differ in a REACHABLE range: for 0 < sum < 6.1e-5 the " +
				"clamp divides by 6.1e-5 and distorts a legitimate small sum, where " +
				"the addition divides by ~sum and does not. Below an f32 ulp of any " +
				"sum above ~1e-13, so it is invisible everywhere the clamp would " +
				"bite -- the strictly less invasive of the two, and the one the " +
				"weights were trained against",
		},
		{
			"n_group_used == n_expert_groups",
			"REFUSED: GGML_ASSERT(n_group_used < n_expert_groups) at load",
			"allowed -- torch.topk(k=n_group) is an ordinary call",
			"allowed: every group is kept, which is what the definition says and " +
				"is one fewer shape to refuse",
		},
		{
			"whether k == 1 renormalises",
			"yes -- norm_w alone, so a single weight becomes exactly 1.0",
			"no -- `if self.top_k > 1 and self.norm_topk_prob`",
			"llama.cpp: Norm alone, at every k",
		},
		{
			"whether a scale of 1 is applied",
			"no -- `if (w_scale != 0.0f && w_scale != 1.0f)`",
			"yes, always -- `topk_weight * self.routed_scaling_factor`",
			"llama.cpp: x*1 is x for every finite float, so the two agree in value " +
				"and jitllm emits no multiply",
		},
	}
}

// MoERouter is the reference. scores is the post-gating score of every expert
// -- sigmoid(logits) for V3/R1/K2/GLM-4-MoE, softmax(logits) for everything
// before it -- because which gating function ran is the caller's choice of
// kernel and not this function's business (nn.MoERouteJIT runs the generated
// one, and the engine has always computed the softmax as its own kernel).
//
// Selection is the lexicographic walk over (score, -index), the kernel's
// contract: equal scores go to the lowest index, as ggml_argsort's descending
// order does (it matters when a fresh router emits identical logits). A NaN is
// never eligible.
//
// The method is deliberately not the kernel's: the kernel finds the kept groups
// from a threshold pair left by the last group pass, this builds the kept set
// directly, so the comparison is between two derivations.
func MoERouter(scores []float32, k int, g MoEGate, f MoEFault) MoERoute {
	n := len(scores)
	scale := g.Scale
	if scale == 0 {
		scale = 1
	}

	// The selection plane: the scores plus the correction bias. The weights
	// never come from here.
	bi := make([]float32, n)
	copy(bi, scores)
	if g.Bias != nil {
		for e := range bi {
			bi[e] = scores[e] + g.Bias[e]
		}
	}

	if g.Grouped() && f != MoEFaultMaskAfterTopK {
		sz := n / g.NGroup
		take := 2
		if sz < take {
			take = sz
		}
		if f == MoEFaultGroupTop1 {
			take = 1
		}
		gs := make([]float32, g.NGroup)
		for grp := range gs {
			span := bi[grp*sz : (grp+1)*sz]
			var s float32 // ggml_sum_rows accumulates from zero
			for _, e := range moeLexTopK(span, take, MoEFaultNone) {
				s += span[e]
			}
			gs[grp] = s
		}
		keep := make([]bool, g.NGroup)
		for _, grp := range moeLexTopK(gs, g.NGroupUsed, MoEFaultNone) {
			keep[grp] = true
		}
		negInf := float32(math.Inf(-1))
		for grp, ok := range keep {
			if ok {
				continue
			}
			for e := grp * sz; e < (grp+1)*sz; e++ {
				bi[e] = negInf
			}
		}
	}

	selI := moeLexTopK(bi, k, f)
	ordI := append([]int(nil), selI...)
	for i := 1; i < len(ordI); i++ {
		for j := i; j > 0 && ordI[j] < ordI[j-1]; j-- {
			ordI[j], ordI[j-1] = ordI[j-1], ordI[j]
		}
	}

	// The weight of a selected expert is its unbiased score.
	w := make([]float32, len(selI))
	for i, e := range selI {
		w[i] = scores[e]
		if f == MoEFaultBiasedWeights {
			w[i] = bi[e]
		}
		if f == MoEFaultScaleBeforeNorm {
			w[i] *= scale
		}
	}

	sum := float32(1)
	if g.Norm {
		sum = 0
		// In selection order, one addition at a time: that is the float32 sum
		// model.Forward, model.Prefill and model.ForwardBatch are gated
		// bit-identical against each other on a mixture.
		src := w
		if f == MoEFaultAscendingSum {
			src = make([]float32, len(ordI))
			for i, e := range ordI {
				for j, s := range selI {
					if s == e {
						src[i] = w[j]
					}
				}
			}
		}
		for _, v := range src {
			sum += v
		}
		// The floor is DeepSeek's addition rather than llama.cpp's clamp: they
		// agree at zero, and the clamp distorts a legitimate small sum. See
		// MoEDivergences().
		sum += moeSumFloor
	}
	for i := range w {
		w[i] /= sum
		if scale != 1 && f != MoEFaultScaleBeforeNorm {
			w[i] *= scale
		}
		switch {
		case g.ExpScale == nil || f == MoEFaultNoExpScale:
		case f == MoEFaultExpScaleByRank:
			w[i] *= g.ExpScale[i]
		default:
			w[i] *= g.ExpScale[selI[i]]
		}
	}

	if f == MoEFaultOrdInSelectionOrder {
		copy(ordI, selI)
	}

	r := MoERoute{
		Sel: make([]int32, len(selI)), Ord: make([]int32, len(ordI)),
		Wt: make([]float32, len(selI)), OWt: make([]float32, len(ordI)),
		Sum: sum,
	}
	for i, e := range selI {
		r.Sel[i], r.Wt[i] = int32(e), w[i]
	}
	for i, e := range ordI {
		r.Ord[i] = int32(e)
		for j, s := range selI {
			if s == e {
				r.OWt[i] = w[j]
			}
		}
	}
	return r
}

// moeLexTopK is the k largest of v, descending, with the lowest index winning a
// tie and a NaN never eligible. It is the definition every phase of this router
// selects by: the global top-k, a group's top two, and the kept groups.
func moeLexTopK(v []float32, k int, f MoEFault) []int {
	out := make([]int, 0, k)
	var prevVal float32
	prevIdx := -1
	for i := 0; i < k; i++ {
		best := -1
		var bestVal float32
		for e, x := range v {
			if x != x {
				continue // a NaN is ordered against nothing
			}
			if i > 0 && !(x < prevVal || (x == prevVal && e > prevIdx)) {
				continue
			}
			// `>` keeps the lowest index of a tie; `>=` keeps the last, which
			// is MoEFaultLastOfTie.
			if best < 0 || x > bestVal || (f == MoEFaultLastOfTie && x == bestVal) {
				best, bestVal = e, x
			}
		}
		if best < 0 {
			break // nothing eligible is left, which only k > n can reach
		}
		out = append(out, best)
		prevVal, prevIdx = bestVal, best
	}
	return out
}

// MoESigmoid is sigma(x) = 1/(1+exp(-x)) in f64, narrowed once. It is the
// reference for the gating step, which is a separate generated kernel run
// before the router kernel (as the softmax always was).
//
// It is the one place in this router compared with a tolerance: the kernel's
// exp is a minimax polynomial, this is libm. Everything after the gating is
// bit-exact on every tier.
func MoESigmoid(x float32) float32 {
	return float32(1 / (1 + math.Exp(-float64(x))))
}
