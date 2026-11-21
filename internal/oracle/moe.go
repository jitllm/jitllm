package oracle

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
// is that mistake.

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

// moeSumFloor guards the renormalising divide. It is the composed router's
// floor and the two are asserted equal by model.TestRouterFloorMatchesTheOracle:
// an oracle that guards differently from the kernel it gates is not an oracle.
const moeSumFloor = 1e-20

// MoESumFloorForTest exposes that constant so a gate in another package can
// assert it equals the kernel's. See model.TestRouterFloorMatchesTheOracle.
const MoESumFloorForTest = moeSumFloor

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
