package oracle

import (
	"math"
	"sort"
)

// The samplers' definitions in float64, which the composed draws are gated
// against (graph.SampleRow, SampleWalk, SampleCut and their Prob forms). Each
// returns, beside its answer, how far the draw is from flipping, so a gate
// holds only the draws no float32 rounding can turn.

// Argmax is the index of the largest of v, the lower of two equal.
func Argmax(v []float32) int32 {
	best := 0
	for i := range v {
		if v[i] > v[best] {
			best = i
		}
	}
	return int32(best)
}

// PenalizeOnce is v with the repeat penalty once on each token hist holds,
// however often: a positive value divided by pen, any other multiplied.
func PenalizeOnce(v []float32, hist []int, pen float32) {
	seen := map[int]bool{}
	for _, h := range hist {
		if seen[h] {
			continue
		}
		seen[h] = true
		if v[h] > 0 {
			v[h] /= pen
		} else {
			v[h] *= pen
		}
	}
}

// SampleRow is the draw with no top-k and no cut: the repeat penalty once on
// each token the window holds, in float32 (PenalizeOnce), the row's softmax
// at 1/T in float64, its order (the larger first, the lower id of two equal),
// and the walk at u -- the first token whose mass up to and including it
// passes u times the row's mass, the last where none does. It returns the
// token and how far u's point is from the nearest edge, as a share of the
// mass.
func SampleRow(logits []float32, hist []int, pen float32, invT float32, u float64) (tok uint32, margin float64) {
	v := append([]float32(nil), logits...)
	PenalizeOnce(v, hist, pen)
	n := len(v)
	m := math.Inf(-1)
	s := make([]float64, n)
	for i := range v {
		s[i] = float64(v[i] * invT)
		m = max(m, s[i])
	}
	q := make([]float64, n)
	mass := 0.0
	for i := range q {
		q[i] = math.Exp(s[i] - m)
		mass += q[i]
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return q[order[a]] > q[order[b]] })
	target, upto := u*mass, 0.0
	margin = math.Inf(1)
	tok = uint32(order[n-1])
	landed := false
	for _, i := range order {
		upto += q[i]
		margin = min(margin, math.Abs(upto-target)/mass)
		if !landed && upto > target {
			tok, landed = uint32(i), true
		}
	}
	return tok, margin
}

// SampleWalk is the top-k draw with no penalty: the K best logits (the lower
// id on a tie), the first k of them, top-p and then min-p over their softmax
// at temperature 1 with the best always kept, then the walk at u over the
// survivors' softmax at 1/T -- the first whose mass up to and including it
// passes u times their mass, the last where none does. It returns the token
// and how far u's point is from the nearest edge, as a share of the mass.
func SampleWalk(logits []float32, K, k int, invT, topp, minp, u float64) (tok uint32, margin float64) {
	order := make([]int, len(logits))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return logits[order[a]] > logits[order[b]] })
	K = min(K, len(logits))
	v0 := float64(logits[order[0]])
	p := make([]float64, K)
	sum := 0.0
	for i := range min(k, K) {
		p[i] = math.Exp(float64(logits[order[i]]) - v0)
		sum += p[i]
	}
	var kept []int
	before := 0.0
	for i := range K {
		p[i] /= sum
		if i == 0 || i < k && before < topp && p[i] >= minp*p[0] {
			kept = append(kept, i)
		}
		before += p[i]
	}
	et := make([]float64, len(kept))
	mass := 0.0
	for j, i := range kept {
		et[j] = math.Exp((float64(logits[order[i]]) - v0) * invT)
		mass += et[j]
	}
	target, upto := u*mass, 0.0
	margin = math.Inf(1)
	tok = uint32(order[kept[len(kept)-1]])
	landed := false
	for j, i := range kept {
		upto += et[j]
		margin = min(margin, math.Abs(upto-target)/mass)
		if !landed && upto > target {
			tok, landed = uint32(order[i]), true
		}
	}
	return tok, margin
}

// SampleCut is the draw with no top-k and a cut: the repeat penalty once a
// history entry in float32, the row's softmax at temperature 1 in float64 and
// its order (the larger first, the lower id of two equal); min-p stops at the
// first token under min-p times the best's probability, before it joins the
// mass, and top-p at the first whose mass up to it reaches top-p, after; the
// best kept. The survivors' softmax at 1/T is walked at u in that order.
// Where neither cut fires the draw is SampleRow's. It returns the token and
// how far the draw is from flipping: the nearest cut's distance from its
// threshold and u's point from an edge.
func SampleCut(logits []float32, hist []int, pen, invT float32, topp, minp, u float64) (tok uint32, margin float64) {
	v := append([]float32(nil), logits...)
	PenalizeOnce(v, hist, pen)
	n := len(v)
	m := math.Inf(-1)
	for i := range v {
		m = max(m, float64(v[i]))
	}
	q1 := make([]float64, n)
	sum := 0.0
	for i := range v {
		q1[i] = math.Exp(float64(v[i]) - m)
		sum += q1[i]
	}
	for i := range q1 {
		q1[i] /= sum
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return q1[order[a]] > q1[order[b]] })
	p0 := q1[order[0]]
	keep, fired, acc := n, false, 0.0
	margin = math.Inf(1)
	for k, i := range order {
		if minp > 0 {
			margin = min(margin, math.Abs(q1[i]-minp*p0)/p0)
		}
		if q1[i] < minp*p0 {
			keep, fired = k, true
			break
		}
		acc += q1[i]
		if !math.IsInf(topp, 1) {
			margin = min(margin, math.Abs(acc-topp))
		}
		if acc >= topp {
			keep, fired = k+1, true
			break
		}
	}
	if !fired {
		t, mg := SampleRow(v, nil, 1, invT, u)
		return t, min(margin, mg)
	}
	keep = max(keep, 1)
	surv := order[:keep]
	top := float64(v[surv[0]] * invT)
	et := make([]float64, keep)
	mass := 0.0
	for j, i := range surv {
		et[j] = math.Exp(float64(v[i]*invT) - top)
		mass += et[j]
	}
	target, upto := u*mass, 0.0
	tok = uint32(surv[keep-1])
	landed := false
	for j, i := range surv {
		upto += et[j]
		margin = min(margin, math.Abs(upto-target)/mass)
		if !landed && upto > target {
			tok, landed = uint32(i), true
		}
	}
	return tok, margin
}

// SampleProb is the probability the draw gives token x: of the candidates
// (the k best penalized logits, or the whole row where k is 0) in their
// order, min-p and top-p over their softmax at temperature 1 (the best
// kept), then x's share of the survivors' softmax at 1/T; with no top-k and
// no cut firing, x's share of the row's softmax at 1/T. It returns the
// probability and how far the cuts are from flipping.
func SampleProb(logits []float32, hist []int, pen, invT float32, k int, topp, minp float64, x int) (p, margin float64) {
	v := append([]float32(nil), logits...)
	PenalizeOnce(v, hist, pen)
	n := len(v)
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return v[order[a]] > v[order[b]] })
	cand := order
	if k > 0 {
		cand = order[:k]
	}
	v0 := float64(v[cand[0]])
	q1 := make([]float64, len(cand))
	sum := 0.0
	for j, i := range cand {
		q1[j] = math.Exp(float64(v[i]) - v0)
		sum += q1[j]
	}
	keep, fired, acc := len(cand), false, 0.0
	margin = math.Inf(1)
	for j := range cand {
		q1[j] /= sum
		if minp > 0 {
			margin = min(margin, math.Abs(q1[j]-minp*q1[0])/q1[0])
		}
		if q1[j] < minp*q1[0] {
			keep, fired = j, true
			break
		}
		acc += q1[j]
		if !math.IsInf(topp, 1) {
			margin = min(margin, math.Abs(acc-topp))
		}
		if acc >= topp {
			keep, fired = j+1, true
			break
		}
	}
	surv := cand[:max(keep, 1)]
	if k == 0 && !fired {
		surv = order
	}
	top := float64(v[surv[0]] * invT)
	mass, px := 0.0, 0.0
	for _, i := range surv {
		e := math.Exp(float64(v[i]*invT) - top)
		mass += e
		if i == x {
			px = e
		}
	}
	return px / mass, margin
}
