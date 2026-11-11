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

// Penalize is v with llama.cpp's penalties on each token hist holds: the
// repeat penalty once (PenalizeOnce), then frequency times its count plus
// presence subtracted, each step rounded to float32.
func Penalize(v []float32, hist []int, rep, presence, frequency float32) {
	PenalizeOnce(v, hist, rep)
	count := map[int]int{}
	for _, h := range hist {
		count[h]++
	}
	for t, c := range count {
		f := float32(float32(c) * frequency)
		v[t] = float32(v[t] - float32(f+presence))
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

// typicalCut is llama.cpp's locally typical cut over p, the candidates'
// probabilities: of the candidates in (in), in the order of their surprise
// -log p's distance from the entropy, the nearest first (the earlier of two
// equal), each while the mass before it is at most typ, the nearest always;
// every candidate where typ is +Inf. It returns the survivors and how near
// the cut is to moving: the mass before each from typ, and 0 where two
// candidates on either side of it are within rounding of each other's
// distance.
func typicalCut(p []float64, in []bool, typ float64) (keep []bool, margin float64) {
	keep = append([]bool(nil), in...)
	margin = math.Inf(1)
	if math.IsInf(typ, 1) {
		return keep, margin
	}
	h := 0.0
	for i, x := range p {
		if in[i] && x > 0 {
			h -= x * math.Log(x)
		}
	}
	shift := make([]float64, len(p))
	var idx []int
	for i, x := range p {
		if in[i] {
			shift[i] = math.Abs(-math.Log(x) - h)
			idx = append(idx, i)
		}
	}
	sort.SliceStable(idx, func(a, b int) bool { return shift[idx[a]] < shift[idx[b]] })
	before := 0.0
	for j, i := range idx {
		keep[i] = j == 0 || before <= typ
		if j > 0 {
			margin = min(margin, math.Abs(before-typ))
			if keep[i] != keep[idx[j-1]] && shift[i]-shift[idx[j-1]] < 1e-4*(1+shift[i]) {
				margin = 0
			}
		}
		before += p[i]
	}
	return keep, margin
}

// TopNSigma is v with llama.cpp's top-n-sigma: every value under the largest
// less n standard deviations of the finite values at -Inf (none where n is
// not above 0), and how near a value is to that edge.
func TopNSigma(v []float32, n float64) ([]float32, float64) {
	out := append([]float32(nil), v...)
	if n <= 0 {
		return out, math.Inf(1)
	}
	mx, sum, cnt := math.Inf(-1), 0.0, 0
	for _, x := range v {
		if !math.IsInf(float64(x), -1) {
			mx, sum, cnt = max(mx, float64(x)), sum+float64(x), cnt+1
		}
	}
	mean, ss := sum/float64(cnt), 0.0
	for _, x := range v {
		if !math.IsInf(float64(x), -1) {
			ss += (float64(x) - mean) * (float64(x) - mean)
		}
	}
	edge := mx - n*math.Sqrt(ss/float64(cnt))
	margin := math.Inf(1)
	for i, x := range v {
		margin = min(margin, math.Abs(float64(x)-edge)/(1+math.Abs(edge)))
		if float64(x) < edge {
			out[i] = float32(math.Inf(-1))
		}
	}
	return out, margin
}

// xtcCut is llama.cpp's XTC over the survivors surv (in their order, largest
// first) with probabilities p, the share of their softmax at temperature 1:
// those at or above thr, every one but the last of them, where two or more
// are; +Inf cuts none. It returns what is left and how near a survivor is to
// the threshold.
func xtcCut(surv []int, p func(i int) float64, thr float64) ([]int, float64) {
	margin := math.Inf(1)
	if math.IsInf(thr, 1) {
		return surv, margin
	}
	sum := 0.0
	for _, i := range surv {
		sum += p(i)
	}
	last := 0
	for j, i := range surv {
		ps := p(i) / sum
		margin = min(margin, math.Abs(ps-thr))
		if ps < thr {
			break
		}
		last = j
	}
	return surv[last:], margin
}

// SampleWalk is the top-k draw with no penalty: the K best logits (the lower
// id on a tie), the first k of them, typical-p (typicalCut; +Inf where off),
// then top-p and min-p over the survivors' softmax at temperature 1 with the
// best survivor always kept, XTC at threshold xtc (xtcCut), then the walk at
// u over the
// survivors' softmax at 1/T -- the first whose mass up to and including it
// passes u times their mass, the last where none does. It returns the token
// and how far u's point is from the nearest edge, as a share of the mass.
func SampleWalk(logits []float32, K, k int, invT, topp, minp, typ, xtc, u float64) (tok uint32, margin float64) {
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
	in := make([]bool, K)
	for i := range K {
		p[i] /= sum
		in[i] = i < k
	}
	keepT, tm := typicalCut(p, in, typ)
	first, tsum := -1, 0.0
	for i := range K {
		if keepT[i] {
			tsum += p[i]
			if first < 0 {
				first = i
			}
		}
	}
	if !math.IsInf(typ, 1) {
		for i := range K {
			if keepT[i] {
				p[i] /= tsum
			} else {
				p[i] = 0
			}
		}
	}
	pmax := p[first]
	var kept []int
	before := 0.0
	for i := range K {
		if i == first || keepT[i] && before < topp && p[i] >= minp*pmax {
			kept = append(kept, i)
		}
		before += p[i]
	}
	kept, xm := xtcCut(kept, func(i int) float64 { return p[i] }, xtc)
	et := make([]float64, len(kept))
	mass := 0.0
	for j, i := range kept {
		et[j] = math.Exp((float64(logits[order[i]]) - v0) * invT)
		mass += et[j]
	}
	target, upto := u*mass, 0.0
	margin = min(tm, xm)
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
// best kept. Typical-p (typicalCut; +Inf where off) cuts the row before
// them, the cuts then reading its survivors renormalized, and their walk is
// the draw whatever the cuts do. The survivors' softmax at 1/T is walked at
// u in that order. Where no cut fires the draw is SampleRow's. It returns the token and
// how far the draw is from flipping: the nearest cut's distance from its
// threshold and u's point from an edge.
func SampleCut(logits []float32, hist []int, pen, invT float32, topp, minp, typ, xtc, u float64) (tok uint32, margin float64) {
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
	all := make([]bool, n)
	for i := range all {
		all[i] = true
	}
	order, margin = typicalOrder(q1, order, all, typ)
	p0 := q1[order[0]]
	keep, fired, acc := len(order), !math.IsInf(typ, 1) || !math.IsInf(xtc, 1), 0.0
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
	surv, xm := xtcCut(order[:keep], func(i int) float64 { return q1[i] }, xtc)
	margin = min(margin, xm)
	keep = len(surv)
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

// typicalOrder is order (a row's tokens, the larger q first) cut to
// typical-p's survivors over q (typicalCut, the candidates those in), q
// renormalized over them in place where typ is not +Inf, and how near the
// cut is to moving.
func typicalOrder(q []float64, order []int, in []bool, typ float64) ([]int, float64) {
	keep, margin := typicalCut(q, in, typ)
	if math.IsInf(typ, 1) {
		return order, margin
	}
	var out []int
	sum := 0.0
	for _, i := range order {
		if keep[i] {
			out = append(out, i)
			sum += q[i]
		}
	}
	for i := range q {
		if keep[i] {
			q[i] /= sum
		} else {
			q[i] = 0
		}
	}
	return out, margin
}

// SampleProb is the probability the draw gives token x: of the candidates
// (the k best penalized logits, or the whole row where k is 0) in their
// order, typical-p (+Inf where off), then min-p and top-p over the
// survivors' softmax at temperature 1 (the best kept), then x's share of the
// survivors' softmax at 1/T; with no top-k and no cut firing, x's share of
// the row's softmax at 1/T. It returns the probability and how far the cuts
// are from flipping.
func SampleProb(logits []float32, hist []int, pen, invT float32, k int, topp, minp, typ, xtc, mix float64, x int) (p, margin float64) {
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
	q := make([]float64, n)
	in := make([]bool, n)
	sum := 0.0
	for _, i := range cand {
		q[i] = math.Exp(float64(v[i]) - v0)
		in[i] = true
		sum += q[i]
	}
	for _, i := range cand {
		q[i] /= sum
	}
	cand, margin = typicalOrder(q, cand, in, typ)
	q1 := make([]float64, len(cand))
	for j, i := range cand {
		q1[j] = q[i]
	}
	keep, fired, acc := len(cand), !math.IsInf(typ, 1) || !math.IsInf(xtc, 1), 0.0
	for j := range cand {
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
	share := func(surv []int) float64 {
		top := float64(v[surv[0]] * invT)
		mass, px := 0.0, 0.0
		for _, i := range surv {
			e := math.Exp(float64(v[i]*invT) - top)
			mass += e
			if i == x {
				px = e
			}
		}
		return px / mass
	}
	pc := share(surv)
	if math.IsInf(xtc, 1) {
		return pc, margin
	}
	cut, xm := xtcCut(surv, func(i int) float64 { return q[i] }, xtc)
	return pc + (share(cut)-pc)*mix, min(margin, xm)
}
