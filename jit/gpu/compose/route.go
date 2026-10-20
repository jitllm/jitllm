package compose

import (
	"math"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// Route is a mixture's router as a group ranks it: k of NExpert experts a
// row of logits, scored by the logit itself, its sigmoid or its
// sqrt-softplus, ranked on that score plus a selection bias (one shared,
// or a row's own), within the NGroupUsed best of NGroup groups (their masked
// plane given), the weights renormalised over the selection (Norm) and
// scaled.
type Route struct {
	NExpert, K            int
	Sigmoid, SqrtSoftplus bool
	Bias, RowBias         bool
	NGroup                int
	Norm                  bool
	Scale                 float32
	// Rows is the rows one group routes, 0 and 1 one.
	Rows int
}

func (r Route) rows() int     { return max(r.Rows, 1) }
func (r Route) gated() bool   { return r.Sigmoid || r.SqrtSoftplus }
func (r Route) grouped() bool { return r.NGroup > 1 }

// softBias is a softmax router's bias, added to the probability: the rank
// needs the row's softmax first.
func (r Route) softBias() bool { return !r.gated() && r.Bias }

func (r Route) scale() float32 {
	if r.Scale == 0 {
		return 1
	}
	return r.Scale
}

// score is the router's gate of logit l.
func (g *Row) score(r Route, l int) int {
	switch {
	case r.SqrtSoftplus:
		return g.Add(Node{Kind: NUnary, Op: OpSqrt, A: g.Act(kernels.ActSoftplus, l)})
	case r.Sigmoid:
		one := g.Const(1)
		return g.Bin(OpDiv, one, g.Bin(OpAdd, one, g.exp(g.Bin(OpMul, l, g.Const(-1)))))
	}
	return l
}

func (g *Row) exp(x int) int { return g.Add(Node{Kind: NUnary, Op: OpExp, A: x}) }

// at is row's offset of stride added to index i, i itself at one row.
func (g *Row) at(r Route, row, stride, i int) int {
	if r.rows() == 1 {
		return i
	}
	return g.UBin(OpAdd, g.UBin(OpMul, row, g.UConst(stride)), i)
}

// RouteRank ranks a row's experts, an element each (a group of NExpert by
// the rows): each expert's place by its key among the row's, ties to the
// lower id, its id and its unbiased score written at that place of sel and
// top (k+1 a row, place k the bin every unchosen expert writes). l is the
// logits, bm the masked plane of a grouped route, bias the selection bias
// (-1 for none).
func (g *Row) RouteRank(r Route, l, sel, top, bm, bias int) {
	u, ub := g.UConst, g.UBin
	e, row := g.Column(), g.RowIdx()
	atL := func(i int) int { return g.at(r, row, r.NExpert, i) }
	if r.rows() > 1 {
		e = ub(OpMin, e, u(r.NExpert-1))
	}
	var m, z int
	if r.softBias() {
		m = g.FoldMax(r.NExpert, func(i int) int { return g.LoadAt(l, atL(i)) })
		z = g.FoldSum(r.NExpert, func(j int) int { return g.exp(g.Bin(OpSub, g.LoadAt(l, atL(j)), m)) })
	}
	key := func(i int) int {
		switch {
		case r.grouped():
			return g.LoadAt(bm, atL(i))
		case r.gated():
			v := g.score(r, g.LoadAt(l, atL(i)))
			if r.Bias {
				bi := i
				if r.RowBias {
					bi = atL(i)
				}
				v = g.Bin(OpAdd, v, g.LoadAt(bias, bi))
			}
			return v
		case r.softBias():
			p := g.Bin(OpDiv, g.exp(g.Bin(OpSub, g.LoadAt(l, atL(i)), m)), z)
			return g.Bin(OpAdd, p, g.LoadAt(bias, i))
		}
		return g.LoadAt(l, atL(i))
	}
	mine := key(e)
	rank := g.Rank(r.NExpert, e, mine, key)
	slot := g.at(r, row, r.K+1, ub(OpMin, rank, u(r.K)))
	g.StoreAt(sel, e, slot)
	t := mine
	if r.gated() || r.softBias() {
		t = g.score(r, g.LoadAt(l, atL(e)))
	}
	g.StoreAt(top, t, slot)
}

// RouteWeights is a row's mixture weights from its ranked scores, a thread
// a row: a softmax over the selection (Norm) or over every logit l, or for
// a gated router the scores renormalised by their sum, and scaled.
func (g *Row) RouteWeights(r Route, top, w, l int) {
	row := g.RowIdx()
	atTop := func(i int) int { return g.at(r, row, r.K+1, i) }
	atW := func(i int) int { return g.at(r, row, r.K, i) }
	sc := r.scale()
	scaled := func(v int) int {
		if sc != 1 {
			return g.Bin(OpMul, v, g.Const(sc))
		}
		return v
	}
	if r.gated() {
		var sum int
		if r.Norm {
			sum = g.Bin(OpAdd, g.FoldSum(r.K, func(j int) int { return g.LoadAt(top, atTop(j)) }), g.Const(1e-20))
		}
		c := g.Loop(r.K)
		n := g.Iter(c)
		v := g.LoadAt(top, atTop(n))
		if r.Norm {
			v = g.Bin(OpDiv, v, sum)
		}
		g.StoreAt(w, scaled(v), atW(n))
		return
	}
	src, at, red := top, atTop, r.K
	if !r.Norm {
		src, at, red = l, func(i int) int { return g.at(r, row, r.NExpert, i) }, r.NExpert
	}
	m := g.FoldMax(red, func(i int) int { return g.LoadAt(src, at(i)) })
	sum := g.FoldSum(red, func(j int) int { return g.exp(g.Bin(OpSub, g.LoadAt(src, at(j)), m)) })
	c := g.Loop(r.K)
	n := g.Iter(c)
	g.StoreAt(w, scaled(g.Bin(OpDiv, g.exp(g.Bin(OpSub, g.LoadAt(top, atTop(n)), m)), sum)), atW(n))
}

// RouteLanes is the renormalised softmax route of one row on one subgroup
// (lane rows of the target's lanes): each lane holds its NExpert/lanes
// logits, expert lane+j*lanes; k rounds each take the best by a butterfly
// of (logit, id), ties to the lower id, write it and strike it out; then
// the weights, a softmax over the selection in a workgroup array.
func (g *Row) RouteLanes(r Route, l, sel, top, w int) {
	u, ub := g.UConst, g.UBin
	L := g.Width
	per := (r.NExpert + L - 1) / L
	lane := g.Column()
	negInf := g.Const(float32(math.Inf(-1)))
	v, id := make([]int, per), make([]int, per)
	for j := range v {
		id[j] = ub(OpAdd, lane, u(j*L))
		if (j+1)*L <= r.NExpert {
			v[j] = g.LoadAt(l, id[j])
			continue
		}
		ld := g.LoadAt(l, ub(OpMin, id[j], u(r.NExpert-1)))
		v[j] = g.Select(g.Less(id[j], u(r.NExpert)), ld, negInf)
	}
	eq := func(x, y int) int { return ub(OpSub, u(1), ub(OpAdd, g.Flag(g.Less(x, y)), g.Flag(g.Less(y, x)))) }
	sh := g.Shared(r.K, false)
	c := g.Loop(r.K)
	n := g.Iter(c)
	pv := make([]int, per)
	for j := range pv {
		pv[j] = g.Fold(c, v[j])
	}
	best, bi := pv[0], id[0]
	for j := 1; j < per; j++ {
		gt := g.Less(best, pv[j])
		best, bi = g.Select(gt, pv[j], best), g.Select(gt, id[j], bi)
	}
	for m := L / 2; m >= 1; m /= 2 {
		ov, oi := g.Shuffle(best, m), g.Shuffle(bi, m)
		take := ub(OpAdd, g.Flag(g.Less(best, ov)), ub(OpMul, eq(best, ov), g.Flag(g.Less(oi, bi))))
		t := g.Less(u(0), take)
		best, bi = g.Select(t, ov, best), g.Select(t, oi, bi)
	}
	g.StoreAt(sel, bi, n)
	g.StoreAt(top, best, n)
	g.StoreAt(sh, best, n)
	for j := range pv {
		g.SetFold(pv[j], g.Select(g.Less(u(0), eq(id[j], bi)), negInf, pv[j]))
	}
	mk := g.FoldMax(r.K, func(i int) int { return g.LoadAt(sh, i) })
	sum := g.FoldSum(r.K, func(j int) int { return g.exp(g.Bin(OpSub, g.LoadAt(sh, j), mk)) })
	c2 := g.Loop(r.K)
	o := g.Iter(c2)
	wv := g.Bin(OpDiv, g.exp(g.Bin(OpSub, g.LoadAt(sh, o), mk)), sum)
	if sc := r.scale(); sc != 1 {
		wv = g.Bin(OpMul, wv, g.Const(sc))
	}
	g.StoreAt(w, wv, o)
}

// RouteShared is the softmax route of one row on one workgroup, an expert a
// thread: the logits shared, each expert's rank and the row's maximum in one
// pass, its id and logit written at its place, then its weight, a softmax
// over the selection (Norm, the selected logits shared at their places) or
// over every expert (each exponential shared), and scaled.
func (g *Row) RouteShared(r Route, l, sel, top, w int) {
	u, ub := g.UConst, g.UBin
	e := g.UBin(OpMin, g.Column(), g.UConst(r.NExpert-1))
	le := g.LoadAt(l, e)
	shL := g.Shared(r.NExpert, false)
	g.StoreAt(shL, le, e)
	c := g.Loop(r.NExpert)
	f := g.Iter(c)
	rank := g.Fold(c, u(0))
	m := g.Fold(c, g.LoadAt(shL, u(0)))
	lf := g.LoadAt(shL, f)
	gt, lt := g.Flag(g.Less(le, lf)), g.Flag(g.Less(lf, le))
	eq := ub(OpSub, u(1), ub(OpAdd, gt, lt))
	lower := g.Flag(g.Less(f, e))
	g.SetFold(rank, ub(OpAdd, rank, ub(OpAdd, gt, ub(OpMul, eq, lower))))
	g.SetFold(m, g.Bin(OpMax, m, lf))
	slot := ub(OpMin, rank, u(r.K))
	g.StoreAt(sel, e, slot)
	g.StoreAt(top, le, slot)
	var wv int
	if !r.Norm {
		ee := g.exp(g.Bin(OpSub, le, m))
		shE := g.Shared(r.NExpert, false)
		g.StoreAt(shE, ee, e)
		wv = g.Bin(OpDiv, ee, g.FoldSum(r.NExpert, func(j int) int { return g.LoadAt(shE, j) }))
	} else {
		sh := g.Shared(r.K+1, false)
		g.StoreAt(sh, le, slot)
		mk := g.FoldMax(r.K, func(i int) int { return g.LoadAt(sh, i) })
		sum := g.FoldSum(r.K, func(j int) int { return g.exp(g.Bin(OpSub, g.LoadAt(sh, j), mk)) })
		wv = g.Bin(OpDiv, g.exp(g.Bin(OpSub, le, mk)), sum)
	}
	if sc := r.scale(); sc != 1 {
		wv = g.Bin(OpMul, wv, g.Const(sc))
	}
	g.StoreAt(w, wv, slot)
}

// Combine is an element of the slots' outputs summed by their weights:
// out = sum_j w[j] * e[j*n + element], n the group's width.
func (g *Row) Combine(e, w, slots int) int {
	u, ub := g.UConst, g.UBin
	i := g.Column()
	c := g.Loop(slots)
	j := g.Iter(c)
	acc := g.Fold(c, g.Const(0))
	g.SetFold(acc, g.fma(g.LoadAt(w, j), g.LoadAt(e, ub(OpAdd, i, ub(OpMul, j, u(g.Width)))), acc))
	return acc
}
