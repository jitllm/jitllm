package compose

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// A mixture's routing at one row, as op fragments over a row of experts:
// each is a function of the route's numbers and flags (Route), its
// selection counted by Rank and its row-wide maxima and sums folded in
// index order, so every value is the float the router's arithmetic gives
// in that order.

// Score is how a route scores an expert's logit.
type Score uint8

const (
	ScoreLogit        Score = iota // the logit itself (a softmax route ranks logits)
	ScoreSigmoid                   // 1/(1 + e^-l)
	ScoreSqrtSoftplus              // sqrt(softplus(l))
)

// Route is a router's selection and weights at one row: K of Experts, the
// score of a logit, a selection bias on the key, Groups of experts of which
// the best GroupsUsed are kept, the weights renormalised (Norm) and scaled,
// or, where Jitter is set, sparsemixer's weights over a plain top two.
type Route struct {
	Experts, K         int
	Score              Score
	Bias               bool
	Groups, GroupsUsed int
	Norm               bool
	Scale              float32
	Jitter             float32
}

// gated is a route whose weights are its scores (renormalised), not a
// softmax of its logits.
func (rt Route) gated() bool { return rt.Score != ScoreLogit }

// grouped is a route keeping whole groups of experts.
func (rt Route) grouped() bool { return rt.Groups > 1 }

// softBias is a softmax route with a selection bias: the bias is added to
// the probability, so the key needs the softmax over every logit first.
func (rt Route) softBias() bool { return !rt.gated() && rt.Bias }

// scale is the weights' multiply, 1 for none (0 and 1 both).
func (rt Route) scale() float32 {
	if rt.Scale == 0 {
		return 1
	}
	return rt.Scale
}

// Check refuses a route these fragments do not form.
func (rt Route) Check() error {
	switch {
	case rt.K < 1 || rt.Experts < 1 || rt.K > rt.Experts:
		return fmt.Errorf("compose: a route of %d of %d experts", rt.K, rt.Experts)
	case rt.grouped() && (rt.Experts%rt.Groups != 0 || rt.GroupsUsed < 1 || rt.GroupsUsed > rt.Groups):
		return fmt.Errorf("compose: %d experts in %d groups, %d kept", rt.Experts, rt.Groups, rt.GroupsUsed)
	case rt.grouped() && !rt.gated():
		return fmt.Errorf("compose: a softmax route over groups needs the softmax before the groups' scores")
	case rt.Jitter != 0 && (rt.K != 2 || rt.gated() || rt.Bias || rt.grouped() || rt.Norm || rt.scale() != 1):
		return fmt.Errorf("compose: sparsemixer's weights are over a plain top two")
	}
	return nil
}

// negBig is a masked key: every masked expert ties at it, so the lower
// index ranks first, as an infinity would.
const negBig = -math.MaxFloat32

// GateScore is score s of logit node l.
func (g *Row) GateScore(s Score, l int) int {
	switch s {
	case ScoreSigmoid:
		one := g.Const(1)
		e := g.Add(Node{Kind: NUnary, Op: OpExp, A: g.Bin(OpMul, l, g.Const(-1))})
		return g.Bin(OpDiv, one, g.Bin(OpAdd, one, e))
	case ScoreSqrtSoftplus:
		return g.Act(kernels.ActSqrtSoftplus, l)
	}
	return l
}

// scored is the route's score of the logit at index node i of buffer l,
// plus the selection bias at i where bias is a buffer (>= 0).
func (g *Row) scored(rt Route, l, bias, i int) int {
	v := g.GateScore(rt.Score, g.LoadAt(l, i))
	if bias >= 0 {
		v = g.Bin(OpAdd, v, g.LoadAt(bias, i))
	}
	return v
}

// TopK writes the route's selection over a row of Experts logits (buffer
// l, the group's row): each expert's rank by its key, its id into sel and
// its score into top at slot min(rank, K) -- slot K a bin every expert not
// kept writes and nothing reads, so both are K+1 long. The key is the
// logit, its score plus the selection bias (buffer bias), the softmax's
// probability plus the bias, or a grouped route's masked plane (buffer
// plane, GroupMask's); bias and plane are -1 where the route has none.
func (g *Row) TopK(rt Route, l, sel, top, plane, bias int) {
	e := g.Column()
	var sm, sz int
	if rt.softBias() {
		sm = g.FoldMax(rt.Experts, func(f int) int { return g.LoadAt(l, f) })
		sz = g.FoldSum(rt.Experts, func(f int) int {
			return g.Add(Node{Kind: NUnary, Op: OpExp, A: g.Bin(OpSub, g.LoadAt(l, f), sm)})
		})
	}
	key := func(i int) int {
		switch {
		case rt.grouped():
			return g.LoadAt(plane, i)
		case rt.gated():
			return g.scored(rt, l, bias, i)
		case rt.softBias():
			p := g.Bin(OpDiv, g.Add(Node{Kind: NUnary, Op: OpExp, A: g.Bin(OpSub, g.LoadAt(l, i), sm)}), sz)
			return g.Bin(OpAdd, p, g.LoadAt(bias, i))
		}
		return g.LoadAt(l, i)
	}
	mine := key(e)
	slot := g.UBin(OpMin, g.Rank(rt.Experts, e, mine, key), g.UConst(rt.K))
	g.StoreAt(sel, e, slot)
	// The score, unbiased: the bias moves the selection, never a weight.
	t := mine
	if rt.gated() || rt.softBias() {
		t = g.GateScore(rt.Score, g.LoadAt(l, e))
	}
	g.StoreAt(top, t, slot)
}

// GroupMask writes a grouped route's selection plane over a row of Experts
// logits (buffer l) into out: each expert's biased score, or negBig where
// its group is not among the GroupsUsed best. A group's score is the sum of
// its two highest biased scores (one where a group is one expert), carried
// as two running maxima; groups rank as experts do (Rank).
func (g *Row) GroupMask(rt Route, l, bias, out int) {
	sz := rt.Experts / rt.Groups
	gscore := func(h int) int {
		c := g.Loop(sz)
		big := g.Const(negBig)
		m1, m2 := g.Fold(c, big), g.Fold(c, big)
		v := g.scored(rt, l, bias, g.UBin(OpAdd, g.UBin(OpMul, h, g.UConst(sz)), g.Iter(c)))
		// A new maximum pushes the old one down; otherwise the candidate
		// competes for second place. Both read the trip's start.
		g.SetFold(m2, g.Select(g.Less(m1, v), m1, g.Bin(OpMax, m2, v)))
		g.SetFold(m1, g.Bin(OpMax, m1, v))
		if sz > 1 {
			return g.Bin(OpAdd, m1, m2)
		}
		return m1
	}
	e := g.Column()
	grp := g.UBin(OpDiv, e, g.UConst(sz))
	rank := g.Rank(rt.Groups, grp, gscore(grp), gscore)
	keep := g.Less(rank, g.UConst(rt.GroupsUsed))
	g.Store(out, g.Select(keep, g.scored(rt, l, bias, e), g.Const(negBig)))
}

// moeSumFloor is the renormalising divisor's floor, the router's own (kernels.MoESumFloor).
const moeSumFloor = kernels.MoESumFloor

// RouteWeights writes the route's K weights (a row of K, buffer w) from the
// selected scores (buffer top, TopK's) and, where a softmax is over every
// expert or the route is sparsemixer's, the row's logits (buffer l, -1
// where unread): the scores renormalised by their sum (in slot order, plus
// the floor) for a gated route; a softmax over the selected logits, or over
// every logit, for a softmax one; sparsemixer's two softmaxes within its
// jitter of each pick. Then the scale.
func (g *Row) RouteWeights(rt Route, top, l, w int) {
	n := g.Column()
	var v int
	switch {
	case rt.gated():
		v = g.LoadAt(top, n)
		if rt.Norm {
			sum := g.Bin(OpAdd, g.FoldSum(rt.K, func(f int) int { return g.LoadAt(top, f) }), g.Const(moeSumFloor))
			v = g.Bin(OpDiv, v, sum)
		}
	case rt.Jitter != 0:
		v = g.sparseMixer(rt, top, l, n)
	default:
		src, red := top, rt.K
		if !rt.Norm {
			src, red = l, rt.Experts
		}
		m := g.FoldMax(red, func(f int) int { return g.LoadAt(src, f) })
		ex := func(x int) int { return g.Add(Node{Kind: NUnary, Op: OpExp, A: g.Bin(OpSub, x, m)}) }
		sum := g.FoldSum(red, func(f int) int { return ex(g.LoadAt(src, f)) })
		v = g.Bin(OpDiv, ex(g.LoadAt(top, n)), sum)
	}
	if sc := rt.scale(); sc != 1 {
		v = g.Bin(OpMul, v, g.Const(sc))
	}
	g.Store(w, v)
}

// sparseMixer is sparsemixer's weight of pick n (0 or 1): one over the sum
// of exp(s_j - m) over the logits s_j within a relative 2*Jitter of the
// pick's logit m, the second leaving out the first pick (the lowest index
// holding it, so a tie keeps the twin). A term left out is selected away,
// not multiplied by zero, as exp past the maximum can overflow.
func (g *Row) sparseMixer(rt Route, top, l, n int) int {
	ne := rt.Experts
	m1, m2 := g.LoadAt(top, g.UConst(0)), g.LoadAt(top, g.UConst(1))
	// i1: the first index holding m1.
	c := g.Loop(ne)
	cn := g.UConst(ne)
	i1 := g.Fold(c, cn)
	j := g.Iter(c)
	x := g.LoadAt(l, j)
	eq := g.UBin(OpSub, g.UConst(1), g.UBin(OpAdd, g.Flag(g.Less(x, m1)), g.Flag(g.Less(m1, x))))
	cand := g.UBin(OpAdd, g.UBin(OpMul, eq, j), g.UBin(OpMul, g.UBin(OpSub, g.UConst(1), eq), cn))
	g.SetFold(i1, g.UBin(OpMin, i1, cand))
	thr := g.Const(float32(2 * float64(rt.Jitter)))
	term := func(s, m int) int {
		abs := g.Bin(OpMax, s, g.Bin(OpMul, s, g.Const(-1)))
		gap := g.Bin(OpDiv, g.Bin(OpSub, m, s), g.Bin(OpMax, abs, m))
		return g.Select(g.Less(thr, gap), g.Const(0), g.Add(Node{Kind: NUnary, Op: OpExp, A: g.Bin(OpSub, s, m)}))
	}
	s1 := g.FoldSum(ne, func(f int) int { return term(g.LoadAt(l, f), m1) })
	s2 := g.FoldSum(ne, func(f int) int {
		t := term(g.LoadAt(l, f), m2)
		return g.Select(g.Less(f, i1), t, g.Select(g.Less(i1, f), t, g.Const(0)))
	})
	one := g.Const(1)
	return g.Select(g.Less(n, g.UConst(1)), g.Bin(OpDiv, one, s1), g.Bin(OpDiv, one, s2))
}

// ScaleWeights writes each of a row's K weights (buffer w) times its
// expert's own factor (buffer scale, read at the expert the selection sel
// names at the weight's slot) into out.
func (g *Row) ScaleWeights(w, sel, scale, out int) {
	g.Store(out, g.Bin(OpMul, g.Load(w), g.LoadAt(scale, g.LoadAt(sel, g.Column()))))
}

// RowDot stores the dot of each row's a and b (nodes at the element) at
// the row's index of out: the products summed over the group's threads,
// strided, then the group's tree.
func (g *Row) RowDot(a, b, out int) {
	g.StoreAt(out, g.Reduce(RedSum, g.Bin(OpMul, a, b)), g.RowIdx())
}
