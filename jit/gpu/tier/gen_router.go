package tier

import (
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// A mixture's routing, generated from its ops: the router's logits and a
// shared expert's gate as row dots, the logits' bias, the group mask, the
// top k and the weights as the route's fragments (compose/route.go), each
// stated by the op's attributes -- its graph op's route (graph.Attrs: the
// router's score, groups, renormalisation and scale), its bound vectors,
// and its inputs -- and the per-expert weight scale. The dots, top k and
// weights hold the menu's arithmetic in its order (TestComposedRouteMatches
// TheMenu, TestComposedRouterDotsMatchTheMenu).

// dotGroup is a row dot's group: the router's dot sums 128 strided
// partials in a tree of 128, and that tree is its arithmetic's order.
const dotGroup = 128

// opRoute is op o's route as its attributes state it; false where they do
// not state one the fragments form.
func opRoute(o *devOp) (compose.Route, bool) {
	a := &o.A
	rt := compose.Route{Experts: a.Experts, K: a.TopK, Groups: a.Groups, GroupsUsed: a.GroupsUsed,
		Norm: a.RouteNorm, Scale: float32(a.Scale)}
	switch a.Router {
	case graph.RouterSoftmax:
	case graph.RouterSigmoid:
		rt.Score = compose.ScoreSigmoid
	case graph.RouterSqrtSoftplus:
		rt.Score = compose.ScoreSqrtSoftplus
	case graph.RouterSparseMixer:
		// Its jitter, stated by the builder on the weights' op.
		rt.Jitter = float32(o.Alpha)
		if o.Kind == dkRouteWeights && rt.Jitter == 0 {
			return rt, false
		}
	default:
		return rt, false
	}
	return rt, rt.Check() == nil
}

// genRoute is routing op o's launches, or nil where it is not one or its
// group cannot be formed (o.Why says which). The dots of the router and of
// a shared expert's gate run at any rows; the rest at one row. Callers
// hold g.mu.
func (g *devTier) genRoute(o *devOp, prog *devProgram, rows int) []genLaunch {
	switch o.Kind {
	case dkRouterLogits, dkGateLogit, dkRouterBias, dkScaleWeights, dkGroupMask, dkTopK, dkRouteWeights:
	default:
		return nil
	}
	if len(o.In) == 0 || len(o.Out) == 0 {
		return nil
	}
	if o.Kind != dkGateLogit && rows != 1 {
		o.Why = whyNoBatch
		return nil
	}
	w := func(v devVal) int { return prog.vals[v].Width }
	switch o.Kind {
	case dkRouterLogits, dkGateLogit:
		// One logit a row: the dot of the row with the router's (an expert
		// a row of the group) or with the gate's one vector (a token a row).
		k := w(o.In[0])
		if o.V == nil || k < 1 {
			o.Why = whyNoWeight
			return nil
		}
		rb := &rowBuild{r: compose.Row{Width: k, Rows: rows}}
		r := &rb.r
		var a, b int
		if o.Kind == dkRouterLogits {
			l := g.curLayer
			if l == nil || w(o.Out[0]) < 1 {
				o.Why = whyNoWeight
				return nil
			}
			wk := compose.BufRow
			if l.routerF16 {
				wk = compose.BufRowF16
			}
			r.Rows = w(o.Out[0])
			a, b = r.Load(rb.buf(wk, gaV)), r.Load(rb.buf(compose.BufVec, gaIn0))
		} else {
			if w(o.Out[0]) != 1 {
				return nil
			}
			a, b = r.Load(rb.buf(compose.BufVec, gaV)), r.Load(rb.buf(compose.BufRow, gaIn0))
		}
		r.RowDot(a, b, rb.buf(compose.BufOut, gaOut0))
		return g.routeLaunch(o, rb, compose.Schedule{Group: dotGroup})
	case dkRouterBias:
		if o.V == nil {
			o.Why = whyNoWeight
			return nil
		}
		rb := &rowBuild{r: compose.Row{Width: w(o.Out[0]), Rows: 1}}
		rb.store(rb.bin(compose.OpAdd, rb.load(compose.BufRow, gaIn0), rb.load(compose.BufVec, gaV)), gaOut0)
		return g.routeLaunch(o, rb, rowSchedule(&rb.r))
	case dkScaleWeights:
		if o.V == nil || len(o.In) != 2 {
			o.Why = whyNoWeight
			return nil
		}
		rb := &rowBuild{r: compose.Row{Width: w(o.Out[0]), Rows: 1}}
		r := &rb.r
		r.ScaleWeights(rb.buf(compose.BufRow, gaIn0), rb.buf(compose.BufU32, gaIn1), rb.buf(compose.BufVec, gaV),
			rb.buf(compose.BufOut, gaOut0))
		return g.routeLaunch(o, rb, rowSchedule(r))
	}
	rt, ok := opRoute(o)
	if !ok {
		o.Why = whyRoute
		return nil
	}
	E, K := rt.Experts, rt.K
	switch o.Kind {
	case dkGroupMask:
		if !(rt.Groups > 1) || w(o.In[0]) != E || w(o.Out[0]) != E {
			o.Why = whyRoute
			return nil
		}
		rt.Bias = o.V != nil
		rb := &rowBuild{r: compose.Row{Width: E, Rows: 1}}
		l, bias := rb.buf(compose.BufRow, gaIn0), -1
		if rt.Bias {
			bias = rb.buf(compose.BufVec, gaV)
		}
		rb.r.GroupMask(rt, l, bias, rb.buf(compose.BufOut, gaOut0))
		return g.routeLaunch(o, rb, rowSchedule(&rb.r))
	case dkTopK:
		if w(o.In[0]) != E || len(o.Out) != 2 || w(o.Out[0]) != K || w(o.Out[1]) != K {
			o.Why = whyRoute
			return nil
		}
		rb := &rowBuild{r: compose.Row{Width: E, Rows: 1}}
		l := rb.buf(compose.BufRow, gaIn0)
		sel, top := rb.buf(compose.BufOutU32, gaOut0), rb.buf(compose.BufOut, gaOut1)
		plane, bias := -1, -1
		switch {
		case rt.Groups > 1:
			// The masked plane, the group mask's: it carries the bias.
			if len(o.In) < 2 {
				o.Why = whyRoute
				return nil
			}
			plane = rb.buf(compose.BufRow, gaIn1)
		case len(o.In) > 1:
			// A bias per row, an input of its own.
			bias = rb.buf(compose.BufRow, gaIn0+genArg(len(o.In)-1))
		case o.V != nil:
			bias = rb.buf(compose.BufVec, gaV)
		}
		rt.Bias = bias >= 0
		rb.r.TopK(rt, l, sel, top, plane, bias)
		return g.routeLaunch(o, rb, rowSchedule(&rb.r))
	case dkRouteWeights:
		if w(o.In[0]) != K || w(o.Out[0]) != K {
			o.Why = whyRoute
			return nil
		}
		rb := &rowBuild{r: compose.Row{Width: K, Rows: 1}}
		top, out, l := rb.buf(compose.BufRow, gaIn0), rb.buf(compose.BufOut, gaOut0), -1
		if rt.Score == compose.ScoreLogit && (!rt.Norm || rt.Jitter != 0) {
			// A softmax over every logit, or sparsemixer's bands over them.
			if len(o.In) < 2 || w(o.In[1]) != E {
				o.Why = whyRoute
				return nil
			}
			l = rb.buf(compose.BufRow, gaIn1)
		}
		rb.r.RouteWeights(rt, top, l, out)
		return g.routeLaunch(o, rb, rowSchedule(&rb.r))
	}
	return nil
}

// routeLaunch is row group rb's one launch under schedule s: a group a
// row. Callers hold g.mu.
func (g *devTier) routeLaunch(o *devOp, rb *rowBuild, s compose.Schedule) []genLaunch {
	k, _ := g.genKern(append([]uint32{encRow}, rb.r.Encode(s)...), func() (*ir.Kernel, error) { return rb.r.Emit(s) })
	if k == nil {
		o.Why = whyNoKernel
		return nil
	}
	l := genLaunch{k: k, groups: rb.r.Rows, width: s.Group, n: uint8(len(rb.args))}
	copy(l.args[:], rb.args)
	return []genLaunch{l}
}
