package tier

import (
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit_old/gpu/compose"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

// routeOf is the op's router as the composer ranks it, false for a router
// it has no form of.
func routeOf(o *devOp, prog *devProgram, rows int) (compose.Route, bool) {
	a := o.A
	r := compose.Route{NExpert: a.Experts, K: a.TopK, Norm: a.RouteNorm, NGroup: a.Groups, NGroupUsed: a.GroupsUsed, Rows: rows}
	if a.Scale != 0 && a.Scale != 1 {
		r.Scale = float32(a.Scale)
	}
	switch a.Router {
	case graph.RouterSoftmax:
	case graph.RouterSigmoid:
		r.Sigmoid = true
	case graph.RouterSqrtSoftplus:
		r.SqrtSoftplus = true
	default:
		return r, false
	}
	if o.Kind == dkTopK {
		r.RowBias = len(o.In) > 1+b2i(r.NGroup > 1)
	}
	r.Bias = o.Role == jlm.RoleExpProbsB || r.RowBias
	return r, r.NExpert > 0 && r.K > 0
}

// fused reports whether a route's rank and weights run as one launch: the
// plain softmax route of one row.
func fused(r compose.Route) bool {
	return !r.Sigmoid && !r.SqrtSoftplus && !r.Bias && r.NGroup <= 1 && max(r.Rows, 1) == 1 && r.NExpert <= 1024
}

// lanesRoute reports whether a fused route runs on one subgroup.
func (g *devTier) lanesRoute(r compose.Route) bool {
	L := g.composeTarget().Lanes
	return r.Norm && L == ir.SubgroupLanes && (r.NExpert+L-1)/L <= 32 && r.K <= 64
}

// genRoute is a mixture's routing ops at one row: the router's logits (a
// workgroup an expert, the products combined by a tree), their bias, the
// rank and the weights -- one launch for a plain softmax route (on a
// subgroup, or a workgroup an expert a thread), the rank and then the
// weights otherwise -- and the grouping, which a row does not need.
func (g *devTier) genRoute(o *devOp, prog *devProgram, bs *blockScratch, rows int) []genLaunch {
	o.Why = whyNoForm
	l := g.curLayer
	if l == nil || rows != 1 && (o.Kind == dkRouterBias || o.Kind == dkGroupByExpert) {
		return nil
	}
	launch := func(rb *rowBuild, f compose.Facts) []genLaunch {
		k := g.rowKernWith(&rb.r, f)
		if k.k == nil {
			return nil
		}
		gl := genLaunch{k: k.k, groups: k.groups, width: k.group, n: uint8(len(rb.args))}
		copy(gl.args[:], rb.args)
		o.Why = whyNone
		return []genLaunch{gl}
	}
	switch o.Kind {
	case dkGroupByExpert:
		o.Why = whyNothing
		return nil
	case dkRouterLogits:
		if l.router == nil || len(o.In) != 1 {
			return nil
		}
		k, ex := prog.vals[o.In[0]].Width, prog.vals[o.Out[0]].Width
		rb := &rowBuild{r: compose.Row{Width: k, Rows: ex * rows}}
		wk := compose.BufRow
		if l.routerF16 {
			wk = compose.BufRowF16
		}
		w, x, out := rb.buf(wk, gaRouter), rb.buf(compose.BufRow, gaIn0), rb.buf(compose.BufOut, gaOut0)
		rb.r.RouterLogits(w, x, out, ex, k)
		return launch(rb, compose.Facts{RowGroup: 128, Tree: true})
	case dkGateLogit:
		if l.shRouter == nil || len(o.In) != 1 {
			return nil
		}
		k := prog.vals[o.In[0]].Width
		rb := &rowBuild{r: compose.Row{Width: k, Rows: rows}}
		w, x, out := rb.buf(compose.BufRow, gaShRouter), rb.buf(compose.BufRow, gaIn0), rb.buf(compose.BufOut, gaOut0)
		rb.r.RouterLogits(w, x, out, 1, k)
		return launch(rb, compose.Facts{RowGroup: 128, Tree: true})
	case dkRouterBias:
		if l.routerB == nil {
			return nil
		}
		w := prog.vals[o.Out[0]].Width
		rb := &rowBuild{r: compose.Row{Width: w, Rows: 1}}
		rb.store(rb.bin(compose.OpAdd, rb.load(compose.BufRow, gaIn0), rb.load(compose.BufVec, gaRouterB)), gaOut0)
		return launch(rb, compose.Facts{})
	}
	r, ok := routeOf(o, prog, rows)
	if !ok {
		return nil
	}
	if o.Kind == dkGroupMask {
		if r.NGroup < 2 || r.NExpert%r.NGroup != 0 || r.Bias && l.expSelB == nil || len(o.In) != 1 {
			return nil
		}
		if o.Own[0] = g.ownBuf(bs, o, 0, rows*r.NGroup, nil); o.Own[0] == nil {
			return nil
		}
		gs := &rowBuild{r: compose.Row{Width: r.NGroup, Rows: rows}}
		lg, sc := gs.buf(compose.BufRow, gaIn0), gs.buf(compose.BufOut, gaOwn0)
		bias := -1
		if r.Bias {
			bias = gs.buf(compose.BufRow, gaSelB)
		}
		gs.r.RouteGroupScore(r, lg, sc, bias)
		gm := &rowBuild{r: compose.Row{Width: r.NExpert, Rows: rows}}
		lg, in, bm := gm.buf(compose.BufRow, gaIn0), gm.buf(compose.BufRow, gaOwn0), gm.buf(compose.BufOut, gaOut0)
		bias = -1
		if r.Bias {
			bias = gm.buf(compose.BufRow, gaSelB)
		}
		gm.r.RouteGroupMask(r, lg, in, bm, bias)
		ls := launch(gs, compose.Facts{})
		if ls == nil {
			return nil
		}
		lm := launch(gm, compose.Facts{})
		if lm == nil {
			return nil
		}
		return append(ls, lm...)
	}
	if o.Kind == dkRouteWeights {
		if fused(r) {
			o.Why = whyNothing
			return nil
		}
		rb := &rowBuild{r: compose.Row{Width: 1, Rows: rows}}
		top, w := rb.buf(compose.BufRow, gaIn0), rb.buf(compose.BufOut, gaOut0)
		lg := -1
		if !r.Norm && !r.Sigmoid && !r.SqrtSoftplus {
			lg = rb.buf(compose.BufRow, gaIn1)
		}
		rb.r.RouteWeights(r, top, w, lg)
		return launch(rb, compose.Facts{})
	}
	if fused(r) {
		width := r.NExpert
		if g.lanesRoute(r) {
			width = ir.SubgroupLanes
		}
		rb := &rowBuild{r: compose.Row{Width: width, Rows: 1}}
		lg, sel, top, w := rb.buf(compose.BufRow, gaIn0), rb.buf(compose.BufOutU32, gaOut0), rb.buf(compose.BufOut, gaOut1), rb.buf(compose.BufOut, gaRouteW)
		if g.lanesRoute(r) {
			rb.r.RouteLanes(r, lg, sel, top, w)
		} else {
			rb.r.RouteShared(r, lg, sel, top, w)
		}
		return launch(rb, compose.Facts{})
	}
	rb := &rowBuild{r: compose.Row{Width: r.NExpert, Rows: rows}}
	lg, sel, top := rb.buf(compose.BufRow, gaIn0), rb.buf(compose.BufOutU32, gaOut0), rb.buf(compose.BufOut, gaOut1)
	bm, bias := -1, -1
	switch {
	case r.NGroup > 1:
		bm = rb.buf(compose.BufRow, gaIn1)
	case r.RowBias:
		bias = rb.buf(compose.BufRow, gaIn0+genArg(len(o.In)-1))
	case r.Bias:
		if l.expSelB == nil {
			return nil
		}
		bias = rb.buf(compose.BufRow, gaSelB)
	}
	rb.r.RouteRank(r, lg, sel, top, bm, bias)
	return launch(rb, compose.Facts{})
}

// genCombine is a mixture's slots' outputs summed by their weights.
func (g *devTier) genCombine(o *devOp, prog *devProgram, bs *blockScratch) []genLaunch {
	if len(o.In) != 2 || len(o.Out) != 1 {
		return nil
	}
	n, slots := prog.vals[o.Out[0]].Width, prog.vals[o.In[1]].Width
	if n < 1 || slots < 1 || prog.vals[o.In[0]].Width != n {
		return nil
	}
	rb := &rowBuild{r: compose.Row{Width: n, Rows: 1}}
	e, w := rb.buf(compose.BufRow, gaIn0), rb.buf(compose.BufRow, gaIn1)
	rb.store(rb.r.Combine(e, w, slots), gaOut0)
	k := g.rowKern(&rb.r)
	if k.k == nil {
		return nil
	}
	gl := genLaunch{k: k.k, groups: k.groups, width: k.group, n: uint8(len(rb.args))}
	copy(gl.args[:], rb.args)
	return []genLaunch{gl}
}
