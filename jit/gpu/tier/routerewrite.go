package tier

import (
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// rwRoute: a route stated as primitives -- the k largest of the router's
// logits, their logits taken and softmaxed (the selection's renormalised
// weights), optionally scaled -- as the route's rank and weights.
func rwRoute(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	writer := func(v devVal) int {
		for j, o := range p.ops {
			for _, x := range o.Out {
				if x == v {
					return j
				}
			}
		}
		return -1
	}
	// only is the op reading v, where exactly one does.
	only := func(v devVal) int {
		if p.uses(v) != 1 {
			return -1
		}
		for j, o := range p.ops {
			for _, x := range o.In {
				if x == v {
					return j
				}
			}
		}
		return -1
	}
	is := func(j int, k devKind, fn graph.MapFn, in ...devVal) bool {
		if j < 0 || p.ops[j].Kind != k || p.ops[j].A.Fn != fn || len(p.ops[j].In) != len(in) || len(p.ops[j].Out) != 1 {
			return false
		}
		for i, v := range in {
			if p.ops[j].In[i] != v {
				return false
			}
		}
		return true
	}
	for i, o := range p.ops {
		if o.Kind != dkSelectK || len(o.In) != 1 || len(o.Out) != 2 || o.A.Span != 0 {
			continue
		}
		if gatedRoute(p, i, writer, only, is) {
			return true
		}
		lg, role := o.In[0], jlm.Role(0)
		var keyOps []int
		if b := writer(lg); b >= 0 && p.ops[b].Kind == dkMap && p.ops[b].A.Fn == graph.MapAdd && p.ops[b].Role == jlm.RoleExpProbsB && len(p.ops[b].In) == 1 {
			// The key the softmax's probability with the selection bias
			// added: the rank computes both from the logits.
			x, ops, ok := softmaxOf(p, p.ops[b].In[0], writer, only, is)
			if !ok || p.uses(lg) != 1 {
				continue
			}
			lg, role, keyOps = x, jlm.RoleExpProbsB, append(ops, b)
		}
		if w := writer(lg); w < 0 || p.ops[w].Kind != dkRouterLogits {
			continue
		}
		sel, top := o.Out[0], o.Out[1]
		// take(logits, sel), then its softmax.
		t := -1
		for j, q := range p.ops {
			if q.Kind == dkTake && len(q.In) == 2 && q.In[0] == lg && q.In[1] == sel && len(q.Out) == 1 {
				t = j
			}
		}
		if t < 0 {
			if probsRoute(p, i, only, is) {
				return true
			}
			continue
		}
		tl := p.ops[t].Out[0]
		var rmax, csub, cexp, rsum, wdiv int
		chain := func() bool {
			// tl feeds the maximum and the centring: two readers.
			if p.uses(tl) != 2 {
				return false
			}
			rmax, csub = -1, -1
			for j, q := range p.ops {
				if len(q.In) > 0 && q.In[0] == tl {
					if q.Kind == dkReduce {
						rmax = j
					} else {
						csub = j
					}
				}
			}
			if !is(rmax, dkReduce, graph.MapMax, tl) || !is(csub, dkMap, graph.MapSub, tl, p.ops[rmax].Out[0]) {
				return false
			}
			cexp = only(p.ops[csub].Out[0])
			if !is(cexp, dkMap, graph.MapExp, p.ops[csub].Out[0]) || p.uses(p.ops[cexp].Out[0]) != 2 {
				return false
			}
			e := p.ops[cexp].Out[0]
			rsum, wdiv = -1, -1
			for j, q := range p.ops {
				if len(q.In) > 0 && q.In[0] == e {
					if q.Kind == dkReduce {
						rsum = j
					} else {
						wdiv = j
					}
				}
			}
			return is(rsum, dkReduce, graph.MapAdd, e) && is(wdiv, dkMap, graph.MapDiv, e, p.ops[rsum].Out[0])
		}
		if !chain() {
			continue
		}
		w := p.ops[wdiv].Out[0]
		end := wdiv
		scale := 0.0
		if s := only(w); s >= 0 && p.ops[s].Kind == dkMap && p.ops[s].A.Fn == graph.MapMul && len(p.ops[s].In) == 1 {
			scale, end, w = p.ops[s].A.Const, s, p.ops[s].Out[0]
		}
		a := graph.Attrs{Experts: p.vals[lg].Width, TopK: o.A.TopK, Router: graph.RouterSoftmax, RouteNorm: true, Scale: scale}
		p.ops[i] = devOp{Kind: dkTopK, Src: o.Src, In: []devVal{lg}, Out: []devVal{sel, top}, Role: role, A: a}
		p.ops[end] = devOp{Kind: dkRouteWeights, Src: p.ops[end].Src, In: []devVal{top, lg}, Out: []devVal{w}, Role: role, A: a}
		drop := map[int]bool{t: true, rmax: true, csub: true, cexp: true, rsum: true}
		for _, j := range keyOps {
			drop[j] = true
		}
		if end != wdiv {
			drop[wdiv] = true
		}
		ops := p.ops[:0]
		for j, q := range p.ops {
			if !drop[j] {
				ops = append(ops, q)
			}
		}
		p.ops = ops
		return true
	}
	return false
}

// gatedRoute rewrites select_k op i over a gated router's scores -- the
// sigmoid (or sqrt of the softplus) of the router's logits, a selection
// bias added -- with the selected unbiased scores taken, renormalised by
// their sum plus the floor and scaled, as the route's rank and weights.
func gatedRoute(p *devProgram, i int, writer, only func(devVal) int, is func(int, devKind, graph.MapFn, ...devVal) bool) bool {
	o := p.ops[i]
	sel, top := o.Out[0], o.Out[1]
	if p.uses(top) != 0 {
		return false
	}
	chain := []int{}
	key, role := o.In[0], jlm.Role(0)
	if b := writer(key); b >= 0 && p.ops[b].Kind == dkMap && p.ops[b].A.Fn == graph.MapAdd && p.ops[b].Role == jlm.RoleExpProbsB && len(p.ops[b].In) == 1 {
		if p.uses(key) != 1 {
			return false
		}
		chain, key, role = append(chain, b), p.ops[b].In[0], jlm.RoleExpProbsB
	}
	sc := writer(key)
	if sc < 0 || p.ops[sc].Kind != dkMap || len(p.ops[sc].In) != 1 || p.uses(key) != 2 {
		return false
	}
	router, lg := graph.RouterSigmoid, p.ops[sc].In[0]
	switch p.ops[sc].A.Fn {
	case graph.MapSigmoid:
	case graph.MapSqrt:
		sp := writer(lg)
		if sp < 0 || !is(sp, dkMap, graph.MapSoftplus, p.ops[sp].In...) || len(p.ops[sp].In) != 1 || p.uses(lg) != 1 {
			return false
		}
		chain, lg, router = append(chain, sp), p.ops[sp].In[0], graph.RouterSqrtSoftplus
	default:
		return false
	}
	if w := writer(lg); w < 0 || p.ops[w].Kind != dkRouterLogits {
		return false
	}
	t := -1
	for j, q := range p.ops {
		if q.Kind == dkTake && len(q.In) == 2 && q.In[0] == key && q.In[1] == sel && len(q.Out) == 1 {
			t = j
		}
	}
	if t < 0 {
		return false
	}
	chain = append(chain, sc, t)
	w, norm := p.ops[t].Out[0], false
	if r := only(w); r < 0 && p.uses(w) == 2 {
		var rs, dv int
		rs, dv = -1, -1
		for j, q := range p.ops {
			if len(q.In) > 0 && q.In[0] == w {
				if q.Kind == dkReduce {
					rs = j
				} else {
					dv = j
				}
			}
		}
		if !is(rs, dkReduce, graph.MapAdd, w) {
			return false
		}
		f := only(p.ops[rs].Out[0])
		if !is(f, dkMap, graph.MapAdd, p.ops[rs].Out[0]) || float32(p.ops[f].A.Const) != float32(graph.RouteSumFloor) ||
			!is(dv, dkMap, graph.MapDiv, w, p.ops[f].Out[0]) {
			return false
		}
		chain = append(chain, rs, f, dv)
		w, norm = p.ops[dv].Out[0], true
	}
	scale := 0.0
	if s := only(w); s >= 0 && p.ops[s].Kind == dkMap && p.ops[s].A.Fn == graph.MapMul && len(p.ops[s].In) == 1 {
		chain = append(chain, s)
		scale, w = p.ops[s].A.Const, p.ops[s].Out[0]
	}
	end := chain[len(chain)-1]
	a := graph.Attrs{Experts: p.vals[lg].Width, TopK: o.A.TopK, Router: router, RouteNorm: norm, Scale: scale}
	p.ops[i] = devOp{Kind: dkTopK, Src: o.Src, In: []devVal{lg}, Out: []devVal{sel, top}, Role: role, A: a}
	p.ops[end] = devOp{Kind: dkRouteWeights, Src: p.ops[end].Src, In: []devVal{top, lg}, Out: []devVal{w}, Role: role, A: a}
	drop := map[int]bool{}
	for _, j := range chain[:len(chain)-1] {
		drop[j] = true
	}
	ops := p.ops[:0]
	for j, q := range p.ops {
		if !drop[j] {
			ops = append(ops, q)
		}
	}
	p.ops = ops
	return true
}

// probsRoute rewrites select_k op i over the router's logits whose weights
// are the full softmax's probabilities of the selected experts, optionally
// scaled, as the route's rank and weights.
func probsRoute(p *devProgram, i int, only func(devVal) int, is func(int, devKind, graph.MapFn, ...devVal) bool) bool {
	o := p.ops[i]
	lg, sel, top := o.In[0], o.Out[0], o.Out[1]
	if p.uses(top) != 0 || p.uses(lg) != 3 {
		return false
	}
	mx, cs := -1, -1
	for j, q := range p.ops {
		if j != i && len(q.In) > 0 && q.In[0] == lg {
			if q.Kind == dkReduce {
				mx = j
			} else {
				cs = j
			}
		}
	}
	if !is(mx, dkReduce, graph.MapMax, lg) || !is(cs, dkMap, graph.MapSub, lg, p.ops[mx].Out[0]) {
		return false
	}
	ex := only(p.ops[cs].Out[0])
	if !is(ex, dkMap, graph.MapExp, p.ops[cs].Out[0]) || p.uses(p.ops[ex].Out[0]) != 2 {
		return false
	}
	e := p.ops[ex].Out[0]
	rs, dv := -1, -1
	for j, q := range p.ops {
		if len(q.In) > 0 && q.In[0] == e {
			if q.Kind == dkReduce {
				rs = j
			} else {
				dv = j
			}
		}
	}
	if !is(rs, dkReduce, graph.MapAdd, e) || !is(dv, dkMap, graph.MapDiv, e, p.ops[rs].Out[0]) {
		return false
	}
	t := only(p.ops[dv].Out[0])
	if t < 0 || p.ops[t].Kind != dkTake || len(p.ops[t].In) != 2 || p.ops[t].In[1] != sel || len(p.ops[t].Out) != 1 {
		return false
	}
	chain := []int{mx, cs, ex, rs, dv, t}
	w := p.ops[t].Out[0]
	scale := 0.0
	if s := only(w); s >= 0 && p.ops[s].Kind == dkMap && p.ops[s].A.Fn == graph.MapMul && len(p.ops[s].In) == 1 {
		chain = append(chain, s)
		scale, w = p.ops[s].A.Const, p.ops[s].Out[0]
	}
	end := chain[len(chain)-1]
	a := graph.Attrs{Experts: p.vals[lg].Width, TopK: o.A.TopK, Router: graph.RouterSoftmax, Scale: scale}
	p.ops[i] = devOp{Kind: dkTopK, Src: o.Src, In: []devVal{lg}, Out: []devVal{sel, top}, A: a}
	p.ops[end] = devOp{Kind: dkRouteWeights, Src: p.ops[end].Src, In: []devVal{top, lg}, Out: []devVal{w}, A: a}
	drop := map[int]bool{}
	for _, j := range chain[:len(chain)-1] {
		drop[j] = true
	}
	ops := p.ops[:0]
	for j, q := range p.ops {
		if !drop[j] {
			ops = append(ops, q)
		}
	}
	p.ops = ops
	return true
}

// softmaxOf reports whether v is the softmax of a value x over its row
// (its maximum subtracted, exponentiated, divided by the sum), with x and
// the ops that compute it.
func softmaxOf(p *devProgram, v devVal, writer, only func(devVal) int, is func(int, devKind, graph.MapFn, ...devVal) bool) (devVal, []int, bool) {
	dv := writer(v)
	if dv < 0 || p.ops[dv].Kind != dkMap || p.ops[dv].A.Fn != graph.MapDiv || len(p.ops[dv].In) != 2 || p.uses(v) != 1 {
		return 0, nil, false
	}
	e, sum := p.ops[dv].In[0], p.ops[dv].In[1]
	rs, ex := writer(sum), writer(e)
	if !is(rs, dkReduce, graph.MapAdd, e) || ex < 0 || !is(ex, dkMap, graph.MapExp, p.ops[ex].In...) || len(p.ops[ex].In) != 1 || p.uses(e) != 2 {
		return 0, nil, false
	}
	cs := writer(p.ops[ex].In[0])
	if cs < 0 || p.ops[cs].Kind != dkMap || p.ops[cs].A.Fn != graph.MapSub || len(p.ops[cs].In) != 2 || only(p.ops[cs].Out[0]) != ex {
		return 0, nil, false
	}
	x, m := p.ops[cs].In[0], p.ops[cs].In[1]
	mx := writer(m)
	if !is(mx, dkReduce, graph.MapMax, x) || only(m) != cs {
		return 0, nil, false
	}
	return x, []int{mx, cs, ex, rs, dv}, true
}
