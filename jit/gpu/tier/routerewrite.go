package tier

import "github.com/jitllm/jitllm/engine/graph"

// rwRoute: a route stated as primitives -- the k largest of the router's
// logits, their logits taken and softmaxed (the selection's renormalised
// weights), optionally scaled -- as the route's rank and weights.
func rwRoute(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	if nseq > 0 {
		return false // a ragged step's route has no composed form yet
	}
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
		lg := o.In[0]
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
		p.ops[i] = devOp{Kind: dkTopK, Src: o.Src, In: []devVal{lg}, Out: []devVal{sel, top}, A: a}
		p.ops[end] = devOp{Kind: dkRouteWeights, Src: p.ops[end].Src, In: []devVal{top, lg}, Out: []devVal{w}, A: a}
		drop := map[int]bool{t: true, rmax: true, csub: true, cexp: true, rsum: true}
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
