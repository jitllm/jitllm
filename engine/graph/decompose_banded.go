package graph

// bandedRoute writes route op o under RouterSparseMixer as primitives: the
// router's top two logits m1 and m2 (the plain selection, ties to the lower
// index), each weighted by 1/SUM exp(s - m) over the experts within the band
// of it, not renormalised. An expert s is kept against m unless
// (m - s)/max(|s|, m) exceeds 2*Margin (a 0/0 gap is kept); the second sum
// leaves out the first pick. Every sum runs from zero in expert order, and the
// pick's own term, exp(0), is the sum's 1 once divided out. A route that states
// more than that (a renormalisation, a scale, groups, a selection bias, a pick
// count other than two, no stated margin) stays whole.
func (d *Graph) bandedRoute(o Op, router, bias, selBias *Binding) (ValueID, bool) {
	a := o.Attr
	E := a.Experts
	if router == nil || selBias != nil || E < 2 || a.TopK != 2 || a.RouteNorm || a.Groups > 1 ||
		a.Scale != 0 && a.Scale != 1 || !(a.Margin > 0) {
		return 0, false
	}
	p := prims{d: d}
	logits := p.out("route.logits", E, Op{Kind: OpMatVec, In: []ValueID{o.In[0]}, Weight: []Binding{*router}})
	if bias != nil {
		logits = p.mapw("route.logits_b", E, MapAdd, *bias, logits)
	}
	sel := o.Out[0]
	d.Values[sel].Width, d.Values[sel].Elem = 2, ElemU32
	top := d.value("route.top", 2)
	d.op(Op{Kind: OpTopK, In: []ValueID{logits}, Out: []ValueID{sel, top}, Attr: Attrs{TopK: 2}})

	// place is every logit's rank, so 0 is the first pick (equal logits to the
	// lower index, as the selection ranks them); the second pick's logit is the
	// largest of the rest.
	place := p.outElem("route.place", E, ElemU32, Op{Kind: OpRank, In: []ValueID{logits}})
	maxOf := func(name string, x ValueID) ValueID {
		return p.out(name, 1, Op{Kind: OpReduce, In: []ValueID{x}, Attr: Attrs{Fn: MapMax, Order: OrderSeq}})
	}
	m1 := maxOf("route.m1", logits)
	rest := p.out("route.rest", E, Op{Kind: OpMap, In: []ValueID{place, logits}, Attr: Attrs{Fn: MapWhere, Const: MaskedOut}})
	m2 := maxOf("route.m2", rest)
	zero := p.out("route.zero", 1, Op{Kind: OpMap, In: []ValueID{m1, m1}, Attr: Attrs{Fn: MapSub}})
	one := p.out("route.one", 1, Op{Kind: OpMap, In: []ValueID{zero}, Attr: Attrs{Fn: MapAdd, Const: 1}})

	// |s|, shared by both picks: max(s, -s).
	neg := p.out("route.neg", E, Op{Kind: OpMap, In: []ValueID{logits}, Attr: Attrs{Fn: MapMul, Const: -1}})
	abs := p.out("route.abs", E, Op{Kind: OpMap, In: []ValueID{logits, neg}, Attr: Attrs{Fn: MapMax}})
	thr := float64(float32(2 * a.Margin))
	// weight is 1 over the sum of the terms exp(s - m) kept against m; keep,
	// where not -1, further masks out the experts it is 0 for (the first
	// pick, in the second sum).
	weight := func(name string, m ValueID, keep ValueID) ValueID {
		den := p.out(name+".denominator", E, Op{Kind: OpMap, In: []ValueID{abs, m}, Attr: Attrs{Fn: MapMax}})
		diff := p.out(name+".gap_num", E, Op{Kind: OpMap, In: []ValueID{m, logits}, Attr: Attrs{Fn: MapSub}})
		gap := p.out(name+".gap", E, Op{Kind: OpMap, In: []ValueID{diff, den}, Attr: Attrs{Fn: MapDiv}})
		// Out of the band where the gap exceeds the threshold, gap - thr > 0;
		// a NaN gap (0/0) is not, and is kept.
		over := p.out(name+".over", E, Op{Kind: OpMap, In: []ValueID{gap}, Attr: Attrs{Fn: MapSub, Const: thr}})
		c := p.out(name+".centred", E, Op{Kind: OpMap, In: []ValueID{logits, m}, Attr: Attrs{Fn: MapSub}})
		e := p.out(name+".exp", E, Op{Kind: OpMap, In: []ValueID{c}, Attr: Attrs{Fn: MapExp}})
		// A masked term is selected away, not multiplied by zero: the first
		// pick's exp(s - m2) can overflow, and 0*Inf is NaN.
		t := p.out(name+".terms", E, Op{Kind: OpMap, In: []ValueID{over, zero, e}, Attr: Attrs{Fn: MapWhere}})
		if keep >= 0 {
			t = p.out(name+".kept", E, Op{Kind: OpMap, In: []ValueID{keep, t}, Attr: Attrs{Fn: MapWhere, Const: 0}})
		}
		z := p.out(name+".sum", 1, Op{Kind: OpReduce, In: []ValueID{t}, Attr: Attrs{Fn: MapAdd, Order: OrderSeq}})
		return p.out(name, 1, Op{Kind: OpMap, In: []ValueID{one, z}, Attr: Attrs{Fn: MapDiv}})
	}
	w1 := weight("route.w1", m1, -1)
	w2 := weight("route.w2", m2, place)
	// The pair, in selection order: the second pick's weight where the pick's
	// place among the two is past the first.
	pick := p.outElem("route.pick", 2, ElemU32, Op{Kind: OpRank, In: []ValueID{top}})
	w := p.out("route.weights_b", 2, Op{Kind: OpMap, In: []ValueID{pick, w2, w1}, Attr: Attrs{Fn: MapWhere}})
	return w, true
}

// gateScalar writes OpGateScalar o as primitives into d: the gate's dot with
// the row (a one-row matvec, so the dot's order is the matvec's, as the host's
// shared expert takes it), its sigmoid, and the broadcast product with the
// scaled row. It reports false for a gate that is not a vector over the row.
func (d *Graph) gateScalar(o Op) bool {
	if len(o.In) != 2 || len(o.Out) != 1 || len(o.Weight) != 1 {
		return false
	}
	w := o.Weight[0]
	if w.Rows > 1 || w.Sheets != 0 || w.K != d.Values[o.In[0]].Width || d.Values[o.In[1]].Width != d.Values[o.Out[0]].Width {
		return false
	}
	w.Rows = 1
	p := prims{d: d}
	logit := p.out("gate.logit", 1, Op{Kind: OpMatVec, In: []ValueID{o.In[0]}, Weight: []Binding{w}})
	s := p.out("gate.score", 1, Op{Kind: OpMap, In: []ValueID{logit}, Attr: Attrs{Fn: MapSigmoid}})
	d.op(Op{Kind: OpMap, In: []ValueID{o.In[1], s}, Out: o.Out, Attr: Attrs{Fn: MapMul}})
	return true
}
