package graph

import "github.com/jitllm/jitllm/format/jlm"

// altStates is the stream count and row width of an AltUp op o whose first
// input is the streams (stream-major, N rows of d), or false where o does not
// state a shape the decomposition below expresses.
func altStates(g *Graph, o Op) (n, d int, ok bool) {
	n = o.Attr.Streams
	if n < 2 || len(o.In) == 0 || len(o.Out) == 0 {
		return 0, 0, false
	}
	w := g.Values[o.In[0]].Width
	if w%n != 0 || w == 0 {
		return 0, 0, false
	}
	return n, w / n, true
}

// altWeights reports whether o binds exactly the roles want, in order.
func altWeights(o Op, want ...jlm.Role) bool {
	if len(o.Weight) != len(want) {
		return false
	}
	for i, r := range want {
		if o.Weight[i].Role != r {
			return false
		}
	}
	return true
}

// altUp writes AltUp's prologue (OpAltPredict) or epilogue (OpAltCorrect) o
// as primitives into d, or returns false, leaving d as it was, where o is not
// a shape they express. Both read a router over the active stream's norm:
// the norm's row divided by its width, projected to one number a stream, and
// squashed by tanh; the stored coefficient table, one row a stream, weighed
// by those numbers in stream order, each row fused into the sum.
//
// The prologue predicts stream k as stream k plus every stream j weighed by
// c[k*N+j], the sum taken in j order, each term fused; the active prediction
// is stream 0's. The epilogue adds to each prediction the active stream's
// change (the block's output less the active prediction) weighed by the
// coefficient plus one, gates the per-layer input by the corrected active
// stream scaled per element, projects it back, norms it, and adds it to every
// stream but the active one.
func (d *Graph) altUp(o Op) bool {
	n, w, ok := altStates(d, o)
	if !ok {
		return false
	}
	eps := o.Attr.Eps
	switch o.Kind {
	case OpAltPredict:
		if !altWeights(o, jlm.RoleAltUpRouterNorm, jlm.RoleAltUpRouter, jlm.RoleAltUpPredCoef) || len(o.In) != 1 ||
			len(o.Out) != 2 || d.Values[o.Out[0]].Width != n*w || d.Values[o.Out[1]].Width != w ||
			o.Weight[0].K != w || o.Weight[1].K != w || o.Weight[1].Rows != n || o.Weight[2].K*max(o.Weight[2].Rows, 1) != n*n*n {
			return false
		}
	case OpAltCorrect:
		if !altWeights(o, jlm.RoleAltUpRouterNorm, jlm.RoleAltUpRouter, jlm.RoleAltUpCorrCoef, jlm.RoleAltUpCorrScale,
			jlm.RolePLEGate, jlm.RolePLEProj, jlm.RolePLEPostNorm) || len(o.In) != 3 || len(o.Out) != 1 ||
			d.Values[o.In[1]].Width != w || d.Values[o.Out[0]].Width != n*w ||
			o.Weight[0].K != w || o.Weight[1].K != w || o.Weight[1].Rows != n || o.Weight[2].K*max(o.Weight[2].Rows, 1) != n*n ||
			o.Weight[3].K != w || o.Weight[4].K != w || o.Weight[4].Rows != d.Values[o.In[2]].Width ||
			o.Weight[5].K != o.Weight[4].Rows || o.Weight[5].Rows != w || o.Weight[6].K != w {
			return false
		}
	default:
		return false
	}
	p := prims{d: d}
	slice := func(name string, x ValueID, off, width int) ValueID {
		return p.out(name, width, Op{Kind: OpSlice, In: []ValueID{x}, Attr: Attrs{Offset: off, Width: width}})
	}
	// tile is x repeated n times end to end, the wide operand an OpMap
	// weighs by a per-stream coefficient.
	tile := func(name string, x ValueID, times int) ValueID {
		in := make([]ValueID, times)
		for i := range in {
			in[i] = x
		}
		return p.out(name, times*d.Values[x].Width, Op{Kind: OpConcat, In: in})
	}
	// router is the modalities of the row x: tanh(R * norm(x) / width).
	router := func(x ValueID) ValueID {
		h := p.out("alt.normed", w, Op{Kind: OpNorm, In: []ValueID{x}, Weight: []Binding{o.Weight[0]}, Attr: Attrs{Eps: eps}})
		h = p.out("alt.scaled", w, Op{Kind: OpMap, In: []ValueID{h}, Attr: Attrs{Fn: MapMul, Const: float64(float32(1 / float64(w)))}})
		r := p.out("alt.route", n, Op{Kind: OpMatVec, In: []ValueID{h}, Weight: []Binding{o.Weight[1]}})
		return p.tanh("alt.modalities", r, n)
	}
	// coefs is the table (binding t, rows of width) weighed by the
	// modalities m: row 0 times m0 from zero, each later row fused onto the
	// sum.
	coefs := func(m ValueID, t Binding, width int) ValueID {
		var c ValueID
		for i := 0; i < n; i++ {
			mi := slice("alt.modality", m, i, 1)
			row := p.out("alt.coef_row", width, Op{Kind: OpSlice, Weight: []Binding{t}, Attr: Attrs{Offset: i * width, Width: width}})
			if i == 0 {
				c = p.out("alt.coefs", width, Op{Kind: OpMap, In: []ValueID{mi, row}, Attr: Attrs{Fn: MapFma, Const: 0}})
				continue
			}
			c = p.out("alt.coefs", width, Op{Kind: OpMap, In: []ValueID{mi, row, c}, Attr: Attrs{Fn: MapFma}})
		}
		return c
	}

	switch o.Kind {
	case OpAltPredict:
		x := o.In[0]
		m := router(slice("alt.active", x, 0, w))
		c := coefs(m, o.Weight[2], n*n)
		pred := x
		for j := 0; j < n; j++ {
			// Stream k's weight on stream j, a column of the coefficients,
			// against stream j tiled across the streams.
			cj := p.out("alt.coef_col", n, Op{Kind: OpSlice, In: []ValueID{c}, Attr: Attrs{HeadDim: n, Offset: j, Width: 1}})
			xj := tile("alt.stream_tiled", slice("alt.stream", x, j*w, w), n)
			fma := Op{Kind: OpMap, In: []ValueID{cj, xj, pred}, Attr: Attrs{Fn: MapFma}}
			if j == n-1 {
				fma.Out = []ValueID{o.Out[0]}
				d.op(fma)
				pred = o.Out[0]
				continue
			}
			pred = p.out("alt.pred", n*w, fma)
		}
		d.op(Op{Kind: OpSlice, In: []ValueID{pred}, Out: []ValueID{o.Out[1]}, Attr: Attrs{Offset: 0, Width: w}})
	case OpAltCorrect:
		pred, out, ple := o.In[0], o.In[1], o.In[2]
		m := router(out)
		cc := coefs(m, o.Weight[2], n)
		cc = p.out("alt.coefs_one", n, Op{Kind: OpMap, In: []ValueID{cc}, Attr: Attrs{Fn: MapAdd, Const: 1}})
		change := p.out("alt.change", w, Op{Kind: OpMap, In: []ValueID{out, slice("alt.pred_active", pred, 0, w)}, Attr: Attrs{Fn: MapSub}})
		x := p.out("alt.corrected", n*w, Op{Kind: OpMap, In: []ValueID{cc, tile("alt.change_tiled", change, n), pred}, Attr: Attrs{Fn: MapFma}})
		x0 := slice("alt.corrected_active", x, 0, w)
		// The per-layer input, gated by the corrected active stream times
		// its scale, projected back, normed, and added to every other stream.
		h := p.mapw("alt.gate_in", w, MapMul, o.Weight[3], x0)
		g := p.out("alt.ple_gate", o.Weight[4].Rows, Op{Kind: OpMatVec, In: []ValueID{h}, Weight: []Binding{o.Weight[4]}})
		a := p.out("alt.ple_act", o.Weight[4].Rows, Op{Kind: OpActMul, In: []ValueID{g, ple}, Attr: Attrs{Act: o.Attr.Act}})
		po := p.out("alt.ple_out", w, Op{Kind: OpMatVec, In: []ValueID{a}, Weight: []Binding{o.Weight[5]}})
		pn := p.out("alt.ple_normed", w, Op{Kind: OpNorm, In: []ValueID{po}, Weight: []Binding{o.Weight[6]}, Attr: Attrs{Eps: eps}})
		rest := p.out("alt.others", (n-1)*w, Op{Kind: OpMap, In: []ValueID{slice("alt.others_in", x, w, (n-1)*w), tile("alt.ple_tiled", pn, n-1)},
			Attr: Attrs{Fn: MapAdd}})
		d.op(Op{Kind: OpConcat, In: []ValueID{x0, rest}, Out: []ValueID{o.Out[0]}})
	}
	return true
}

// tanh is x's hyperbolic tangent over its row of w, as the engine's softcap
// computes it from the one exp: 1 - 2/(exp(2x) + 1), so the saturations are
// exp's. The ones row is min(exp(2x)+1, 1), which is one wherever the sum is
// at least one, and which stands in for the constant numerators an OpMap has
// no operand for.
func (p prims) tanh(name string, x ValueID, w int) ValueID {
	t := p.out(name+".twice", w, Op{Kind: OpMap, In: []ValueID{x}, Attr: Attrs{Fn: MapMul, Const: 2}})
	e := p.out(name+".exp", w, Op{Kind: OpMap, In: []ValueID{t}, Attr: Attrs{Fn: MapExp}})
	z := p.out(name+".denominator", w, Op{Kind: OpMap, In: []ValueID{e}, Attr: Attrs{Fn: MapAdd, Const: 1}})
	one := p.out(name+".one", w, Op{Kind: OpMap, In: []ValueID{z}, Attr: Attrs{Fn: MapMin, Const: 1}})
	two := p.out(name+".two", w, Op{Kind: OpMap, In: []ValueID{one}, Attr: Attrs{Fn: MapAdd, Const: 1}})
	q := p.out(name+".quotient", w, Op{Kind: OpMap, In: []ValueID{two, z}, Attr: Attrs{Fn: MapDiv}})
	return p.out(name, w, Op{Kind: OpMap, In: []ValueID{one, q}, Attr: Attrs{Fn: MapSub}})
}
