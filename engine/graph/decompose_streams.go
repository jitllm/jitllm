package graph

// streamMix writes the low-rank gated read of a residual of several streams
// as primitives, or returns false for the forms it does not state (the
// Sinkhorn-normalised mix, and the softmax bank over scores, which binds one
// weight and writes no placement).
//
// The op reads In[0], Streams streams of d stream-major, and binds the
// streams' grouped norm (Weight[0], Streams*d), a down projection to the low
// rank (Weight[1]), an up projection back to Streams*d (Weight[2]) and, with
// a placement (Out[1], Streams wide), the placement's projection (Weight[3],
// Streams*d to Streams). In order:
//
//	xn   = headnorm(x, d) * norm            each stream normed, then scaled
//	lo   = silu(down xn / Streams)
//	gate = sigmoid(up lo) * xn
//	row  = (gate_0 + gate_1 + ... ) / Streams   in stream order
//	w    = 2 * sigmoid(inject xn / Streams)
//
// Out[0] is row, Out[1] w.
func (d *Graph) streamMix(o Op) bool {
	a := o.Attr
	h := a.Streams
	if !a.Sinkhorn && a.Groups != 0 {
		return d.bankMix(o)
	}
	if a.Sinkhorn || a.Groups != 0 || h < 1 || len(o.In) != 1 || len(o.Out) < 1 || len(o.Out) > 2 ||
		len(o.Weight) != 2+len(o.Out) {
		return false
	}
	hd, w := d.Values[o.In[0]].Width, d.Values[o.Out[0]].Width
	if d.Values[o.In[0]].Elem != ElemF32 || hd != h*w || w < 1 {
		return false
	}
	norm, down, up := o.Weight[0], o.Weight[1], o.Weight[2]
	rank := down.Rows
	if norm.K != hd || norm.Rows != 0 || down.K != hd || rank < 1 || up.K != rank || up.Rows != hd {
		return false
	}
	var inj *Binding
	if len(o.Out) == 2 {
		inj = &o.Weight[3]
		if inj.K != hd || inj.Rows != h || d.Values[o.Out[1]].Width != h {
			return false
		}
	}
	p := prims{d: d}
	inv := float64(1 / float32(h))
	// The streams normed, each against its own row of the scale.
	hn := p.out("mix.normed", hd, Op{Kind: OpHeadNorm, In: []ValueID{o.In[0]}, Attr: Attrs{Eps: a.Eps, HeadDim: w}})
	xn := p.mapw("mix.xn", hd, MapMul, norm, hn)
	// The gate: a low-rank pair, the rank's input over the stream count.
	lo := p.out("mix.lo", rank, Op{Kind: OpMatVec, In: []ValueID{xn}, Weight: []Binding{down}})
	lo = p.out("mix.lo_s", rank, Op{Kind: OpMap, In: []ValueID{lo}, Attr: Attrs{Fn: MapMul, Const: inv}})
	lo = p.out("mix.lo_act", rank, Op{Kind: OpAct, In: []ValueID{lo}, Attr: Attrs{Act: ActSiLU}})
	g := p.out("mix.up", hd, Op{Kind: OpMatVec, In: []ValueID{lo}, Weight: []Binding{up}})
	g = p.out("mix.gate", hd, Op{Kind: OpActMul, In: []ValueID{g, xn}, Attr: Attrs{Act: ActSigmoid}})
	// The mean over the streams: the first, then each added in order.
	acc := p.out("mix.stream0", w, Op{Kind: OpSlice, In: []ValueID{g}, Attr: Attrs{Offset: 0, Width: w}})
	for k := 1; k < h; k++ {
		s := p.out("mix.stream", w, Op{Kind: OpSlice, In: []ValueID{g}, Attr: Attrs{Offset: k * w, Width: w}})
		acc = p.out("mix.sum", w, Op{Kind: OpAdd, In: []ValueID{acc, s}})
	}
	d.op(Op{Kind: OpMap, In: []ValueID{acc}, Out: []ValueID{o.Out[0]}, Attr: Attrs{Fn: MapMul, Const: inv}})
	if inj == nil {
		return true
	}
	// The placement weights: the projection over the stream count, through
	// a sigmoid, doubled.
	pw := p.out("mix.place", h, Op{Kind: OpMatVec, In: []ValueID{xn}, Weight: []Binding{*inj}})
	pw = p.out("mix.place_s", h, Op{Kind: OpMap, In: []ValueID{pw}, Attr: Attrs{Fn: MapMul, Const: inv}})
	pw = p.out("mix.place_g", h, Op{Kind: OpMap, In: []ValueID{pw}, Attr: Attrs{Fn: MapSigmoid}})
	d.op(Op{Kind: OpMap, In: []ValueID{pw}, Out: []ValueID{o.Out[1]}, Attr: Attrs{Fn: MapMul, Const: 2}})
	return true
}

// streamPlace writes the placement of a sublayer's output row (In[1]) into
// every stream of the residual (In[0]) at the Streams weights In[2] as
// primitives: stream k becomes x_k + w_k*y, the product fused into the sum.
// The row is tiled over the streams and the weights broadcast a group of d
// each, so the whole residual is one fused multiply-add.
//
// The Sinkhorn form (which recombines the streams) and the bank's (which
// takes no weights) stay whole.
func (d *Graph) streamPlace(o Op) bool {
	a := o.Attr
	h := a.Streams
	if !a.Sinkhorn && len(o.In) == 2 {
		return d.bankPlace(o)
	}
	if a.Sinkhorn || a.Block != 0 || h < 1 || len(o.In) != 3 || len(o.Out) != 1 || len(o.Weight) != 0 {
		return false
	}
	hd, w := d.Values[o.In[0]].Width, d.Values[o.In[1]].Width
	if hd != h*w || w < 1 || d.Values[o.In[2]].Width != h || d.Values[o.Out[0]].Width != hd ||
		d.Values[o.In[0]].Elem != ElemF32 || d.Values[o.In[1]].Elem != ElemF32 || d.Values[o.In[2]].Elem != ElemF32 {
		return false
	}
	p := prims{d: d}
	y := p.out("place.tiled", hd, Op{Kind: OpTile, In: []ValueID{o.In[1]}, Attr: Attrs{Span: w}})
	d.op(Op{Kind: OpMap, In: []ValueID{y, o.In[2], o.In[0]}, Out: []ValueID{o.Out[0]}, Attr: Attrs{Fn: MapFma}})
	return true
}

// bankMix writes a residual attention's read as primitives: over Streams
// streams of d (In[0]; stream 0 the running residual, stream 1+j checkpoint
// j), the first Groups checkpoints and the running residual, in that
// order, each scored against the bound row (Weight[0], d) as
// dot(w, v)/sqrt(mean(v*v) + eps), and the sublayer's row (Out[0]) their
// sum weighted by the scores' softmax, added in order.
func (d *Graph) bankMix(o Op) bool {
	a := o.Attr
	h, nb := a.Streams, a.Groups
	if nb < 1 || nb >= h || len(o.In) != 1 || len(o.Out) != 1 || len(o.Weight) != 1 {
		return false
	}
	w := d.Values[o.Out[0]].Width
	if w < 1 || d.Values[o.In[0]].Width != h*w || d.Values[o.In[0]].Elem != ElemF32 || o.Weight[0].K != w || o.Weight[0].Rows != 0 {
		return false
	}
	p := prims{d: d}
	terms := make([]ValueID, nb+1)
	score := make([]ValueID, nb+1)
	for j := range terms {
		k := j + 1
		if j == nb {
			k = 0
		}
		v := p.out("bank.term", w, Op{Kind: OpSlice, In: []ValueID{o.In[0]}, Attr: Attrs{Offset: k * w, Width: w}})
		sq := p.out("bank.sq", w, Op{Kind: OpMap, In: []ValueID{v, v}, Attr: Attrs{Fn: MapMul}})
		ss := p.out("bank.ss", 1, Op{Kind: OpReduce, In: []ValueID{sq}, Attr: Attrs{Fn: MapAdd, Order: OrderSeq}})
		wv := p.mapw("bank.wv", w, MapMul, o.Weight[0], v)
		dot := p.out("bank.dot", 1, Op{Kind: OpReduce, In: []ValueID{wv}, Attr: Attrs{Fn: MapAdd, Order: OrderSeq}})
		ms := p.out("bank.mean", 1, Op{Kind: OpMap, In: []ValueID{ss}, Attr: Attrs{Fn: MapDiv, Const: float64(w)}})
		ms = p.out("bank.mean_eps", 1, Op{Kind: OpMap, In: []ValueID{ms}, Attr: Attrs{Fn: MapAdd, Const: float64(float32(a.Eps))}})
		rt := p.out("bank.rms", 1, Op{Kind: OpMap, In: []ValueID{ms}, Attr: Attrs{Fn: MapSqrt}})
		terms[j] = v
		score[j] = p.out("bank.score", 1, Op{Kind: OpMap, In: []ValueID{dot, rt}, Attr: Attrs{Fn: MapDiv}})
	}
	mx := score[0]
	for _, s := range score[1:] {
		mx = p.out("bank.max", 1, Op{Kind: OpMap, In: []ValueID{mx, s}, Attr: Attrs{Fn: MapMax}})
	}
	e := make([]ValueID, len(score))
	var sum ValueID
	for j, s := range score {
		c := p.out("bank.centred", 1, Op{Kind: OpMap, In: []ValueID{s, mx}, Attr: Attrs{Fn: MapSub}})
		e[j] = p.out("bank.exp", 1, Op{Kind: OpMap, In: []ValueID{c}, Attr: Attrs{Fn: MapExp}})
		if j == 0 {
			sum = e[0]
			continue
		}
		sum = p.out("bank.sum", 1, Op{Kind: OpMap, In: []ValueID{sum, e[j]}, Attr: Attrs{Fn: MapAdd}})
	}
	var acc ValueID
	for j, v := range terms {
		pj := p.out("bank.p", 1, Op{Kind: OpMap, In: []ValueID{e[j], sum}, Attr: Attrs{Fn: MapDiv}})
		switch {
		case j == len(terms)-1 && j > 0:
			d.op(Op{Kind: OpMap, In: []ValueID{pj, v, acc}, Out: []ValueID{o.Out[0]}, Attr: Attrs{Fn: MapFma}})
		case j == 0:
			acc = p.out("bank.acc", w, Op{Kind: OpMap, In: []ValueID{pj, v}, Attr: Attrs{Fn: MapMul}})
		default:
			acc = p.out("bank.acc", w, Op{Kind: OpMap, In: []ValueID{pj, v, acc}, Attr: Attrs{Fn: MapFma}})
		}
	}
	return true
}

// bankPlace writes a residual attention's placement of a sublayer's output
// row y (In[1]) into the streams (In[0]) as primitives: the running
// residual plus y, every checkpoint as it was; on a checkpoint block (Block
// k) the running residual restarts from y and stream k banks the block's
// input, the running residual as it stood.
func (d *Graph) bankPlace(o Op) bool {
	a := o.Attr
	h, k := a.Streams, a.Block
	if h < 2 || k < 0 || k >= h || len(o.Out) != 1 || len(o.Weight) != 0 {
		return false
	}
	w := d.Values[o.In[1]].Width
	if w < 1 || d.Values[o.In[0]].Width != h*w || d.Values[o.Out[0]].Width != h*w ||
		d.Values[o.In[0]].Elem != ElemF32 || d.Values[o.In[1]].Elem != ElemF32 {
		return false
	}
	p := prims{d: d}
	x := o.In[0]
	slice := func(from, to int) ValueID {
		return p.out("place.streams", (to-from)*w, Op{Kind: OpSlice, In: []ValueID{x}, Attr: Attrs{Offset: from * w, Width: (to - from) * w}})
	}
	x0 := slice(0, 1)
	var parts []ValueID
	if k == 0 {
		parts = []ValueID{p.out("place.running", w, Op{Kind: OpAdd, In: []ValueID{x0, o.In[1]}}), slice(1, h)}
	} else {
		parts = []ValueID{o.In[1]}
		if k > 1 {
			parts = append(parts, slice(1, k))
		}
		parts = append(parts, x0)
		if k < h-1 {
			parts = append(parts, slice(k+1, h))
		}
	}
	d.op(Op{Kind: OpConcat, In: parts, Out: []ValueID{o.Out[0]}})
	return true
}
