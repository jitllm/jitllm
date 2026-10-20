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
