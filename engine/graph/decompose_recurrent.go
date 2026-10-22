package graph

import "github.com/jitllm/jitllm/format/jlm"

// DecomposeRecurrent is g with each recurrent mixer whose kind it states
// written as primitives, the mixer's own state an OpStateConv's window;
// the rest of g as it is. It is the composer's, apart from Decompose:
// main's device runs these mixers whole.
//
// LFM2's gated short convolution (jlm.LayerShortConv): B, C and x the
// input row's projections, B*x convolved over the window, C times that,
// then the output projection.
//
// Mamba-2's mixer (jlm.LayerSSD), the host's ssdAttn: xBC, z and dt the
// row's projections; xBC convolved over the window, its bias added, SiLU;
// dt = softplus(dt + dt_bias), decay = exp(A dt); the scan; silu(z) times
// its output; the RMS norm over each of c.SSM.Groups groups with its slice
// of the weight; the output projection. c states the scan's heads.
func DecomposeRecurrent(g *Graph, c *jlm.Config) *Graph {
	d := &Graph{Version: g.Version, Block: g.Block, Values: append([]Value(nil), g.Values...),
		In: append([]ValueID(nil), g.In...), Out: append([]ValueID(nil), g.Out...)}
	for _, o := range g.Ops {
		if o.Kind == OpRecurrent && o.Attr.Mixer == uint8(jlm.LayerShortConv) && d.shortConv(o) {
			continue
		}
		if o.Kind == OpRecurrent && o.Attr.Mixer == uint8(jlm.LayerSSD) && d.ssd(o, c) {
			continue
		}
		d.op(o)
	}
	return d
}

// shortConv writes LFM2's mixer o as primitives, false where o does not
// bind what it reads.
func (d *Graph) shortConv(o Op) bool {
	w := map[jlm.Role]Binding{}
	for _, b := range o.Weight {
		w[b.Role] = b
	}
	conv, out := w[jlm.RoleSSMConv1d], w[jlm.RoleSSMOut]
	bx, bb, bc := w[jlm.RoleSCX], w[jlm.RoleSCB], w[jlm.RoleSCC]
	if len(o.In) != 1 || len(o.Out) != 1 || conv.K < 2 || out.Rows == 0 || bx.Rows == 0 || bb.Rows != bx.Rows || bc.Rows != bx.Rows ||
		conv.Rows != bx.Rows || out.K != bx.Rows {
		return false
	}
	n := bx.Rows
	name := d.Values[o.Out[0]].Name
	val := func(s string) ValueID { return d.value(name+"."+s, n) }
	mv := func(s string, b Binding) ValueID {
		v := val(s)
		d.op(Op{Kind: OpMatVec, In: []ValueID{o.In[0]}, Out: []ValueID{v}, Weight: []Binding{b}})
		return v
	}
	mul := func(s string, a, b ValueID) ValueID {
		v := val(s)
		d.op(Op{Kind: OpMap, In: []ValueID{a, b}, Out: []ValueID{v}, Attr: Attrs{Fn: MapMul}})
		return v
	}
	x, b, c := mv("x", bx), mv("b", bb), mv("c", bc)
	gated := mul("bx", b, x)
	cv := val("conv")
	d.op(Op{Kind: OpStateConv, In: []ValueID{gated}, Out: []ValueID{cv}, Weight: []Binding{conv}, Attr: Attrs{Taps: conv.K}})
	y := mul("y", c, cv)
	d.op(Op{Kind: OpMatVec, In: []ValueID{y}, Out: []ValueID{o.Out[0]}, Weight: []Binding{out}})
	return true
}

// ssd writes Mamba-2's mixer o as primitives, false where o does not bind
// what it reads or c states no scan that fits it.
func (d *Graph) ssd(o Op, c *jlm.Config) bool {
	w := map[jlm.Role]Binding{}
	for _, b := range o.Weight {
		w[b.Role] = b
	}
	in, gate, ba, conv, out := w[jlm.RoleAttnQKV], w[jlm.RoleAttnGate], w[jlm.RoleSSMBA], w[jlm.RoleSSMConv1d], w[jlm.RoleSSMOut]
	cb, a, skip, dtb := w[jlm.RoleSSMConvBias], w[jlm.RoleSSMA], w[jlm.RoleSSMD], w[jlm.RoleSSMDtBias]
	nrm, hasNorm := w[jlm.RoleSSMNorm]
	kh, kd, vh := int(c.SSM.Groups), int(c.SSM.StateSize), int(c.SSM.NHeadV)
	inner := gate.Rows
	if len(o.In) != 1 || len(o.Out) != 1 || kh < 1 || kd < 1 || vh < 1 || inner%vh != 0 || vh%kh != 0 ||
		in.Rows != inner+2*kh*kd || ba.Rows != vh || conv.Rows != in.Rows || conv.K < 2 || cb.K != in.Rows ||
		a.Role == 0 || skip.Role == 0 || dtb.Role == 0 || out.K != inner {
		return false
	}
	groups := max(int(c.SSM.Groups), 1)
	if hasNorm && inner%groups != 0 {
		return false
	}
	name := d.Values[o.Out[0]].Name
	val := func(s string, n int) ValueID { return d.value(name+"."+s, n) }
	mv := func(s string, b Binding, x ValueID) ValueID {
		v := val(s, b.Rows)
		d.op(Op{Kind: OpMatVec, In: []ValueID{x}, Out: []ValueID{v}, Weight: []Binding{b}})
		return v
	}
	mapw := func(s string, f MapFn, x ValueID, b Binding, n int) ValueID {
		v := val(s, n)
		d.op(Op{Kind: OpMap, In: []ValueID{x}, Out: []ValueID{v}, Weight: []Binding{{Role: b.Role, K: n}}, Attr: Attrs{Fn: f}})
		return v
	}
	map1 := func(s string, f MapFn, x ValueID, n int) ValueID {
		v := val(s, n)
		d.op(Op{Kind: OpMap, In: []ValueID{x}, Out: []ValueID{v}, Attr: Attrs{Fn: f}})
		return v
	}
	xbc, z, dtr := mv("xbc", in, o.In[0]), mv("z", gate, o.In[0]), mv("dt_raw", ba, o.In[0])
	cv := val("conv", in.Rows)
	d.op(Op{Kind: OpStateConv, In: []ValueID{xbc}, Out: []ValueID{cv}, Weight: []Binding{conv}, Attr: Attrs{Taps: conv.K}})
	cbv := mapw("conv_b", MapAdd, cv, cb, in.Rows)
	xs := val("conv_silu", in.Rows)
	d.op(Op{Kind: OpAct, In: []ValueID{cbv}, Out: []ValueID{xs}, Attr: Attrs{Act: ActSiLU}})
	dt := map1("dt", MapSoftplus, mapw("dt_b", MapAdd, dtr, dtb, vh), vh)
	decay := map1("decay", MapExp, mapw("a_dt", MapMul, dt, a, vh), vh)
	y := val("y", inner)
	d.op(Op{Kind: OpStateScan, In: []ValueID{xs, dt, decay}, Out: []ValueID{y}, Weight: []Binding{{Role: skip.Role, K: vh}},
		Attr: Attrs{NHead: vh, NKVHead: kh, HeadDim: kd, KeyTiled: c.Flags.Has(jlm.FlagDeltaKeyTiled)}})
	gated := val("gated", inner)
	d.op(Op{Kind: OpActMul, In: []ValueID{z, y}, Out: []ValueID{gated}, Attr: Attrs{Act: ActSiLU}})
	last := gated
	if hasNorm {
		nv := val("normed", inner)
		d.op(Op{Kind: OpHeadNorm, In: []ValueID{gated}, Out: []ValueID{nv}, Attr: Attrs{Eps: float64(c.RMSEps), HeadDim: inner / groups}})
		last = mapw("norm_w", MapMul, nv, nrm, inner)
	}
	d.op(Op{Kind: OpMatVec, In: []ValueID{last}, Out: []ValueID{o.Out[0]}, Weight: []Binding{out}})
	return true
}
