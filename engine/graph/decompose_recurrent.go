package graph

import (
	"math"

	"github.com/jitllm/jitllm/format/jlm"
)

// DecomposeRecurrent is g with each recurrent mixer whose kind it states
// written as primitives, the mixer's own state an OpStateConv's window;
// the rest of g as it is. It is the composer's, apart from Decompose:
// main's device runs these mixers whole.
//
// LFM2's gated short convolution (jlm.LayerShortConv): B, C and x the
// input row's projections, B*x convolved over the window, C times that,
// then the output projection.
//
// The gated delta rule (jlm.LayerLinearAttn), the host's linearAttn: q,
// k and v the row's projection; z the output gate's projection, low rank
// where the block binds ssm_g_a; the convolution and SiLU; beta the
// sigmoid of its logits and the decay exp(A softplus(alpha + dt_bias)) --
// exp(lb sigmoid(-A (alpha + dt_bias))) where Kimi-K3 states a lower bound
// lb -- alpha deinterleaved from the logits' row (Qwen3-Next) or its own
// low-rank projection a value head's state columns wide (Kimi Delta
// Attention); q and k L2-normed per key head, q scaled twice by
// 1/sqrt(HeadDim) and k once; the rule; each value head RMS-normed against
// ssm_norm, gated by silu(z) or, for KDA and where the container states
// it, sigmoid(z); the output projection.
//
// Mamba-1's mixer (jlm.LayerMamba1), the host's mamba1Attn: x and z the
// row's projections; x convolved, its bias added, SiLU; dt's low rank, B
// and C its projections, each RMS-normed where the block binds a norm;
// dt = softplus(dt's projection + dt_bias); the scan, each channel a value
// head; silu(z) times its output; the output projection.
//
// Mamba-2's mixer (jlm.LayerSSD, and Falcon-H1's beside its attention,
// jlm.LayerSSDAttn), the host's ssdAttn: xBC, z and dt the
// row's projections; xBC convolved over the window, its bias added, SiLU;
// dt = softplus(dt + dt_bias), decay = exp(A dt); the scan; silu(z) times
// its output; the RMS norm over each of c.SSM.Groups groups with its slice
// of the weight; the output projection. c states the scan's heads.
func DecomposeRecurrent(g *Graph, c *jlm.Config) *Graph {
	d := &Graph{Version: g.Version, Block: g.Block, Values: append([]Value(nil), g.Values...),
		In: append([]ValueID(nil), g.In...), Out: append([]ValueID(nil), g.Out...)}
	for _, o := range g.Ops {
		if o.Kind == OpRecurrent && o.Attr.Mixer == uint8(jlm.LayerShortConv) && d.shortConv(o, c) {
			continue
		}
		ssd := o.Attr.Mixer == uint8(jlm.LayerSSD) || o.Attr.Mixer == uint8(jlm.LayerSSDAttn)
		if o.Kind == OpRecurrent && ssd && d.ssd(o, c) {
			continue
		}
		if o.Kind == OpRecurrent && o.Attr.Mixer == uint8(jlm.LayerMamba1) && d.mamba1(o, c) {
			continue
		}
		if o.Kind == OpRecurrent && o.Attr.Mixer == uint8(jlm.LayerLinearAttn) && d.delta(o, c) {
			continue
		}
		if o.Kind == OpIndex && d.index(o, g, c) {
			continue
		}
		if o.Kind == OpIndex && d.blockSelect(o, g, c) {
			continue
		}
		d.op(o)
	}
	return d
}

// shortConv writes LFM2's mixer o as primitives, false where o does not
// bind what it reads.
func (d *Graph) shortConv(o Op, cfg *jlm.Config) bool {
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
	kind := OpStateConv
	if cfg.Flags.Has(jlm.FlagNonCausal) {
		// A bidirectional model's convolution is centred over the
		// sequence's rows (LFM2's encoder).
		kind = OpRowConv
	}
	d.op(Op{Kind: kind, In: []ValueID{gated}, Out: []ValueID{cv}, Weight: []Binding{conv}, Attr: Attrs{Taps: conv.K}})
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
	slice := func(s string, off, n int) ValueID {
		v := val(s, n)
		d.op(Op{Kind: OpSlice, In: []ValueID{xs}, Out: []ValueID{v}, Attr: Attrs{Offset: off, Width: n}})
		return v
	}
	cs, bs, xv := slice("c", 0, kh*kd), slice("b", kh*kd, kh*kd), slice("x", 2*kh*kd, inner)
	y := val("y", inner)
	d.op(Op{Kind: OpStateScan, In: []ValueID{xv, bs, cs, dt, decay}, Out: []ValueID{y}, Weight: []Binding{{Role: skip.Role, K: vh}},
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

// mamba1 writes Mamba-1's mixer o as primitives, false where o does not
// bind what it reads.
func (d *Graph) mamba1(o Op, c *jlm.Config) bool {
	w := map[jlm.Role]Binding{}
	for _, b := range o.Weight {
		w[b.Role] = b
	}
	in, gate, conv, out := w[jlm.RoleAttnQKV], w[jlm.RoleAttnGate], w[jlm.RoleSSMConv1d], w[jlm.RoleSSMOut]
	cb, a, skip, dtb := w[jlm.RoleSSMConvBias], w[jlm.RoleSSMA], w[jlm.RoleSSMD], w[jlm.RoleSSMDtBias]
	xdt, xb, xc, dtp := w[jlm.RoleSSMXDt], w[jlm.RoleSSMXB], w[jlm.RoleSSMXC], w[jlm.RoleSSMDtProj]
	nd, hasNorm := w[jlm.RoleSSMDtNorm]
	nb, nc := w[jlm.RoleSSMBNorm], w[jlm.RoleSSMCNorm]
	inner, kd := in.Rows, int(c.SSM.StateSize)
	if len(o.In) != 1 || len(o.Out) != 1 || inner < 1 || kd < 1 || gate.Rows != inner || conv.Rows != inner || conv.K < 2 ||
		cb.K != inner || xb.Rows != kd || xc.Rows != kd || xdt.Rows < 1 || dtp.K != xdt.Rows || dtp.Rows != inner ||
		a.Rows != inner || a.K != kd || skip.Role == 0 || dtb.Role == 0 || out.K != inner ||
		hasNorm && (nb.Role == 0 || nc.Role == 0) {
		return false
	}
	name := d.Values[o.Out[0]].Name
	val := func(s string, n int) ValueID { return d.value(name+"."+s, n) }
	mv := func(s string, b Binding, x ValueID) ValueID {
		v := val(s, b.Rows)
		d.op(Op{Kind: OpMatVec, In: []ValueID{x}, Out: []ValueID{v}, Weight: []Binding{b}})
		return v
	}
	norm := func(s string, x ValueID, b Binding, n int) ValueID {
		if !hasNorm {
			return x
		}
		v := val(s, n)
		d.op(Op{Kind: OpNorm, In: []ValueID{x}, Out: []ValueID{v}, Weight: []Binding{{Role: b.Role, K: n}}, Attr: Attrs{Eps: float64(c.RMSEps)}})
		return v
	}
	add := func(s string, x ValueID, b Binding, n int) ValueID {
		v := val(s, n)
		d.op(Op{Kind: OpMap, In: []ValueID{x}, Out: []ValueID{v}, Weight: []Binding{{Role: b.Role, K: n}}, Attr: Attrs{Fn: MapAdd}})
		return v
	}
	x, z := mv("x", in, o.In[0]), mv("z", gate, o.In[0])
	cv := val("conv", inner)
	d.op(Op{Kind: OpStateConv, In: []ValueID{x}, Out: []ValueID{cv}, Weight: []Binding{conv}, Attr: Attrs{Taps: conv.K}})
	xs := val("conv_silu", inner)
	d.op(Op{Kind: OpAct, In: []ValueID{add("conv_b", cv, cb, inner)}, Out: []ValueID{xs}, Attr: Attrs{Act: ActSiLU}})
	low := norm("dt_low_n", mv("dt_low", xdt, xs), nd, xdt.Rows)
	bs := norm("b_n", mv("b", xb, xs), nb, kd)
	cs := norm("c_n", mv("c", xc, xs), nc, kd)
	dt := val("dt", inner)
	d.op(Op{Kind: OpMap, In: []ValueID{add("dt_b", mv("dt_raw", dtp, low), dtb, inner)}, Out: []ValueID{dt}, Attr: Attrs{Fn: MapSoftplus}})
	y := val("y", inner)
	d.op(Op{Kind: OpStateScan, In: []ValueID{xs, bs, cs, dt}, Out: []ValueID{y}, Weight: []Binding{{Role: skip.Role, K: inner}, a},
		Attr: Attrs{NHead: inner, NKVHead: 1, HeadDim: kd}})
	gated := val("gated", inner)
	d.op(Op{Kind: OpActMul, In: []ValueID{z, y}, Out: []ValueID{gated}, Attr: Attrs{Act: ActSiLU}})
	d.op(Op{Kind: OpMatVec, In: []ValueID{gated}, Out: []ValueID{o.Out[0]}, Weight: []Binding{out}})
	return true
}

// kdaL2Eps is Kimi Delta Attention's L2-norm epsilon, hardcoded in its
// reference where the delta rule takes rms_norm_eps.
const kdaL2Eps = 1e-6

// delta writes the gated delta rule o as primitives, false where o does not
// bind what it reads, or c states a decay bound on a scalar decay.
func (d *Graph) delta(o Op, c *jlm.Config) bool {
	w := map[jlm.Role]Binding{}
	for _, b := range o.Weight {
		w[b.Role] = b
	}
	in, conv, out, ba := w[jlm.RoleAttnQKV], w[jlm.RoleSSMConv1d], w[jlm.RoleSSMOut], w[jlm.RoleSSMBA]
	a, dtb, nrm := w[jlm.RoleSSMA], w[jlm.RoleSSMDtBias], w[jlm.RoleSSMNorm]
	gate, hasGate := w[jlm.RoleAttnGate]
	ga, lowGate := w[jlm.RoleSSMGA]
	gb := w[jlm.RoleSSMGB]
	fa, chans := w[jlm.RoleSSMFA]
	fb := w[jlm.RoleSSMFB]
	kh, kd, vh := int(c.SSM.Groups), int(c.SSM.StateSize), int(c.SSM.NHeadV)
	inner := int(c.SSM.Inner)
	if len(o.In) != 1 || len(o.Out) != 1 || kh < 1 || kd < 1 || vh < 1 || vh%kh != 0 || inner%vh != 0 ||
		in.Rows != inner+2*kh*kd || conv.Rows != in.Rows || conv.K < 2 || out.K != inner || nrm.K != inner/vh ||
		hasGate == lowGate || c.KDALowerBound != 0 && !chans {
		return false
	}
	rep, vd := vh/kh, inner/vh
	decayW := vh
	if chans {
		decayW = vh * kd
		if ba.Rows != vh || fb.Rows != decayW || fb.K != fa.Rows {
			return false
		}
	} else if ba.Rows != 2*vh {
		return false
	}
	if a.K != decayW || dtb.K != decayW || lowGate && (gb.K != ga.Rows || gb.Rows != inner) || hasGate && gate.Rows != inner {
		return false
	}
	name := d.Values[o.Out[0]].Name
	val := func(s string, n int) ValueID { return d.value(name+"."+s, n) }
	mv := func(s string, b Binding, x ValueID) ValueID {
		v := val(s, b.Rows)
		d.op(Op{Kind: OpMatVec, In: []ValueID{x}, Out: []ValueID{v}, Weight: []Binding{b}})
		return v
	}
	mapv := func(s string, f MapFn, n int, k float64, x ...ValueID) ValueID {
		v := val(s, n)
		d.op(Op{Kind: OpMap, In: x, Out: []ValueID{v}, Attr: Attrs{Fn: f, Const: k}})
		return v
	}
	mapw := func(s string, f MapFn, x ValueID, b Binding, n int) ValueID {
		v := val(s, n)
		d.op(Op{Kind: OpMap, In: []ValueID{x}, Out: []ValueID{v}, Weight: []Binding{{Role: b.Role, K: b.K}}, Attr: Attrs{Fn: f}})
		return v
	}
	slice := func(s string, x ValueID, off, n, head int) ValueID {
		width := n
		if head > 0 {
			width = d.Values[x].Width / head * n
		}
		v := val(s, width)
		d.op(Op{Kind: OpSlice, In: []ValueID{x}, Out: []ValueID{v}, Attr: Attrs{Offset: off, Width: n, HeadDim: head}})
		return v
	}
	mixed := mv("qkv", in, o.In[0])
	var z ValueID
	if lowGate {
		z = mv("z", gb, mv("z_low", ga, o.In[0]))
	} else {
		z = mv("z", gate, o.In[0])
	}
	var alpha, braw ValueID
	if chans {
		alpha, braw = mv("alpha", fb, mv("alpha_low", fa, o.In[0])), mv("b", ba, o.In[0])
	} else {
		bav := mv("ba", ba, o.In[0])
		braw, alpha = slice("b", bav, 0, rep, 2*rep), slice("alpha", bav, rep, rep, 2*rep)
	}
	cv := val("conv", in.Rows)
	d.op(Op{Kind: OpStateConv, In: []ValueID{mixed}, Out: []ValueID{cv}, Weight: []Binding{conv}, Attr: Attrs{Taps: conv.K}})
	xs := val("conv_silu", in.Rows)
	d.op(Op{Kind: OpAct, In: []ValueID{cv}, Out: []ValueID{xs}, Attr: Attrs{Act: ActSiLU}})

	ad := mapw("alpha_dt", MapAdd, alpha, dtb, decayW)
	var decay ValueID
	if lb := c.KDALowerBound; lb != 0 {
		// Kimi-K3's lower bound: exp(lb sigmoid(-A z)), A the stored
		// -exp(A_log).
		neg := mapv("a_z_neg", MapMul, decayW, -1, mapw("a_z", MapMul, ad, a, decayW))
		decay = mapv("decay", MapExp, decayW, 0, mapv("bounded", MapMul, decayW, float64(lb), mapv("a_z_sig", MapSigmoid, decayW, 0, neg)))
	} else {
		sp := mapv("alpha_sp", MapSoftplus, decayW, 0, ad)
		decay = mapv("decay", MapExp, decayW, 0, mapw("a_sp", MapMul, sp, a, decayW))
	}
	beta := mapv("beta", MapSigmoid, vh, 0, braw)

	eps := float64(c.RMSEps)
	if chans {
		eps = kdaL2Eps
	}
	scale := 1 / math.Sqrt(float64(kd))
	l2 := func(s string, off int) ValueID {
		x := slice(s, xs, off, kh*kd, 0)
		v := val(s+"_l2", kh*kd)
		d.op(Op{Kind: OpHeadNorm, In: []ValueID{x}, Out: []ValueID{v}, Attr: Attrs{Eps: eps / float64(kd), HeadDim: kd}})
		return mapv(s+"_scaled", MapMul, kh*kd, scale, v)
	}
	q := mapv("q_scaled2", MapMul, kh*kd, scale, l2("q", 0))
	k := l2("k", kh*kd)
	v := slice("v", xs, 2*kh*kd, inner, 0)
	y := val("y", inner)
	d.op(Op{Kind: OpStateDelta, In: []ValueID{q, k, v, decay, beta}, Out: []ValueID{y},
		Attr: Attrs{NHead: vh, NKVHead: kh, HeadDim: kd, KeyTiled: c.Flags.Has(jlm.FlagDeltaKeyTiled)}})
	normed := val("normed", inner)
	d.op(Op{Kind: OpHeadNorm, In: []ValueID{y}, Out: []ValueID{normed}, Weight: []Binding{{Role: nrm.Role, K: vd}},
		Attr: Attrs{Eps: float64(c.RMSEps), HeadDim: vd}})
	act := ActSiLU
	if chans || c.Flags2.Has(jlm.Flag2DeltaSigmoidGate) {
		act = ActSigmoid
	}
	gated := val("gated", inner)
	d.op(Op{Kind: OpActMul, In: []ValueID{z, normed}, Out: []ValueID{gated}, Attr: Attrs{Act: act}})
	d.op(Op{Kind: OpMatVec, In: []ValueID{gated}, Out: []ValueID{o.Out[0]}, Weight: []Binding{out}})
	return true
}

// idxEps is the indexer key norm's epsilon, hardcoded in the reference.
const idxEps = 1e-6

// index writes an MLA block's indexer o (DeepSeek V3.2's, over the normed
// row and the query latent) as the host's indexer.go runs it: the key the
// row's projection, LayerNormed, the first NRot of it turned in NEOX pairs
// by the block's rotary; the query the latent's projection, each head
// turned the same way; each head's weight the row's projection over
// sqrt(heads*HeadDim); then the selection. It is false for the GQA form
// (MiniMax's), whose key is one more kv head of the row.
func (d *Graph) index(o Op, g *Graph, c *jlm.Config) bool {
	a := o.Attr
	if len(o.In) != 2 || len(o.Out) != 1 || len(o.Weight) != 5 || a.NHead < 1 || a.HeadDim < 1 || a.TopK < 1 {
		return false
	}
	w := map[jlm.Role]Binding{}
	for _, b := range o.Weight {
		w[b.Role] = b
	}
	qb, k, proj, kn, knb := w[jlm.RoleIdxQB], w[jlm.RoleIdxK], w[jlm.RoleIdxProj], w[jlm.RoleIdxKNorm], w[jlm.RoleIdxKNormB]
	H, D := a.NHead, a.HeadDim
	if qb.Rows != H*D || k.Rows != D || proj.Rows != H || kn.K != D || knb.K != D {
		return false
	}
	var rope *Attrs
	if !c.Flags.Has(jlm.FlagNoPosEnc) {
		for _, r := range g.Ops {
			if r.Kind == OpRope {
				at := r.Attr
				rope = &at
				break
			}
		}
		if rope == nil || rope.NRot > D {
			return false
		}
		rope.HeadDim, rope.Layout = D, RopeNeox
	}
	name := d.Values[o.Out[0]].Name
	val := func(s string, n int) ValueID { return d.value(name+"."+s, n) }
	mv := func(s string, b Binding, x ValueID) ValueID {
		v := val(s, b.Rows)
		d.op(Op{Kind: OpMatVec, In: []ValueID{x}, Out: []ValueID{v}, Weight: []Binding{b}})
		return v
	}
	turn := func(s string, x ValueID) ValueID {
		if rope == nil {
			return x
		}
		v := val(s, d.Values[x].Width)
		d.op(Op{Kind: OpRope, In: []ValueID{x}, Out: []ValueID{v}, Attr: *rope})
		return v
	}
	kv := mv("k", k, o.In[0])
	kl := val("k_normed", D)
	d.op(Op{Kind: OpLayerNorm, In: []ValueID{kv}, Out: []ValueID{kl}, Weight: []Binding{{Role: kn.Role, K: D}, {Role: knb.Role, K: D}},
		Attr: Attrs{Eps: idxEps}})
	key := turn("k_rot", kl)
	q := turn("q_rot", mv("q", qb, o.In[1]))
	ws := val("w_scaled", H)
	d.op(Op{Kind: OpMap, In: []ValueID{mv("w", proj, o.In[0])}, Out: []ValueID{ws},
		Attr: Attrs{Fn: MapMul, Const: float64(float32(1 / math.Sqrt(float64(H*D))))}})
	d.op(Op{Kind: OpIndexSelect, In: []ValueID{q, ws, key}, Out: []ValueID{o.Out[0]}, Reads: o.Reads, Writes: o.Writes,
		Attr: Attrs{NHead: H, HeadDim: D, TopK: a.TopK}})
	return true
}

// blockSelect writes a GQA block's indexer o (MiniMax Sparse Attention,
// over the normed row alone) as the host's msa.go runs it: the query the
// row's projection, each group RMS-normed against RoleIdxQNorm and turned by
// the block's rotary; the key the row's projection, RMS-normed against
// RoleIdxKNorm and turned the same way; then the selection. It is false for
// QSA's pooled form.
func (d *Graph) blockSelect(o Op, g *Graph, c *jlm.Config) bool {
	a := o.Attr
	if len(o.In) != 1 || len(o.Out) != 1 || len(o.Weight) != 4 || a.Pooled || a.NHead < 1 || a.HeadDim < 1 ||
		a.TopK < 1 || a.Block < 1 || a.Local < 0 {
		return false
	}
	w := map[jlm.Role]Binding{}
	for _, b := range o.Weight {
		w[b.Role] = b
	}
	qp, kp, qn, kn := w[jlm.RoleIdxQ], w[jlm.RoleIdxK], w[jlm.RoleIdxQNorm], w[jlm.RoleIdxKNorm]
	G, D := a.NHead, a.HeadDim
	if qp.Rows != G*D || kp.Rows != D || qn.K != D || kn.K != D {
		return false
	}
	var rope *Attrs
	if c.NRot > 0 && !c.Flags.Has(jlm.FlagNoPosEnc) {
		for _, r := range g.Ops {
			if r.Kind == OpRope {
				at := r.Attr
				rope = &at
				break
			}
		}
		if rope == nil || rope.NRot > D {
			return false
		}
		rope.HeadDim = D
	}
	name := d.Values[o.Out[0]].Name
	val := func(s string, n int) ValueID { return d.value(name+"."+s, n) }
	turn := func(s string, x ValueID) ValueID {
		if rope == nil {
			return x
		}
		v := val(s, d.Values[x].Width)
		d.op(Op{Kind: OpRope, In: []ValueID{x}, Out: []ValueID{v}, Attr: *rope})
		return v
	}
	q, k := val("q", G*D), val("k", D)
	d.op(Op{Kind: OpMatVec, In: []ValueID{o.In[0]}, Out: []ValueID{q}, Weight: []Binding{qp}})
	d.op(Op{Kind: OpMatVec, In: []ValueID{o.In[0]}, Out: []ValueID{k}, Weight: []Binding{kp}})
	qh, kh := val("q_normed", G*D), val("k_normed", D)
	d.op(Op{Kind: OpHeadNorm, In: []ValueID{q}, Out: []ValueID{qh}, Weight: []Binding{{Role: qn.Role, K: D}},
		Attr: Attrs{Eps: float64(c.RMSEps), HeadDim: D}})
	d.op(Op{Kind: OpNorm, In: []ValueID{k}, Out: []ValueID{kh}, Weight: []Binding{{Role: kn.Role, K: D}},
		Attr: Attrs{Eps: float64(c.RMSEps)}})
	d.op(Op{Kind: OpBlockSelect, In: []ValueID{turn("q_rot", qh), turn("k_rot", kh)}, Out: []ValueID{o.Out[0]},
		Reads: []StateRef{{"idx", 0}}, Writes: []StateRef{{"idx", 1}},
		Attr: Attrs{NHead: G, HeadDim: D, TopK: a.TopK, Block: a.Block, Local: a.Local}})
	return true
}
