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
func DecomposeRecurrent(g *Graph) *Graph {
	d := &Graph{Version: g.Version, Block: g.Block, Values: append([]Value(nil), g.Values...),
		In: append([]ValueID(nil), g.In...), Out: append([]ValueID(nil), g.Out...)}
	for _, o := range g.Ops {
		if o.Kind == OpRecurrent && o.Attr.Mixer == uint8(jlm.LayerShortConv) && d.shortConv(o) {
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
