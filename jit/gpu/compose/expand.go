package compose

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// expanded is g with expandActs applied, g itself untouched (its Encode is
// the cache key), or g where it holds no activation; and where each of g's
// nodes now is.
func (g *Row) expanded() (*Row, []int, error) {
	has := false
	for _, n := range g.Nodes {
		if n.Kind == NUnary && n.Op == OpAct || n.Kind == NBinary && (n.Op == OpActMul || n.Op == OpDivExact) {
			has = true
			break
		}
	}
	if !has {
		at := make([]int, len(g.Nodes))
		for i := range at {
			at[i] = i
		}
		return g, at, nil
	}
	c := *g
	c.Nodes = append([]Node(nil), g.Nodes...)
	c.Stores = append([]Store(nil), g.Stores...)
	c.Stages = make([]Stage, len(g.Stages))
	for i, st := range g.Stages {
		c.Stages[i] = Stage{Width: st.Width, Stores: append([]Store(nil), st.Stores...)}
	}
	if g.Q8 != nil {
		q := *g.Q8
		c.Q8 = &q
	}
	at, err := c.expandActs()
	return &c, at, err
}

// expandActs rewrites every activation node of g (OpAct, OpActMul) as its
// formula over primitive nodes -- maps, constants, comparisons, selects --
// in place, so the former emits primitives only. The constants are the
// vocabulary's (kernels.GELUErfA, TanhP, ...), the ones the host tiers
// evaluate, so a device and the host compute one function. Every reference
// to a rewritten node, a fold's back edge and a store's included, moves to
// its formula's result.
func (g *Row) expandActs() ([]int, error) {
	old := g.Nodes
	at := make([]int, len(old)) // old index -> new
	x := &expander{}
	for i, n := range old {
		var err error
		switch {
		case n.Kind == NUnary && n.Op == OpAct:
			at[i], err = x.act(at[n.A], n.Act)
		case n.Kind == NBinary && n.Op == OpActMul:
			at[i], err = x.actMul(n.Act, at[n.A], at[n.B])
		case n.Kind == NBinary && n.Op == OpDivExact:
			at[i] = x.divExact(old, n, at)
		default:
			at[i] = x.add(n) // its operands remapped below
		}
		if err != nil {
			return nil, err
		}
	}
	// The kept nodes' operands, back edges included, now that every old
	// node has a place.
	for i, n := range old {
		if n.Kind == NUnary && n.Op == OpAct || n.Kind == NBinary && (n.Op == OpActMul || n.Op == OpDivExact) {
			continue
		}
		nd := &x.nodes[at[i]]
		switch n.Kind {
		case NUnary, NReduce, NWindow, NShuffle:
			nd.A = at[n.A]
		case NDot4:
			nd.A, nd.B, nd.C = at[n.A], at[n.B], at[n.C]
		case NAt:
			nd.A, nd.Ix = at[n.A], at[n.Ix]
		case NBinary, NUBinary, NLess:
			nd.A, nd.B = at[n.A], at[n.B]
			if n.Kind == NBinary && n.Op == OpFma {
				nd.C = at[n.C]
			}
		case NSelect:
			nd.A, nd.B, nd.C = at[n.A], at[n.B], at[n.C]
		case NLoad:
			if n.Indexed {
				nd.Ix = at[n.Ix]
			}
		case NFold:
			nd.B = at[n.B]
			if n.A >= 0 {
				nd.A = at[n.A]
			}
		}
	}
	g.Nodes = x.nodes
	remap := func(sts []Store) {
		for i := range sts {
			sts[i].Val = at[sts[i].Val]
			if sts[i].Indexed {
				sts[i].Ix = at[sts[i].Ix]
			}
		}
	}
	remap(g.Stores)
	for _, st := range g.Stages {
		remap(st.Stores)
	}
	if g.Q8 != nil {
		g.Q8.Val = at[g.Q8.Val]
	}
	return at, nil
}

// expander builds the rewritten node list.
type expander struct{ nodes []Node }

func (x *expander) add(n Node) int  { x.nodes = append(x.nodes, n); return len(x.nodes) - 1 }
func (x *expander) c(f float32) int { return x.add(Node{Kind: NConst, F: f}) }
func (x *expander) bin(op Op, a, b int) int {
	return x.add(Node{Kind: NBinary, Op: op, A: a, B: b})
}
func (x *expander) un(op Op, a int) int    { return x.add(Node{Kind: NUnary, Op: op, A: a}) }
func (x *expander) mul(a, b int) int       { return x.bin(OpMul, a, b) }
func (x *expander) neg(a int) int          { return x.bin(OpSub, x.c(0), a) }
func (x *expander) abs(a int) int          { return x.un(OpAbs, a) }
func (x *expander) less(a, b int) int      { return x.add(Node{Kind: NLess, A: a, B: b}) }
func (x *expander) sel(cond, a, b int) int { return x.add(Node{Kind: NSelect, A: a, B: b, C: cond}) }
func (x *expander) clamp(v int, lo, hi float32) int {
	return x.bin(OpMax, x.bin(OpMin, v, x.c(hi)), x.c(lo))
}

// fma is a*b + c with one rounding.
func (x *expander) fma(a, b, c int) int {
	return x.add(Node{Kind: NBinary, Op: OpFma, A: a, B: b, C: c})
}

// The forms below are the ones the host tiers evaluate (jit/cpu), term for
// term and rounding for rounding, so a device and the host compute one
// function: the same constants (the vocabulary's), the same order.

// tanhRational is tanh(v) as v P(v^2)/Q(v^2) on v clamped to +-TanhClamp,
// P and Q in Horner form by fused multiply-adds, highest power first.
func (x *expander) tanhRational(v int) int {
	v = x.clamp(v, -kernels.TanhClamp, kernels.TanhClamp)
	v2 := x.mul(v, v)
	p := x.c(kernels.TanhP[0])
	for _, k := range kernels.TanhP[1:] {
		p = x.fma(p, v2, x.c(k))
	}
	q := x.c(kernels.TanhQ[0])
	for _, k := range kernels.TanhQ[1:] {
		q = x.fma(q, v2, x.c(k))
	}
	return x.bin(OpDiv, x.mul(p, v), q)
}

// softplus is log(1 + e^z) with no logarithm among the primitives:
// max(z, 0) + 2w(1 + w^2/3 + ... + w^8/9), w = e^-|z| / (2 + e^-|z|).
func (x *expander) softplus(z int) int {
	zero, two := x.c(0), x.c(2)
	az := x.bin(OpMax, z, x.bin(OpSub, zero, z))
	e := x.un(OpExp, x.bin(OpSub, zero, az))
	w := x.bin(OpDiv, e, x.bin(OpAdd, two, e))
	w2 := x.mul(w, w)
	s := x.c(1.0 / 9)
	for _, k := range []float32{1.0 / 7, 1.0 / 5, 1.0 / 3, 1} {
		s = x.fma(s, w2, x.c(k))
	}
	return x.bin(OpAdd, x.bin(OpMax, z, zero), x.mul(two, x.mul(w, s)))
}

// act is k(v), or an error for a kind with no formula.
func (x *expander) act(v int, k kernels.ActKind) (int, error) {
	one := x.c(1)
	switch k {
	case kernels.ActIdentity:
		return v, nil
	case kernels.ActReLU:
		return x.bin(OpMax, v, x.c(0)), nil
	case kernels.ActReLU2:
		r := x.bin(OpMax, v, x.c(0))
		return x.mul(r, r), nil
	case kernels.ActSiLU, kernels.ActQuickGELU:
		// v / (1 + e^-z), z = v or 1.702v.
		z := v
		if k == kernels.ActQuickGELU {
			z = x.mul(x.c(1.702), v)
		}
		return x.bin(OpDiv, v, x.bin(OpAdd, one, x.un(OpExp, x.neg(z)))), nil
	case kernels.ActSqrtSoftplus:
		return x.un(OpSqrt, x.softplus(v)), nil
	case kernels.ActSqrtSigmoid:
		// sigmoid(sign(v) sqrt(max(|v|, floor))).
		av := x.bin(OpMax, v, x.neg(v))
		m := x.un(OpSqrt, x.bin(OpMax, av, x.c(kernels.SqrtSigmoidFloor)))
		sm := x.sel(x.less(v, x.c(0)), x.neg(m), m)
		return x.bin(OpDiv, one, x.bin(OpAdd, one, x.un(OpExp, x.neg(sm)))), nil
	case kernels.ActSigmoid:
		return x.bin(OpDiv, one, x.bin(OpAdd, one, x.un(OpExp, x.neg(v)))), nil
	case kernels.ActTanh:
		return x.tanhRational(v), nil
	case kernels.ActSoftplus:
		return x.softplus(v), nil
	case kernels.ActGELUErf:
		// 0.5 ((v + |v|) - |v| q), q = 1 - erf(|v|/sqrt 2) by Abramowitz and
		// Stegun 7.1.26: t = 1/(1 + p z), q = poly(t) t e^-z^2.
		av := x.bin(OpMax, v, x.neg(v))
		z := x.mul(x.c(0.70710678118654752), av)
		t := x.bin(OpDiv, one, x.fma(x.c(kernels.GELUErfP), z, one))
		p := x.c(kernels.GELUErfA[4])
		for i := 3; i >= 0; i-- {
			p = x.fma(p, t, x.c(kernels.GELUErfA[i]))
		}
		q := x.mul(x.mul(p, t), x.un(OpExp, x.neg(x.mul(z, z))))
		return x.mul(x.c(0.5), x.bin(OpSub, x.bin(OpAdd, v, av), x.mul(av, q))), nil
	case kernels.ActGELU:
		// 0.5 v (1 + tanh u), u = sqrt(2/pi) (v + 0.044715 v^3), tanh u as
		// 1 - 2/(e^2u + 1).
		u := x.mul(x.c(0.7978845608028654), x.bin(OpAdd, v, x.mul(x.c(0.044715), x.mul(v, x.mul(v, v)))))
		e := x.un(OpExp, x.mul(x.c(2), u))
		th := x.bin(OpSub, one, x.bin(OpDiv, x.c(2), x.bin(OpAdd, e, one)))
		return x.mul(x.mul(x.c(0.5), v), x.bin(OpAdd, one, th)), nil
	}
	return 0, fmt.Errorf("compose: no formula for the activation %v", k)
}

// actMul is gated activation k of gate g and up u, by its definition.
func (x *expander) actMul(k kernels.ActKind, g, u int) (int, error) {
	switch k {
	case kernels.ActSwiGLUOAI:
		// quick_gelu(min(g, 7)) (clamp(u, -7, 7) + 1).
		a, err := x.act(x.bin(OpMin, g, x.c(7)), kernels.ActQuickGELU)
		return x.mul(a, x.bin(OpAdd, x.clamp(u, -7, 7), x.c(1))), err
	case kernels.ActSwiGLUClamp:
		// silu(min(g, 10)) clamp(u, -10, 10).
		a, err := x.act(x.bin(OpMin, g, x.c(10)), kernels.ActSiLU)
		return x.mul(a, x.clamp(u, -10, 10)), err
	case kernels.ActSitu:
		// b tanh(g/b) sigmoid(g) times b' tanh(u/b'), b and b' the
		// vocabulary's bounds, tanh the bounded one.
		bound := func(v int, c float32) int { return x.mul(x.c(c), x.tanhRational(x.mul(v, x.c(1/c)))) }
		sig := x.bin(OpDiv, x.c(1), x.bin(OpAdd, x.c(1), x.un(OpExp, x.neg(g))))
		return x.mul(x.mul(bound(g, kernels.SituBeta), sig), bound(u, kernels.SituLinearBeta)), nil
	}
	a, err := x.act(g, k)
	return x.mul(a, u), err
}

// divExact is node n, a correctly rounded quotient, in integer arithmetic
// over its operands at their new places: by the constant 127, the
// significand's division (Div127); otherwise rounded from an estimate --
// the reciprocal of a correctly rounded quotient c/x from x/c refined by a
// Newton step (two floats either side), any other quotient from the
// division itself (three, SPIR-V's 2.5 ulp).
func (x *expander) divExact(old []Node, n Node, at []int) int {
	g := &Row{Nodes: x.nodes}
	num, den := at[n.A], at[n.B]
	var v int
	switch dn := old[n.B]; {
	case dn.Kind == NConst && dn.F == 127:
		v = g.Div127(num)
	case old[n.A].Kind == NConst && old[n.A].F == 1 && dn.Kind == NBinary && dn.Op == OpDivExact &&
		old[dn.B].Kind == NConst:
		// 1/(a/c): c/a estimates it; one Newton step against the rounded
		// quotient den refines it.
		y := g.Bin(OpDiv, g.Const(old[dn.B].F), at[dn.A])
		y = g.Bin(OpAdd, y, g.Bin(OpMul, y, g.Bin(OpSub, g.Const(1), g.Bin(OpMul, den, y))))
		v = g.RoundQuotient(num, den, y, 2)
	default:
		v = g.RoundQuotient(num, den, g.Bin(OpDiv, num, den), 3)
	}
	x.nodes = g.Nodes
	return v
}
