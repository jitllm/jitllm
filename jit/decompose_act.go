package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/ir"
)

// The activations' formulas are main's (jit_old/gpu/compose/expand.go), term
// for term: its constants, its order, its fused multiply-adds.

const (
	tanhClamp = float32(7.90531110763549805)
	geluErfP  = float32(0.3275911)
)

var (
	tanhP = [7]float32{-2.76076847742355e-16, 2.00018790482477e-13, -8.60467152213735e-11,
		5.12229709037114e-08, 1.48572235717979e-05, 6.37261928875436e-04, 4.89352455891786e-03}
	tanhQ    = [4]float32{1.19825839466702e-06, 1.18534705686654e-04, 2.26843463243900e-03, 4.89352518554385e-03}
	geluErfA = [5]float32{0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429}
)

func (x *exprs) fma(a, b, c int) int { return x.add(Expr{Op: OpFma, Type: ir.F32, A: a, B: b, C: c}) }

func (x *exprs) neg(a int) int { return x.op(OpSub, ir.F32, x.f(0), a) }

func (x *exprs) clamp(v int, lo, hi float32) int {
	return x.op(OpMax, ir.F32, x.op(OpMin, ir.F32, v, x.f(hi)), x.f(lo))
}

func (x *exprs) sigmoid(v int) int {
	return x.op(OpDiv, ir.F32, x.f(1), x.op(OpAdd, ir.F32, x.f(1), x.un(OpExp, ir.F32, x.neg(v))))
}

// tanh is v P(v^2)/Q(v^2) on v clamped to +-tanhClamp.
func (x *exprs) tanh(v int) int {
	F := ir.F32
	v = x.clamp(v, -tanhClamp, tanhClamp)
	v2 := x.op(OpMul, F, v, v)
	p := x.f(tanhP[0])
	for _, k := range tanhP[1:] {
		p = x.fma(p, v2, x.f(k))
	}
	q := x.f(tanhQ[0])
	for _, k := range tanhQ[1:] {
		q = x.fma(q, v2, x.f(k))
	}
	return x.op(OpDiv, F, x.op(OpMul, F, p, v), q)
}

// act is activation a of v.
func (x *exprs) act(v int, a graph.Act) (int, error) {
	F := ir.F32
	one := x.f(1)
	switch a {
	case graph.ActReLU:
		return x.op(OpMax, F, v, x.f(0)), nil
	case graph.ActReLU2:
		r := x.op(OpMax, F, v, x.f(0))
		return x.op(OpMul, F, r, r), nil
	case graph.ActSiLU:
		return x.op(OpDiv, F, v, x.op(OpAdd, F, one, x.un(OpExp, F, x.neg(v)))), nil
	case graph.ActSigmoid:
		return x.sigmoid(v), nil
	case graph.ActGELUErf:
		av := x.op(OpMax, F, v, x.neg(v))
		z := x.op(OpMul, F, x.f(0.70710678118654752), av)
		t := x.op(OpDiv, F, one, x.fma(x.f(geluErfP), z, one))
		p := x.f(geluErfA[4])
		for i := 3; i >= 0; i-- {
			p = x.fma(p, t, x.f(geluErfA[i]))
		}
		q := x.op(OpMul, F, x.op(OpMul, F, p, t), x.un(OpExp, F, x.neg(x.op(OpMul, F, z, z))))
		return x.op(OpMul, F, x.f(0.5), x.op(OpSub, F, x.op(OpAdd, F, v, av), x.op(OpMul, F, av, q))), nil
	case graph.ActGELUTanh:
		u := x.op(OpMul, F, x.f(0.7978845608028654), x.op(OpAdd, F, v,
			x.op(OpMul, F, x.f(0.044715), x.op(OpMul, F, v, x.op(OpMul, F, v, v)))))
		e := x.un(OpExp, F, x.op(OpMul, F, x.f(2), u))
		th := x.op(OpSub, F, one, x.op(OpDiv, F, x.f(2), x.op(OpAdd, F, e, one)))
		return x.op(OpMul, F, x.op(OpMul, F, x.f(0.5), v), x.op(OpAdd, F, one, th)), nil
	}
	return 0, fmt.Errorf("no formula for activation %d yet", a)
}

// actMul is gated activation a of gate g and up u.
func (x *exprs) actMul(a graph.Act, g, u int) (int, error) {
	F := ir.F32
	switch a {
	case graph.ActSwiGLUOAI:
		gc := x.op(OpMin, F, g, x.f(7))
		qg := x.op(OpDiv, F, gc, x.op(OpAdd, F, x.f(1), x.un(OpExp, F, x.neg(x.op(OpMul, F, x.f(1.702), gc)))))
		return x.op(OpMul, F, qg, x.op(OpAdd, F, x.clamp(u, -7, 7), x.f(1))), nil
	case graph.ActSwiGLUClamp:
		s, err := x.act(x.op(OpMin, F, g, x.f(10)), graph.ActSiLU)
		return x.op(OpMul, F, s, x.clamp(u, -10, 10)), err
	case graph.ActSitu:
		bound := func(v int, c float32) int { return x.op(OpMul, F, x.f(c), x.tanh(x.op(OpMul, F, v, x.f(1/c)))) }
		return x.op(OpMul, F, x.op(OpMul, F, bound(g, 4), x.sigmoid(g)), bound(u, 25)), nil
	}
	s, err := x.act(g, a)
	return x.op(OpMul, F, s, u), err
}

// decomposeAct is an ungated activation, elementwise.
func decomposeAct(o graph.Op, c *Context) ([]Prim, error) {
	if xi := o.Attr.Act == graph.ActXIELU; len(o.In) != 1 || len(o.Out) != 1 || xi != (len(o.Weight) == 1) || len(o.Weight) > 1 {
		return nil, fmt.Errorf("an act of %d operands, %d weights", len(o.In), len(o.Weight))
	}
	sh := c.rows(o.Out[0])
	var x exprs
	if o.Attr.Act == graph.ActXIELU {
		// Apertus's xIELU, its four numbers alpha_p, alpha_n, beta and eps
		// in their effective form: alpha_p x^2 + beta x above zero, else
		// (expm1(min(x, eps)) - x) alpha_n + beta x.
		F := ir.F32
		v := x.at(0, sh, x.coord(0), x.coord(1))
		p := func(i uint32) int { return x.load(1, x.u(i)) }
		ap, an, beta, eps := p(0), p(1), p(2), p(3)
		bx := x.op(OpMul, F, beta, v)
		pos := x.op(OpAdd, F, x.op(OpMul, F, x.op(OpMul, F, ap, v), v), bx)
		em := x.op(OpSub, F, x.un(OpExp, F, x.op(OpMin, F, v, eps)), x.f(1))
		neg := x.op(OpAdd, F, x.op(OpMul, F, x.op(OpSub, F, em, v), an), bx)
		x.sel(F, x.op(OpLess, ir.Pred, x.f(0), v), pos, neg)
		w := o.Weight[0]
		return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0]), {Kind: OperandVector, Role: w.Role, Name: w.Role.String()}},
			Out: c.value(o.Out[0]), Expr: x.e, Facts: Facts{Extents: map[int]int{1: 4}}}}, nil
	}
	if _, err := x.act(x.at(0, sh, x.coord(0), x.coord(1)), o.Attr.Act); err != nil {
		return nil, err
	}
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0])}, Out: c.value(o.Out[0]), Expr: x.e}}, nil
}
