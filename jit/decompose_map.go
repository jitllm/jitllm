package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/ir"
)

// softplus is log(1 + e^z) as main composes it, no logarithm among the
// primitives: max(z, 0) + 2w(1 + w^2/3 + ... + w^8/9), w = e^-|z|/(2 + e^-|z|).
func (x *exprs) softplus(z int) int {
	F := ir.F32
	az := x.op(OpMax, F, z, x.neg(z))
	e := x.un(OpExp, F, x.neg(az))
	w := x.op(OpDiv, F, e, x.op(OpAdd, F, x.f(2), e))
	w2 := x.op(OpMul, F, w, w)
	s := x.f(1.0 / 9)
	for _, k := range []float32{1.0 / 7, 1.0 / 5, 1.0 / 3, 1} {
		s = x.fma(s, w2, x.f(k))
	}
	return x.op(OpAdd, F, x.op(OpMax, F, z, x.f(0)), x.op(OpMul, F, x.f(2), x.op(OpMul, F, w, s)))
}

// mapFn is f over operands a.
func (x *exprs) mapFn(f graph.MapFn, a []int) (int, error) {
	F := ir.F32
	switch f {
	case graph.MapAdd:
		return x.op(OpAdd, F, a[0], a[1]), nil
	case graph.MapSub:
		return x.op(OpSub, F, a[0], a[1]), nil
	case graph.MapMul:
		return x.op(OpMul, F, a[0], a[1]), nil
	case graph.MapDiv:
		return x.op(OpDiv, F, a[0], a[1]), nil
	case graph.MapMax:
		return x.op(OpMax, F, a[0], a[1]), nil
	case graph.MapMin:
		return x.op(OpMin, F, a[0], a[1]), nil
	case graph.MapExp:
		return x.un(OpExp, F, a[0]), nil
	case graph.MapSigmoid:
		return x.sigmoid(a[0]), nil
	case graph.MapSoftplus:
		return x.softplus(a[0]), nil
	case graph.MapSqrt:
		return x.un(OpSqrt, F, a[0]), nil
	case graph.MapWhere:
		return x.sel(F, x.op(OpLess, ir.Pred, x.f(0), a[0]), a[1], a[2]), nil
	case graph.MapLess:
		return x.sel(F, x.op(OpLess, ir.Pred, a[0], a[1]), x.f(1), x.f(0)), nil
	case graph.MapFma:
		return x.fma(a[0], a[1], a[2]), nil
	}
	return 0, fmt.Errorf("no formula for the map %v", f)
}

// decomposeMap is an elementwise function of its operands -- its inputs,
// then its bound vector, then its constant -- an operand narrower than the
// output read at element i/(w/n), an index row read as its float.
func decomposeMap(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.Out) != 1 || len(o.Weight) > 1 {
		return nil, fmt.Errorf("a map of %d outputs, %d weights", len(o.Out), len(o.Weight))
	}
	if c.Graph.Values[o.Out[0]].Elem != graph.ElemF32 {
		return nil, fmt.Errorf("a map to %s values", c.Graph.Values[o.Out[0]].Name)
	}
	sh := c.rows(o.Out[0])
	R, W := sh[0], sh[1]
	var x exprs
	r, i := x.coord(0), x.coord(1)
	at := func(j, n int) (int, error) {
		if n <= 0 || W%n != 0 {
			return 0, fmt.Errorf("an operand of %d elements under a map of %d", n, W)
		}
		e := i
		if n != W {
			e = x.op(OpDiv, ir.U32, i, x.u(uint32(W/n)))
		}
		return x.at(j, []int{R, n}, r, e), nil
	}
	var in []Operand
	var args []int
	for _, v := range o.In {
		a, err := at(len(in), c.width(v))
		if err != nil {
			return nil, err
		}
		if c.Graph.Values[v].Elem == graph.ElemU32 {
			// An index row reads as its float, as main's map takes it.
			a = x.un(OpCvtF32, ir.F32, x.un(OpBitcast, ir.I32, a))
		}
		in, args = append(in, c.value(v)), append(args, a)
	}
	for _, w := range o.Weight {
		if w.Rows > 1 {
			return nil, fmt.Errorf("a map over a matrix")
		}
		e := i
		if w.K != W {
			if w.K <= 0 || W%w.K != 0 {
				return nil, fmt.Errorf("a vector of %d under a map of %d", w.K, W)
			}
			e = x.op(OpDiv, ir.U32, i, x.u(uint32(W/w.K)))
		}
		args = append(args, x.at(len(in), []int{w.K}, e))
		in = append(in, Operand{Kind: OperandVector, Role: w.Role, Name: w.Role.String()})
	}
	n := o.Attr.Fn.Arity()
	if len(args) < n {
		args = append(args, x.f(float32(o.Attr.Const)))
	}
	if len(args) != n {
		return nil, fmt.Errorf("a %v of %d operands", o.Attr.Fn, len(args))
	}
	if _, err := x.mapFn(o.Attr.Fn, args); err != nil {
		return nil, err
	}
	return []Prim{{Kind: PrimMap, Shape: sh, In: in, Out: c.value(o.Out[0]), Expr: x.e}}, nil
}

// decomposeLayerNorm is a LayerNorm over each row: the row's mean, its
// variance about the mean, then each element's deviation over the root of
// the variance plus Eps, times the weight, plus the bias where there is one.
func decomposeLayerNorm(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 || len(o.Weight) < 1 || len(o.Weight) > 2 {
		return nil, fmt.Errorf("a layer norm of %d weights", len(o.Weight))
	}
	F := ir.F32
	sh := c.rows(o.In[0])
	R, W := sh[0], sh[1]
	name := c.Graph.Values[o.Out[0]].Name
	sum := Operand{Kind: OperandScratch, Name: name + ".sum"}
	dev := Operand{Kind: OperandScratch, Name: name + ".devsq"}
	x := c.value(o.In[0])
	vec := func(w graph.Binding) Operand {
		return Operand{Kind: OperandVector, Role: w.Role, Name: w.Role.String()}
	}

	var s exprs
	s.at(0, sh, s.coord(0), s.coord(1))

	var v exprs
	vr := v.coord(0)
	d := v.op(OpSub, F, v.at(0, sh, vr, v.coord(1)), v.op(OpMul, F, v.at(1, []int{R}, vr), v.f(1/float32(W))))
	v.op(OpMul, F, d, d)

	var m exprs
	r, i := m.coord(0), m.coord(1)
	y := m.op(OpSub, F, m.at(0, sh, r, i), m.op(OpMul, F, m.at(1, []int{R}, r), m.f(1/float32(W))))
	vv := m.op(OpMul, F, m.at(2, []int{R}, r), m.f(1/float32(W)))
	y = m.op(OpMul, F, y, m.op(OpDiv, F, m.f(1), m.un(OpSqrt, F, m.op(OpAdd, F, vv, m.f(float32(o.Attr.Eps))))))
	y = m.op(OpMul, F, y, m.at(3, []int{W}, i))
	in := []Operand{x, sum, dev, vec(o.Weight[0])}
	if len(o.Weight) == 2 {
		m.op(OpAdd, F, y, m.at(4, []int{W}, i))
		in = append(in, vec(o.Weight[1]))
	}
	return []Prim{
		{Kind: PrimReduce, Shape: sh, In: []Operand{x}, Out: sum, Combine: OpAdd, Expr: s.e},
		{Kind: PrimReduce, Shape: sh, In: []Operand{x, sum}, Out: dev, Combine: OpAdd, Expr: v.e},
		{Kind: PrimMap, Shape: sh, In: in, Out: c.value(o.Out[0]), Expr: m.e},
	}, nil
}
