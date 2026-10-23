package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
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

// decomposeScale is the row times the stored scalar.
func decomposeScale(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 || len(o.Weight) != 1 {
		return nil, fmt.Errorf("a scale of %d operands, %d weights", len(o.In), len(o.Weight))
	}
	sh := c.rows(o.Out[0])
	var x exprs
	x.op(OpMul, ir.F32, x.at(0, sh, x.coord(0), x.coord(1)), x.at(1, []int{1}, x.u(0)))
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0]),
		{Kind: OperandVector, Role: o.Weight[0].Role, Name: o.Weight[0].Role.String()}}, Out: c.value(o.Out[0]), Expr: x.e}}, nil
}

// decomposeSlice copies Width elements from Offset: of each row, of each
// head of HeadDim where it is set, or of the bound vector where the op has
// no input.
func decomposeSlice(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	if len(o.Out) != 1 || a.Width < 1 || len(o.In) > 1 || (len(o.In) == 0) != (len(o.Weight) == 1) {
		return nil, fmt.Errorf("a slice of %d operands, %d weights", len(o.In), len(o.Weight))
	}
	U := ir.U32
	sh := c.rows(o.Out[0])
	var x exprs
	r, e := x.coord(0), x.coord(1)
	if len(o.In) == 0 {
		if sh[1] != a.Width {
			return nil, fmt.Errorf("a slice of %d from a table into %d", a.Width, sh[1])
		}
		w := o.Weight[0]
		if a.Offset+a.Width > w.K*max(1, w.Rows) {
			return nil, fmt.Errorf("a slice of %d from %d of a table of %d", a.Width, a.Offset, w.K*max(1, w.Rows))
		}
		x.load(0, x.op(OpAdd, U, x.u(uint32(a.Offset)), e))
		return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{{Kind: OperandVector, Role: w.Role, Name: w.Role.String()}},
			Out: c.value(o.Out[0]), Expr: x.e, Facts: Facts{Extents: map[int]int{0: w.K * max(1, w.Rows)}}}}, nil
	}
	in := c.rows(o.In[0])
	col := x.op(OpAdd, U, x.u(uint32(a.Offset)), e)
	want := a.Width
	if a.HeadDim > 0 {
		if in[1]%a.HeadDim != 0 || a.Offset+a.Width > a.HeadDim {
			return nil, fmt.Errorf("a slice of %d from %d in heads of %d", a.Width, a.Offset, a.HeadDim)
		}
		want = in[1] / a.HeadDim * a.Width
		h := x.op(OpDiv, U, e, x.u(uint32(a.Width)))
		col = x.op(OpAdd, U, x.op(OpMul, U, h, x.u(uint32(a.HeadDim))), x.op(OpAdd, U, x.u(uint32(a.Offset)), x.op(OpRem, U, e, x.u(uint32(a.Width)))))
	} else if a.Offset+a.Width > in[1] {
		return nil, fmt.Errorf("a slice of %d from %d of %d", a.Width, a.Offset, in[1])
	}
	if sh[1] != want {
		return nil, fmt.Errorf("a slice of %d into %d", want, sh[1])
	}
	x.at(0, in, r, col)
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0])}, Out: c.value(o.Out[0]), Expr: x.e}}, nil
}

// decomposeTile is the row repeated: element i of the output its input's
// element i mod Span (the whole row where Span is 0).
func decomposeTile(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 || len(o.Weight) != 0 {
		return nil, fmt.Errorf("a tile of %d operands", len(o.In))
	}
	in, sh := c.rows(o.In[0]), c.rows(o.Out[0])
	span := o.Attr.Span
	if span == 0 {
		span = in[1]
	}
	if span < 1 || span > in[1] || sh[1]%span != 0 {
		return nil, fmt.Errorf("a tile of %d over a row of %d into %d", span, in[1], sh[1])
	}
	var x exprs
	x.at(0, in, x.coord(0), x.op(OpRem, ir.U32, x.coord(1), x.u(uint32(span))))
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0])}, Out: c.value(o.Out[0]), Expr: x.e}}, nil
}

// decomposeConcat is its inputs end to end: each input's elements at its
// offset, one store a input.
func decomposeConcat(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) < 1 || len(o.Out) != 1 || len(o.Weight) != 0 {
		return nil, fmt.Errorf("a concatenation of %d operands", len(o.In))
	}
	out := c.value(o.Out[0])
	osh := c.rows(o.Out[0])
	var prims []Prim
	off := 0
	for _, v := range o.In {
		sh := c.rows(v)
		var x exprs
		r, i := x.coord(0), x.coord(1)
		val := x.at(0, sh, r, i)
		at := x.op(OpAdd, ir.U32, x.u(uint32(off)), i)
		prims = append(prims, Prim{Kind: PrimMap, Shape: sh, In: []Operand{c.value(v)}, Expr: x.e,
			Stores: []PrimStore{{Out: out, Ax: []int{r, at}, Ext: osh, Value: val}}})
		off += sh[1]
	}
	if off != osh[1] {
		return nil, fmt.Errorf("a concatenation of %d into %d", off, osh[1])
	}
	return prims, nil
}

// decomposePool is the step's live rows pooled into every row: the first
// row's, the last live one's, or their mean -- their sum in order over the
// live count. The live count is the positions table's first word.
func decomposePool(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 || c.width(o.In[0]) != c.width(o.Out[0]) {
		return nil, fmt.Errorf("a pool of %d operands", len(o.In))
	}
	U, F := ir.U32, ir.F32
	sh := c.rows(o.Out[0])
	R, W := sh[0], sh[1]
	in := []Operand{c.value(o.In[0]), tablePos}
	facts := Facts{Extents: map[int]int{1: R + 1}}
	live := func(x *exprs) int { return x.op(OpMax, U, x.load(1, x.u(0)), x.u(1)) }
	if p := jlm.Pooling(o.Attr.Pooling); p == jlm.PoolCLS || p == jlm.PoolLast {
		var x exprs
		row := x.u(0)
		if p == jlm.PoolLast {
			row = x.op(OpSub, U, live(&x), x.u(1))
		}
		x.at(0, sh, row, x.coord(1))
		return []Prim{{Kind: PrimMap, Shape: sh, In: in, Out: c.value(o.Out[0]), Expr: x.e, Facts: facts}}, nil
	}
	sum := Operand{Kind: OperandScratch, Name: c.value(o.Out[0]).Name + ".sum", Temp: true}
	var s exprs
	u := s.coord(2)
	v := s.at(0, sh, s.op(OpMin, U, u, s.u(uint32(R-1))), s.coord(1))
	s.sel(F, s.op(OpLess, ir.Pred, u, live(&s)), v, s.f(0))
	var m exprs
	n := m.un(OpCvtF32, F, m.un(OpBitcast, ir.I32, live(&m)))
	m.op(OpDiv, F, m.at(0, []int{R, W}, m.coord(0), m.coord(1)), n)
	return []Prim{
		{Kind: PrimReduce, Shape: []int{R, W, R}, In: in, Out: sum, Combine: OpAdd, Expr: s.e, Facts: facts},
		{Kind: PrimMap, Shape: sh, In: []Operand{sum, tablePos}, Out: c.value(o.Out[0]), Expr: m.e, Facts: facts},
	}, nil
}
