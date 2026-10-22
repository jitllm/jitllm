package jit

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/ir"
)

// groups is an op's row seen as groups of span elements: the row's width
// and the groups' count (one group where span is 0).
func (c *Context) groups(v graph.ValueID, span int) (W, G, S int, err error) {
	W = c.width(v)
	S = span
	if S == 0 {
		S = W
	}
	if S <= 0 || W%S != 0 {
		return 0, 0, 0, fmt.Errorf("groups of %d over a row of %d", span, W)
	}
	return W, W / S, S, nil
}

// decomposeReduce combines each group of Span elements in order from the
// first.
func decomposeReduce(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 || o.Attr.Order != graph.OrderSeq {
		return nil, fmt.Errorf("a reduce of %d operands in order %d", len(o.In), o.Attr.Order)
	}
	_, G, S, err := c.groups(o.In[0], o.Attr.Span)
	if err != nil {
		return nil, err
	}
	var comb Op
	init := int64(0)
	switch o.Attr.Fn {
	case graph.MapAdd:
		comb = OpAdd
	case graph.MapMax:
		comb, init = OpMax, int64(math.Float32bits(float32(math.Inf(-1))))
	case graph.MapMin:
		comb, init = OpMin, int64(math.Float32bits(float32(math.Inf(1))))
	default:
		return nil, fmt.Errorf("a reduce by %v", o.Attr.Fn)
	}
	R := c.Step.Rows
	sh := []int{R, G, S}
	var x exprs
	if G == 1 {
		sh = []int{R, S}
		x.at(0, sh, x.coord(0), x.coord(1))
	} else {
		x.at(0, sh, x.coord(0), x.coord(1), x.coord(2))
	}
	return []Prim{{Kind: PrimReduce, Shape: sh, In: []Operand{c.value(o.In[0])}, Out: c.value(o.Out[0]), Combine: comb,
		Init: init, Expr: x.e}}, nil
}

// decomposeTake is In[0]'s element at each index of In[1], held to the row.
func decomposeTake(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 2 || len(o.Out) != 1 {
		return nil, fmt.Errorf("a take of %d operands", len(o.In))
	}
	U := ir.U32
	R, W, K := c.Step.Rows, c.width(o.In[0]), c.width(o.In[1])
	var x exprs
	r, j := x.coord(0), x.coord(1)
	ix := x.op(OpMin, U, x.at(1, []int{R, K}, r, j), x.u(uint32(W-1)))
	x.load(0, x.op(OpAdd, U, x.op(OpMul, U, r, x.u(uint32(W))), ix))
	return []Prim{{Kind: PrimMap, Shape: []int{R, K}, In: []Operand{c.value(o.In[0]), c.value(o.In[1])}, Out: c.value(o.Out[0]),
		Expr: x.e, Facts: Facts{Extents: map[int]int{0: R * W}}}}, nil
}

// decomposeTopKRanked is the TopK largest of each group: each element's
// rank among its group -- how many are larger, or equal and before it --
// then each of the K places the index and the value of the element ranked
// there, the one element of the group whose rank it is.
func decomposeTopKRanked(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 2 || o.Attr.TopK <= 0 {
		return nil, fmt.Errorf("a top %d of %d operands", o.Attr.TopK, len(o.In))
	}
	_, G, S, err := c.groups(o.In[0], o.Attr.Span)
	if err != nil {
		return nil, err
	}
	K := o.Attr.TopK
	if K > S {
		return nil, fmt.Errorf("a top %d of groups of %d", K, S)
	}
	U, I, F := ir.U32, ir.I32, ir.F32
	R := c.Step.Rows
	in := c.value(o.In[0])
	rank := Operand{Kind: OperandScratch, Name: c.value(o.Out[1]).Name + ".rank", Elem: I, Temp: true}

	// The element at place s: its index, and its value, alone of the group.
	at := func(x *exprs, val func(r, g, i int) int, t ir.Type, zero int) {
		r, g, s, i := x.coord(0), x.coord(1), x.coord(2), x.coord(3)
		k := x.at(1, []int{R, G, S}, r, g, i)
		ks := x.un(OpBitcast, I, s)
		x.sel(t, x.op(OpLess, ir.Pred, k, ks), zero, x.sel(t, x.op(OpLess, ir.Pred, k, x.op(OpAdd, I, ks, x.i(1))), val(r, g, i), zero))
	}
	var ix exprs
	at(&ix, func(_, _, i int) int { return i }, U, ix.u(0))
	var vx exprs
	at(&vx, func(r, g, i int) int { return vx.at(0, []int{R, G, S}, r, g, i) }, F, vx.f(0))

	sel := c.value(o.Out[0])
	sel.Elem = U
	return []Prim{
		rankPrim(c, in, rank, G, S),
		{Kind: PrimReduce, Shape: []int{R, G, K, S}, In: []Operand{in, rank}, Out: sel, Combine: OpAdd, Type: U, Expr: ix.e},
		{Kind: PrimReduce, Shape: []int{R, G, K, S}, In: []Operand{in, rank}, Out: c.value(o.Out[1]), Combine: OpAdd, Expr: vx.e},
	}, nil
}

// decomposeGather is the row once per pair of the selection, scaled by the
// pair's weight where the op weights its inputs.
func decomposeGather(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.Out) != 1 || len(o.In) < 2 || o.Attr.WeightIn != (len(o.In) == 3) {
		return nil, fmt.Errorf("a gather of %d operands", len(o.In))
	}
	R, W, K := c.Step.Rows, c.width(o.In[0]), c.width(o.In[1])
	var x exprs
	r, j, e := x.coord(0), x.coord(1), x.coord(2)
	v := x.at(0, []int{R, W}, r, e)
	in := []Operand{c.value(o.In[0])}
	if o.Attr.WeightIn {
		x.op(OpMul, ir.F32, v, x.at(1, []int{R, K}, r, j))
		in = append(in, c.value(o.In[2]))
	}
	return []Prim{{Kind: PrimMap, Shape: []int{R, K, W}, In: in, Out: c.value(o.Out[0]), Expr: x.e}}, nil
}

// decomposeCombine is the selection's pairs summed in its order, each by its
// weight (and the expert's scale where the op binds one), unweighted where
// the gather weighted the inputs.
func decomposeCombine(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.Out) != 1 || len(o.In) < 2 || len(o.Weight) > 1 {
		return nil, fmt.Errorf("a combine of %d operands", len(o.In))
	}
	R, W := c.Step.Rows, c.width(o.Out[0])
	K := c.width(o.In[1])
	if c.width(o.In[0]) != K*W {
		return nil, fmt.Errorf("a combine of %d pairs of %d over %d", K, W, c.width(o.In[0]))
	}
	U, F := ir.U32, ir.F32
	var x exprs
	r, e, j := x.coord(0), x.coord(1), x.coord(2)
	v := x.at(0, []int{R, K, W}, r, j, e)
	in := []Operand{c.value(o.In[0])}
	if !o.Attr.WeightIn {
		if len(o.In) != 3 {
			return nil, fmt.Errorf("a weighted combine without its weights")
		}
		w := x.at(1, []int{R, K}, r, j)
		in = append(in, c.value(o.In[2]))
		if len(o.Weight) == 1 {
			sel := x.at(2, []int{R, K}, r, j)
			w = x.op(OpMul, F, w, x.load(3, x.op(OpMin, U, sel, x.u(uint32(o.Weight[0].K-1)))))
			in = append(in, c.value(o.In[1]), Operand{Kind: OperandVector, Role: o.Weight[0].Role, Name: o.Weight[0].Role.String()})
		}
		x.op(OpMul, F, w, v)
	}
	p := Prim{Kind: PrimReduce, Shape: []int{R, W, K}, In: in, Out: c.value(o.Out[0]), Combine: OpAdd, Expr: x.e}
	if len(o.Weight) == 1 {
		p.Facts.Extents = map[int]int{3: o.Weight[0].K}
	}
	return []Prim{p}, nil
}

// decomposeExpertMatVec is a matvec per pair of the selection: pair j's row
// of the input against the sheet of the bank the selection names at j --
// over the input's int8 form where the format and the form allow, as a
// plain matvec runs, else the weights decoded -- and the bank's bias row
// for that expert added after where the op binds one.
func decomposeExpertMatVec(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 2 || len(o.Out) != 1 || len(o.Weight) < 1 || len(o.Weight) > 2 {
		return nil, fmt.Errorf("an expert matvec of %d operands, %d weights", len(o.In), len(o.Weight))
	}
	w := o.Weight[0]
	E, P, K, rows := w.Sheets, c.width(o.In[1]), w.K, w.Rows
	if E < 1 || c.width(o.In[0]) != P*K || c.width(o.Out[0]) != P*rows {
		return nil, fmt.Errorf("an expert matvec of %d pairs of %d over %d", P, K, c.width(o.In[0]))
	}
	wo, err := c.weight(w)
	if err != nil {
		return nil, err
	}
	out := c.value(o.Out[0])
	if len(o.Weight) == 2 {
		out = Operand{Kind: OperandScratch, Name: out.Name + ".unbiased", Temp: true}
	}
	var prims []Prim
	if l, ok := jlm.Layout(wo.Format); ok {
		if ps, ok := expertInt8(o, c, wo, l, out); ok {
			prims = ps
		}
	}
	if prims == nil {
		p, err := expertDecoded(o, c, wo, out)
		if err != nil {
			return nil, err
		}
		prims = []Prim{p}
	}
	if len(o.Weight) == 2 {
		U := ir.U32
		R := c.Step.Rows
		var x exprs
		r, j, row := x.coord(0), x.coord(1), x.coord(2)
		sel := x.op(OpMin, U, x.at(1, []int{R, P}, r, j), x.u(uint32(E-1)))
		b := x.load(2, x.op(OpAdd, U, x.op(OpMul, U, sel, x.u(uint32(rows))), row))
		x.op(OpAdd, ir.F32, x.at(0, []int{R, P, rows}, r, j, row), b)
		bias := Operand{Kind: OperandVector, Role: o.Weight[1].Role, Name: o.Weight[1].Role.String()}
		prims = append(prims, Prim{Kind: PrimMap, Shape: []int{R, P, rows}, In: []Operand{out, c.value(o.In[1]), bias},
			Out: c.value(o.Out[0]), Expr: x.e, Facts: Facts{Extents: map[int]int{2: E * rows}}})
	}
	return prims, nil
}

// expertDecoded is each pair's sum over K of its sheet's weights, decoded,
// times the pair's input.
func expertDecoded(o graph.Op, c *Context, wo Operand, out Operand) (Prim, error) {
	w := o.Weight[0]
	E, P, K, rows := w.Sheets, c.width(o.In[1]), w.K, w.Rows
	U := ir.U32
	R := c.Step.Rows
	var x exprs
	r, j, row, col := x.coord(0), x.coord(1), x.coord(2), x.coord(3)
	in := []Operand{c.value(o.In[0]), c.value(o.In[1])}
	facts := Facts{Extents: map[int]int{0: R * P * K, 1: R * P}}
	sel := x.op(OpMin, U, x.at(1, []int{R, P}, r, j), x.u(uint32(E-1)))
	xv := x.at(0, []int{R, P, K}, r, j, col)
	var wv int
	if jlm.Verbatim(wo.Format) {
		v, err := weightElement(&x, wo, E*rows, K, &in, &facts, x.op(OpAdd, U, x.op(OpMul, U, sel, x.u(uint32(rows))), row), col)
		if err != nil {
			return Prim{}, err
		}
		wv = v
	} else {
		l, ok := jlm.Layout(wo.Format)
		if !ok || K%(l.Sub*l.SuperSub) != 0 {
			return Prim{}, fmt.Errorf("no reading rules for a bank of rows of %d", K)
		}
		p := packedOperands(&x, wo, l, rows, K, &in, &facts)
		p.bank(sel, E)
		wv = jlm.DecodeElement[int](p, l, rows, row, col)
	}
	x.op(OpMul, ir.F32, wv, xv)
	return Prim{Kind: PrimReduce, Shape: []int{R, P, rows, K}, In: in, Out: out, Combine: OpAdd, Expr: x.e, Facts: facts}, nil
}

// expertInt8 is decomposeMatVecInt8 a pair at a time: pair j's slice of the
// input's int8 form against the codes of the sheet the selection names.
func expertInt8(o graph.Op, c *Context, wo Operand, l jlm.TypeLayout, out Operand) (prims []Prim, ok bool) {
	w, q, v := o.Weight[0], c.Int8, o.In[0]
	if q == nil || q.Word != 4 || l.Sub%q.Sums != 0 || w.K%(l.Sub*l.SuperSub) != 0 {
		return nil, false
	}
	win := c.window(v)
	if win%l.Sub != 0 || w.K%win != 0 {
		return nil, false
	}
	if !c.formed[v] {
		c.formed[v] = true
		prims = c.int8(v)
	}
	U, I, F := ir.U32, ir.I32, ir.F32
	E, P, R, rows, K := w.Sheets, c.width(o.In[1]), c.Step.Rows, w.Rows, w.K
	nsub, nwd, nw, ns := K/l.Sub, P*K/q.Word, P*K/win, P*K/q.Sums
	x := c.value(v)
	words := Operand{Kind: OperandScratch, Name: x.Name + ".q8a", Elem: U}
	scales := Operand{Kind: OperandScratch, Name: x.Name + ".q8x"}
	dot := Operand{Kind: OperandScratch, Name: c.value(o.Out[0]).Name + ".dot", Elem: I, Temp: true}
	sel := c.value(o.In[1])

	var d exprs
	r, j, row, s, t := d.coord(0), d.coord(1), d.coord(2), d.coord(3), d.coord(4)
	in := []Operand{words, sel}
	facts := Facts{Extents: map[int]int{0: R * nwd, 1: R * P}}
	p := packedOperands(&d, wo, l, rows, K, &in, &facts)
	p.bank(d.op(OpMin, U, d.at(1, []int{R, P}, r, j), d.u(uint32(E-1))), E)
	at := func(col int) int { return d.op(OpAdd, U, d.op(OpMul, U, j, d.u(uint32(K))), col) }
	per := l.Sub
	if l.WordCodes() {
		per = l.Sub / 4
		col := d.op(OpAdd, U, d.op(OpMul, U, s, d.u(uint32(l.Sub))), d.op(OpMul, U, t, d.u(4)))
		wq, _ := jlm.DecodeWord[int](p, l, rows, row, col)
		xw := d.load(0, d.op(OpAdd, U, d.op(OpMul, U, r, d.u(uint32(nwd))), d.op(OpShr, U, at(col), d.u(2))))
		d.add(Expr{Op: OpDot4, Type: I, A: wq, B: xw, C: d.i(0)})
	} else {
		col := d.op(OpAdd, U, d.op(OpMul, U, s, d.u(uint32(l.Sub))), t)
		code, _, _ := jlm.DecodeParts[int](p, l, rows, row, col)
		xw := d.load(0, d.op(OpAdd, U, d.op(OpMul, U, r, d.u(uint32(nwd))), d.op(OpShr, U, at(col), d.u(2))))
		xb := d.op(OpAnd, U, d.op(OpShr, U, xw, d.op(OpMul, U, d.op(OpAnd, U, col, d.u(3)), d.u(8))), d.u(0xff))
		xc := d.op(OpSub, U, d.op(OpXor, U, xb, d.u(0x80)), d.u(0x80))
		d.op(OpMul, I, d.un(OpBitcast, I, code), d.un(OpBitcast, I, xc))
	}
	prims = append(prims, Prim{Kind: PrimReduce, Shape: []int{R, P, rows, nsub, per}, In: in, Out: dot, Combine: OpAdd, Type: I,
		Expr: d.e, Facts: facts})

	var e exprs
	er, ej, erow, es := e.coord(0), e.coord(1), e.coord(2), e.coord(3)
	ein := []Operand{dot, scales, sel}
	efacts := Facts{Extents: map[int]int{0: R * P * rows * nsub, 1: R * (nw + ns), 2: R * P}}
	ep := packedOperands(&e, wo, l, rows, K, &ein, &efacts)
	ep.bank(e.op(OpMin, U, e.at(2, []int{R, P}, er, ej), e.u(uint32(E-1))), E)
	first := e.op(OpMul, U, es, e.u(uint32(l.Sub)))
	_, scale, bias := jlm.DecodeParts[int](ep, l, rows, erow, first)
	df := e.un(OpCvtF32, F, e.at(0, []int{R, P, rows, nsub}, er, ej, erow, es))
	col := e.op(OpAdd, U, e.op(OpMul, U, ej, e.u(uint32(K))), first)
	xs := e.load(1, e.op(OpAdd, U, e.op(OpMul, U, er, e.u(uint32(nw))), e.op(OpDiv, U, col, e.u(uint32(win)))))
	sumAt := e.op(OpAdd, U, e.u(uint32(R*nw)), e.op(OpAdd, U, e.op(OpMul, U, er, e.u(uint32(ns))), e.op(OpDiv, U, col, e.u(uint32(q.Sums)))))
	csum := e.load(1, sumAt)
	for k := 1; k < l.Sub/q.Sums; k++ {
		csum = e.op(OpAdd, F, csum, e.load(1, e.op(OpAdd, U, sumAt, e.u(uint32(k)))))
	}
	e.op(OpMul, F, xs, e.op(OpSub, F, e.op(OpMul, F, scale, df), e.op(OpMul, F, bias, csum)))
	return append(prims, Prim{Kind: PrimReduce, Shape: []int{R, P, rows, nsub}, In: ein, Out: out, Combine: OpAdd,
		Expr: e.e, Facts: efacts}), true
}

// decomposeRank is each element's place among its group, largest first,
// equal values to the lower index: how many of the group come before it.
func decomposeRank(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 {
		return nil, fmt.Errorf("a rank of %d operands", len(o.In))
	}
	_, G, S, err := c.groups(o.In[0], o.Attr.Span)
	if err != nil {
		return nil, err
	}
	out := c.value(o.Out[0])
	out.Elem = ir.U32
	return []Prim{rankPrim(c, c.value(o.In[0]), out, G, S)}, nil
}

// rankPrim is element i's count of the j of its group with x_i < x_j, or
// x_j == x_i and j < i.
func rankPrim(c *Context, in, out Operand, G, S int) Prim {
	I := out.Elem
	if I == 0 {
		I = ir.I32
	}
	one, zero := func(x *exprs) int { return x.add(Expr{Op: OpConst, Type: I, Imm: 1}) }, func(x *exprs) int { return x.add(Expr{Op: OpConst, Type: I}) }
	R := c.Step.Rows
	var rk exprs
	r, g, i, j := rk.coord(0), rk.coord(1), rk.coord(2), rk.coord(3)
	xi, xj := rk.at(0, []int{R, G, S}, r, g, i), rk.at(0, []int{R, G, S}, r, g, j)
	before := rk.sel(I, rk.op(OpLess, ir.Pred, j, i), one(&rk), zero(&rk))
	rk.sel(I, rk.op(OpLess, ir.Pred, xi, xj), one(&rk), rk.sel(I, rk.op(OpLess, ir.Pred, xj, xi), zero(&rk), before))
	return Prim{Kind: PrimReduce, Shape: []int{R, G, S, S}, In: []Operand{in}, Out: out, Combine: OpAdd, Type: I, Expr: rk.e}
}

// decomposeClamp is the input held to [-Bound, Bound], or to the bounds
// table's pair at Offset.
func decomposeClamp(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 || len(o.Weight) > 1 {
		return nil, fmt.Errorf("a clamp of %d operands", len(o.In))
	}
	F := ir.F32
	sh := c.rows(o.Out[0])
	var x exprs
	v := x.at(0, sh, x.coord(0), x.coord(1))
	in := []Operand{c.value(o.In[0])}
	lo, hi := x.f(float32(-o.Attr.Bound)), x.f(float32(o.Attr.Bound))
	if len(o.Weight) == 1 {
		in = append(in, Operand{Kind: OperandVector, Role: o.Weight[0].Role, Name: o.Weight[0].Role.String()})
		lo, hi = x.load(1, x.u(uint32(o.Attr.Offset))), x.load(1, x.u(uint32(o.Attr.Offset+1)))
	}
	x.op(OpMax, F, x.op(OpMin, F, v, hi), lo)
	p := Prim{Kind: PrimMap, Shape: sh, In: in, Out: c.value(o.Out[0]), Expr: x.e}
	if len(o.Weight) == 1 {
		p.Facts.Extents = map[int]int{1: o.Weight[0].K}
	}
	return []Prim{p}, nil
}

// decomposeStateConv is a causal depthwise convolution over the part's
// window state ("rec.conv", Taps-1 rows of the channels): each channel's
// row and window against its taps, the newest tap the row's; then the
// window less its oldest row and with the row, into a scratch row and
// back, so no kernel reads the window while writing it.
func decomposeStateConv(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 || len(o.Weight) != 1 || c.Step.Rows != 1 {
		return nil, fmt.Errorf("a state convolution of %d operands over %d rows", len(o.In), c.Step.Rows)
	}
	U, F := ir.U32, ir.F32
	C, T := c.width(o.In[0]), o.Attr.Taps
	if T < 2 || o.Weight[0].K != T || o.Weight[0].Rows != C {
		return nil, fmt.Errorf("a state convolution of %d taps over %d channels", T, C)
	}
	wo, err := c.weight(o.Weight[0])
	if err != nil {
		return nil, err
	}
	if !jlm.Verbatim(wo.Format) {
		return nil, fmt.Errorf("a state convolution over %s taps", wo.Format)
	}
	state := Operand{Kind: OperandTable, Name: "rec.conv", PerPart: true}
	next := Operand{Kind: OperandScratch, Name: c.value(o.Out[0]).Name + ".window"}
	in := c.value(o.In[0])
	W := (T - 1) * C

	var x exprs
	ch := x.coord(1)
	tapIn := []Operand{in, state}
	facts := Facts{Extents: map[int]int{1: W}}
	tap := func(t int) int {
		v, _ := weightElement(&x, wo, C, T, &tapIn, &facts, ch, x.u(uint32(t)))
		return v
	}
	acc := x.op(OpMul, F, x.at(0, []int{1, C}, x.coord(0), ch), tap(T-1))
	for t := 0; t < T-1; t++ {
		acc = x.op(OpAdd, F, acc, x.op(OpMul, F, x.load(1, x.op(OpAdd, U, x.u(uint32(t*C)), ch)), tap(t)))
	}

	var n exprs
	nt, nc := n.coord(0), n.coord(1)
	older := n.load(1, n.op(OpAdd, U, n.op(OpMul, U, n.op(OpMin, U, n.op(OpAdd, U, nt, n.u(1)), n.u(uint32(T-2))), n.u(uint32(C))), nc))
	n.sel(F, n.op(OpLess, ir.Pred, n.op(OpAdd, U, nt, n.u(1)), n.u(uint32(T-1))), older, n.at(0, []int{1, C}, n.u(0), nc))

	var k exprs
	kt, kc := k.coord(0), k.coord(1)
	ix := k.op(OpAdd, U, k.op(OpMul, U, kt, k.u(uint32(C))), kc)
	v := k.load(0, ix)
	return []Prim{
		{Kind: PrimMap, Shape: []int{1, C}, In: tapIn, Out: c.value(o.Out[0]), Expr: x.e, Facts: facts},
		{Kind: PrimMap, Shape: []int{T - 1, C}, In: []Operand{in, state}, Out: next, Expr: n.e, Facts: Facts{Extents: map[int]int{1: W}}},
		{Kind: PrimMap, Shape: []int{T - 1, C}, In: []Operand{next}, Expr: k.e, Facts: Facts{Extents: map[int]int{0: W}},
			Stores: []PrimStore{{Out: state, Index: ix, Value: v, Extent: W}}},
	}, nil
}

// decomposeStateScan is Mamba-2's selective state update over the part's
// state ("rec.ssm", value heads by x's width by HeadDim): the advanced
// state into a scratch, its rows against C, the output with D times x,
// and the state copied back, so no kernel reads the state while writing it.
func decomposeStateScan(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	if len(o.In) != 3 || len(o.Out) != 1 || len(o.Weight) != 1 || c.Step.Rows != 1 || a.NHead < 1 || a.NKVHead < 1 ||
		a.HeadDim < 1 || a.NHead%a.NKVHead != 0 {
		return nil, fmt.Errorf("a state scan of %d operands over %d rows", len(o.In), c.Step.Rows)
	}
	U, F := ir.U32, ir.F32
	vh, kh, kd := a.NHead, a.NKVHead, a.HeadDim
	inner := c.width(o.Out[0])
	if inner%vh != 0 || c.width(o.In[0]) != inner+2*kh*kd || c.width(o.In[1]) != vh || c.width(o.In[2]) != vh {
		return nil, fmt.Errorf("a state scan of %d value heads over %d", vh, c.width(o.In[0]))
	}
	vd, rep := inner/vh, vh/kh
	N := vh * vd * kd
	xs, dt, decay := c.value(o.In[0]), c.value(o.In[1]), c.value(o.In[2])
	state := Operand{Kind: OperandTable, Name: "rec.ssm", PerPart: true}
	name := c.value(o.Out[0]).Name
	next := Operand{Kind: OperandScratch, Name: name + ".state"}
	acc := Operand{Kind: OperandScratch, Name: name + ".sc", Temp: true}
	key := func(x *exprs, v int) int {
		if a.KeyTiled {
			return x.op(OpRem, U, v, x.u(uint32(kh)))
		}
		return x.op(OpDiv, U, v, x.u(uint32(rep)))
	}
	flat := func(x *exprs, v, j, i int) int {
		return x.op(OpAdd, U, x.op(OpMul, U, x.op(OpAdd, U, x.op(OpMul, U, v, x.u(uint32(vd))), j), x.u(uint32(kd))), i)
	}
	sh := []int{vh, vd, kd}

	// S*decay + (x*dt)*B.
	var n exprs
	v, j, i := n.coord(0), n.coord(1), n.coord(2)
	xv := n.load(0, n.op(OpAdd, U, n.u(uint32(2*kh*kd)), n.op(OpAdd, U, n.op(OpMul, U, v, n.u(uint32(vd))), j)))
	b := n.load(0, n.op(OpAdd, U, n.u(uint32(kh*kd)), n.op(OpAdd, U, n.op(OpMul, U, key(&n, v), n.u(uint32(kd))), i)))
	sv := n.load(3, flat(&n, v, j, i))
	n.op(OpAdd, F, n.op(OpMul, F, sv, n.load(2, v)), n.op(OpMul, F, n.op(OpMul, F, xv, n.load(1, v)), b))

	// The advanced state's rows against C.
	var r exprs
	rv, rj, ri := r.coord(0), r.coord(1), r.coord(2)
	cv := r.load(1, r.op(OpAdd, U, r.op(OpMul, U, key(&r, rv), r.u(uint32(kd))), ri))
	r.op(OpMul, F, r.load(0, flat(&r, rv, rj, ri)), cv)

	// The output: the rows' sum and D times x.
	var y exprs
	ye := y.coord(1)
	yx := y.load(0, y.op(OpAdd, U, y.u(uint32(2*kh*kd)), ye))
	y.op(OpAdd, F, y.load(1, ye), y.op(OpMul, F, y.load(2, y.op(OpDiv, U, ye, y.u(uint32(vd)))), yx))

	var k exprs
	kx := flat(&k, k.coord(0), k.coord(1), k.coord(2))
	kv := k.load(0, kx)
	D := Operand{Kind: OperandVector, Role: o.Weight[0].Role, Name: o.Weight[0].Role.String()}
	return []Prim{
		{Kind: PrimMap, Shape: sh, In: []Operand{xs, dt, decay, state}, Out: next, Expr: n.e,
			Facts: Facts{Extents: map[int]int{0: c.width(o.In[0]), 1: vh, 2: vh, 3: N}}},
		{Kind: PrimReduce, Shape: sh, In: []Operand{next, xs}, Out: acc, Combine: OpAdd, Expr: r.e,
			Facts: Facts{Extents: map[int]int{0: N, 1: c.width(o.In[0])}}},
		{Kind: PrimMap, Shape: []int{1, inner}, In: []Operand{xs, acc, D}, Out: c.value(o.Out[0]), Expr: y.e,
			Facts: Facts{Extents: map[int]int{0: c.width(o.In[0]), 1: inner, 2: vh}}},
		{Kind: PrimMap, Shape: sh, In: []Operand{next}, Expr: k.e, Facts: Facts{Extents: map[int]int{0: N}},
			Stores: []PrimStore{{Out: state, Index: kx, Value: kv, Extent: N}}},
	}, nil
}
