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
	sel := c.value(o.Out[0])
	sel.Elem = ir.U32
	if G == 1 && S > selectMin && K < S {
		return decomposeTopKSelect(c, c.value(o.In[0]), sel, c.value(o.Out[1]), S, K), nil
	}
	return rankedPlaces(c, c.value(o.In[0]), nil, sel, c.value(o.Out[1]), G, S, K), nil
}

// rankedPlaces is the K largest of each group of S of in: each element's
// rank among its group, then each of the K places the index and the value
// of the element ranked there, the one element of the group whose rank it
// is. Where ids is given, an element's index is its word of ids rather than
// its place.
func rankedPlaces(c *Context, in Operand, ids *Operand, sel, val Operand, G, S, K int) []Prim {
	U, I, F := ir.U32, ir.I32, ir.F32
	gs := grouped{c.Step.Rows, G, S}
	rank := Operand{Kind: OperandScratch, Name: val.Name + ".rank", Elem: I, Temp: true}

	// The element at place s: its index, and its value, alone of the group.
	at := func(x *exprs, val func(r, g, i int) int, t ir.Type, zero func() int) {
		r, g, si := gs.coords(x, 2)
		s, i := si[0], si[1]
		k := gs.at(x, 1, r, g, i)
		ks := x.un(OpBitcast, I, s)
		x.sel(t, x.op(OpLess, ir.Pred, k, ks), zero(), x.sel(t, x.op(OpLess, ir.Pred, k, x.op(OpAdd, I, ks, x.i(1))), val(r, g, i), zero()))
	}
	ixIn := []Operand{in, rank}
	var ix exprs
	if ids == nil {
		at(&ix, func(_, _, i int) int { return i }, U, func() int { return ix.u(0) })
	} else {
		ixIn = append(ixIn, *ids)
		at(&ix, func(r, g, i int) int { return gs.at(&ix, 2, r, g, i) }, U, func() int { return ix.u(0) })
	}
	var vx exprs
	at(&vx, func(r, g, i int) int { return gs.at(&vx, 0, r, g, i) }, F, func() int { return vx.f(0) })

	return []Prim{
		rankPrim(c, in, rank, G, S),
		{Kind: PrimReduce, Shape: gs.shape(K, S), In: ixIn, Out: sel, Combine: OpAdd, Type: U, Expr: ix.e},
		{Kind: PrimReduce, Shape: gs.shape(K, S), In: []Operand{in, rank}, Out: val, Combine: OpAdd, Expr: vx.e},
	}
}

// decomposeExpertMatVec is a matvec per pair of the selection: pair j's row
// of the input against the sheet of the bank the selection names at j --
// over the input's int8 form where the format and the form allow, as a
// plain matvec runs, else the weights decoded -- and the bank's bias row
// for that expert added after where the op binds one. With no selection
// pair j reads sheet j.
func decomposeExpertMatVec(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 && len(o.In) != 2 || len(o.Out) != 1 || len(o.Weight) < 1 || len(o.Weight) > 2 {
		return nil, fmt.Errorf("an expert matvec of %d operands, %d weights", len(o.In), len(o.Weight))
	}
	w := o.Weight[0]
	E, P, K, rows := w.Sheets, pairsOf(o, c), w.K, w.Rows
	if E < 1 || len(o.In) == 1 && P != E || c.width(o.In[0]) != P*K || c.width(o.Out[0]) != P*rows {
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
		in := []Operand{out}
		var f Facts
		sel := sheetAt(o, c, &x, &in, &f, 1, r, j)
		bias := Operand{Kind: OperandVector, Role: o.Weight[1].Role, Name: o.Weight[1].Role.String()}
		b := x.load(len(in), x.op(OpAdd, U, x.op(OpMul, U, sel, x.u(uint32(rows))), row))
		x.op(OpAdd, ir.F32, x.at(0, []int{R, P, rows}, r, j, row), b)
		if f.Extents == nil {
			f.Extents = map[int]int{}
		}
		f.Extents[len(in)] = E * rows
		prims = append(prims, Prim{Kind: PrimMap, Shape: []int{R, P, rows}, In: append(in, bias),
			Out: c.value(o.Out[0]), Expr: x.e, Facts: f})
	}
	return prims, nil
}

// pairStride is how far apart in the input two pairs' rows are: a row each
// for an expert matvec, none for a row-selected matvec, whose pairs all read
// the one input row.
func pairStride(o graph.Op) int {
	if o.Kind == graph.OpMatVec {
		return 0
	}
	return o.Weight[0].K
}

// decomposeMatVecRows is a matvec over the rows its selection names: pair
// j the selected row of the matrix against the input, the selection bounded
// by the matrix. Over Q8_0 it is the plain matvec's form at those rows;
// over any other format the matrix is read by expert matvec's decode, its
// sheets the matrix's rows.
func decomposeMatVecRows(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.Weight) != 1 || len(o.In) != 2 || len(o.Out) != 1 || c.width(o.Out[0]) != c.width(o.In[1]) {
		return nil, fmt.Errorf("a row-selected matvec of %d weights over %d operands", len(o.Weight), len(o.In))
	}
	w := o.Weight[0]
	if c.width(o.In[0]) != w.K {
		return nil, fmt.Errorf("a row-selected matvec of %d over a row of %d", w.K, c.width(o.In[0]))
	}
	rows := o
	rows.Weight = []graph.Binding{w}
	rows.Weight[0].Sheets, rows.Weight[0].Rows = w.Rows, 1
	wo, err := c.weight(w)
	if err != nil {
		return nil, err
	}
	if wo.Format == jlm.TypeQ8 && c.Int8 != nil && c.Graph.Values[o.In[0]].Elem != graph.ElemI8 {
		// The plain matvec's Q8_0 form, at the selected rows.
		return decomposeMatVecQ8(o, c, wo)
	}
	out := c.value(o.Out[0])
	if l, ok := jlm.Layout(wo.Format); ok {
		if ps, ok := expertInt8(rows, c, wo, l, out); ok {
			return ps, nil
		}
	}
	p, err := expertDecoded(rows, c, wo, out)
	if err != nil {
		return nil, err
	}
	return []Prim{p}, nil
}

// sheetRow is the packed weight a pair reads and the row of it: the pair's
// sheet of a bank, its row the coordinate's, or for a row-selected matvec
// the one matrix of the bank's sheets as rows, its row the selection's.
func sheetRow(o graph.Op, x *exprs, wo Operand, l jlm.TypeLayout, in *[]Operand, facts *Facts, sheet, row int) (p *packed, rows, at int) {
	w := o.Weight[0]
	if pairStride(o) == 0 {
		return packedOperands(x, wo, l, w.Sheets, w.K, in, facts), w.Sheets, sheet
	}
	p = packedOperands(x, wo, l, w.Rows, w.K, in, facts)
	p.bank(sheet, w.Sheets)
	return p, w.Rows, row
}

// pairsOf is how many pairs an expert matvec runs: its selection's width,
// or with none its bank's sheets.
func pairsOf(o graph.Op, c *Context) int {
	if len(o.In) == 1 {
		return o.Weight[0].Sheets
	}
	return c.width(o.In[1])
}

// sheetAt is the sheet pair j of row r reads: the selection's word, which
// it appends to in as operand at, bounded by the bank, or with no selection
// j itself.
func sheetAt(o graph.Op, c *Context, x *exprs, in *[]Operand, f *Facts, at, r, j int) int {
	if len(o.In) == 1 {
		return j
	}
	*in = append(*in, c.value(o.In[1]))
	P, E := c.width(o.In[1]), o.Weight[0].Sheets
	if f.Extents == nil {
		f.Extents = map[int]int{}
	}
	f.Extents[at] = c.Step.Rows * P
	return x.op(OpMin, ir.U32, x.at(at, []int{c.Step.Rows, P}, r, j), x.u(uint32(E-1)))
}

// expertDecoded is each pair's sum over K of its sheet's weights, decoded,
// times the pair's input.
func expertDecoded(o graph.Op, c *Context, wo Operand, out Operand) (Prim, error) {
	w := o.Weight[0]
	E, P, K, rows := w.Sheets, pairsOf(o, c), w.K, w.Rows
	U := ir.U32
	R := c.Step.Rows
	var x exprs
	r, j, row, col := x.coord(0), x.coord(1), x.coord(2), x.coord(3)
	in := []Operand{c.value(o.In[0])}
	stride := pairStride(o)
	W := (P-1)*stride + K
	facts := Facts{Extents: map[int]int{0: R * W}}
	sel := sheetAt(o, c, &x, &in, &facts, 1, r, j)
	// The input read as the row its producer writes and its other readers
	// see, pair j's K from j*stride.
	xv := x.at(0, []int{R, W}, r, x.op(OpAdd, U, x.op(OpMul, U, j, x.u(uint32(stride))), col))
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
		p, dr, dw := sheetRow(o, &x, wo, l, &in, &facts, sel, row)
		wv = jlm.DecodeElement[int](p, l, dr, dw, col)
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
	P, R, rows, K := pairsOf(o, c), c.Step.Rows, w.Rows, w.K
	stride := pairStride(o)
	W := (P-1)*stride + K
	nsub, nwd, nw, ns := K/l.Sub, W/q.Word, W/win, W/q.Sums
	x := c.value(v)
	words := Operand{Kind: OperandScratch, Name: x.Name + ".q8a", Elem: U}
	scales := Operand{Kind: OperandScratch, Name: x.Name + ".q8x"}
	dot := Operand{Kind: OperandScratch, Name: c.value(o.Out[0]).Name + ".dot", Elem: I, Temp: true}

	var d exprs
	r, j, row, s, t := d.coord(0), d.coord(1), d.coord(2), d.coord(3), d.coord(4)
	in := []Operand{words}
	facts := Facts{Extents: map[int]int{0: R * nwd}}
	sheet := sheetAt(o, c, &d, &in, &facts, 1, r, j)
	p, dr, dw := sheetRow(o, &d, wo, l, &in, &facts, sheet, row)
	at := func(col int) int { return d.op(OpAdd, U, d.op(OpMul, U, j, d.u(uint32(stride))), col) }
	per := l.Sub
	if l.WordCodes() {
		per = l.Sub / 4
		col := d.op(OpAdd, U, d.op(OpMul, U, s, d.u(uint32(l.Sub))), d.op(OpMul, U, t, d.u(4)))
		wq, _ := jlm.DecodeWord[int](p, l, dr, dw, col)
		xw := d.load(0, d.op(OpAdd, U, d.op(OpMul, U, r, d.u(uint32(nwd))), d.op(OpShr, U, at(col), d.u(2))))
		d.add(Expr{Op: OpDot4, Type: I, A: wq, B: xw, C: d.i(0)})
	} else {
		col := d.op(OpAdd, U, d.op(OpMul, U, s, d.u(uint32(l.Sub))), t)
		code, _, _ := jlm.DecodeParts[int](p, l, dr, dw, col)
		xw := d.load(0, d.op(OpAdd, U, d.op(OpMul, U, r, d.u(uint32(nwd))), d.op(OpShr, U, at(col), d.u(2))))
		xb := d.op(OpAnd, U, d.op(OpShr, U, xw, d.op(OpMul, U, d.op(OpAnd, U, col, d.u(3)), d.u(8))), d.u(0xff))
		xc := d.op(OpSub, U, d.op(OpXor, U, xb, d.u(0x80)), d.u(0x80))
		d.op(OpMul, I, d.un(OpBitcast, I, code), d.un(OpBitcast, I, xc))
	}
	prims = append(prims, Prim{Kind: PrimReduce, Shape: []int{R, P, rows, nsub, per}, In: in, Out: dot, Combine: OpAdd, Type: I,
		Expr: d.e, Facts: facts})

	var e exprs
	er, ej, erow, es := e.coord(0), e.coord(1), e.coord(2), e.coord(3)
	ein := []Operand{dot, scales}
	efacts := Facts{Extents: map[int]int{0: R * P * rows * nsub, 1: R * (nw + ns)}}
	esheet := sheetAt(o, c, &e, &ein, &efacts, 2, er, ej)
	ep, edr, edw := sheetRow(o, &e, wo, l, &ein, &efacts, esheet, erow)
	first := e.op(OpMul, U, es, e.u(uint32(l.Sub)))
	_, scale, bias := jlm.DecodeParts[int](ep, l, edr, edw, first)
	df := e.un(OpCvtF32, F, e.at(0, []int{R, P, rows, nsub}, er, ej, erow, es))
	col := e.op(OpAdd, U, e.op(OpMul, U, ej, e.u(uint32(stride))), first)
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
	gs := grouped{c.Step.Rows, G, S}
	var rk exprs
	r, g, ij := gs.coords(&rk, 2)
	i, j := ij[0], ij[1]
	xi, xj := gs.at(&rk, 0, r, g, i), gs.at(&rk, 0, r, g, j)
	before := rk.sel(I, rk.op(OpLess, ir.Pred, j, i), one(&rk), zero(&rk))
	rk.sel(I, rk.op(OpLess, ir.Pred, xi, xj), one(&rk), rk.sel(I, rk.op(OpLess, ir.Pred, xj, xi), zero(&rk), before))
	return Prim{Kind: PrimReduce, Shape: gs.shape(S, S), In: []Operand{in}, Out: out, Combine: OpAdd, Type: I, Expr: rk.e}
}

// grouped is a row seen as G groups of S, the group axis left out where
// there is one group: a value of one group keeps the [rows][width] its
// producer and its other readers see it at.
type grouped struct{ R, G, S int }

// shape is the rows, the group where there are several, then more.
func (g grouped) shape(more ...int) []int {
	if g.G == 1 {
		return append([]int{g.R}, more...)
	}
	return append([]int{g.R, g.G}, more...)
}

// coords is a primitive of g.shape's coordinates: the row's, the group's (a
// constant 0 where there is one group), and the n after them.
func (g grouped) coords(x *exprs, n int) (r, grp int, rest []int) {
	r, at := x.coord(0), 1
	if g.G == 1 {
		grp = x.u(0)
	} else {
		grp, at = x.coord(1), 2
	}
	for k := range n {
		rest = append(rest, x.coord(at+k))
	}
	return r, grp, rest
}

// at reads operand j, a row of g's groups, at element i of group grp.
func (g grouped) at(x *exprs, j, r, grp, i int) int {
	if g.G == 1 {
		return x.at(j, []int{g.R, g.S}, r, i)
	}
	return x.at(j, []int{g.R, g.G, g.S}, r, grp, i)
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
// window state (a slot of "rec.conv", Taps-1 rows of the channels): each
// channel's row and window against its taps, the newest tap the row's;
// then the window less its oldest row and with the row, into a scratch
// row and to the slot written, so no kernel reads the window while
// writing it.
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
	next := Operand{Kind: OperandScratch, Name: c.value(o.Out[0]).Name + ".window"}
	in := c.value(o.In[0])
	W := (T - 1) * C

	var x exprs
	ch := x.coord(1)
	tapIn := []Operand{in, tableRecConv, tableRecRows}
	facts := Facts{Extents: map[int]int{1: W, 2: 3}}
	tap := func(t int) int {
		v, _ := weightElement(&x, wo, C, T, &tapIn, &facts, ch, x.u(uint32(t)))
		return v
	}
	base := x.recSlot(2, 1, W)
	acc := x.op(OpMul, F, x.at(0, []int{1, C}, x.coord(0), ch), tap(T-1))
	for t := 0; t < T-1; t++ {
		acc = x.op(OpAdd, F, acc, x.op(OpMul, F, x.load(1, x.op(OpAdd, U, base, x.op(OpAdd, U, x.u(uint32(t*C)), ch))), tap(t)))
	}

	// The window less its oldest row, then the row after it.
	var n exprs
	nt, nc := n.coord(0), n.coord(1)
	nix := n.op(OpAdd, U, n.op(OpMul, U, nt, n.u(uint32(C))), nc)
	n.load(0, n.op(OpAdd, U, n.recSlot(1, 1, W), n.op(OpAdd, U, nix, n.u(uint32(C)))))
	var m exprs
	mix := m.op(OpAdd, U, m.u(uint32((T-2)*C)), m.coord(1))
	mv := m.at(0, []int{1, C}, m.coord(0), m.coord(1))

	var k exprs
	kt, kc := k.coord(0), k.coord(1)
	ix := k.op(OpAdd, U, k.op(OpMul, U, kt, k.u(uint32(C))), kc)
	v := k.load(0, ix)
	to := k.op(OpAdd, U, k.recSlot(1, 2, W), ix)
	ps := []Prim{{Kind: PrimMap, Shape: []int{1, C}, In: tapIn, Out: c.value(o.Out[0]), Expr: x.e, Facts: facts}}
	if T > 2 {
		ps = append(ps, Prim{Kind: PrimMap, Shape: []int{T - 2, C}, In: []Operand{tableRecConv, tableRecRows}, Expr: n.e,
			Facts: Facts{Extents: map[int]int{0: W, 1: 3}}, Stores: []PrimStore{{Out: next, Index: nix, Value: len(n.e) - 1, Extent: W}}})
	}
	return append(ps,
		Prim{Kind: PrimMap, Shape: []int{1, C}, In: []Operand{in}, Expr: m.e, Stores: []PrimStore{{Out: next, Index: mix, Value: mv, Extent: W}}},
		Prim{Kind: PrimMap, Shape: []int{T - 1, C}, In: []Operand{next, tableRecRows}, Expr: k.e, Facts: Facts{Extents: map[int]int{0: W, 1: 3}},
			Stores: []PrimStore{{Out: tableRecConvOut, Index: to, Value: v, Extent: W}}},
	), nil
}

// decomposeRowConv is a centred depthwise convolution across the step's
// live rows: each channel's centre tap on its own row first, then its
// other taps in order on the rows before and after, a row past the live
// ones zero -- the host's order (lfConv). The live count is the positions
// table's first word; a row past the live ones is the last live row's, as
// every row op of the step repeats it (run.Session.tables).
func decomposeRowConv(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 1 || len(o.Out) != 1 || len(o.Weight) != 1 {
		return nil, fmt.Errorf("a row convolution of %d operands", len(o.In))
	}
	U, F := ir.U32, ir.F32
	sh := c.rows(o.In[0])
	R, C, T := sh[0], sh[1], o.Attr.Taps
	if T < 1 || T%2 == 0 || o.Weight[0].K != T || o.Weight[0].Rows != C {
		return nil, fmt.Errorf("a row convolution of %d taps over %d channels", T, C)
	}
	wo, err := c.weight(o.Weight[0])
	if err != nil {
		return nil, err
	}
	if !jlm.Verbatim(wo.Format) {
		return nil, fmt.Errorf("a row convolution over %s taps", wo.Format)
	}
	var x exprs
	ch := x.coord(1)
	in := []Operand{c.value(o.In[0]), tablePos}
	facts := Facts{Extents: map[int]int{1: R + 1}}
	live := x.op(OpMax, U, x.load(1, x.u(0)), x.u(1))
	r := x.op(OpMin, U, x.coord(0), x.op(OpSub, U, live, x.u(1)))
	tap := func(t int) int {
		v, _ := weightElement(&x, wo, C, T, &in, &facts, ch, x.u(uint32(t)))
		return v
	}
	h := T / 2
	acc := x.op(OpMul, F, x.at(0, sh, r, ch), tap(h))
	for t := 0; t < T; t++ {
		if t == h {
			continue
		}
		// Row r+t-h, held in the step's rows; zero past the live ones.
		at := x.op(OpSub, U, x.op(OpAdd, U, r, x.u(uint32(t))), x.u(uint32(h)))
		ok := x.op(OpLess, ir.Pred, at, live)
		v := x.at(0, sh, x.op(OpMin, U, at, x.u(uint32(R-1))), ch)
		acc = x.op(OpAdd, F, acc, x.sel(F, ok, x.op(OpMul, F, v, tap(t)), x.f(0)))
	}
	return []Prim{{Kind: PrimMap, Shape: sh, In: in, Out: c.value(o.Out[0]), Expr: x.e, Facts: facts}}, nil
}
