package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/ir"
)

// packed is a packed weight's reading rules (jlm.DecodeElement) over a
// primitive's expressions: the weight's planes are three of its inputs, a
// format's code table one more of constant words.
type packed struct {
	x      *exprs
	planes [3]int
	in     *[]Operand
	facts  *Facts
	codes  int
}

// packedOperands appends weight wo's three planes to a primitive's inputs,
// sized for a tensor of rows rows of k, and is the arithmetic that reads
// them.
func packedOperands(x *exprs, wo Operand, l jlm.TypeLayout, rows, k int, in *[]Operand, facts *Facts) *packed {
	p := &packed{x: x, in: in, facts: facts, codes: -1}
	n := [3]int{}
	n[0], n[1], n[2] = l.Words(rows, k)
	for i := range p.planes {
		if n[i] == 0 {
			p.planes[i] = -1
			continue
		}
		w := wo
		w.Region = i
		p.planes[i] = len(*in)
		*in = append(*in, w)
		if facts.Extents == nil {
			facts.Extents = map[int]int{}
		}
		facts.Extents[p.planes[i]] = n[i]
	}
	return p
}

func (p *packed) U(v uint32) int       { return p.x.u(v) }
func (p *packed) Add(a, b int) int     { return p.x.op(OpAdd, ir.U32, a, b) }
func (p *packed) Sub(a, b int) int     { return p.x.op(OpSub, ir.U32, a, b) }
func (p *packed) Mul(a, b int) int     { return p.x.op(OpMul, ir.U32, a, b) }
func (p *packed) Div(a, b int) int     { return p.x.op(OpDiv, ir.U32, a, b) }
func (p *packed) Rem(a, b int) int     { return p.x.op(OpRem, ir.U32, a, b) }
func (p *packed) Min(a, b int) int     { return p.x.op(OpMin, ir.U32, a, b) }
func (p *packed) Shr(a, b int) int     { return p.x.op(OpShr, ir.U32, a, b) }
func (p *packed) Shl(a, b int) int     { return p.x.op(OpShl, ir.U32, a, b) }
func (p *packed) And(a, b int) int     { return p.x.op(OpAnd, ir.U32, a, b) }
func (p *packed) Xor(a, b int) int     { return p.x.op(OpXor, ir.U32, a, b) }
func (p *packed) Half(a int) int       { return p.x.un(OpWidenHalf, ir.F32, a) }
func (p *packed) F32(a int) int        { return p.x.un(OpBitcast, ir.F32, a) }
func (p *packed) Int(a int) int        { return p.x.un(OpCvtF32, ir.F32, p.x.un(OpBitcast, ir.I32, a)) }
func (p *packed) FConst(v float32) int { return p.x.f(v) }
func (p *packed) FMul(a, b int) int    { return p.x.op(OpMul, ir.F32, a, b) }
func (p *packed) FSub(a, b int) int    { return p.x.op(OpSub, ir.F32, a, b) }
func (p *packed) FAdd(a, b int) int    { return p.x.op(OpAdd, ir.F32, a, b) }

// Float converts a u32 the rules only ever hold below 2^31 (a byte, a field,
// a code), where the signed conversion is exact.
func (p *packed) Float(a int) int { return p.Int(a) }

func (p *packed) Word(pl jlm.Plane, i int) int { return p.x.load(p.planes[pl], i) }

// Code is entry q of the format's code table, its sixteen bytes four words
// of constants.
func (p *packed) Code(t *[16]uint8, q int) int {
	if p.codes < 0 {
		var w []uint32
		for i := 0; i < 16; i += 4 {
			w = append(w, uint32(t[i])|uint32(t[i+1])<<8|uint32(t[i+2])<<16|uint32(t[i+3])<<24)
		}
		p.codes = len(*p.in)
		*p.in = append(*p.in, Operand{Kind: OperandTable, Name: fmt.Sprintf("codes.%x", w), Elem: ir.U32})
		if p.facts.Words == nil {
			p.facts.Words = map[int][]uint32{}
		}
		p.facts.Words[p.codes] = w
		p.facts.Extents[p.codes] = len(w)
	}
	word := p.x.load(p.codes, p.Shr(q, p.U(2)))
	return p.And(p.Shr(word, p.Mul(p.And(q, p.U(3)), p.U(8))), p.U(0xff))
}

// weightElement is element (row, col) of weight wo, rows by k, decoded from
// the format the container stores it in: the format's own reading rules
// (jlm.DecodeElement, jlm.DecodeVerbatim) over the planes it appends to in.
func weightElement(x *exprs, wo Operand, rows, k int, in *[]Operand, facts *Facts, row, col int) (int, error) {
	if jlm.Verbatim(wo.Format) {
		width := 2
		if wo.Format == jlm.TypeF32 {
			width = 4
		}
		p := &packed{x: x, in: in, facts: facts, codes: -1, planes: [3]int{len(*in), -1, -1}}
		facts.Extents[len(*in)] = (rows*k*width + 3) / 4
		*in = append(*in, wo)
		return jlm.DecodeVerbatim[int](p, wo.Format, k, row, col), nil
	}
	l, ok := jlm.Layout(wo.Format)
	if !ok || k%(l.Sub*l.SuperSub) != 0 {
		return 0, fmt.Errorf("no reading rules for a row of %d", k)
	}
	p := packedOperands(x, wo, l, rows, k, in, facts)
	return jlm.DecodeElement[int](p, l, rows, row, col), nil
}

// decomposeMatVecDecoded is each output row's sum over K of the weight's
// elements, each decoded from the container's format, times the input.
func decomposeMatVecDecoded(rows, k int, wo Operand, x, out Operand, R int) (Prim, error) {
	var e exprs
	r, row, col := e.coord(0), e.coord(1), e.coord(2)
	in := []Operand{x}
	facts := Facts{Extents: map[int]int{0: R * k}}
	xv := e.load(0, e.op(OpAdd, ir.U32, e.op(OpMul, ir.U32, r, e.u(uint32(k))), col))
	w, err := weightElement(&e, wo, rows, k, &in, &facts, row, col)
	if err != nil {
		return Prim{}, err
	}
	e.op(OpMul, ir.F32, w, xv)
	return Prim{Kind: PrimReduce, Shape: []int{R, rows, k}, In: in, Out: out, Combine: OpAdd, Expr: e.e, Facts: facts}, nil
}

// decomposeMatVecInt8 is a matvec over a packed weight against its input's
// int8 form, main's arithmetic: each sub-block's integer dot of the
// weight's codes and the input's, then the sum over sub-blocks of the
// input's window scale times the weight's scale times that dot, less the
// weight's bias times the input's codes' sum over the sub-block. ok is false
// where the format's sub-blocks do not sit in the form's windows.
func decomposeMatVecInt8(o graph.Op, c *Context, wo Operand, l jlm.TypeLayout) (prims []Prim, ok bool) {
	w, q, v := o.Weight[0], c.Int8, o.In[0]
	if q == nil || q.Word != 4 || l.Sub%q.Sums != 0 || w.K%(l.Sub*l.SuperSub) != 0 {
		return nil, false
	}
	win := c.window(v)
	if win%l.Sub != 0 || c.width(v)%win != 0 {
		return nil, false
	}
	if !c.formed[v] {
		c.formed[v] = true
		prims = c.int8(v)
	}
	U, I, F := ir.U32, ir.I32, ir.F32
	R, rows, K := c.Step.Rows, w.Rows, w.K
	nsub, nwd, nw, ns := K/l.Sub, K/q.Word, c.width(v)/win, c.width(v)/q.Sums
	x := c.value(v)
	words := Operand{Kind: OperandScratch, Name: x.Name + ".q8a", Elem: U}
	scales := Operand{Kind: OperandScratch, Name: x.Name + ".q8x"}
	dot := Operand{Kind: OperandScratch, Name: c.value(o.Out[0]).Name + ".dot", Elem: I, Temp: true}

	// Each sub-block's integer dot: the weight's codes against the input's.
	var d exprs
	r, row, s, j := d.coord(0), d.coord(1), d.coord(2), d.coord(3)
	col := d.op(OpAdd, U, d.op(OpMul, U, s, d.u(uint32(l.Sub))), j)
	in := []Operand{words}
	facts := Facts{Extents: map[int]int{0: R * nwd}}
	p := packedOperands(&d, wo, l, rows, K, &in, &facts)
	code, _, _ := jlm.DecodeParts[int](p, l, rows, row, col)
	xw := d.load(0, d.op(OpAdd, U, d.op(OpMul, U, r, d.u(uint32(nwd))), d.op(OpShr, U, col, d.u(2))))
	xb := d.op(OpAnd, U, d.op(OpShr, U, xw, d.op(OpMul, U, d.op(OpAnd, U, col, d.u(3)), d.u(8))), d.u(0xff))
	xc := d.op(OpSub, U, d.op(OpXor, U, xb, d.u(0x80)), d.u(0x80))
	d.op(OpMul, I, d.un(OpBitcast, I, code), d.un(OpBitcast, I, xc))
	prims = append(prims, Prim{Kind: PrimReduce, Shape: []int{R, rows, nsub, l.Sub}, In: in, Out: dot, Combine: OpAdd, Type: I,
		Expr: d.e, Facts: facts})

	// The sum over sub-blocks: the window's scale times the weight's scale
	// times the dot, less the bias times the window's scale times the codes'
	// sum.
	var e exprs
	er, erow, es := e.coord(0), e.coord(1), e.coord(2)
	ein := []Operand{dot, scales}
	efacts := Facts{Extents: map[int]int{0: R * rows * nsub, 1: R * (nw + ns)}}
	ep := packedOperands(&e, wo, l, rows, K, &ein, &efacts)
	first := e.op(OpMul, U, es, e.u(uint32(l.Sub)))
	_, scale, bias := jlm.DecodeParts[int](ep, l, rows, erow, first)
	df := e.un(OpCvtF32, F, e.at(0, []int{R, rows, nsub}, er, erow, es))
	xs := e.load(1, e.op(OpAdd, U, e.op(OpMul, U, er, e.u(uint32(nw))), e.op(OpDiv, U, first, e.u(uint32(win)))))
	sumAt := e.op(OpAdd, U, e.u(uint32(R*nw)), e.op(OpAdd, U, e.op(OpMul, U, er, e.u(uint32(ns))), e.op(OpDiv, U, first, e.u(uint32(q.Sums)))))
	csum := e.load(1, sumAt)
	for t := 1; t < l.Sub/q.Sums; t++ {
		csum = e.op(OpAdd, F, csum, e.load(1, e.op(OpAdd, U, sumAt, e.u(uint32(t)))))
	}
	e.op(OpMul, F, xs, e.op(OpSub, F, e.op(OpMul, F, scale, df), e.op(OpMul, F, bias, csum)))
	return append(prims, Prim{Kind: PrimReduce, Shape: []int{R, rows, nsub}, In: ein, Out: c.value(o.Out[0]), Combine: OpAdd,
		Expr: e.e, Facts: efacts}), true
}
