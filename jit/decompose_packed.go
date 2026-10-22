package jit

import (
	"fmt"

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
