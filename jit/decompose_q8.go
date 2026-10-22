package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/ir"
)

// The Q8_0 layout the container stores (kernels.PackWeights): a row's
// signed bytes four to a word, and word u of every row contiguous, so word u
// of row r is at u*rows + r; and a 32-wide block's binary16 scale, two rows to
// a word, so block b of row r is half r&1 of word b*ceil(rows/2) + r/2.
const (
	q8Block = 32
	q8Words = q8Block / 4
)

// decomposeMatVecQ8 is a matvec over Q8_0 weights against the input's int8
// form: each block's integer dot of eight words, then each output the sum
// over blocks of that dot times the weight's scale and the input's.
func decomposeMatVecQ8(o graph.Op, c *Context, wo Operand) ([]Prim, error) {
	w := o.Weight[0]
	q := c.Int8
	v := o.In[0]
	if q == nil || q.Window != q8Block || q.Word != 4 || w.K%q8Block != 0 {
		return nil, fmt.Errorf("a Q8_0 matvec of %d over an int8 form of windows of %v", w.K, q)
	}
	var prims []Prim
	if !c.formed[v] {
		c.formed[v] = true
		prims = c.int8(v)
	}
	U, F, I := ir.U32, ir.F32, ir.I32
	R, rows, K := c.Step.Rows, w.Rows, w.K
	nb, nwd := K/q8Block, K/4
	ns := c.width(v) / q.Sums
	x := c.value(v)
	qs := wo
	qs.Region = 0
	ds := wo
	ds.Region = 1
	words := Operand{Kind: OperandScratch, Name: x.Name + ".q8a", Elem: U}
	scales := Operand{Kind: OperandScratch, Name: x.Name + ".q8x"}
	dot := Operand{Kind: OperandScratch, Name: c.value(o.Out[0]).Name + ".dot", Elem: I, Temp: true}
	half := (rows + 1) / 2

	// The block's integer dot: the eight words of row row's block b against
	// the input row's.
	var d exprs
	r, row, b, j := d.coord(0), d.coord(1), d.coord(2), d.coord(3)
	u := d.op(OpAdd, U, d.op(OpMul, U, b, d.u(q8Words)), j)
	wq := d.load(0, d.op(OpAdd, U, d.op(OpMul, U, u, d.u(uint32(rows))), row))
	xq := d.load(1, d.op(OpAdd, U, d.op(OpMul, U, r, d.u(uint32(nwd))), u))
	d.add(Expr{Op: OpDot4, Type: I, A: wq, B: xq, C: d.i(0)})

	// The output: each block's dot by the weight's scale and the input's.
	var s exprs
	sr, srow, sb := s.coord(0), s.coord(1), s.coord(2)
	at := s.op(OpAdd, U, s.op(OpMul, U, sb, s.u(uint32(half))), s.op(OpShr, U, srow, s.u(1)))
	word := s.load(1, at)
	shift := s.op(OpMul, U, s.op(OpAnd, U, srow, s.u(1)), s.u(16))
	wd := s.un(OpWidenHalf, F, s.op(OpShr, U, word, shift))
	xd := s.load(2, s.op(OpAdd, U, s.op(OpMul, U, sr, s.u(uint32(nb))), sb))
	id := s.un(OpCvtF32, F, s.at(0, []int{R, rows, nb}, sr, srow, sb))
	s.op(OpMul, F, s.op(OpMul, F, id, wd), xd)

	return append(prims,
		Prim{Kind: PrimReduce, Shape: []int{R, rows, nb, q8Words}, In: []Operand{qs, words}, Out: dot, Combine: OpAdd, Type: I, Expr: d.e,
			Facts: Facts{Extents: map[int]int{0: nwd * rows, 1: R * nwd}}},
		Prim{Kind: PrimReduce, Shape: []int{R, rows, nb}, In: []Operand{dot, ds, scales}, Out: c.value(o.Out[0]), Combine: OpAdd, Expr: s.e,
			Facts: Facts{Extents: map[int]int{1: nb * half, 2: R * (c.width(v)/q.Window + ns)}}},
	), nil
}

// q8Element is element col of row row of a Q8_0 weight of rows rows, its
// quants input qi and its scales input di: the row's signed byte times its
// block's scale.
func q8Element(x *exprs, qi, di, row, col, rows int) int {
	U, I, F := ir.U32, ir.I32, ir.F32
	word := x.load(qi, x.op(OpAdd, U, x.op(OpMul, U, x.op(OpShr, U, col, x.u(2)), x.u(uint32(rows))), row))
	// The byte, its sign extended by flipping the top bit and taking 128
	// away: no device shift keeps a sign.
	b := x.op(OpAnd, U, x.op(OpShr, U, word, x.op(OpMul, U, x.op(OpAnd, U, col, x.u(3)), x.u(8))), x.u(255))
	signed := x.op(OpAdd, U, x.op(OpXor, U, b, x.u(0x80)), x.u(0xffffff80))
	q := x.un(OpCvtF32, F, x.un(OpBitcast, I, signed))
	half := (rows + 1) / 2
	at := x.op(OpAdd, U, x.op(OpMul, U, x.op(OpShr, U, col, x.u(5)), x.u(uint32(half))), x.op(OpShr, U, row, x.u(1)))
	sw := x.load(di, at)
	scale := x.un(OpWidenHalf, F, x.op(OpShr, U, sw, x.op(OpMul, U, x.op(OpAnd, U, row, x.u(1)), x.u(16))))
	return x.op(OpMul, F, q, scale)
}
