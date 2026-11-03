package composer

import (
	"fmt"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/graph"
)

// Philox2x32-10's round multiplier and key increment (Salmon et al.,
// "Parallel random numbers: as easy as 1, 2, 3", SC 2011).
const (
	philoxM = 0xD256D193
	philoxW = 0x9E3779B9
)

// decomposeUniform is OpUniform in words of 32 bits: Philox2x32-10 over the
// counter (the key, the position) keyed by the seed, a map a round small
// enough for a CPU's registers, each round's 32x32 product as four limbs of
// 16 bits (ngramProduct) and its two words kept in a scratch for the next;
// then the first word's top 24 bits over 2^24 -- exact in a float, so every
// device draws the same value.
func decomposeUniform(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 2 || len(o.Out) != 1 {
		return nil, fmt.Errorf("a uniform of %d inputs into %d outputs", len(o.In), len(o.Out))
	}
	W := c.width(o.In[0])
	if c.width(o.In[1]) != 2 || c.width(o.Out[0]) != W {
		return nil, fmt.Errorf("a uniform of %d keys under a counter of %d", W, c.width(o.In[1]))
	}
	U, F, R := ir.U32, ir.F32, c.Step.Rows
	keys, ctr, out := c.value(o.In[0]), c.value(o.In[1]), c.value(o.Out[0])
	keys.Elem, ctr.Elem = U, U
	state := Operand{Kind: OperandScratch, Name: out.Name + ".philox", Elem: U}
	ext := []int{R, 2 * W}
	var prims []Prim
	for round := range 10 {
		var x exprs
		r, e := x.coord(0), x.coord(1)
		e1 := x.op(OpAdd, U, e, x.u(uint32(W)))
		var x0, x1 int
		in := []Operand{ctr}
		if round == 0 {
			in = append(in, keys)
			x0, x1 = x.at(1, []int{R, W}, r, e), x.at(0, []int{R, 2}, r, x.u(1))
		} else {
			in = append(in, state)
			x0, x1 = x.at(1, ext, r, e), x.at(1, ext, r, e1)
		}
		k := x.op(OpAdd, U, x.at(0, []int{R, 2}, r, x.u(0)), x.u(uint32(round*philoxW)))
		p := ngramProduct(&x, x0, philoxM)
		hi := x.op(OpAdd, U, p[2], x.op(OpShl, U, p[3], x.u(16)))
		lo := x.op(OpAdd, U, p[0], x.op(OpShl, U, p[1], x.u(16)))
		y0 := x.op(OpXor, U, x.op(OpXor, U, hi, k), x1)
		prims = append(prims, Prim{Kind: PrimMap, Shape: []int{R, W}, In: in, Expr: x.e, Stores: []PrimStore{
			{Out: state, Ax: []int{r, e}, Ext: ext, Value: y0},
			{Out: state, Ax: []int{r, e1}, Ext: ext, Value: lo},
		}})
	}
	var x exprs
	r, e := x.coord(0), x.coord(1)
	top := x.op(OpShr, U, x.at(0, ext, r, e), x.u(8))
	x.op(OpMul, F, x.un(OpCvtF32, F, x.un(OpBitcast, ir.I32, top)), x.f(1.0/(1<<24)))
	return append(prims, Prim{Kind: PrimMap, Shape: []int{R, W}, In: []Operand{state}, Out: out, Expr: x.e}), nil
}
