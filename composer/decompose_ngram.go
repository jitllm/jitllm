package composer

import (
	"fmt"
	"math/bits"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/engine/graph"
)

// decomposeNGramHash is OpNGramHash in words of 32 bits, in maps small
// enough for a CPU's general registers: each row's window read into its
// ids (a predecessor that is none, or past an end of sequence where the op
// resets, read as the end of sequence); each g-gram's mixed product as four
// limbs of 16 bits -- the (g-1)-gram's limbs xored with the last id times
// its multiplier, a column at a time: the products of the id's halves by
// the multiplier's limbs, each exact in a word, their carries walked up;
// each head's row, its g-gram's limbs reduced by its vocabulary from the
// top a few bits at a time (a remainder below the vocabulary, shifted by
// those bits, stays in a word) and offset; and the next position's
// predecessors. No 64-bit word and no integer compare: equality is a
// difference's bits, a choice a product by it.
func decomposeNGramHash(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	if len(o.In) != 1 || len(o.Out) != 2 {
		return nil, fmt.Errorf("an n-gram hash of %d inputs into %d outputs", len(o.In), len(o.Out))
	}
	n, per := c.width(o.In[0]), a.Span
	heads := (n - 1) * per
	if n < 2 || per < 1 || len(a.Words) != 1+n+2*heads {
		return nil, fmt.Errorf("an n-gram hash of a window of %d, %d heads a gram, %d constants", n, per, len(a.Words))
	}
	if a.Words[0] >= 1<<32 {
		return nil, fmt.Errorf("an n-gram hash whose end of sequence %d is not a word", a.Words[0])
	}
	for h := range heads {
		off, voc := a.Words[1+n+2*h], a.Words[2+n+2*h]
		if voc < 1 || voc > 1<<31 || off+voc > 1<<32 {
			return nil, fmt.Errorf("an n-gram head of %d rows from %d: its rows are not words", voc, off)
		}
	}
	U := ir.U32
	R := c.Step.Rows
	win := c.value(o.In[0])
	rows, next := c.value(o.Out[0]), c.value(o.Out[1])
	rows.Elem, next.Elem = U, U
	limbs := Operand{Kind: OperandScratch, Name: rows.Name + ".limbs", Elem: U}
	nl := 4 * (n - 1)
	store := func(x *exprs, to Operand, r, w, at, v int) PrimStore {
		return PrimStore{Out: to, Ax: []int{r, x.u(uint32(at))}, Ext: []int{R, w}, Value: v}
	}

	// Each g-gram's limbs: the one before's, xored with the last id's
	// product, the 2-gram's the first two ids' products. The ids are read
	// from the window where a map reads them: a predecessor's cut is the
	// ones before it.
	var prims []Prim
	for g := 2; g <= n; g++ {
		var y exprs
		r := y.coord(0)
		one, zero, eos := y.u(1), y.u(0), y.u(uint32(a.Words[0]))
		cut, last := zero, 0
		for s := 1; s < g; s++ {
			w := y.at(0, []int{R, n}, r, y.u(uint32(s)))
			none := y.op(OpSub, U, one, y.op(OpMin, U, w, one))
			last = ngramPick(&y, y.op(OpMax, U, none, cut), eos, y.op(OpSub, U, w, one))
			if a.Reset && s < g-1 {
				cut = y.op(OpMax, U, cut, y.op(OpSub, U, one, y.op(OpMin, U, y.op(OpXor, U, last, eos), one)))
			}
		}
		p := ngramProduct(&y, last, a.Words[g])
		var acc [4]int
		if g == 2 {
			acc = ngramProduct(&y, y.at(0, []int{R, n}, r, zero), a.Words[1])
		} else {
			for q := range acc {
				acc[q] = y.at(1, []int{R, nl}, r, y.u(uint32(4*(g-3)+q)))
			}
		}
		var cs []PrimStore
		for q := range p {
			cs = append(cs, store(&y, limbs, r, nl, 4*(g-2)+q, y.op(OpXor, U, acc[q], p[q])))
		}
		in := []Operand{win}
		if g > 2 {
			in = append(in, limbs)
		}
		prims = append(prims, Prim{Kind: PrimMap, Shape: []int{R}, In: in, Expr: y.e, Stores: cs})
	}

	// Each head's row from its gram's limbs.
	for h := range heads {
		var z exprs
		r := z.coord(0)
		g := h/per + 2
		limb := func(q int) int {
			return z.at(0, []int{R, nl}, r, z.u(uint32(4*(g-2)+q)))
		}
		v := z.op(OpAdd, U, ngramMod(&z, limb, a.Words[2+n+2*h]), z.u(uint32(a.Words[1+n+2*h])))
		prims = append(prims, Prim{Kind: PrimMap, Shape: []int{R}, In: []Operand{limbs}, Expr: z.e,
			Stores: []PrimStore{store(&z, rows, r, heads, h, v)}})
	}

	var w exprs
	wr, j := w.coord(0), w.coord(1)
	first := w.op(OpSub, U, w.u(1), w.op(OpMin, U, j, w.u(1)))
	w.op(OpAdd, U, w.at(0, []int{R, n}, wr, j), first)
	return append(prims, Prim{Kind: PrimMap, Shape: []int{R, n - 1}, In: []Operand{win}, Out: next, Expr: w.e}), nil
}

const ngramLimb = 0xFFFF

// ngramPick is a where p is 1, b where it is 0.
func ngramPick(x *exprs, p, a, b int) int {
	U := ir.U32
	return x.op(OpAdd, U, b, x.op(OpMul, U, p, x.op(OpSub, U, a, b)))
}

// ngramProduct is id times m modulo 2^64, as four limbs of 16 bits, a
// column at a time: each product of the id's halves by m's limbs is formed
// where its column reads it, so few words are live at once.
func ngramProduct(x *exprs, id int, m uint64) [4]int {
	U := ir.U32
	half := func(i int) int {
		if i == 0 {
			return x.op(OpAnd, U, id, x.u(ngramLimb))
		}
		return x.op(OpShr, U, id, x.u(16))
	}
	term := func(i, k int) int { return x.op(OpMul, U, half(i), x.u(uint32(m>>(16*k)&ngramLimb))) }
	var limbs [4]int
	carry := x.u(0)
	for q := range limbs {
		sum := carry
		for i := range 2 {
			if k := q - i; k >= 0 && m>>(16*k)&ngramLimb != 0 {
				sum = x.op(OpAdd, U, sum, x.op(OpAnd, U, term(i, k), x.u(ngramLimb)))
			}
			if k := q - 1 - i; k >= 0 && m>>(16*k)&ngramLimb != 0 {
				sum = x.op(OpAdd, U, sum, x.op(OpShr, U, term(i, k), x.u(16)))
			}
		}
		limbs[q], carry = x.op(OpAnd, U, sum, x.u(ngramLimb)), x.op(OpShr, U, sum, x.u(16))
	}
	return limbs
}

// ngramMod is the 64-bit number whose limb q limb gives modulo voc, from
// the top: a remainder below voc shifted by step bits stays in a word.
func ngramMod(x *exprs, limb func(q int) int, voc uint64) int {
	U := ir.U32
	bitsOf := func(lo, ln int) int {
		q, sh := lo/16, lo%16
		v := limb(q)
		if sh > 0 {
			v = x.op(OpShr, U, v, x.u(uint32(sh)))
		}
		if sh+ln > 16 {
			v = x.op(OpAdd, U, v, x.op(OpShl, U, limb(q+1), x.u(uint32(16-sh))))
		}
		return x.op(OpAnd, U, v, x.u(uint32(1)<<ln-1))
	}
	step := min(16, 32-bits.Len64(voc-1))
	rem := x.u(0)
	for top := 64; top > 0; {
		ln := min(step, top)
		top -= ln
		rem = x.op(OpRem, U, x.op(OpAdd, U, x.op(OpShl, U, rem, x.u(uint32(ln))), bitsOf(top, ln)), x.u(uint32(voc)))
	}
	return rem
}
