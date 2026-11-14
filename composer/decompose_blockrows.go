package composer

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/graph"
)

// blockElems is the elements one llama.cpp block holds.
const blockElems = 32

// decomposeBlockRows is each row's floats in Attr.Block's llama.cpp blocks.
func decomposeBlockRows(o graph.Op, c *Context) ([]Prim, error) {
	f := o.Attr.Block
	if len(o.In) != 1 || len(o.Out) != 1 || f.Bytes() == 0 || c.width(o.Out[0]) != graph.BlockRowWords(f, c.width(o.In[0])) {
		return nil, fmt.Errorf("%v blocks of %d inputs into %d outputs", f, len(o.In), len(o.Out))
	}
	out := c.value(o.Out[0])
	out.Elem = ir.U32
	return blockRows(c, c.value(o.In[0]), c.rows(o.In[0]), out, f), nil
}

// blockRows is each of src's sh rows of floats into out's words in format
// f, as ggml's reference quantizer (quantize_row_q4_0_ref and its kin)
// quantizes a row zero-padded to whole blocks: each block's scale from its
// signed extreme (the first element of the greatest magnitude) or from its
// range, the scale's reciprocal, each element's code truncated as the
// reference's integer conversion does; then the row's regions
// (graph.BlockRowWords): the scales, the fifth bits, the nibbles.
func blockRows(c *Context, src Operand, sh []int, out Operand, f graph.BlockFmt) []Prim {
	U, F, I := ir.U32, ir.F32, ir.I32
	R, W := sh[0], sh[1]
	nb := (W + blockElems - 1) / blockElems
	rw := graph.BlockRowWords(f, W)
	scales, fifths, nibbles := graph.BlockRegions(f, W)
	temp := func(name string, t ir.Type) Operand {
		return Operand{Kind: OperandScratch, Name: out.Name + "." + name, Elem: t, Temp: true}
	}
	blocks, padded := []int{R, nb}, []int{R, nb * blockElems}
	elem := func(e *exprs, r, i int) int {
		if W%blockElems == 0 {
			return e.at(0, sh, r, i)
		}
		return e.sel(F, e.op(OpLess, ir.Pred, i, e.u(uint32(W))), e.at(0, sh, r, e.op(OpMin, U, i, e.u(uint32(W-1)))), e.f(0))
	}
	block := func(e *exprs) int {
		return e.op(OpAdd, U, e.op(OpMul, U, e.coord(1), e.u(blockElems)), e.coord(2))
	}
	quot := func(e *exprs, a, b int) int {
		if c.exactDiv {
			return e.op(OpDiv, F, a, b)
		}
		return e.div(a, b)
	}
	recip := func(e *exprs, a int) int {
		if c.exactDiv {
			return e.op(OpDiv, F, e.f(1), a)
		}
		return e.recip(a)
	}
	// inverse is 1/d, zero where d is.
	inverse := func() []Expr {
		var n exprs
		dd := n.at(0, blocks, n.coord(0), n.coord(1))
		n.sel(F, n.op(OpLess, ir.Pred, n.f(0), n.abs(dd)), recip(&n, dd), n.f(0))
		return n.e
	}
	d, inv, lo, codes := temp("d", F), temp("inv", F), temp("m", F), temp("q", U)
	var prims []Prim
	// levels is the codes' top and bias the one a code is read back less.
	levels, bias := float32(15), float32(8)
	if f.Fifths() {
		levels, bias = 31, 16
	}
	if f.Mins() {
		hi, nlo := temp("hi", F), temp("nlo", F)
		// The range from -FLT_MAX, as the reference's loop starts it.
		var a, b exprs
		elem(&a, a.coord(0), block(&a))
		b.neg(elem(&b, b.coord(0), block(&b)))
		lowest := int64(math.Float32bits(-math.MaxFloat32))
		var s exprs
		mx, mn := s.at(0, blocks, s.coord(0), s.coord(1)), s.neg(s.at(1, blocks, s.coord(0), s.coord(1)))
		quot(&s, s.op(OpSub, F, mx, mn), s.f(levels))
		var m exprs
		m.neg(m.at(0, blocks, m.coord(0), m.coord(1)))
		var k exprs
		r, i := k.coord(0), k.coord(1)
		blk := k.op(OpShr, U, i, k.u(5))
		t := k.op(OpMul, F, k.op(OpSub, F, elem(&k, r, i), k.at(1, blocks, r, blk)), k.at(2, blocks, r, blk))
		q := k.un(OpBitcast, U, k.un(OpCvtI32, I, k.op(OpAdd, F, t, k.f(0.5))))
		// q4_1 clamps its code to 15; q5_1 converts it to a byte unclamped.
		if f == graph.BlockQ4_1 {
			k.op(OpMin, U, q, k.u(15))
		} else {
			k.op(OpAnd, U, q, k.u(255))
		}
		prims = append(prims,
			Prim{Kind: PrimReduce, Shape: []int{R, nb, blockElems}, In: []Operand{src}, Out: hi, Combine: OpMax, Init: lowest, Expr: a.e},
			Prim{Kind: PrimReduce, Shape: []int{R, nb, blockElems}, In: []Operand{src}, Out: nlo, Combine: OpMax, Init: lowest, Expr: b.e},
			Prim{Kind: PrimMap, Shape: blocks, In: []Operand{hi, nlo}, Out: d, Expr: s.e},
			Prim{Kind: PrimMap, Shape: blocks, In: []Operand{d}, Out: inv, Expr: inverse()},
			Prim{Kind: PrimMap, Shape: blocks, In: []Operand{nlo}, Out: lo, Expr: m.e},
			Prim{Kind: PrimMap, Shape: padded, In: []Operand{src, lo, inv}, Out: codes, Expr: k.e})
	} else {
		// The signed extreme is the first element of the block's greatest
		// magnitude: of the elements not below the magnitude, the greatest
		// of twice 32 less the index, one more where the element is
		// negative, names the first and its sign.
		amax, first := temp("amax", F), temp("first", F)
		var a, fi exprs
		a.abs(elem(&a, a.coord(0), block(&a)))
		v := elem(&fi, fi.coord(0), block(&fi))
		below := fi.op(OpLess, ir.Pred, fi.abs(v), fi.at(1, blocks, fi.coord(0), fi.coord(1)))
		key := fi.op(OpMul, F, fi.op(OpSub, F, fi.f(blockElems), fi.un(OpCvtF32, F, fi.un(OpBitcast, I, fi.coord(2)))), fi.f(2))
		key = fi.op(OpAdd, F, key, fi.sel(F, fi.op(OpLess, ir.Pred, v, fi.f(0)), fi.f(1), fi.f(0)))
		fi.sel(F, below, fi.f(0), key)
		var s exprs
		neg := s.op(OpAnd, U, s.un(OpBitcast, U, s.un(OpCvtI32, I, s.at(1, blocks, s.coord(0), s.coord(1)))), s.u(1))
		m := s.at(0, blocks, s.coord(0), s.coord(1))
		mx := s.sel(F, s.op(OpLess, ir.Pred, s.u(0), neg), s.neg(m), m)
		s.op(OpMul, F, mx, s.f(-1/bias))
		var k exprs
		r, i := k.coord(0), k.coord(1)
		t := k.op(OpMul, F, elem(&k, r, i), k.at(1, blocks, r, k.op(OpShr, U, i, k.u(5))))
		q := k.un(OpBitcast, U, k.un(OpCvtI32, I, k.op(OpAdd, F, t, k.f(bias+0.5))))
		k.op(OpMin, U, q, k.u(uint32(levels)))
		prims = append(prims,
			Prim{Kind: PrimReduce, Shape: []int{R, nb, blockElems}, In: []Operand{src}, Out: amax, Combine: OpMax, Expr: a.e},
			Prim{Kind: PrimReduce, Shape: []int{R, nb, blockElems}, In: []Operand{src, amax}, Out: first, Combine: OpMax, Expr: fi.e},
			Prim{Kind: PrimMap, Shape: blocks, In: []Operand{amax, first}, Out: d, Expr: s.e},
			Prim{Kind: PrimMap, Shape: blocks, In: []Operand{d}, Out: inv, Expr: inverse()},
			Prim{Kind: PrimMap, Shape: padded, In: []Operand{src, inv}, Out: codes, Expr: k.e})
	}
	ext := R * rw
	at := func(e *exprs, region, w int) int {
		return e.op(OpAdd, U, e.op(OpMul, U, e.coord(0), e.u(uint32(rw))), e.op(OpAdd, U, e.u(uint32(region)), w))
	}

	// The scales: each block's d (and m) as a half, then a block's d and m
	// in a word, or two blocks' d.
	halves := func(of Operand, name string) Operand {
		o := temp(name, U)
		var x exprs
		x.halfBits(x.at(0, blocks, x.coord(0), x.coord(1)))
		prims = append(prims, Prim{Kind: PrimMap, Shape: blocks, In: []Operand{of}, Out: o, Expr: x.e})
		return o
	}
	dh := halves(d, "dh")
	var h exprs
	if f.Mins() {
		mh := halves(lo, "mh")
		b := h.coord(1)
		v := h.op(OpOr, U, h.at(0, blocks, h.coord(0), b), h.op(OpShl, U, h.at(1, blocks, h.coord(0), b), h.u(16)))
		ix := at(&h, scales, b)
		prims = append(prims, Prim{Kind: PrimMap, Shape: blocks, In: []Operand{dh, mh}, Expr: h.e,
			Stores: []PrimStore{{Out: out, Index: ix, Value: v, Extent: ext}}})
	} else {
		w := h.coord(1)
		b0 := h.op(OpShl, U, w, h.u(1))
		b1 := h.op(OpMin, U, h.op(OpAdd, U, b0, h.u(1)), h.u(uint32(nb-1)))
		hi := h.sel(U, h.op(OpLess, ir.Pred, b0, h.u(uint32(nb-1))), h.at(0, blocks, h.coord(0), b1), h.u(0))
		v := h.op(OpOr, U, h.at(0, blocks, h.coord(0), b0), h.op(OpShl, U, hi, h.u(16)))
		ix := at(&h, scales, w)
		prims = append(prims, Prim{Kind: PrimMap, Shape: []int{R, (nb + 1) / 2}, In: []Operand{dh}, Expr: h.e,
			Stores: []PrimStore{{Out: out, Index: ix, Value: v, Extent: ext}}})
	}

	// The fifth bits: element j's bit j of its block's word, summed.
	if f.Fifths() {
		qh := temp("qh", I)
		var u exprs
		bit := u.op(OpAnd, U, u.op(OpShr, U, u.at(0, padded, u.coord(0), block(&u)), u.u(4)), u.u(1))
		u.un(OpBitcast, I, u.op(OpShl, U, bit, u.coord(2)))
		var p exprs
		v := p.un(OpBitcast, U, p.at(0, blocks, p.coord(0), p.coord(1)))
		ix := at(&p, fifths, p.coord(1))
		prims = append(prims,
			Prim{Kind: PrimReduce, Shape: []int{R, nb, blockElems}, In: []Operand{codes}, Out: qh, Combine: OpAdd, Type: I, Expr: u.e},
			Prim{Kind: PrimMap, Shape: blocks, In: []Operand{qh}, Expr: p.e,
				Stores: []PrimStore{{Out: out, Index: ix, Value: v, Extent: ext}}})
	}

	// The nibbles: byte j of a block element j's low four bits and element
	// j+16's high, four bytes a word, the lower byte low.
	bytes := temp("qs", U)
	var y exprs
	j := y.coord(1)
	e0 := y.op(OpAdd, U, y.op(OpShl, U, y.op(OpShr, U, j, y.u(4)), y.u(5)), y.op(OpAnd, U, j, y.u(15)))
	y.op(OpOr, U, y.op(OpAnd, U, y.at(0, padded, y.coord(0), e0), y.u(15)),
		y.op(OpShl, U, y.op(OpAnd, U, y.at(0, padded, y.coord(0), y.op(OpAdd, U, e0, y.u(16))), y.u(15)), y.u(4)))
	var p exprs
	nbytes := []int{R, nb * 16}
	word := -1
	for i := 0; i < 4; i++ {
		b := p.at(0, nbytes, p.coord(0), p.op(OpAdd, U, p.op(OpShl, U, p.coord(1), p.u(2)), p.u(uint32(i))))
		if i == 0 {
			word = b
			continue
		}
		word = p.op(OpOr, U, word, p.op(OpShl, U, b, p.u(uint32(8*i))))
	}
	ix := at(&p, nibbles, p.coord(1))
	return append(prims,
		Prim{Kind: PrimMap, Shape: nbytes, In: []Operand{codes}, Out: bytes, Expr: y.e},
		Prim{Kind: PrimMap, Shape: []int{R, nb * 4}, In: []Operand{bytes}, Expr: p.e,
			Stores: []PrimStore{{Out: out, Index: ix, Value: word, Extent: ext}}})
}

// halfBits is a's binary16 in the low half of a word, rounded to nearest
// even, a NaN its sign and 0x7E00, in integer arithmetic: the word a
// block's scale is stored in, whatever lanes the device keeps halves in.
//
//	s = x & 0x80000000; x ^= s
//	n = (x + (x>>13 & 1) + 0xC8000FFF) >> 13     a normal, rebiased
//	d = bits(float(x) + 0.5) - bits(0.5)          a subnormal, rounded by the add
//	o = x < 2^-14 ? d : n
//	o = x > 0x477FFFFF ? (x > Inf ? 0x7E00 : 0x7C00) : o
//	h = o | s>>16
func (x *exprs) halfBits(a int) int {
	U := ir.U32
	b := x.un(OpBitcast, U, a)
	s := x.op(OpAnd, U, b, x.u(0x80000000))
	m := x.op(OpXor, U, b, s)
	n := x.op(OpShr, U, x.op(OpAdd, U, x.op(OpAdd, U, m, x.op(OpAnd, U, x.op(OpShr, U, m, x.u(13)), x.u(1))), x.u(0xC8000FFF)), x.u(13))
	d := x.op(OpSub, U, x.un(OpBitcast, U, x.op(OpAdd, ir.F32, x.un(OpBitcast, ir.F32, m), x.f(0.5))), x.u(0x3F000000))
	o := x.sel(U, x.op(OpLess, ir.Pred, m, x.u(0x38800000)), d, n)
	inf := x.sel(U, x.op(OpLess, ir.Pred, x.u(0x7F800000), m), x.u(0x7E00), x.u(0x7C00))
	o = x.sel(U, x.op(OpLess, ir.Pred, x.u(0x477FFFFF), m), inf, o)
	return x.op(OpOr, U, o, x.op(OpShr, U, s, x.u(16)))
}

// blockElem is element e of a row held in format f's regions, its words
// read by word: the element's code from its nibble (and fifth bit), less
// the format's bias times the block's scale, or times the scale plus the
// block's minimum, as ggml's dequantizer reads it.
func blockElem(x *exprs, f graph.BlockFmt, w int, word func(i int) int, e int) int {
	U, F := ir.U32, ir.F32
	scales, fifths, nibbles := graph.BlockRegions(f, w)
	b := x.op(OpShr, U, e, x.u(5))
	j := x.op(OpAnd, U, e, x.u(31))
	var dw, d int
	if f.Mins() {
		dw = word(x.op(OpAdd, U, x.u(uint32(scales)), b))
		d = x.un(OpWidenHalf, F, dw)
	} else {
		dw = word(x.op(OpAdd, U, x.u(uint32(scales)), x.op(OpShr, U, b, x.u(1))))
		d = x.un(OpWidenHalf, F, x.op(OpShr, U, dw, x.op(OpShl, U, x.op(OpAnd, U, b, x.u(1)), x.u(4))))
	}
	nw := word(x.op(OpAdd, U, x.u(uint32(nibbles)), x.op(OpAdd, U, x.op(OpShl, U, b, x.u(2)), x.op(OpShr, U, x.op(OpAnd, U, j, x.u(15)), x.u(2)))))
	shift := x.op(OpAdd, U, x.op(OpShl, U, x.op(OpAnd, U, j, x.u(3)), x.u(3)), x.op(OpShl, U, x.op(OpShr, U, j, x.u(4)), x.u(2)))
	q := x.op(OpAnd, U, x.op(OpShr, U, nw, shift), x.u(15))
	if f.Fifths() {
		hw := word(x.op(OpAdd, U, x.u(uint32(fifths)), b))
		q = x.op(OpOr, U, q, x.op(OpShl, U, x.op(OpAnd, U, x.op(OpShr, U, hw, j), x.u(1)), x.u(4)))
	}
	if f.Mins() {
		m := x.un(OpWidenHalf, F, x.op(OpShr, U, dw, x.u(16)))
		return x.op(OpAdd, F, x.op(OpMul, F, x.un(OpCvtF32, F, x.un(OpBitcast, ir.I32, q)), d), m)
	}
	bias := uint32(8)
	if f.Fifths() {
		bias = 16
	}
	c := x.un(OpCvtF32, F, x.un(OpBitcast, ir.I32, x.op(OpSub, U, q, x.u(bias))))
	return x.op(OpMul, F, c, d)
}

// decomposeWidenBlockRows is each element of rows held in Attr.Block's
// blocks as a float.
func decomposeWidenBlockRows(o graph.Op, c *Context) ([]Prim, error) {
	f := o.Attr.Block
	if len(o.In) != 1 || len(o.Out) != 1 || f.Bytes() == 0 || c.width(o.In[0]) != graph.BlockRowWords(f, c.width(o.Out[0])) {
		return nil, fmt.Errorf("floats of %d %v rows into %d outputs", len(o.In), f, len(o.Out))
	}
	W := c.width(o.Out[0])
	sh := c.rows(o.In[0])
	in := c.value(o.In[0])
	in.Elem = ir.U32
	var x exprs
	r := x.coord(0)
	blockElem(&x, f, W, func(i int) int { return x.at(0, sh, r, i) }, x.coord(1))
	return []Prim{{Kind: PrimMap, Shape: []int{sh[0], W}, In: []Operand{in}, Out: c.value(o.Out[0]), Expr: x.e}}, nil
}
