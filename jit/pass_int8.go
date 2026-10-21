package jit

import (
	"math"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/ir"
)

// Int8 adds the int8 form of a contraction's input where the input is a row
// a map of the same part produced, the part is of a kind in Parts and the
// weight's format is one of Formats (every format where Formats is nil):
// windows of Window by an exact
// scale amax/127, codes rounded half away from zero packed Word to a word,
// and the codes' sums a Sums. The form is naive IR placed after the map.
type Int8 struct {
	Window, Word, Sums int
	Formats            []jlm.Type
	Parts              []PartKind
}

func (Int8) Name() string { return "int8" }

func (q Int8) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	var out []*Kernel
	done := map[Operand]bool{}
	for i, k := range ks {
		out = append(out, k)
		if k.Kind != PrimMap || len(k.Loops) != 0 || !q.covers(k.Part) {
			continue
		}
		for _, s := range k.Stores {
			o := k.Bufs[s.Buf].Operand
			sh := k.Bufs[s.Buf].Shape
			if s.Barrier || s.Run || o.Kind != OperandValue || done[o] || !k.storesOwn(s) || len(sh) != 2 ||
				sh[1]%q.Window != 0 || q.Window%q.Sums != 0 || q.Sums%q.Word != 0 || !q.consumed(ks[i+1:], o) {
				continue
			}
			done[o] = true
			for _, n := range q.form(o, sh[0], sh[1]) {
				n.Passes, n.Part = []string{q.Name()}, k.Part
				out = append(out, n)
			}
		}
	}
	return out, nil
}

func (q Int8) covers(p PartKind) bool {
	for _, x := range q.Parts {
		if x == p {
			return true
		}
	}
	return false
}

// consumed reports whether a contraction of a format the pass covers reads o.
func (q Int8) consumed(ks []*Kernel, o Operand) bool {
	for _, k := range ks {
		if k.Kind != PrimContract || !k.reads(o) {
			continue
		}
		for _, b := range k.Bufs {
			if b.Operand.Kind != OperandWeight {
				continue
			}
			if q.Formats == nil {
				return true
			}
			for _, f := range q.Formats {
				if f == b.Operand.Format {
					return true
				}
			}
		}
	}
	return false
}

// form is the int8 form of value v, rows of w, as naive kernels: the
// windows' amax, their scales, the scales' inverses, the codes, the packed
// words, the words' sums and the sums a Sums, the scales and the sums one
// buffer, windows first.
func (q Int8) form(v Operand, rows, w int) []*Kernel {
	U, F, I := ir.U32, ir.F32, ir.I32
	nw, nwd, ns := w/q.Window, w/q.Word, w/q.Sums
	scratch := func(name string, t ir.Type) Operand {
		return Operand{Kind: OperandScratch, Name: v.Name + "." + name, Elem: t}
	}
	amax, inv, codes := scratch("amax", F), scratch("inv", F), scratch("q", I)
	wsum := scratch("wsum", I)
	words := Operand{Kind: OperandScratch, Name: v.Name + ".q8a", Elem: U}
	scales := Operand{Kind: OperandScratch, Name: v.Name + ".q8x"}
	in := func(k *Kernel, o Operand, shape []int) int {
		return k.buf(Buf{Kind: BufIn, Elem: elemOf(o.Elem), Operand: o, Shape: shape})
	}
	outBuf := func(k *Kernel, o Operand, shape []int) int {
		return k.buf(Buf{Kind: BufOut, Elem: elemOf(o.Elem), Operand: o, Shape: shape})
	}
	space := func(name string, kind PrimKind, ext int) (*Kernel, int, int) {
		k := &Kernel{Name: name, Kind: kind, Space: []int{rows, ext}}
		return k, k.add(Node{Op: OpCoord, A: 0, Type: U}), k.add(Node{Op: OpCoord, A: 1, Type: U})
	}
	load := func(k *Kernel, b int, ax ...int) int {
		return k.add(Node{Op: OpLoad, Type: k.Bufs[b].Elem, Buf: b, Ax: ax})
	}
	at := func(k *Kernel, r, x, m int) int {
		return k.bin(OpAdd, U, k.bin(OpMul, U, x, k.u32(m)), r)
	}

	// The windows' amax: a reduction over each window of |x|.
	ka, r, c := space("reduce", PrimReduce, nw)
	l := ka.loop(q.Window, -1)
	it := ka.add(Node{Op: OpIter, At: l, Type: U})
	x := load(ka, in(ka, v, []int{rows, w}), r, at(ka, it, c, q.Window))
	acc := ka.add(Node{Op: OpFold, At: l, Type: F, B: ka.f32(0)})
	ka.Nodes[acc].A = ka.bin(OpMax, F, acc, abs(ka, x))
	ka.Stores = []Store{{Buf: outBuf(ka, amax, []int{rows, nw}), Val: acc, At: -1, Ax: []int{r, c}}}

	// The scales, the flat buffer's first rows*nw words.
	kd, r, c := space("map", PrimMap, nw)
	a := load(kd, in(kd, amax, []int{rows, nw}), r, c)
	sb := outBuf(kd, scales, nil)
	kd.Stores = []Store{{Buf: sb, Ix: at(kd, c, r, nw), Val: scale(kd, a), At: -1}}

	// The inverses.
	kr, r, c := space("map", PrimMap, nw)
	a = load(kr, in(kr, amax, []int{rows, nw}), r, c)
	dv := kr.load(in(kr, scales, nil), at(kr, c, r, nw))
	kr.Stores = []Store{{Buf: outBuf(kr, inv, []int{rows, nw}), Val: inverse(kr, a, dv), At: -1, Ax: []int{r, c}}}

	// The codes: x times its window's inverse, rounded half away from zero.
	kc, r, c := space("map", PrimMap, w)
	t := kc.bin(OpMul, F, load(kc, in(kc, v, []int{rows, w}), r, c),
		load(kc, in(kc, inv, []int{rows, nw}), r, kc.bin(OpDiv, U, c, kc.u32(q.Window))))
	sign := kc.bin(OpAnd, U, kc.un(OpBitcast, U, t), kc.u32(0x80000000))
	half := kc.un(OpBitcast, F, kc.bin(OpAdd, U, sign, kc.u32(0x3f000000)))
	code := kc.un(OpCvtI32, I, kc.bin(OpAdd, F, t, half))
	kc.Stores = []Store{{Buf: outBuf(kc, codes, []int{rows, w}), Val: code, At: -1, Ax: []int{r, c}}}

	// The words: Word codes' low bytes each.
	kp, r, c := space("map", PrimMap, nwd)
	cb := in(kp, codes, []int{rows, w})
	packed := -1
	for j := 0; j < q.Word; j++ {
		e := load(kp, cb, r, kp.bin(OpAdd, U, kp.bin(OpMul, U, c, kp.u32(q.Word)), kp.u32(j)))
		b := kp.bin(OpAnd, U, e, kp.u32(255))
		if j == 0 {
			packed = b
			continue
		}
		packed = kp.bin(OpAdd, U, packed, kp.bin(OpShl, U, b, kp.u32(8*j)))
	}
	kp.Stores = []Store{{Buf: outBuf(kp, words, []int{rows, nwd}), Val: packed, At: -1, Ax: []int{r, c}}}

	// The words' sums of their codes.
	kw, r, c := space("reduce", PrimReduce, nwd)
	l = kw.loop(q.Word, -1)
	it = kw.add(Node{Op: OpIter, At: l, Type: U})
	e := load(kw, in(kw, codes, []int{rows, w}), r, at(kw, it, c, q.Word))
	acc = kw.add(Node{Op: OpFold, At: l, Type: I, B: kw.add(Node{Op: OpConst, Type: I})})
	kw.Nodes[acc].A = kw.bin(OpAdd, I, acc, e)
	kw.Stores = []Store{{Buf: outBuf(kw, wsum, []int{rows, nwd}), Val: acc, At: -1, Ax: []int{r, c}}}

	// The sums a Sums, from the first word's on, after the scales.
	per := q.Sums / q.Word
	ks, r, c := space("reduce", PrimReduce, ns)
	wb := in(ks, wsum, []int{rows, nwd})
	first := ks.un(OpCvtF32, F, load(ks, wb, r, ks.bin(OpMul, U, c, ks.u32(per))))
	l = ks.loop(per-1, -1)
	it = ks.add(Node{Op: OpIter, At: l, Type: U})
	nx := ks.un(OpCvtF32, F, load(ks, wb, r, ks.bin(OpAdd, U, at(ks, it, c, per), ks.u32(1))))
	acc = ks.add(Node{Op: OpFold, At: l, Type: F, B: first})
	ks.Nodes[acc].A = ks.bin(OpAdd, F, acc, nx)
	ks.Stores = []Store{{Buf: outBuf(ks, scales, nil), Ix: ks.bin(OpAdd, U, at(ks, c, r, ns), ks.u32(rows*nw)), Val: acc, At: -1}}

	return []*Kernel{ka, kd, kr, kc, kp, kw, ks}
}

// abs is |x| by its sign bit.
func abs(k *Kernel, x int) int {
	return k.un(OpBitcast, ir.F32, k.bin(OpAnd, ir.U32, k.un(OpBitcast, ir.U32, x), k.u32(0x7fffffff)))
}

func (k *Kernel) fconst(v float32) int { return k.f32(math.Float32bits(v)) }

// scale is amax/127 correctly rounded: the mantissa's quotient in integers
// where the exponent allows it, the float division where it does not.
func scale(k *Kernel, amax int) int {
	U := ir.U32
	op := func(o Op, a, b int) int { return k.bin(o, U, a, b) }
	b := k.un(OpBitcast, U, amax)
	e := op(OpShr, b, k.u32(23))
	m := op(OpAdd, op(OpAnd, b, k.u32(0x7fffff)), k.u32(0x800000))
	q := op(OpDiv, op(OpShl, m, k.u32(8)), k.u32(127))
	sh := op(OpAdd, op(OpShr, q, k.u32(25)), k.u32(1))
	half := op(OpShl, k.u32(1), op(OpSub, sh, k.u32(1)))
	mask := op(OpSub, op(OpShl, half, k.u32(1)), k.u32(1))
	round := k.add(Node{Op: OpSelect, Type: U, A: k.add(Node{Op: OpLess, Type: ir.Pred, A: op(OpAnd, q, mask), B: half}),
		B: k.u32(0), C: k.u32(1)})
	q = op(OpAdd, op(OpShr, q, sh), round)
	over := op(OpShr, q, k.u32(24))
	exp := op(OpAdd, op(OpSub, op(OpAdd, e, sh), k.u32(8)), over)
	bits := op(OpAdd, op(OpShl, exp, k.u32(23)), op(OpAnd, op(OpShr, q, over), k.u32(0x7fffff)))
	exact := k.add(Node{Op: OpLess, Type: ir.Pred, A: op(OpSub, e, k.u32(8)), B: k.u32(247)})
	return k.add(Node{Op: OpSelect, Type: ir.F32, A: exact, B: k.un(OpBitcast, ir.F32, bits),
		C: k.bin(OpDiv, ir.F32, amax, k.fconst(127))})
}

// inverse is 1/d for d = scale(amax): 127/amax refined once by Newton, then
// corrected to the float nearest 1/d by testing the four candidates below
// it in integers where every exponent allows, zero where d is not positive.
func inverse(k *Kernel, amax, d int) int {
	U, F := ir.U32, ir.F32
	op := func(o Op, a, b int) int { return k.bin(o, U, a, b) }
	less := func(a, b int) int { return k.add(Node{Op: OpLess, Type: ir.Pred, A: a, B: b}) }
	sel := func(t ir.Type, p, a, b int) int { return k.add(Node{Op: OpSelect, Type: t, A: p, B: a, C: b}) }
	r := k.bin(OpDiv, F, k.fconst(127), amax)
	r = k.bin(OpAdd, F, r, k.bin(OpMul, F, r, k.bin(OpSub, F, k.fconst(1), k.bin(OpMul, F, d, r))))
	pos := less(k.fconst(0), d)
	ob, db := k.un(OpBitcast, U, k.fconst(1)), k.un(OpBitcast, U, d)
	oe, de := op(OpShr, ob, k.u32(23)), op(OpShr, db, k.u32(23))
	om := op(OpAdd, op(OpAnd, ob, k.u32(0x7fffff)), k.u32(0x800000))
	dm := op(OpAdd, op(OpAnd, db, k.u32(0x7fffff)), k.u32(0x800000))
	rb := k.un(OpBitcast, U, r)
	base := op(OpSub, rb, k.u32(2))
	acc := base
	for j := 0; j < 4; j++ {
		c := op(OpAdd, base, k.u32(j))
		sh := op(OpSub, op(OpAdd, oe, k.u32(151)), op(OpAdd, op(OpShr, c, k.u32(23)), de))
		cm := op(OpAdd, op(OpShl, op(OpAdd, op(OpAnd, c, k.u32(0x7fffff)), k.u32(0x800000)), k.u32(1)), k.u32(1))
		diff := op(OpSub, op(OpSub, op(OpShl, om, sh), op(OpMul, cm, dm)), k.u32(1))
		acc = op(OpAdd, acc, sel(U, less(diff, k.u32(0x7fffffff)), k.u32(1), k.u32(0)))
	}
	okO := less(op(OpSub, oe, k.u32(1)), k.u32(254))
	okD := less(op(OpSub, de, k.u32(1)), k.u32(254))
	okR := less(op(OpSub, op(OpShr, rb, k.u32(23)), k.u32(2)), k.u32(252))
	v := sel(F, okR, k.un(OpBitcast, F, acc), r)
	v = sel(F, okD, v, r)
	v = sel(F, okO, v, r)
	return sel(F, pos, v, k.fconst(0))
}
