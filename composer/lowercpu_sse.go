package composer

import (
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/composer/x86"
)

// The SSE lowering is the x86 lowering on a host without AVX2, FMA and F16C:
// a kernel formed four floats a vector (the profile's Narrow) in the legacy
// encodings, SSE4.1 and SSSE3 at most. Legacy SSE is two-operand and
// destructive, and its 16-byte memory operands fault unaligned, so a memory
// operand is loaded first; a scratch vector the kernel's registers leave none
// for is borrowed, its value kept below the red zone and put back after.

// sseSave is where an operand's register is kept while it makes a result
// whose register is the other operand's.
func sseSave() x86.Mem { return x86.At(x86.RSP, -144) }

// sseSlot is where borrowed vector register j is kept.
func sseSlot(j int) x86.Mem { return x86.At(x86.RSP, int32(-160-16*j)) }

// sseNarrow reports whether the lowering emits the legacy SSE encodings: a
// stated ISA short of the AVX2 tier, as the CPU profile's Narrow passes are
// chosen.
func sseNarrow(isa x86.ISA) bool { return isa != 0 && isa&x86.ISATierAVX2 != x86.ISATierAVX2 }

// borrow is n vector registers none of avoid is, their values kept until
// giveBack.
func (e *x86Lowerer) borrow(n int, avoid ...x86.Reg) []x86.Reg {
	var out []x86.Reg
	for r := x86.Reg(0); r < 16 && len(out) < n; r++ {
		taken := false
		for _, x := range avoid {
			taken = taken || x == r
		}
		if !taken {
			e.a.MOVUPSStore(sseSlot(len(out)), r)
			out = append(out, r)
		}
	}
	return out
}

func (e *x86Lowerer) giveBack(rs []x86.Reg) {
	for j, r := range rs {
		e.a.MOVUPSLoad(r, sseSlot(j))
	}
}

// bin is dst = x op y in the two-operand form: where dst is y's register and
// not x's, y and x swap where op commutes; where not, x's register makes the
// result and is put back after.
func (e *x86Lowerer) bin(op func(dst, x, y x86.Reg), commutes bool, dst, x, y x86.Reg) {
	switch {
	case dst != y || dst == x:
		op(dst, x, y)
	case commutes:
		op(dst, y, x)
	default:
		e.a.MOVUPSStore(sseSave(), x)
		op(x, x, y)
		e.a.MOVAPS(dst, x)
		e.a.MOVUPSLoad(x, sseSave())
	}
}

// binMem is dst = x op the vector at m: loaded into dst where dst is not x,
// else into a borrowed register.
func (e *x86Lowerer) binMem(op func(dst, x, y x86.Reg), dst, x x86.Reg, m x86.Mem) {
	if dst != x {
		e.a.MOVDQULoad(dst, m)
		op(dst, dst, x)
		return
	}
	t := e.borrow(1, dst)
	e.a.MOVDQULoad(t[0], m)
	op(dst, x, t[0])
	e.giveBack(t)
}

// splat sets each word of r to v, through the reload register.
func (e *x86Lowerer) splat(r x86.Reg, v uint32) {
	e.a.MOVimm32(reloadReg, int32(v))
	e.a.MOVDToX(r, reloadReg)
	e.a.PSHUFD(r, r, 0)
}

// widenSSE turns the binary16 in the low half of each of x's words into its
// float, exactly: (h&0x7FFF)<<13 times 2^112 rebiases a normal and makes a
// subnormal's denormal exact, an all-ones exponent is set back to all ones,
// and the sign is or'd in.
func (e *x86Lowerer) widenSSE(x x86.Reg) {
	a := e.a
	t := e.borrow(3, x)
	t0, s, m := t[0], t[1], t[2]
	a.MOVAPS(s, x)
	a.PSLLD(x, x, 17)
	a.PSRLD(x, x, 4)
	a.PSLLD(t0, x, 4)
	a.PSRAD(t0, t0, 27)
	a.PCMPEQD(m, m, m)
	a.PCMPEQD(t0, t0, m)
	a.PSRLD(t0, t0, 24)
	a.PSLLD(t0, t0, 23)
	e.splat(m, 0x77800000)
	a.MULPS(x, x, m)
	a.ORPS(x, x, t0)
	a.PSRLD(t0, s, 15)
	a.PSLLD(t0, t0, 31)
	a.ORPS(x, x, t0)
	e.giveBack(t)
}

// narrowSSE turns each of x's floats into its binary16, rounded to nearest
// even, in the low half of its word, a NaN its sign and 0x7E00:
//
//	s = x & 0x80000000; x ^= s
//	n = (x + (x>>13 & 1) + 0xC8000FFF) >> 13     a normal, rebiased
//	d = bits(float(x) + 0.5) - bits(0.5)          a subnormal, rounded by the add
//	o = x < 2^-14 ? d : n
//	o = x > 0x477FFFFF ? (x > Inf ? 0x7E00 : 0x7C00) : o
//	h = o | s>>16
func (e *x86Lowerer) narrowSSE(x x86.Reg) {
	a := e.a
	t := e.borrow(5, x)
	s, n, d, c, m := t[0], t[1], t[2], t[3], t[4]
	a.MOVAPS(s, x)
	a.PSRLD(s, s, 31)
	a.PSLLD(s, s, 31)
	a.PXOR(x, x, s)
	a.PSRLD(n, x, 13)
	a.PSLLD(n, n, 31)
	a.PSRLD(n, n, 31)
	a.PADDD(n, n, x)
	e.splat(c, 0xC8000FFF)
	a.PADDD(n, n, c)
	a.PSRLD(n, n, 13)
	e.splat(c, 0x3F000000)
	a.MOVAPS(d, x)
	a.ADDPS(d, d, c)
	a.PSUBD(d, d, c)
	e.splat(m, 0x38800000)
	a.PCMPGTD(m, m, x)
	a.PAND(d, d, m)
	a.PANDN(m, m, n)
	a.POR(d, d, m)
	e.splat(c, 0x477FFFFF)
	a.MOVAPS(m, x)
	a.PCMPGTD(m, m, c)
	e.splat(c, 0x7F800000)
	a.PCMPGTD(x, x, c)
	a.PSRLD(x, x, 31)
	a.PSLLD(x, x, 9)
	e.splat(c, 0x7C00)
	a.POR(x, x, c)
	a.PAND(x, x, m)
	a.PANDN(m, m, d)
	a.POR(x, x, m)
	a.PSRLD(s, s, 16)
	a.POR(x, x, s)
	e.giveBack(t)
}

// broadcastSSE sets every lane of r to the word at m.
func (e *x86Lowerer) broadcastSSE(r x86.Reg, m x86.Mem) {
	e.a.MOVDLoad(r, m)
	e.a.PSHUFD(r, r, 0)
}

func (e *x86Lowerer) nodeSSE(i int) {
	k, a := e.k, e.a
	nd := k.Nodes[i]
	switch nd.Op {
	case OpParam:
		if nd.Reg == 0 && e.reloaded(i) || nd.Spill < 0 {
			return
		}
		a.MOVLoad(e.reg(i), x86.At(x86.RDI, int32(nd.Imm)))
	case OpFold:
		init := k.Nodes[nd.B]
		switch {
		case init.Op == OpFold:
			if nd.Reg != init.Reg {
				e.fail("a sum carried on out of its register")
			}
		case init.Op == OpConst && init.Imm == 0 && e.gpr(i):
			r := e.reg(i)
			a.XORQ(r, r)
		case init.Op == OpConst && init.Imm == 0:
			r := e.reg(i)
			a.PXOR(r, r, r)
		case init.Op == OpLoad && e.home[nd.B] < 0 && init.Imm > 0:
			a.MOVDQULoad(e.reg(i), e.addr(init.Buf, init.A, init.Ptr))
		case init.Op == OpLoad && e.home[nd.B] < 0:
			e.broadcastSSE(e.reg(i), e.addr(init.Buf, init.A, init.Ptr))
		case init.Op == OpConst && e.words[nd.B] && init.Reg > 0:
			a.MOVAPS(e.reg(i), e.reg(nd.B))
		case init.Op != OpConst && (init.Reg > 0 || init.Spill != 0) && (e.home[nd.B] < 0 || !k.within(e.home[nd.B], nd.At)):
			r, from := e.reg(i), e.reg(nd.B)
			switch {
			case r == from:
			case e.gpr(i):
				a.MOVQ(r, from)
			default:
				a.MOVAPS(r, from)
			}
		default:
			e.fail("a sum from %s", init.Op)
		}
	case OpLoad:
		if nd.Mem {
			return
		}
		home := e.home[i]
		m := e.addr(nd.Buf, nd.A, nd.Ptr)
		if e.gpr(i) && !k.Bufs[nd.Buf].Half {
			a.MOVLLoad(e.reg(i), m)
			return
		}
		if k.Bufs[nd.Buf].Half {
			if e.lanes(home) == 0 {
				r := e.reg(i)
				a.PXOR(r, r, r)
				a.PINSRWLoad(r, r, m, 0)
			} else {
				a.PMOVZXWDLoad(e.reg(i), m)
			}
			e.widenSSE(e.reg(i))
			return
		}
		switch {
		case home < 0:
			e.broadcastSSE(e.reg(i), m)
		case e.lanes(home) > 0:
			a.MOVDQULoad(e.reg(i), m)
		default:
			a.MOVSSLoad(e.reg(i), m)
		}
	case OpBitcast:
		switch {
		case e.gpr(nd.A) && !e.gpr(i):
			a.MOVDToX(e.reg(i), e.reg(nd.A))
		case e.gpr(nd.A) && e.reg(i) != e.reg(nd.A):
			a.MOVQ(e.reg(i), e.reg(nd.A))
		case k.BitsGPR && k.Nodes[nd.A].Type == ir.F32 && e.gpr(i):
			a.MOVDFromX(e.reg(i), e.reg(nd.A))
		case e.reg(i) != e.reg(nd.A):
			a.MOVAPS(e.reg(i), e.reg(nd.A))
		}
	case OpDot4, OpDot4U:
		e.dot4SSE(i)
	case OpWidenHalf:
		r := e.reg(i)
		if e.gpr(nd.A) {
			a.MOVDToX(r, e.reg(nd.A))
		} else {
			a.MOVAPS(r, e.reg(nd.A))
		}
		e.widenSSE(r)
	case OpConst:
		if nd.Type == ir.F32 && nd.Imm == 0 && nd.Reg > 0 {
			r := e.reg(i)
			a.PXOR(r, r, r)
			return
		}
		if !e.words[i] || k.dotZero(i) {
			return
		}
		if nd.Tmp == 0 {
			e.fail("a splat with no scratch word")
			return
		}
		t := x86.Reg(nd.Tmp - 1)
		a.MOVimm32(t, int32(uint32(nd.Imm)))
		a.MOVDToX(e.reg(i), t)
		a.PSHUFD(e.reg(i), e.reg(i), 0)
	case OpHalf:
		r := e.reg(i)
		a.MOVAPS(r, e.reg(nd.A))
		e.narrowSSE(r)
		if e.lanes(e.home[i]) > 0 {
			a.PACKUSDW(r, r, r)
		}
	case OpGtW:
		e.bin(a.PCMPGTW, false, e.reg(i), e.reg(nd.A), e.reg(nd.B))
	case OpLess:
		if k.Nodes[nd.A].Type != ir.F32 && !e.words[i] && !k.needsVector(i) {
			e.lessGPR(i)
			return
		}
		if k.Nodes[nd.A].Type != ir.F32 {
			e.bin(a.PCMPGTD, false, e.reg(i), e.reg(nd.B), e.reg(nd.A))
			return
		}
		e.bin(func(d, x, y x86.Reg) { a.CMPPS(d, x, y, 1) }, false, e.reg(i), e.reg(nd.A), e.reg(nd.B))
	case OpSelect:
		if nd.Tmp == 0 {
			e.fail("a select with no scratch register")
			return
		}
		t := x86.Reg(nd.Tmp - 1)
		e.bin(a.PANDN, false, t, e.reg(nd.A), e.reg(nd.C))
		a.PAND(e.reg(i), e.reg(nd.A), e.reg(nd.B))
		a.POR(e.reg(i), e.reg(i), t)
	case OpAndNot:
		if e.gpr(i) {
			e.andNotGPR(i)
			return
		}
		e.bin(a.PANDN, false, e.reg(i), e.reg(nd.B), e.reg(nd.A))
	case OpOr:
		if e.gpr(i) {
			e.integer(i)
			return
		}
		a.POR(e.reg(i), e.reg(nd.A), e.reg(nd.B))
	case OpXor:
		if e.gpr(i) {
			e.integer(i)
			return
		}
		a.PXOR(e.reg(i), e.reg(nd.A), e.reg(nd.B))
	case OpDupEven:
		a.MOVSLDUP(e.reg(i), e.reg(nd.A))
	case OpDupOdd:
		a.MOVSHDUP(e.reg(i), e.reg(nd.A))
	case OpSwapPairs:
		a.PSHUFD(e.reg(i), e.reg(nd.A), 0xB1)
	case OpCopy:
		a.MOVAPS(e.reg(i), e.reg(nd.A))
	case OpFirstLane:
		// Lane 0 from A, the rest from B.
		if r, x, y := e.reg(i), e.reg(nd.A), e.reg(nd.B); r == x && r != y {
			a.BLENDPS(r, r, y, 0xE)
		} else {
			e.bin(func(d, p, q x86.Reg) { a.BLENDPS(d, p, q, 1) }, false, r, y, x)
		}
	case OpAddSub:
		e.bin(a.ADDSUBPS, false, e.reg(i), e.reg(nd.A), e.reg(nd.B))
	case OpSar:
		e.shiftSSE(i)
	case OpEq:
		e.bin(func(d, x, y x86.Reg) { a.CMPPS(d, x, y, 0) }, true, e.reg(i), e.reg(nd.A), e.reg(nd.B))
	case OpHMax, OpHMin:
		r, x := e.reg(i), e.reg(nd.A)
		if nd.Tmp == 0 {
			e.fail("a lane reduction with no scratch register")
			return
		}
		t := x86.Reg(nd.Tmp - 1)
		step, commutes := a.MAXPS, false
		if nd.Op == OpHMin {
			step, commutes = a.PMINSD, true
		}
		a.PSHUFD(t, x, 0x4E)
		e.bin(step, commutes, r, x, t)
		a.PSHUFD(t, r, 0xB1)
		e.bin(step, commutes, r, r, t)
	case OpMul, OpAdd, OpShr, OpAnd, OpShl:
		e.arithSSE(i)
	case OpSub, OpDiv, OpRem, OpMin, OpMax:
		if e.gpr(i) {
			e.integer(i)
			return
		}
		x, y := e.reg(nd.A), e.reg(nd.B)
		if nd.Type != ir.F32 {
			signed := nd.Type == ir.I32
			switch {
			case nd.Op == OpSub:
				e.bin(a.PSUBD, false, e.reg(i), x, y)
			case nd.Op == OpMin && signed:
				e.bin(a.PMINSD, true, e.reg(i), x, y)
			case nd.Op == OpMin:
				e.bin(a.PMINUD, true, e.reg(i), x, y)
			case nd.Op == OpMax && signed:
				e.bin(a.PMAXSD, true, e.reg(i), x, y)
			case nd.Op == OpMax:
				e.bin(a.PMAXUD, true, e.reg(i), x, y)
			default:
				e.fail("node %d: %s of integer lanes", i, nd.Op)
			}
			return
		}
		switch nd.Op {
		case OpSub:
			e.bin(a.SUBPS, false, e.reg(i), x, y)
		case OpDiv:
			e.bin(a.DIVPS, false, e.reg(i), x, y)
		case OpMin:
			e.bin(a.MINPS, false, e.reg(i), x, y)
		case OpMax:
			e.bin(a.MAXPS, false, e.reg(i), x, y)
		}
	case OpSqrt:
		a.SQRTPS(e.reg(i), e.reg(nd.A))
	case OpBroadcast:
		if x := k.Nodes[nd.A]; x.Mem {
			e.broadcastSSE(e.reg(i), e.addr(x.Buf, x.A, x.Ptr))
			return
		}
		if e.gpr(nd.A) || e.gprMask(nd.A) {
			a.MOVDToX(e.reg(i), e.reg(nd.A))
			a.PSHUFD(e.reg(i), e.reg(i), 0)
			return
		}
		a.PSHUFD(e.reg(i), e.reg(nd.A), 0)
	case OpUnpack, OpZip:
		// One 128-bit register is one lane: its zip is its unpack.
		switch {
		case e.shuffles:
			imm := byte(0x44 + 0xAA*nd.Imm)
			e.bin(func(d, x, y x86.Reg) { a.SHUFPS(d, x, y, imm) }, false, e.reg(i), e.reg(nd.A), e.reg(nd.B))
			a.PSHUFD(e.reg(i), e.reg(i), 0xD8)
		case nd.Imm == 0:
			e.bin(a.UNPCKLPS, false, e.reg(i), e.reg(nd.A), e.reg(nd.B))
		default:
			e.bin(a.UNPCKHPS, false, e.reg(i), e.reg(nd.A), e.reg(nd.B))
		}
	case OpRoundI32:
		a.CVTPS2DQ(e.reg(i), e.reg(nd.A))
	case OpCvtI32:
		a.CVTTPS2DQ(e.reg(i), e.reg(nd.A))
	case OpCvtF32:
		if src := e.reg(nd.A); e.gpr(nd.A) {
			if k.Nodes[nd.A].Type == ir.I32 {
				a.MOVSXD(src, src)
			}
			a.CVTSI2SS(e.reg(i), e.reg(i), src)
			return
		}
		a.CVTDQ2PS(e.reg(i), e.reg(nd.A))
	case OpFma:
		e.fmaSSE(i)
	case OpHSum:
		r := e.reg(i)
		if r != e.reg(nd.A) {
			e.fail("a lane sum out of its operand's register")
			return
		}
		a.HADDPS(r, r, r)
		a.HADDPS(r, r, r)
	default:
		e.fail("node %d: no SSE form of %s.%s", i, nd.Op, nd.Type)
	}
}

// dot4SSE is OpDot4 and OpDot4U: each lane's four byte products summed in
// pairs by PMADDUBSW (A's magnitudes unsigned, B with A's signs), the pairs
// by PMADDWD against ones, added to C.
func (e *x86Lowerer) dot4SSE(i int) {
	k, a := e.k, e.a
	nd := k.Nodes[i]
	if nd.Tmp == 0 || nd.Tmp2 == 0 || e.gpr(i) {
		e.fail("node %d: a byte dot outside vector lanes", i)
		return
	}
	t, u := x86.Reg(nd.Tmp-1), x86.Reg(nd.Tmp2-1)
	var p, free x86.Reg
	if nd.Op == OpDot4U || k.unsignedBytes(nd.A) {
		a.MOVAPS(t, e.reg(nd.A))
		a.PMADDUBSW(t, t, e.reg(nd.B))
		p, free = t, u
	} else {
		a.PSIGNB(t, e.reg(nd.B), e.reg(nd.A))
		a.PABSB(u, e.reg(nd.A))
		a.PMADDUBSW(u, u, t)
		p, free = u, t
	}
	ones := x86.Reg(k.Ones - 1)
	if k.Ones == 0 {
		a.PCMPEQW(free, free, free)
		a.PSRLW(free, free, 15)
		ones = free
	}
	a.PMADDWD(p, p, ones)
	if c := k.Nodes[nd.C]; c.Op == OpConst && c.Imm == 0 {
		a.MOVAPS(e.reg(i), p)
		return
	}
	a.PADDD(e.reg(i), e.reg(nd.C), p)
}

// fmaSSE is A*B + C as a multiply and an add: in the result's register
// where it is not C's, else in a borrowed one.
func (e *x86Lowerer) fmaSSE(i int) {
	k, a := e.k, e.a
	nd := k.Nodes[i]
	r := e.reg(i)
	var c x86.Reg
	cm := k.Nodes[nd.C].Op == OpLoad && k.Nodes[nd.C].Mem
	if !cm {
		c = e.reg(nd.C)
	}
	var x, y x86.Reg
	var ym *x86.Mem
	switch {
	case k.Nodes[nd.B].Mem:
		x = e.reg(nd.A)
		m := e.addr(k.Nodes[nd.B].Buf, k.Nodes[nd.B].A, k.Nodes[nd.B].Ptr)
		ym = &m
	case k.Nodes[nd.A].Mem:
		x = e.reg(nd.B)
		m := e.addr(k.Nodes[nd.A].Buf, k.Nodes[nd.A].A, k.Nodes[nd.A].Ptr)
		ym = &m
	default:
		x, y = e.reg(nd.A), e.reg(nd.B)
	}
	prod := func(d x86.Reg) {
		switch {
		case ym != nil && d == x:
			e.binMem(a.MULPS, d, x, *ym)
		case ym != nil:
			a.MOVUPSLoad(d, *ym)
			a.MULPS(d, d, x)
		default:
			e.bin(a.MULPS, true, d, x, y)
		}
	}
	add := func(d x86.Reg) {
		if cm {
			cn := k.Nodes[nd.C]
			e.binMem(a.ADDPS, d, d, e.addr(cn.Buf, cn.A, cn.Ptr))
			return
		}
		a.ADDPS(d, d, c)
	}
	if cm || r != c {
		prod(r)
		add(r)
		return
	}
	avoid := []x86.Reg{r, x}
	if ym == nil {
		avoid = append(avoid, y)
	}
	t := e.borrow(1, avoid...)
	prod(t[0])
	a.ADDPS(r, r, t[0])
	e.giveBack(t)
}

// arithSSE is OpMul, OpAdd, OpShr, OpAnd and OpShl.
func (e *x86Lowerer) arithSSE(i int) {
	k, a := e.k, e.a
	nd := k.Nodes[i]
	if b := k.Nodes[nd.B]; e.words[i] && b.Mem && (nd.Op == OpAdd || nd.Op == OpAnd) {
		op := a.PADDD
		if nd.Op == OpAnd {
			op = a.PAND
		}
		e.binMem(op, e.reg(i), e.reg(nd.A), e.addr(b.Buf, b.A, b.Ptr))
		return
	}
	if (e.words[i] || nd.Type == ir.F32) && nd.Op == OpAnd {
		a.PAND(e.reg(i), e.reg(nd.A), e.reg(nd.B))
		return
	}
	if e.words[i] && nd.Op == OpAdd {
		a.PADDD(e.reg(i), e.reg(nd.A), e.reg(nd.B))
		return
	}
	if e.words[i] && (nd.Op == OpShl || nd.Op == OpShr) {
		e.shiftSSE(i)
		return
	}
	if e.words[i] && nd.Op == OpMul {
		a.PMULLD(e.reg(i), e.reg(nd.A), e.reg(nd.B))
		return
	}
	if nd.Type != ir.F32 {
		e.integer(i)
		return
	}
	op := a.ADDPS
	if nd.Op == OpMul {
		op = a.MULPS
	}
	if b := k.Nodes[nd.B]; b.Mem {
		e.binMem(op, e.reg(i), e.reg(nd.A), e.addr(b.Buf, b.A, b.Ptr))
		return
	}
	op(e.reg(i), e.reg(nd.A), e.reg(nd.B))
}

// shiftSSE is a vector shift of words: by a constant its immediate form, by
// one count for every lane its count register's form, the count moved into
// a borrowed register; each lane by its own count, lane by lane in general
// registers through the save area.
func (e *x86Lowerer) shiftSSE(i int) {
	k, a := e.k, e.a
	nd := k.Nodes[i]
	r, x := e.reg(i), e.reg(nd.A)
	if c := k.Nodes[nd.B]; c.Op == OpConst {
		switch nd.Op {
		case OpShl:
			a.PSLLD(r, x, byte(c.Imm))
		case OpShr:
			a.PSRLD(r, x, byte(c.Imm))
		default:
			a.PSRAD(r, x, byte(c.Imm))
		}
		return
	}
	if !e.words[nd.B] {
		cnt := e.reg(nd.B)
		t := e.borrow(1, r, x)
		a.MOVDToX(t[0], cnt)
		switch nd.Op {
		case OpShl:
			a.PSLLDBy(r, x, t[0])
		case OpShr:
			a.PSRLDBy(r, x, t[0])
		default:
			a.PSRADBy(r, x, t[0])
		}
		e.giveBack(t)
		return
	}
	e.laneShifts(nd.Op, r, x, e.reg(nd.B))
}

// laneShifts is r = x shifted by y lane by lane, as SSE has no shift by a
// count a lane: both kept in the save areas, each word shifted in the reload
// register by CL on 64 bits (a word zero- or sign-extended, so a count past
// 31 leaves what the vector forms do), RCX kept in the shift-count word.
func (e *x86Lowerer) laneShifts(op Op, r, x, y x86.Reg) {
	a := e.a
	words, counts := sseSave(), sseSlot(0)
	a.MOVUPSStore(words, x)
	a.MOVUPSStore(counts, y)
	a.MOVStore(shiftCount(), x86.RCX)
	for l := range int32(4) {
		w := x86.At(x86.RSP, words.Disp+4*l)
		a.MOVLLoad(reloadReg, w)
		a.MOVLLoad(x86.RCX, x86.At(x86.RSP, counts.Disp+4*l))
		switch op {
		case OpShl:
			a.SHLCL(reloadReg)
		case OpShr:
			a.SHRCL(reloadReg)
		default:
			a.MOVSXD(reloadReg, reloadReg)
			a.SARCL(reloadReg)
		}
		a.MOVLStore(w, reloadReg)
	}
	a.MOVLoad(x86.RCX, shiftCount())
	a.MOVDQULoad(r, words)
}

func (e *x86Lowerer) storeSSE(i int) {
	s := e.k.Stores[i]
	m := e.addr(s.Buf, s.Ix, s.Ptr)
	wide := e.lanes(s.At) > 0
	if v := e.k.Nodes[s.Val]; v.Op == OpFold && e.k.Loops[v.At].Wide > 0 {
		wide = true
	}
	if s.Narrow {
		e.fail("store %d: a narrowed pair of vectors on four lanes", i)
		return
	}
	if e.gpr(s.Val) {
		if e.k.Bufs[s.Buf].Half {
			e.fail("store %d: a half from a general register", i)
			return
		}
		e.a.MOVLStore(m, e.reg(s.Val))
		return
	}
	if e.k.Bufs[s.Buf].Half {
		if wide {
			e.a.MOVQStore(m, e.reg(s.Val))
		} else {
			e.a.PEXTRWStore(m, e.reg(s.Val), 0)
		}
		return
	}
	if wide && s.Step > 1 {
		for l := range 4 {
			e.a.PEXTRDStore(laneAt(m, l*s.Step), e.reg(s.Val), byte(l))
		}
		return
	}
	if wide {
		e.a.MOVDQUStore(m, e.reg(s.Val))
		return
	}
	e.a.MOVSSStore(m, e.reg(s.Val))
}

// laneAt is m moved on by n words.
func laneAt(m x86.Mem, n int) x86.Mem {
	m.Disp += int32(4 * n)
	return m
}
