package composer

import (
	"math/bits"

	"github.com/jitllm/jitllm/composer/a64"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/composer/x86"
)

// node makes node i as x86Lowerer.node does, each operation within 128-bit
// lanes its NEON operation on both halves.
func (e *a64Lowerer) node(i int) {
	k, a, x := e.k, e.a, e.x
	nd := k.Nodes[i]
	switch nd.Op {
	case OpParam:
		if nd.Reg == 0 && x.reloaded(i) || nd.Spill < 0 {
			return
		}
		e.access(a64.A64LdrX, uint8(e.xr(i)), a64Mem{a64.X0, int32(nd.Imm)})
	case OpFold:
		init := k.Nodes[nd.B]
		switch {
		case init.Op == OpFold:
			if nd.Reg != init.Reg {
				e.fail("a sum carried on out of its register")
			}
		case init.Op == OpConst && init.Imm == 0 && x.gpr(i):
			a.MOVreg(e.xr(i), a64.XZR)
		case init.Op == OpConst && init.Imm == 0:
			e.zero(e.reg(i))
		case init.Op == OpLoad && x.home[nd.B] < 0 && init.Imm > 0:
			e.load8(e.reg(i), e.addr(init.Buf, init.A, init.Ptr))
		case init.Op == OpLoad && x.home[nd.B] < 0:
			e.splat(e.reg(i), e.addr(init.Buf, init.A, init.Ptr))
		case init.Op == OpConst && x.words[nd.B] && init.Reg > 0:
			e.mov(e.reg(i), e.reg(nd.B))
		case init.Op != OpConst && (init.Reg > 0 || init.Spill != 0) && (x.home[nd.B] < 0 || !k.within(x.home[nd.B], nd.At)):
			r, from := e.reg(i), e.reg(nd.B)
			switch {
			case r == from:
			case x.gpr(i):
				a.MOVreg(e.gp(r), e.gp(from))
			default:
				e.mov(r, from)
			}
		default:
			e.fail("a sum from %s", init.Op)
		}
	case OpLoad:
		if nd.Mem {
			return
		}
		home := x.home[i]
		m := e.addr(nd.Buf, nd.A, nd.Ptr)
		if x.gpr(i) && !k.Bufs[nd.Buf].Half {
			e.access(a64.A64LdrW, uint8(e.xr(i)), m)
			return
		}
		r := e.reg(i)
		if k.Bufs[nd.Buf].Half {
			if x.lanes(home) == 0 {
				a.MOVIzero(lo(r))
				a.LD1h0(lo(r), e.bare(m))
				a.FCVTL(lo(r), lo(r))
				a.MOVIzero(hi(r))
				return
			}
			e.vaccess(a64.A64LdrD, lo(r), m)
			e.vaccess(a64.A64LdrD, hi(r), m.plus(8))
			a.FCVTL(lo(r), lo(r))
			a.FCVTL(hi(r), hi(r))
			return
		}
		switch {
		case home < 0:
			e.splat(r, m)
		case x.lanes(home) > 0:
			e.load8(r, m)
		default:
			e.vaccess(a64.A64LdrS, lo(r), m)
			a.MOVIzero(hi(r))
		}
	case OpBitcast:
		switch {
		case x.gpr(nd.A) && !x.gpr(i):
			r := e.reg(i)
			a.FMOVsW(lo(r), e.xr(nd.A))
			a.MOVIzero(hi(r))
		case x.gpr(nd.A) && e.reg(i) != e.reg(nd.A):
			a.MOVreg(e.xr(i), e.xr(nd.A))
		case e.reg(i) != e.reg(nd.A):
			e.mov(e.reg(i), e.reg(nd.A))
		}
	case OpDot4, OpDot4U:
		e.dot4(i)
	case OpWidenHalf:
		r := e.reg(i)
		switch {
		case x.gpr(nd.A):
			a.FMOVsW(lo(r), e.xr(nd.A))
			a.FCVTL(lo(r), lo(r))
			a.MOVIzero(hi(r))
		case x.lanes(x.home[i]) == 0:
			a.FCVTL(lo(r), lo(e.reg(nd.A)))
			a.MOVIzero(hi(r))
		default:
			s := e.reg(nd.A)
			a.XTN4h(lo(r), lo(s))
			a.FCVTL(lo(r), lo(r))
			a.XTN4h(hi(r), hi(s))
			a.FCVTL(hi(r), hi(r))
		}
	case OpConst:
		if nd.Type == ir.F32 && nd.Imm == 0 && nd.Reg > 0 {
			e.zero(e.reg(i))
			return
		}
		if !x.words[i] || k.dotZero(i) {
			return
		}
		r := e.reg(i)
		a.MOVconst(a64Imm, uint64(uint32(nd.Imm)))
		a.DUPsW(lo(r), a64Imm)
		if x.feedsHalves(i) {
			a.MOVIzero(hi(r))
		} else {
			a.DUPsW(hi(r), a64Imm)
		}
	case OpHalf:
		r, s := e.reg(i), e.reg(nd.A)
		a.FCVTN4h(lo(r), lo(s))
		if x.lanes(x.home[i]) > 0 {
			a.FCVTN2h8(lo(r), hi(s))
		} else {
			// FCVTN packs the halves: a word holds its own, zero above.
			a.UXTL4s(lo(r), lo(r))
		}
		a.MOVIzero(hi(r))
	case OpGtW:
		r := e.reg(i)
		a.CMGT8h(lo(r), lo(e.reg(nd.A)), lo(e.reg(nd.B)))
		a.MOVIzero(hi(r))
	case OpLess:
		if k.Nodes[nd.A].Type != ir.F32 && !x.words[i] && !k.needsVector(i) {
			e.lessGPR(i)
			return
		}
		if k.Nodes[nd.A].Type != ir.F32 {
			each(e.reg(i), e.reg(nd.B), e.reg(nd.A), a.CMGT4s)
			return
		}
		each(e.reg(i), e.reg(nd.B), e.reg(nd.A), a.FCMGT4s)
	case OpSelect:
		r, m, t, f := e.reg(i), e.reg(nd.A), e.reg(nd.B), e.reg(nd.C)
		switch r {
		case m:
			each(r, t, f, a.BSL16b)
		case t:
			each(r, f, m, a.BIF16b)
		case f:
			each(r, t, m, a.BIT16b)
		default:
			e.mov(r, m)
			each(r, t, f, a.BSL16b)
		}
	case OpAndNot:
		if x.gpr(i) {
			e.andNotGPR(i)
			return
		}
		r := e.reg(i)
		if x.half(i) {
			a.BIC16b(lo(r), lo(e.reg(nd.A)), lo(e.reg(nd.B)))
			a.MOVIzero(hi(r))
			return
		}
		each(r, e.reg(nd.A), e.reg(nd.B), a.BIC16b)
	case OpOr:
		if x.gpr(i) {
			e.integer(i)
			return
		}
		each(e.reg(i), e.reg(nd.A), e.reg(nd.B), a.ORR16b)
	case OpDupEven:
		s := e.reg(nd.A)
		each(e.reg(i), s, s, a.TRN1s4)
	case OpDupOdd:
		s := e.reg(nd.A)
		each(e.reg(i), s, s, a.TRN2s4)
	case OpSwapPairs:
		each2(e.reg(i), e.reg(nd.A), a.REV64s4)
	case OpCopy:
		e.mov(e.reg(i), e.reg(nd.A))
	case OpFirstLane:
		r, b, s := e.reg(i), e.reg(nd.B), e.reg(nd.A)
		switch r {
		case b:
			a.INSs(lo(r), 0, lo(s), 0)
		case s:
			a.MOVvec(hi(r), hi(b))
			for l := uint8(1); l < 4; l++ {
				a.INSs(lo(r), l, lo(b), l)
			}
		default:
			e.mov(r, b)
			a.INSs(lo(r), 0, lo(s), 0)
		}
	case OpAddSub:
		r, p, q := e.reg(i), e.reg(nd.A), e.reg(nd.B)
		t := e.temp(a64Halves(r, p, q)...)
		for h := range 2 {
			d, n, m := pair(r)[h], pair(p)[h], pair(q)[h]
			a.FADD4s(t, n, m)
			a.FSUB4s(d, n, m)
			a.INSs(d, 1, t, 1)
			a.INSs(d, 3, t, 3)
		}
		e.release(t)
	case OpSar:
		r, s := e.reg(i), e.reg(nd.A)
		switch {
		case k.Nodes[nd.B].Op == OpConst:
			c := k.Nodes[nd.B].Imm
			switch {
			case c == 0:
				e.mov(r, s)
			default:
				each2(r, s, func(d, n a64.VReg) { a.SSHR4s(d, n, uint8(min(c, 31))) })
			}
		case !x.words[nd.B]:
			e.shiftBy(r, s, e.xr(nd.B), true, a.SSHL4s)
		default:
			e.shiftByLanes(r, s, e.reg(nd.B), true, a.SSHL4s)
		}
	case OpEq:
		each(e.reg(i), e.reg(nd.A), e.reg(nd.B), a.FCMEQ4s)
	case OpHMax, OpHMin:
		r, s := e.reg(i), e.reg(nd.A)
		if nd.Op == OpHMin {
			a.SMIN4s(lo(r), lo(s), hi(s))
			a.SMINP4s(lo(r), lo(r), lo(r))
			a.SMINP4s(lo(r), lo(r), lo(r))
		} else {
			a.FMAX4s(lo(r), lo(s), hi(s))
			a.FMAXP4s(lo(r), lo(r), lo(r))
			a.FMAXP4s(lo(r), lo(r), lo(r))
		}
		a.MOVvec(hi(r), lo(r))
	case OpMul, OpAdd, OpShr, OpAnd, OpShl:
		e.arith(i)
	case OpSub, OpDiv, OpRem, OpMin, OpMax:
		if x.gpr(i) {
			e.integer(i)
			return
		}
		r, p, q := e.reg(i), e.reg(nd.A), e.reg(nd.B)
		if nd.Type != ir.F32 {
			signed := nd.Type == ir.I32
			switch {
			case nd.Op == OpSub:
				each(r, p, q, a.SUB4s)
			case nd.Op == OpMin && signed:
				each(r, p, q, a.SMIN4s)
			case nd.Op == OpMin:
				each(r, p, q, a.UMIN4s)
			case nd.Op == OpMax && signed:
				each(r, p, q, a.SMAX4s)
			case nd.Op == OpMax:
				each(r, p, q, a.UMAX4s)
			default:
				e.fail("node %d: %s of integer lanes", i, nd.Op)
			}
			return
		}
		switch nd.Op {
		case OpSub:
			each(r, p, q, a.FSUB4s)
		case OpDiv:
			each(r, p, q, a.FDIV4s)
		case OpMin:
			each(r, p, q, a.FMIN4s)
		case OpMax:
			each(r, p, q, a.FMAX4s)
		}
	case OpSqrt:
		each2(e.reg(i), e.reg(nd.A), a.FSQRT4s)
	case OpBroadcast:
		r := e.reg(i)
		if s := k.Nodes[nd.A]; s.Mem {
			e.splat(r, e.addr(s.Buf, s.A, s.Ptr))
			return
		}
		if x.gpr(nd.A) || x.gprMask(nd.A) {
			g := e.xr(nd.A)
			a.DUPsW(lo(r), g)
			a.DUPsW(hi(r), g)
			return
		}
		s := e.reg(nd.A)
		a.DUPs4(hi(r), lo(s))
		a.DUPs4(lo(r), lo(s))
	case OpZip:
		r, p, q := e.reg(i), e.reg(nd.A), e.reg(nd.B)
		h := lo
		if nd.Imm != 0 {
			h = hi
		}
		if nd.Imm == 0 {
			a.ZIP2s4(hi(r), h(p), h(q))
			a.ZIP1s4(lo(r), h(p), h(q))
		} else {
			a.ZIP1s4(lo(r), h(p), h(q))
			a.ZIP2s4(hi(r), h(p), h(q))
		}
	case OpUnpack:
		if nd.Imm == 0 {
			each(e.reg(i), e.reg(nd.A), e.reg(nd.B), a.ZIP1s4)
		} else {
			each(e.reg(i), e.reg(nd.A), e.reg(nd.B), a.ZIP2s4)
		}
	case OpUpper:
		r := e.reg(i)
		a.MOVvec(lo(r), hi(e.reg(nd.A)))
		a.MOVIzero(hi(r))
	case OpRoundI32:
		each2(e.reg(i), e.reg(nd.A), a.FCVTNS4s)
	case OpCvtI32:
		each2(e.reg(i), e.reg(nd.A), a.FCVTZS4s)
	case OpXor:
		if x.gpr(i) {
			e.integer(i)
			return
		}
		each(e.reg(i), e.reg(nd.A), e.reg(nd.B), a.EOR16b)
	case OpCvtF32:
		if x.gpr(nd.A) {
			r := e.reg(i)
			if k.Nodes[nd.A].Type == ir.I32 {
				a.SCVTFsW(lo(r), e.xr(nd.A))
			} else {
				a.SCVTFsX(lo(r), e.xr(nd.A))
			}
			a.MOVIzero(hi(r))
			return
		}
		each2(e.reg(i), e.reg(nd.A), a.SCVTF4s)
	case OpFma:
		e.fma(i)
	case OpHSum:
		r := e.reg(i)
		if r != e.reg(nd.A) {
			e.fail("a lane sum out of its operand's register")
		}
		a.FADD4s(lo(r), lo(r), hi(r))
		a.FADDP4s(lo(r), lo(r), lo(r))
		a.FADDP4s(lo(r), lo(r), lo(r))
		a.MOVIzero(hi(r))
	default:
		e.fail("node %d: no arm64 form of %s.%s", i, nd.Op, nd.Type)
	}
}

// shiftBy is r = s shifted by the count in general register c, one count
// for every lane: left, or right where right (a negated count).
func (e *a64Lowerer) shiftBy(r, s x86.Reg, c a64.XReg, right bool, op func(d, n, m a64.VReg)) {
	a := e.a
	if right {
		a.NEG(a64Imm, c)
		c = a64Imm
	}
	t := e.temp(a64Halves(r, s)...)
	a.DUPsW(t, c)
	op(lo(r), lo(s), t)
	op(hi(r), hi(s), t)
	e.release(t)
}

// shiftByLanes is r = s shifted by each lane's count in c.
func (e *a64Lowerer) shiftByLanes(r, s, c x86.Reg, right bool, op func(d, n, m a64.VReg)) {
	if !right {
		each(r, s, c, op)
		return
	}
	t := e.temp(a64Halves(r, s, c)...)
	for h := range 2 {
		e.a.NEG4s(t, pair(c)[h])
		op(pair(r)[h], pair(s)[h], t)
	}
	e.release(t)
}

// arith is a multiply, an add, an and or a shift, as x86Lowerer.node makes
// them.
func (e *a64Lowerer) arith(i int) {
	k, a, x := e.k, e.a, e.x
	nd := k.Nodes[i]
	if b := k.Nodes[nd.B]; x.words[i] && b.Mem && (nd.Op == OpAdd || nd.Op == OpAnd) {
		r, s := e.reg(i), e.reg(nd.A)
		op := a.ADD4s
		if nd.Op == OpAnd {
			op = a.AND16b
		}
		e.withMem(nd.B, a64Halves(r, s), func(h int, v a64.VReg) { op(pair(r)[h], pair(s)[h], v) })
		return
	}
	if (x.words[i] || nd.Type == ir.F32) && nd.Op == OpAnd {
		r := e.reg(i)
		if x.half(i) {
			a.AND16b(lo(r), lo(e.reg(nd.A)), lo(e.reg(nd.B)))
			a.MOVIzero(hi(r))
			return
		}
		each(r, e.reg(nd.A), e.reg(nd.B), a.AND16b)
		return
	}
	if x.words[i] && nd.Op == OpAdd {
		each(e.reg(i), e.reg(nd.A), e.reg(nd.B), a.ADD4s)
		return
	}
	if x.words[i] && (nd.Op == OpShl || nd.Op == OpShr) {
		r, s := e.reg(i), e.reg(nd.A)
		switch c := k.Nodes[nd.B]; {
		case c.Op == OpConst && c.Imm >= 32:
			e.zero(r)
		case c.Op == OpConst && c.Imm == 0:
			e.mov(r, s)
		case c.Op == OpConst && nd.Op == OpShl:
			each2(r, s, func(d, n a64.VReg) { a.SHL4s(d, n, uint8(c.Imm)) })
		case c.Op == OpConst:
			each2(r, s, func(d, n a64.VReg) { a.USHR4s(d, n, uint8(c.Imm)) })
		case !x.words[nd.B]:
			e.shiftBy(r, s, e.xr(nd.B), nd.Op == OpShr, a.USHL4s)
		default:
			e.shiftByLanes(r, s, e.reg(nd.B), nd.Op == OpShr, a.USHL4s)
		}
		return
	}
	if x.words[i] && nd.Op == OpMul {
		each(e.reg(i), e.reg(nd.A), e.reg(nd.B), a.MUL4s)
		return
	}
	if nd.Type != ir.F32 {
		e.integer(i)
		return
	}
	op := a.FADD4s
	if nd.Op == OpMul {
		op = a.FMUL4s
	}
	r, s := e.reg(i), e.reg(nd.A)
	e.withMem(nd.B, a64Halves(r, s), func(h int, v a64.VReg) { op(pair(r)[h], pair(s)[h], v) })
}

// fma is r = a*b + c, fused.
func (e *a64Lowerer) fma(i int) {
	k, a := e.k, e.a
	nd := k.Nodes[i]
	if c := k.Nodes[nd.C]; c.Op != OpLoad || !c.Mem {
		r, c := e.reg(i), e.reg(nd.C)
		if r != c {
			p, q := e.reg(nd.A), e.reg(nd.B)
			if r == p || r == q {
				t := e.temp(a64Halves(r, p, q, c)...)
				for h := range 2 {
					a.MOVvec(t, pair(c)[h])
					a.FMLA4s(t, pair(p)[h], pair(q)[h])
					a.MOVvec(pair(r)[h], t)
				}
				e.release(t)
				return
			}
			e.mov(r, c)
			each(r, p, q, a.FMLA4s)
			return
		}
	}
	r := e.reg(i)
	if r != e.reg(nd.C) {
		e.fail("an fma whose result is not its addend's register")
		return
	}
	switch {
	case k.Nodes[nd.B].Mem:
		p := e.reg(nd.A)
		e.withMem(nd.B, a64Halves(r, p), func(h int, v a64.VReg) { a.FMLA4s(pair(r)[h], pair(p)[h], v) })
	case k.Nodes[nd.A].Mem:
		q := e.reg(nd.B)
		e.withMem(nd.A, a64Halves(r, q), func(h int, v a64.VReg) { a.FMLA4s(pair(r)[h], v, pair(q)[h]) })
	default:
		each(r, e.reg(nd.A), e.reg(nd.B), a.FMLA4s)
	}
}

// dot4Zero is dot4 whose addend is zero: the products' sums alone.
func (e *a64Lowerer) dot4Zero(r, p, q x86.Reg) {
	a := e.a
	if e.dot {
		if r != p && r != q {
			for _, h := range pair(r) {
				a.MOVIzero(h)
			}
			each(r, p, q, a.SDOT)
			return
		}
		t := e.temp(a64Halves(r, p, q)...)
		for h := range 2 {
			a.MOVIzero(t)
			a.SDOT(t, pair(p)[h], pair(q)[h])
			a.MOVvec(pair(r)[h], t)
		}
		e.release(t)
		return
	}
	busy := a64Halves(r, p, q)
	t0 := e.temp(busy...)
	t1 := e.temp(append(busy, t0)...)
	for h := range 2 {
		a.SMULL8b(t0, pair(p)[h], pair(q)[h])
		a.SMULL16b(t1, pair(p)[h], pair(q)[h])
		a.ADDP8h(t0, t0, t1)
		a.SADDLP8h(pair(r)[h], t0)
	}
	e.release(t1)
	e.release(t0)
}

// dot4 is each lane's four signed byte products of A and B added to C: SDOT
// where the device states it, else the products widened to halves, summed in
// pairs, and the pairs added into the lane.
func (e *a64Lowerer) dot4(i int) {
	k, a, x := e.k, e.a, e.x
	nd := k.Nodes[i]
	if x.gpr(i) {
		e.fail("node %d: a byte dot outside vector lanes", i)
		return
	}
	if nd.Op == OpDot4U && !x.k.unsignedBytes(nd.A) {
		e.fail("node %d: an unsigned byte dot, which arm64 has no form of without i8mm", i)
		return
	}
	r, p, q := e.reg(i), e.reg(nd.A), e.reg(nd.B)
	if k.dotZero(nd.C) {
		e.dot4Zero(r, p, q)
		return
	}
	c := e.reg(nd.C)
	if e.dot {
		switch {
		case r == c:
			each(r, p, q, a.SDOT)
		case r != p && r != q:
			e.mov(r, c)
			each(r, p, q, a.SDOT)
		default:
			t := e.temp(a64Halves(r, p, q, c)...)
			for h := range 2 {
				a.MOVvec(t, pair(c)[h])
				a.SDOT(t, pair(p)[h], pair(q)[h])
				a.MOVvec(pair(r)[h], t)
			}
			e.release(t)
		}
		return
	}
	busy := a64Halves(r, p, q, c)
	t0 := e.temp(busy...)
	t1 := e.temp(append(busy, t0)...)
	for h := range 2 {
		a.SMULL8b(t0, pair(p)[h], pair(q)[h])
		a.SMULL16b(t1, pair(p)[h], pair(q)[h])
		a.ADDP8h(t0, t0, t1)
		if r != c {
			a.MOVvec(pair(r)[h], pair(c)[h])
		}
		a.SADALP8h(pair(r)[h], t0)
	}
	e.release(t1)
	e.release(t0)
}

// lessGPR is an integer compare whose mask a general register holds, as
// x86Lowerer.lessGPR makes it.
func (e *a64Lowerer) lessGPR(i int) {
	k, a := e.k, e.a
	nd := k.Nodes[i]
	switch {
	case k.Nodes[nd.B].Op == OpConst:
		a.MOVconst(a64Imm, uint64(uint32(k.Nodes[nd.B].Imm)))
		a.CMPw(e.xr(nd.A), a64Imm)
		a.CSETMw(e.xr(i), a64.CondLT)
	case k.Nodes[nd.A].Op == OpConst:
		a.MOVconst(a64Imm, uint64(uint32(k.Nodes[nd.A].Imm)))
		a.CMPw(e.xr(nd.B), a64Imm)
		a.CSETMw(e.xr(i), a64.CondGT)
	default:
		p, q := e.xr(nd.A), e.xr(nd.B)
		a.CMPw(p, q)
		a.CSETMw(e.xr(i), a64.CondLT)
	}
}

// sext is a 32-bit immediate as x86 sign-extends it.
func sext(imm int64) uint64 { return uint64(int64(int32(imm))) }

// andNotGPR is a &^ b in general registers, cut back to 32 bits.
func (e *a64Lowerer) andNotGPR(i int) {
	k, a := e.k, e.a
	nd := k.Nodes[i]
	dst := e.xr(i)
	switch {
	case k.Nodes[nd.B].Op == OpConst:
		a.MOVconst(a64Imm, sext(int64(^int32(k.Nodes[nd.B].Imm))))
		a.ANDreg(dst, e.xr(nd.A), a64Imm)
	case k.Nodes[nd.A].Op == OpConst:
		a.MOVconst(a64Imm, sext(k.Nodes[nd.A].Imm))
		a.BICreg(dst, a64Imm, e.xr(nd.B))
	default:
		p, q := e.xr(nd.A), e.xr(nd.B)
		if p == q {
			a.MOVreg(dst, a64.XZR)
		} else {
			a.BICreg(dst, p, q)
		}
	}
	a.MOVw(dst, dst)
}

// keep is the condition under which a min or max keeps its first operand.
func keep(nd Node) a64.A64Cond {
	signed := nd.Type == ir.I32
	switch {
	case nd.Op == OpMin && signed:
		return a64.CondLE
	case nd.Op == OpMin:
		return a64.CondLS
	case signed:
		return a64.CondGE
	default:
		return a64.CondHS
	}
}

// integer is an integer node in general registers, as x86Lowerer.integer
// makes it: 64-bit operations, a 32-bit result that can reach past 32 bits
// cut back to them.
func (e *a64Lowerer) integer(i int) {
	k, a := e.k, e.a
	nd := k.Nodes[i]
	dst := e.xr(i)
	A, B := nd.A, nd.B
	commutes := nd.Op == OpAdd || nd.Op == OpMul || nd.Op == OpAnd || nd.Op == OpOr || nd.Op == OpXor
	if commutes && k.Nodes[A].Op == OpConst && k.Nodes[B].Op != OpConst {
		A, B = B, A
	}
	if c := k.Nodes[B]; c.Op == OpConst {
		src := e.xr(A)
		if (nd.Op == OpDiv || nd.Op == OpRem) && nd.Type == ir.U32 && c.Imm > 1 && c.Imm < 1<<31 {
			e.divConst(nd.Op, dst, src, uint32(c.Imm))
			return
		}
		switch nd.Op {
		case OpMul:
			a.MOVconst(a64Imm, sext(c.Imm))
			a.MUL(dst, src, a64Imm)
		case OpShr:
			a.LSRimm(dst, src, uint32(c.Imm&63))
		case OpShl:
			a.LSLimm(dst, src, uint8(c.Imm&63))
		case OpAnd:
			a.MOVconst(a64Imm, sext(c.Imm))
			a.ANDreg(dst, src, a64Imm)
		case OpAdd:
			e.addConstTo(dst, src, int64(int32(c.Imm)))
		case OpSub:
			e.addConstTo(dst, src, -int64(int32(c.Imm)))
		case OpXor:
			a.MOVconst(a64Imm, sext(c.Imm))
			a.EORreg(dst, src, a64Imm)
		case OpOr:
			a.MOVconst(a64Imm, sext(c.Imm))
			a.ORRreg(dst, src, a64Imm)
		case OpMin, OpMax:
			a.MOVconst(a64Imm, uint64(uint32(c.Imm)))
			a.CMPw(src, a64Imm)
			a.CSEL(dst, src, a64Imm, keep(nd))
		default:
			e.fail("node %d: no general-register form of %s by a constant", i, nd.Op)
			return
		}
	} else {
		var p a64.XReg
		if c := k.Nodes[A]; c.Op == OpConst {
			p = a64T
			a.MOVconst(p, uint64(uint32(c.Imm)))
		} else {
			p = e.xr(A)
		}
		q := e.xr(B)
		switch nd.Op {
		case OpShr:
			a.LSRV(dst, p, q)
		case OpShl:
			a.LSLV(dst, p, q)
		case OpSub:
			a.SUBreg(dst, p, q)
		case OpMin, OpMax:
			a.CMPw(p, q)
			a.CSEL(dst, p, q, keep(nd))
		case OpAdd:
			a.ADDreg(dst, p, q)
		case OpMul:
			a.MUL(dst, p, q)
		case OpDiv, OpRem:
			if nd.Type != ir.U32 {
				e.fail("node %d: no general-register form of a signed %s", i, nd.Op)
				return
			}
			// u32 words held zero-extended: the 64-bit quotient is theirs.
			a.UDIV(a64Q, p, q)
			if nd.Op == OpDiv {
				a.MOVreg(dst, a64Q)
				break
			}
			a.MUL(a64Q, a64Q, q)
			a.SUBreg(dst, p, a64Q)
		case OpAnd:
			a.ANDreg(dst, p, q)
		case OpOr:
			a.ORRreg(dst, p, q)
		case OpXor:
			a.EORreg(dst, p, q)
		default:
			e.fail("node %d: no general-register form of %s", i, nd.Op)
			return
		}
	}
	switch nd.Op {
	case OpAdd, OpSub, OpMul, OpShl, OpXor, OpOr, OpMin, OpMax:
		a.MOVw(dst, dst)
	}
}

// divConst is dst = src / c, or src % c, for a 32-bit unsigned src and a
// constant c > 1, as x86Lowerer.divConst makes it.
func (e *a64Lowerer) divConst(op Op, dst, src a64.XReg, c uint32) {
	a := e.a
	if c&(c-1) == 0 {
		if op == OpDiv {
			a.LSRimm(dst, src, uint32(bits.TrailingZeros32(c)))
		} else {
			a.MOVconst(a64Imm, uint64(c-1))
			a.ANDreg(dst, src, a64Imm)
		}
		return
	}
	l := bits.Len32(c - 1)
	m := uint32((uint64(1)<<32)*(uint64(1)<<l-uint64(c))/uint64(c) + 1)
	a.MOVconst(a64Imm, uint64(m))
	a.MUL(a64T, src, a64Imm)
	a.LSRimm(a64T, a64T, 32)
	a.SUBreg(a64Q, src, a64T)
	a.LSRimm(a64Q, a64Q, 1)
	a.ADDreg(a64Q, a64Q, a64T)
	a.LSRimm(a64Q, a64Q, uint32(l-1))
	if op == OpDiv {
		a.MOVreg(dst, a64Q)
		return
	}
	a.MOVconst(a64Imm, uint64(c))
	a.MUL(a64T, a64Q, a64Imm)
	a.SUBreg(dst, src, a64T)
}

func (e *a64Lowerer) store(i int) {
	s := e.k.Stores[i]
	m := e.addr(s.Buf, s.Ix, s.Ptr)
	wide := e.x.lanes(s.At) > 0
	if v := e.k.Nodes[s.Val]; v.Op == OpFold && e.k.Loops[v.At].Wide > 0 {
		wide = true
	}
	if e.x.gpr(s.Val) {
		if e.k.Bufs[s.Buf].Half {
			e.fail("store %d: a half from a general register", i)
			return
		}
		e.access(a64.A64StrW, uint8(e.xr(s.Val)), m)
		return
	}
	r := e.reg(s.Val)
	switch {
	case s.Narrow, e.k.Bufs[s.Buf].Half && wide:
		e.vaccess(a64.A64StrQ, lo(r), m)
	case e.k.Bufs[s.Buf].Half:
		e.vaccess(a64.A64StrH, lo(r), m)
	case wide && s.Step > 1:
		// A jammed store whose lanes are Step elements apart: lane by lane,
		// as x86Lowerer.store makes it.
		for h, v := range pair(r) {
			t := e.temp(v)
			for l := range 4 {
				e.a.DUPs4lane(t, v, uint8(l))
				e.vaccess(a64.A64StrS, t, m.plus(int32((4*h+l)*s.Step*4)))
			}
			e.release(t)
		}
	case wide:
		e.vaccess(a64.A64StrQ, lo(r), m)
		e.vaccess(a64.A64StrQ, hi(r), m.plus(16))
	default:
		e.vaccess(a64.A64StrS, lo(r), m)
	}
}

func (e *a64Lowerer) counter(c int) a64.XReg {
	r := e.k.Loops[c].Reg
	if r == 0 {
		e.fail("loop %d has no counter", c)
	}
	return e.gp(x86.Reg(max(r-1, 0)))
}

func (e *a64Lowerer) enter(c int) {
	k, a := e.k, e.a
	l := k.Loops[c]
	if l.once() {
		e.take(c)
		return
	}
	ctr := e.counter(c)
	runtime := true
	if l.Count > 0 {
		if n := k.Nodes[l.Count-1]; n.Op == OpConst {
			// A constant count of no trips still skips the loop: a tail
			// whose length is a constant multiple of the lanes has none.
			a.MOVconst(ctr, uint64(n.Imm))
			runtime = n.Imm == 0
		} else if src := e.xr(l.Count - 1); src != ctr && !e.preset[c] {
			a.MOVreg(ctr, src)
		}
		if t := c + 1; l.Lanes > 0 && t < len(k.Loops) && k.Loops[t].Of == c+1 && k.passes() && k.Loops[t].Count > 0 && e.x.madeBefore(k.Loops[t].Count-1, c) {
			if tc, src := e.counter(t), e.xr(k.Loops[t].Count-1); tc != ctr && tc != src && k.Nodes[k.Loops[t].Count-1].Op != OpConst {
				a.MOVreg(tc, src)
				e.preset[t] = true
			}
		}
	} else {
		a.MOVconst(ctr, uint64(l.Trip))
		runtime = false
	}
	e.take(c)
	e.end[c] = a.Label()
	if runtime && !l.Nonempty {
		a.CBZ(ctr, e.end[c])
	}
	e.top[c] = a.Label()
	a.Bind(e.top[c])
}

// take zeroes loop c's iteration and takes its copies, after its counter
// has read its count.
func (e *a64Lowerer) take(c int) {
	k, a := e.k, e.a
	l := k.Loops[c]
	if it := k.iterOf(c); it >= 0 && k.Nodes[it].Reg > 0 {
		a.MOVreg(e.xr(it), a64.XZR)
	} else if it >= 0 && k.Nodes[it].Spill > 0 {
		e.slot(a64.A64StrX, uint8(a64.XZR), k.Nodes[it].Spill)
	}
	steps, ok := k.takeOrder(c)
	if !ok {
		e.fail("loop %d's takes each write a register another reads, and every spill slot the stack guard leaves is taken", c)
		return
	}
	zeroed := map[int]bool{}
	for _, st := range steps {
		cu := l.Cursors[st.I]
		if st.To == takeUnpark {
			e.slot(a64.A64LdrX, uint8(e.ptrReg(cu.Buf, cu.Ptr)), st.Slot)
			continue
		}
		if st.To == takePark {
			if cu.Init > 0 {
				src, ix := e.ptrReg(cu.Buf, cu.From), e.xr(cu.Init-1)
				sh := uint8(bits.TrailingZeros(uint(ElemBytes(k.Bufs[cu.Buf]))))
				e.a.ADDlsl(e.gp(reloadReg), src, ix, sh)
			} else if src := e.ptrReg(cu.Buf, cu.From); src != e.gp(reloadReg) {
				a.MOVreg(e.gp(reloadReg), src)
			}
			e.slot(a64.A64StrX, uint8(e.gp(reloadReg)), st.Slot)
			continue
		}
		if cu.Take && cu.Ptr > 0 && k.Ptrs[cu.Ptr-1].Off > 0 {
			if off := k.Ptrs[cu.Ptr-1].Off; !zeroed[off] {
				a.MOVreg(e.gp(x86.Reg(off-1)), a64.XZR)
				zeroed[off] = true
			}
			continue
		}
		if cu.Take && cu.Init > 0 {
			src, ix := e.ptrReg(cu.Buf, cu.From), e.xr(cu.Init-1)
			sh := uint8(bits.TrailingZeros(uint(ElemBytes(k.Bufs[cu.Buf]))))
			if cu.Ptr > 0 && k.Ptrs[cu.Ptr-1].Spill > 0 {
				e.a.ADDlsl(e.gp(reloadReg), src, ix, sh)
				e.slot(a64.A64StrX, uint8(e.gp(reloadReg)), k.Ptrs[cu.Ptr-1].Spill)
			} else {
				e.a.ADDlsl(e.ptrReg(cu.Buf, cu.Ptr), src, ix, sh)
			}
			continue
		}
		if cu.Take && cu.Ptr > 0 && k.Ptrs[cu.Ptr-1].Spill > 0 {
			e.slot(a64.A64StrX, uint8(e.ptrReg(cu.Buf, cu.From)), k.Ptrs[cu.Ptr-1].Spill)
		} else if cu.Take {
			d := e.ptrReg(cu.Buf, cu.Ptr)
			a.MOVreg(d, e.ptrReg(cu.Buf, cu.From))
		}
	}
}

func (e *a64Lowerer) leave(c int) {
	k, a := e.k, e.a
	l := k.Loops[c]
	stepped := map[int]bool{}
	for _, cu := range l.Cursors {
		step := int64(cu.Elems * ElemBytes(k.Bufs[cu.Buf]))
		off := 0
		if cu.Ptr > 0 {
			off = k.Ptrs[cu.Ptr-1].Off
		}
		switch {
		case cu.Elems == 0:
		case off > 0 && stepped[off]:
		case off > 0:
			r := e.gp(x86.Reg(off - 1))
			e.addConstTo(r, r, step/int64(k.Ptrs[cu.Ptr-1].Scale))
			stepped[off] = true
		case cu.Ptr > 0 && k.Ptrs[cu.Ptr-1].Spill > 0:
			s := k.Ptrs[cu.Ptr-1].Spill
			e.slot(a64.A64LdrX, uint8(a64T), s)
			e.addConstTo(a64T, a64T, step)
			e.slot(a64.A64StrX, uint8(a64T), s)
		default:
			p := e.ptrReg(cu.Buf, cu.Ptr)
			e.addConstTo(p, p, step)
		}
	}
	step := int64(max(1, l.Lanes) * max(1, l.Copies))
	if it := k.iterOf(c); it >= 0 && k.Nodes[it].Reg > 0 {
		r := e.xr(it)
		e.addConstTo(r, r, step)
	} else if it >= 0 && k.Nodes[it].Spill > 0 {
		s := k.Nodes[it].Spill
		e.slot(a64.A64LdrX, uint8(a64T), s)
		e.addConstTo(a64T, a64T, step)
		e.slot(a64.A64StrX, uint8(a64T), s)
	}
	if l.once() {
		return
	}
	ctr := e.counter(c)
	a.SUBimm(ctr, ctr, 1)
	a.CBNZ(ctr, e.top[c])
	a.Bind(e.end[c])
}
