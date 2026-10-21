package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/ir"
	"github.com/jitllm/jitllm/jit/x86"
)

// cpuLowering lowers an ordered, bound kernel to x86 through jit/x86: each
// entry of its sequence the instructions that do it, the call's arguments
// addressed from RDI.
type cpuLowering struct{}

func (cpuLowering) Lower(k *Kernel, d Device) (Lowered, error) {
	if k.Seq == nil {
		return Lowered{}, fmt.Errorf("the x86 lowering of %s: the kernel is not ordered", k.Name)
	}
	e := &x86Lowerer{k: k, a: &x86.Buf{}, top: map[int]int{}, end: map[int]int{}, preset: map[int]bool{}, home: k.homes(), words: k.vectorWords()}
	for _, en := range k.Seq {
		if e.err != nil {
			break
		}
		switch en.Kind {
		case EntryNode:
			e.node(en.I)
		case EntryStore:
			e.store(en.I)
		case EntryLoop:
			e.enter(en.I)
		case EntryEnd:
			e.leave(en.I)
		}
	}
	if e.err != nil {
		return Lowered{}, fmt.Errorf("the x86 lowering of %s: %w", k.Name, e.err)
	}
	e.a.VZEROUPPER()
	e.a.RET()
	code, err := e.a.Bytes()
	if err != nil {
		return Lowered{}, err
	}
	return Lowered{CPU: code}, nil
}

type x86Lowerer struct {
	k        *Kernel
	home     []int
	words    map[int]bool
	a        *x86.Buf
	top, end map[int]int
	// preset is the loops whose counter was set as an earlier loop began.
	preset map[int]bool
	err    error
}

func (e *x86Lowerer) fail(format string, args ...any) {
	if e.err == nil {
		e.err = fmt.Errorf(format, args...)
	}
}

func (e *x86Lowerer) reg(i int) x86.Reg {
	r := e.k.Nodes[i].Reg
	if r == 0 {
		e.fail("node %d (%s) has no register", i, e.k.Nodes[i].Op)
	}
	return x86.Reg(max(r-1, 0))
}

// addr is the memory operand of buffer b at constant index node ix through
// pointer ptr: the buffer's address, or one of the kernel's copies.
func (e *x86Lowerer) addr(b, ix, ptr int) x86.Mem {
	k := e.k
	nd := k.Nodes[ix]
	if nd.Op != OpConst {
		e.fail("an access of %s at a computed index", k.Bufs[b].Operand.Name)
	}

	return x86.At(e.ptrReg(b, ptr), int32(nd.Imm)*int32(elemBytes(k.Bufs[b])))
}

// ptrReg is the register of buffer b's pointer ptr.
func (e *x86Lowerer) ptrReg(b, ptr int) x86.Reg {
	r := e.k.Bufs[b].Reg
	if ptr > 0 {
		r = e.k.Ptrs[ptr-1].Reg
	}
	if r == 0 {
		e.fail("%s has no register", e.k.Bufs[b].Operand.Name)
	}
	return x86.Reg(max(r-1, 0))
}

// elemBytes is the bytes of one of buffer b's elements.
func elemBytes(b Buf) int {
	if b.Half {
		return 2
	}
	return 4
}

// lanes is how many lanes loop c's code runs, 0 outside any loop.
// feedsHalves reports whether every reader of node i is made from halves.
func (e *x86Lowerer) feedsHalves(i int) bool {
	n := 0
	for r, nd := range e.k.Nodes {
		for _, a := range e.k.args(&nd) {
			if *a == i {
				if !e.half(r) {
					return false
				}
				n++
			}
		}
	}
	return n > 0
}

// half reports whether node i is made from halves: a narrowed value's
// words, which sit in a vector's low 128 bits.
func (e *x86Lowerer) half(i int) bool {
	return e.k.computedFrom(i, OpHalf)
}

func (e *x86Lowerer) lanes(c int) int {
	if c < 0 {
		return 0
	}
	return e.k.Loops[c].Lanes
}

func (e *x86Lowerer) node(i int) {
	k, a := e.k, e.a
	nd := k.Nodes[i]
	switch nd.Op {
	case OpParam:
		a.MOVLoad(e.reg(i), x86.At(x86.RDI, int32(nd.Imm)))
	case OpFold:
		init := k.Nodes[nd.B]
		switch {
		case init.Op == OpFold:
			if nd.Reg != init.Reg {
				e.fail("a sum carried on out of its register")
			}
		case init.Op == OpConst && init.Imm == 0:
			r := e.reg(i)
			a.VPXOR(r, r, r)
		case init.Op == OpLoad && e.home[nd.B] < 0 && init.Imm > 0:
			a.VMOVDQULoad(e.reg(i), e.addr(init.Buf, init.A, init.Ptr))
		case init.Op == OpLoad && e.home[nd.B] < 0:
			a.VBROADCASTSS(e.reg(i), e.addr(init.Buf, init.A, init.Ptr))
		case init.Op == OpConst && e.words[nd.B] && init.Reg > 0:
			a.VMOVAPSReg(e.reg(i), e.reg(nd.B))
		default:
			e.fail("a sum from %s", init.Op)
		}
	case OpLoad:
		if nd.Mem {
			return
		}
		home := e.home[i]
		m := e.addr(nd.Buf, nd.A, nd.Ptr)
		if k.Bufs[nd.Buf].Half {
			if e.lanes(home) == 0 {
				a.VPINSRWLoad(e.reg(i), e.reg(i), m, 0)
				a.VCVTPH2PSReg(e.reg(i), e.reg(i))
				return
			}
			a.VCVTPH2PS(e.reg(i), m)
			return
		}
		switch {
		case home < 0 && nd.Type != ir.F32:
			a.VPBROADCASTD(e.reg(i), m)
		case home < 0:
			a.VBROADCASTSS(e.reg(i), m)
		case e.lanes(home) > 0:
			a.VMOVDQULoad(e.reg(i), m)
		default:
			a.VMOVSSLoad(e.reg(i), m)
		}
	case OpBitcast:
		if e.reg(i) != e.reg(nd.A) {
			e.fail("a bitcast into another register")
		}
	case OpConst:
		if nd.Type == ir.F32 && nd.Imm == 0 && nd.Reg > 0 {
			r := e.reg(i)
			a.VPXOR(r, r, r)
			return
		}
		if !e.words[i] {
			return
		}
		t := x86.Reg(nd.Tmp - 1)
		if nd.Tmp == 0 {
			e.fail("a splat with no scratch word")
		}
		a.MOVimm32(t, int32(uint32(nd.Imm)))
		a.MOVDToX(e.reg(i), t)
		if e.feedsHalves(i) {
			a.PSHUFD(e.reg(i), e.reg(i), 0)
		} else {
			a.VPBROADCASTDReg(e.reg(i), e.reg(i))
		}
	case OpHalf:
		a.VCVTPS2PH(e.reg(i), e.reg(nd.A), e.lanes(e.home[i]) > 0, 0)
	case OpGtW:
		a.VPCMPGTWx(e.reg(i), e.reg(nd.A), e.reg(nd.B))
	case OpLess:
		if k.Nodes[nd.A].Type != ir.F32 {
			a.VPCMPGTD(e.reg(i), e.reg(nd.B), e.reg(nd.A))
			return
		}
		a.VCMPPS(e.reg(i), e.reg(nd.B), e.reg(nd.A), 14)
	case OpSelect:
		if nd.Tmp == 0 {
			e.fail("a select with no scratch register")
		}
		t := x86.Reg(nd.Tmp - 1)
		a.VPANDN(t, e.reg(nd.A), e.reg(nd.C))
		a.VPAND(e.reg(i), e.reg(nd.A), e.reg(nd.B))
		a.VPOR(e.reg(i), e.reg(i), t)
	case OpAndNot:
		if e.half(i) {
			a.VPANDNx(e.reg(i), e.reg(nd.B), e.reg(nd.A))
			return
		}
		a.VPANDN(e.reg(i), e.reg(nd.B), e.reg(nd.A))
	case OpOr:
		a.VPOR(e.reg(i), e.reg(nd.A), e.reg(nd.B))
	case OpDupEven:
		a.VMOVSLDUP(e.reg(i), e.reg(nd.A))
	case OpDupOdd:
		a.VMOVSHDUP(e.reg(i), e.reg(nd.A))
	case OpSwapPairs:
		a.VPERMILPS(e.reg(i), e.reg(nd.A), 0xB1)
	case OpCopy:
		a.VMOVAPSReg(e.reg(i), e.reg(nd.A))
	case OpFirstLane:
		a.VPBLENDD(e.reg(i), e.reg(nd.B), e.reg(nd.A), 1)
	case OpAddSub:
		a.VADDSUBPS(e.reg(i), e.reg(nd.A), e.reg(nd.B))
	case OpSar:
		a.VPSRAD(e.reg(i), e.reg(nd.A), byte(k.Nodes[nd.B].Imm))
	case OpEq:
		a.VCMPPS(e.reg(i), e.reg(nd.A), e.reg(nd.B), 0)
	case OpHMax, OpHMin:
		r, x := e.reg(i), e.reg(nd.A)
		if nd.Tmp == 0 {
			e.fail("a lane reduction with no scratch register")
		}
		t := x86.Reg(nd.Tmp - 1)
		step := a.VMAXPS
		if nd.Op == OpHMin {
			a.VEXTRACTI128(t, x, 1)
			step = a.VPMINSD
		} else {
			a.VEXTRACTF128(t, x, 1)
		}
		step(r, x, t)
		a.VSHUFPS(t, r, r, 0x4E)
		step(r, r, t)
		a.VSHUFPS(t, r, r, 0xB1)
		step(r, r, t)
	case OpMul, OpAdd, OpShr, OpAnd, OpShl:
		if b := k.Nodes[nd.B]; e.words[i] && b.Mem && (nd.Op == OpAdd || nd.Op == OpAnd) {
			m := e.addr(b.Buf, b.A, b.Ptr)
			if nd.Op == OpAdd {
				a.VPADDDMem(e.reg(i), e.reg(nd.A), m)
			} else {
				a.VPANDMem(e.reg(i), e.reg(nd.A), m)
			}
			return
		}
		if (e.words[i] || nd.Type == ir.F32) && nd.Op == OpAnd {
			if e.half(i) {
				a.VPANDx(e.reg(i), e.reg(nd.A), e.reg(nd.B))
			} else {
				a.VPAND(e.reg(i), e.reg(nd.A), e.reg(nd.B))
			}
			return
		}
		if e.words[i] && nd.Op == OpAdd {
			a.VPADDD(e.reg(i), e.reg(nd.A), e.reg(nd.B))
			return
		}
		if e.words[i] && nd.Op == OpShl {
			a.VPSLLD(e.reg(i), e.reg(nd.A), byte(k.Nodes[nd.B].Imm))
			return
		}
		if e.words[i] && nd.Op == OpShr {
			a.VPSRLD(e.reg(i), e.reg(nd.A), byte(k.Nodes[nd.B].Imm))
			return
		}
		if e.words[i] && nd.Op == OpMul {
			a.VPMULLD(e.reg(i), e.reg(nd.A), e.reg(nd.B))
			return
		}
		if nd.Type != ir.F32 {
			e.integer(i)
			return
		}
		if b := k.Nodes[nd.B]; b.Mem {
			m := e.addr(b.Buf, b.A, b.Ptr)
			if nd.Op == OpMul {
				a.VMULPSMem(e.reg(i), e.reg(nd.A), m)
			} else {
				a.VADDPSMem(e.reg(i), e.reg(nd.A), m)
			}
			return
		}
		if nd.Op == OpMul {
			a.VMULPS(e.reg(i), e.reg(nd.A), e.reg(nd.B))
		} else {
			a.VADDPS(e.reg(i), e.reg(nd.A), e.reg(nd.B))
		}
	case OpSub, OpDiv, OpMin, OpMax:
		x, y := e.reg(nd.A), e.reg(nd.B)
		if nd.Type == ir.I32 && nd.Op == OpMin {
			a.VPMINSD(e.reg(i), x, y)
			return
		}
		switch nd.Op {
		case OpSub:
			a.VSUBPS(e.reg(i), x, y)
		case OpDiv:
			a.VDIVPS(e.reg(i), x, y)
		case OpMin:
			a.VMINPS(e.reg(i), x, y)
		case OpMax:
			a.VMAXPS(e.reg(i), x, y)
		}
	case OpSqrt:
		a.VSQRTPS(e.reg(i), e.reg(nd.A))
	case OpBroadcast:
		if x := k.Nodes[nd.A]; x.Mem {
			a.VBROADCASTSS(e.reg(i), e.addr(x.Buf, x.A, x.Ptr))
			return
		}
		a.VBROADCASTSSReg(e.reg(i), e.reg(nd.A))
	case OpZip:
		if nd.Tmp == 0 {
			e.fail("a zip with no scratch register")
		}
		t := x86.Reg(nd.Tmp - 1)
		a.VUNPCKLPS(t, e.reg(nd.A), e.reg(nd.B))
		a.VUNPCKHPS(e.reg(i), e.reg(nd.A), e.reg(nd.B))
		a.VPERM2F128(e.reg(i), t, e.reg(i), byte(0x20+0x11*nd.Imm))
	case OpRoundI32:
		a.VCVTPS2DQ(e.reg(i), e.reg(nd.A))
	case OpCvtI32:
		a.VCVTTPS2DQ(e.reg(i), e.reg(nd.A))
	case OpXor:
		a.VPXOR(e.reg(i), e.reg(nd.A), e.reg(nd.B))
	case OpCvtF32:
		a.VCVTDQ2PS(e.reg(i), e.reg(nd.A))
	case OpFma:
		if c := k.Nodes[nd.C]; c.Op != OpLoad || !c.Mem {
			if r := e.reg(i); r != e.reg(nd.C) {
				if r != e.reg(nd.A) {
					a.VMOVAPSReg(r, e.reg(nd.A))
				}
				a.VFMADD213PS(r, e.reg(nd.B), e.reg(nd.C))
				return
			}
		}
		if e.reg(i) != e.reg(nd.C) {
			e.fail("an fma whose result is not its addend's register")
		}
		if b := k.Nodes[nd.B]; b.Mem {
			a.VFMADD231PSMem(e.reg(i), e.reg(nd.A), e.addr(b.Buf, b.A, b.Ptr))
			return
		}
		if x := k.Nodes[nd.A]; x.Mem {
			a.VFMADD231PSMem(e.reg(i), e.reg(nd.B), e.addr(x.Buf, x.A, x.Ptr))
			return
		}
		a.VFMADD231PS(e.reg(i), e.reg(nd.A), e.reg(nd.B))
	case OpHSum:
		r := e.reg(i)
		if r != e.reg(nd.A) || nd.Tmp == 0 {
			e.fail("a lane sum out of its operand's register")
		}
		t := x86.Reg(nd.Tmp - 1)
		a.VEXTRACTF128(t, r, 1)
		a.VADDPSx(r, r, t)
		a.VHADDPSx(r, r, r)
		a.VHADDPSx(r, r, r)
	default:
		e.fail("node %d: no x86 form of %s.%s", i, nd.Op, nd.Type)
	}
}

// integer is an integer node on a word register: the operand moved in
// where the result has a register of its own, the operation on it.
func (e *x86Lowerer) integer(i int) {
	k, a := e.k, e.a
	nd := k.Nodes[i]
	b := k.Nodes[nd.B]
	if b.Op != OpConst {
		e.fail("node %d: %s of a computed word", i, nd.Op)
		return
	}
	dst := e.reg(i)
	if src := e.reg(nd.A); src != dst {
		a.MOVQ(dst, src)
	}
	switch nd.Op {
	case OpShr:
		a.SHRimm(dst, byte(b.Imm))
	case OpShl:
		a.SHLimm(dst, byte(b.Imm))
	case OpAnd:
		a.ANDimm8(dst, int8(b.Imm))
	default:
		e.fail("node %d: no word form of %s", i, nd.Op)
	}
}

func (e *x86Lowerer) store(i int) {
	s := e.k.Stores[i]
	m := e.addr(s.Buf, s.Ix, s.Ptr)
	if e.k.Bufs[s.Buf].Half {
		if e.lanes(s.At) > 0 {
			e.a.VMOVDQUStorex(m, e.reg(s.Val))
		} else {
			e.a.VPEXTRWStore(m, e.reg(s.Val), 0)
		}
		return
	}
	if e.lanes(s.At) > 0 {
		e.a.VMOVDQUStore(m, e.reg(s.Val))
		return
	}
	e.a.VMOVSSStore(m, e.reg(s.Val))
}

func (e *x86Lowerer) counter(c int) x86.Reg {
	r := e.k.Loops[c].Reg
	if r == 0 {
		e.fail("loop %d has no counter", c)
	}
	return x86.Reg(max(r-1, 0))
}

func (e *x86Lowerer) enter(c int) {
	k, a := e.k, e.a
	l := k.Loops[c]
	for _, cu := range l.Cursors {
		if cu.Take {
			a.MOVQ(e.ptrReg(cu.Buf, cu.Ptr), e.ptrReg(cu.Buf, 0))
		}
	}
	if l.once() {
		return
	}
	ctr := e.counter(c)
	runtime := true
	if l.Count > 0 {
		if n := k.Nodes[l.Count-1]; n.Op == OpConst {
			a.MOVimm(ctr, n.Imm)
			runtime = false
		} else if src := e.reg(l.Count - 1); src != ctr && !e.preset[c] {
			a.MOVQ(ctr, src)
		}
		if t := c + 1; l.Lanes > 0 && t < len(k.Loops) && k.Loops[t].Of == c+1 && k.passes() && k.Loops[t].Count > 0 {
			if tc, src := e.counter(t), e.reg(k.Loops[t].Count-1); tc != ctr && tc != src && k.Nodes[k.Loops[t].Count-1].Op != OpConst {
				a.MOVQ(tc, src)
				e.preset[t] = true
			}
		}
	} else {
		a.MOVimm(ctr, int64(l.Trip))
		runtime = false
	}
	e.end[c] = a.Label()
	if runtime && !l.Nonempty {
		a.TESTQ(ctr, ctr)
		a.JZ(e.end[c])
	}
	e.top[c] = a.Label()
	a.Bind(e.top[c])
}

func (e *x86Lowerer) leave(c int) {
	k, a := e.k, e.a
	l := k.Loops[c]
	for _, cu := range l.Cursors {
		if cu.Elems != 0 {
			a.ADDimm(e.ptrReg(cu.Buf, cu.Ptr), int32(cu.Elems*elemBytes(k.Bufs[cu.Buf])))
		}
	}
	if l.once() {
		return
	}
	a.DEC(e.counter(c))
	a.JNZ(e.top[c])
	a.Bind(e.end[c])
}
