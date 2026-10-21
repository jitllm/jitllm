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
	e := &x86Lowerer{k: k, a: &x86.Buf{}, top: map[int]int{}, end: map[int]int{}, home: k.homes(), words: k.vectorWords()}
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
	err      error
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
	case OpLoad:
		if nd.Mem {
			return
		}
		home := e.home[i]
		m := e.addr(nd.Buf, nd.A, nd.Ptr)
		if k.Bufs[nd.Buf].Half {
			if e.lanes(home) == 0 {
				e.fail("a scalar load of binary16")
			}
			a.VCVTPH2PS(e.reg(i), m)
			return
		}
		switch {
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
		if !e.words[i] {
			return
		}
		t := x86.Reg(nd.Tmp - 1)
		if nd.Tmp == 0 {
			e.fail("a splat with no scratch word")
		}
		a.MOVimm32(t, int32(uint32(nd.Imm)))
		a.MOVDToX(e.reg(i), t)
		a.PSHUFD(e.reg(i), e.reg(i), 0)
	case OpHalf:
		a.VCVTPS2PH(e.reg(i), e.reg(nd.A), e.lanes(e.home[i]) > 0, 0)
	case OpGtW:
		a.VPCMPGTWx(e.reg(i), e.reg(nd.A), e.reg(nd.B))
	case OpAndNot:
		a.VPANDNx(e.reg(i), e.reg(nd.B), e.reg(nd.A))
	case OpMul, OpAdd, OpShr, OpAnd, OpShl:
		if e.words[i] && nd.Op == OpAnd {
			a.VPANDx(e.reg(i), e.reg(nd.A), e.reg(nd.B))
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
		a.VBROADCASTSSReg(e.reg(i), e.reg(nd.A))
	case OpFma:
		if e.reg(i) != e.reg(nd.C) {
			e.fail("an fma whose result is not its addend's register")
		}
		if b := k.Nodes[nd.B]; b.Mem {
			a.VFMADD231PSMem(e.reg(i), e.reg(nd.A), e.addr(b.Buf, b.A, b.Ptr))
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
	for _, f := range k.foldsOf(c) {
		init := k.Nodes[k.Nodes[f].B]
		switch {
		case init.Op == OpFold:
			if k.Nodes[f].Reg != init.Reg {
				e.fail("a sum carried on out of its register")
			}
		case init.Op == OpConst && init.Imm == 0:
			r := e.reg(f)
			a.VPXOR(r, r, r)
		default:
			e.fail("a sum from %s", init.Op)
		}
	}
	for _, cu := range l.Cursors {
		if cu.Take {
			a.MOVQ(e.ptrReg(cu.Buf, cu.Ptr), e.ptrReg(cu.Buf, 0))
		}
	}
	ctr := e.counter(c)
	runtime := true
	if l.Count > 0 {
		if n := k.Nodes[l.Count-1]; n.Op == OpConst {
			a.MOVimm(ctr, n.Imm)
			runtime = false
		} else if src := e.reg(l.Count - 1); src != ctr {
			a.MOVQ(ctr, src)
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
	a.DEC(e.counter(c))
	a.JNZ(e.top[c])
	a.Bind(e.end[c])
}
