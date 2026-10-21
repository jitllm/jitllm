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
	e := &x86Lowerer{k: k, a: &x86.Buf{}, top: map[int]int{}, end: map[int]int{}}
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

// addr is the memory operand of buffer b at constant index node ix, read or
// written in loop at: from the cursor the innermost loop around it keeps
// for the buffer, else from the buffer's address.
func (e *x86Lowerer) addr(b, ix, at int) x86.Mem {
	k := e.k
	nd := k.Nodes[ix]
	if nd.Op != OpConst {
		e.fail("an access of %s at a computed index", k.Bufs[b].Operand.Name)
	}
	r := k.Bufs[b].Reg
	for c := at; c >= 0; c = k.Loops[c].Parent {
		for _, cu := range k.Loops[c].Cursors {
			if k.Bufs[cu.Buf].Base == k.Bufs[b].Base && cu.Copy {
				r = cu.Reg
			}
		}
		if r != k.Bufs[b].Reg {
			break
		}
	}
	if r == 0 {
		e.fail("%s has no register", k.Bufs[b].Operand.Name)
	}
	return x86.At(x86.Reg(max(r-1, 0)), int32(nd.Imm)*int32(elemBytes(k.Bufs[b].Elem)))
}

func elemBytes(t ir.Type) int {
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
		home := k.loopOf(i)
		m := e.addr(nd.Buf, nd.A, home)
		switch {
		case home < 0:
			a.VBROADCASTSS(e.reg(i), m)
		case e.lanes(home) > 0:
			a.VMOVDQULoad(e.reg(i), m)
		default:
			a.VMOVSSLoad(e.reg(i), m)
		}
	case OpMul:
		a.VMULPS(e.reg(i), e.reg(nd.A), e.reg(nd.B))
	case OpAdd:
		a.VADDPS(e.reg(i), e.reg(nd.A), e.reg(nd.B))
	case OpFma:
		if e.reg(i) != e.reg(nd.C) {
			e.fail("an fma whose result is not its addend's register")
		}
		a.VFMADD231PS(e.reg(i), e.reg(nd.A), e.reg(nd.B))
	default:
		e.fail("node %d: no x86 form of %s.%s", i, nd.Op, nd.Type)
	}
}

func (e *x86Lowerer) store(i int) {
	s := e.k.Stores[i]
	m := e.addr(s.Buf, s.Ix, s.At)
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
		if cu.Copy {
			a.MOVQ(x86.Reg(cu.Reg-1), x86.Reg(k.Bufs[cu.Buf].Reg-1))
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
	if runtime {
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
		r := k.Bufs[cu.Buf].Reg
		if cu.Copy {
			r = cu.Reg
		}
		a.ADDimm(x86.Reg(r-1), int32(cu.Elems*elemBytes(k.Bufs[cu.Buf].Elem)))
	}
	a.DEC(e.counter(c))
	a.JNZ(e.top[c])
	a.Bind(e.end[c])
}
