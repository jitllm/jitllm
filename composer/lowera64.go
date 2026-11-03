package composer

import (
	"fmt"
	"math/bits"
	"runtime"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/composer/x86"
	"github.com/jitllm/jitllm/engine/device/cpu"
)

// a64Lowerer lowers the CPU profile's ordered, bound kernel to arm64 through
// engine/device/cpu's A64 encoder. The profile binds sixteen eight-lane vector
// registers and sixteen general ones; here vector register r is the NEON pair
// V(2r) (lanes 0-3) and V(2r+1) (lanes 4-7), and each general register is
// one X register (a64GPR), the argument block in X0 as the trampoline passes
// it. An operation within 128-bit lanes is its NEON operation on each half;
// one that clears a register's upper half clears V(2r+1). X15, X16, X17 and
// X19 are the lowering's own, and a vector temporary the binder did not give
// a node is one of the pairs' registers saved below SP around its use.
type a64Lowerer struct {
	x        *x86Lowerer
	k        *Kernel
	a        *cpu.A64
	top, end map[int]int
	preset   map[int]bool
	err      error
	cur      int
	reloads  int
	dot      bool
	saved    int
}

// a64GPR is the X register each of the profile's general registers is.
func a64GPR(r x86.Reg) cpu.XReg {
	switch r {
	case x86.RAX:
		return cpu.X1
	case x86.RCX:
		return cpu.X2
	case x86.RDX:
		return cpu.X3
	case x86.RBX:
		return cpu.X4
	case x86.RSP:
		return cpu.SP
	case x86.RBP:
		return cpu.X5
	case x86.RSI:
		return cpu.X6
	case x86.RDI:
		return cpu.X0
	case x86.R8:
		return cpu.X7
	case x86.R9:
		return cpu.X8
	case x86.R10:
		return cpu.X9
	case x86.R11:
		return cpu.X10
	case x86.R12:
		return cpu.X11
	case x86.R13:
		return cpu.X12
	case x86.R14:
		return cpu.X13
	case x86.R15:
		return cpu.X14
	}
	return cpu.X0
}

const (
	a64T    = cpu.X15
	a64Imm  = cpu.X16
	a64Addr = cpu.X17
	a64Q    = cpu.X19
	// a64Save is the bytes below SP a saved vector temporary takes, past
	// the spill slots and the shift-count word.
	a64Save = -160
)

// HostCPULowers reports whether the composer's CPU lowering runs on this
// host: arm64's, or x86's on a host at the AVX2 tier or at the SSE tier
// below it (SSE4.1 and SSSE3: the narrow passes in the legacy encodings).
func HostCPULowers() bool {
	isa := cpu.CPU().ISA()
	return runtime.GOARCH == "arm64" || isa&cpu.ISATierAVX2 == cpu.ISATierAVX2 || isa&cpu.ISATierSSE == cpu.ISATierSSE
}

func lowerA64(k *Kernel, d Device) (lw Lowered, err error) {
	if k.Seq == nil {
		return Lowered{}, fmt.Errorf("the arm64 lowering of %s: the kernel is not ordered", k.Name)
	}
	e := &a64Lowerer{k: k, a: &cpu.A64{}, top: map[int]int{}, end: map[int]int{}, preset: map[int]bool{},
		x:   &x86Lowerer{k: k, home: k.Homes(), words: k.VectorWords()},
		dot: d.ISA&uint64(cpu.ISADotProd) != 0}
	for _, en := range k.Seq {
		if e.err != nil {
			break
		}
		e.cur = -1
		switch en.Kind {
		case EntryNode:
			e.cur = en.I
			e.node(en.I)
			if s := k.Nodes[en.I].Spill; s > 0 && k.Nodes[en.I].Reg > 0 {
				e.slot(cpu.A64StrX, uint8(e.gp(x86.Reg(k.Nodes[en.I].Reg-1))), s)
			}
		case EntryStore:
			e.store(en.I)
		case EntryLoop:
			e.enter(en.I)
		case EntryEnd:
			e.leave(en.I)
		case EntryAhead:
			e.ahead(en.I)
		}
	}
	if e.err != nil {
		return Lowered{}, fmt.Errorf("the arm64 lowering of %s: %w", k.Name, e.err)
	}
	e.a.RET()
	return Lowered{CPU: e.a.Bytes()}, nil
}

func (e *a64Lowerer) fail(format string, args ...any) {
	if e.err == nil {
		e.err = fmt.Errorf(format, args...)
	}
}

func (e *a64Lowerer) gp(r x86.Reg) cpu.XReg { return a64GPR(r & 15) }

func lo(r x86.Reg) cpu.VReg { return cpu.VReg(2 * r) }
func hi(r x86.Reg) cpu.VReg { return cpu.VReg(2*r + 1) }

// slot is op of register rt at spill slot s, 1-based, below SP
// (spillOffset): one past the unscaled offset's reach is addressed from
// a64Imm.
func (e *a64Lowerer) slot(op cpu.A64Access, rt uint8, s int) {
	off := spillOffset(s)
	if e.a.Access(op, rt, cpu.SP, off) {
		return
	}
	e.addConstTo(a64Imm, cpu.SP, int64(off))
	if !e.a.Access(op, rt, a64Imm, 0) {
		e.fail("spill slot %d is out of reach", s)
	}
}

// reg is node i's register, a spilled value loaded into a reload register.
func (e *a64Lowerer) reg(i int) x86.Reg {
	nd := e.k.Nodes[i]
	if nd.Spill < 0 {
		r := spillReg(e.reloads)
		e.reloads++
		e.access(cpu.A64LdrX, uint8(e.gp(r)), a64Mem{cpu.X0, int32(nd.Imm)})
		return r
	}
	if s := nd.Spill; s > 0 && i != e.cur {
		r := spillReg(e.reloads)
		e.reloads++
		e.slot(cpu.A64LdrX, uint8(e.gp(r)), s)
		return r
	}
	if nd.Reg == 0 {
		e.fail("node %d (%s) has no register", i, nd.Op)
	}
	return x86.Reg(max(nd.Reg-1, 0))
}

func (e *a64Lowerer) xr(i int) cpu.XReg { return e.gp(e.reg(i)) }

// a64Mem is an address: a base register and a byte offset.
type a64Mem struct {
	base cpu.XReg
	off  int32
}

func (m a64Mem) plus(n int32) a64Mem { return a64Mem{m.base, m.off + n} }

// access is op of register rt at m, the address made in a64Imm where no
// addressing mode reaches it.
func (e *a64Lowerer) access(op cpu.A64Access, rt uint8, m a64Mem) {
	if e.a.Access(op, rt, m.base, m.off) {
		return
	}
	e.a.MOVconst(a64Imm, uint64(int64(m.off)))
	e.a.ADDreg(a64Imm, m.base, a64Imm)
	e.a.Access(op, rt, a64Imm, 0)
}

// bare is m's address in a register, for the forms that take no offset.
func (e *a64Lowerer) bare(m a64Mem) cpu.XReg {
	if m.off == 0 && m.base != cpu.SP {
		return m.base
	}
	e.addConstTo(a64Imm, m.base, int64(m.off))
	return a64Imm
}

// ahead is a load hint for line i of k.Lines, as x86Lowerer's PREFETCHT0.
func (e *a64Lowerer) ahead(i int) {
	l := e.k.Lines[i]
	nd := e.k.Nodes[l.Load]
	m := e.addr(nd.Buf, nd.A, nd.Ptr).plus(l.Bytes)
	if m.off < 0 || m.off%8 != 0 || m.off/8 > 4095 {
		e.addConstTo(a64Imm, m.base, int64(m.off))
		m = a64Mem{a64Imm, 0}
	}
	e.a.PRFM(m.base, m.off)
}

// addConstTo is dst = src + c.
func (e *a64Lowerer) addConstTo(dst, src cpu.XReg, c int64) {
	switch {
	case c >= 0 && c <= 4095:
		e.a.ADDimm(dst, src, int32(c))
	case c < 0 && c >= -4095:
		e.a.SUBimm(dst, src, int32(-c))
	default:
		t := a64Imm
		if src == a64Imm {
			t = a64T
		}
		e.a.MOVconst(t, uint64(c))
		e.a.ADDreg(dst, src, t)
	}
}

// addr is the address of buffer b at constant index node ix through
// pointer ptr, as x86Lowerer.addr makes it.
func (e *a64Lowerer) addr(b, ix, ptr int) a64Mem {
	k := e.k
	nd := k.Nodes[ix]
	if nd.Op != OpConst {
		switch {
		case k.Bufs[b].Word > 0:
			e.fail("an argument word of %s at a computed index", k.Bufs[b].Operand.Name)
		case (e.x.words[ix] || nd.Type == ir.F32) && e.x.lanes(e.x.home[ix]) == 0:
			return e.laneAddr(b, ix, ptr)
		case e.x.words[ix] || nd.Type == ir.F32:
			e.fail("an index of %s held in a vector register", k.Bufs[b].Operand.Name)
		}
		p := e.ptrReg(b, ptr)
		e.a.ADDlsl(a64Addr, p, e.xr(ix), uint8(bits.TrailingZeros(uint(ElemBytes(k.Bufs[b])))))
		return a64Mem{a64Addr, 0}
	}
	if w := k.Bufs[b].Word; w > 0 {
		return a64Mem{cpu.X0, int32(w-1) + int32(nd.Imm)*int32(ElemBytes(k.Bufs[b]))}
	}
	return a64Mem{e.ptrReg(b, ptr), int32(nd.Imm) * int32(ElemBytes(k.Bufs[b]))}
}

func (e *a64Lowerer) laneAddr(b, ix, ptr int) a64Mem {
	k, a := e.k, e.a
	a.UMOVs0(a64Addr, lo(e.reg(ix)))
	a.LSLimm(a64Addr, a64Addr, uint8(bits.TrailingZeros(uint(ElemBytes(k.Bufs[b])))))
	switch {
	case ptr > 0 && k.Ptrs[ptr-1].Off > 0:
		a.ADDreg(a64Addr, a64Addr, e.offBase(ptr))
		a.ADDlsl(a64Addr, a64Addr, e.gp(x86.Reg(k.Ptrs[ptr-1].Off-1)), uint8(bits.TrailingZeros(uint(k.Ptrs[ptr-1].Scale))))
	case ptr > 0 && k.Ptrs[ptr-1].Reg == 0 && k.Ptrs[ptr-1].Spill > 0:
		e.slot(cpu.A64LdrX, uint8(a64Imm), k.Ptrs[ptr-1].Spill)
		a.ADDreg(a64Addr, a64Addr, a64Imm)
	case ptr > 0 && k.Ptrs[ptr-1].Reg > 0:
		a.ADDreg(a64Addr, a64Addr, e.gp(x86.Reg(k.Ptrs[ptr-1].Reg-1)))
	case ptr == 0 && k.Bufs[b].Reg == 0 && k.Bufs[b].Reload && k.Bufs[b].Base > 0:
		e.access(cpu.A64LdrX, uint8(a64Imm), a64Mem{cpu.X0, int32(k.Nodes[k.Bufs[b].Base-1].Imm)})
		a.ADDreg(a64Addr, a64Addr, a64Imm)
	case ptr == 0 && k.Bufs[b].Reg > 0:
		a.ADDreg(a64Addr, a64Addr, e.gp(x86.Reg(k.Bufs[b].Reg-1)))
	default:
		e.fail("%s has no register", k.Bufs[b].Operand.Name)
	}
	return a64Mem{a64Addr, 0}
}

func (e *a64Lowerer) offBase(ptr int) cpu.XReg {
	p := e.k.Ptrs[ptr-1]
	if p.From > 0 {
		return e.gp(x86.Reg(e.k.Ptrs[p.From-1].Reg - 1))
	}
	return e.gp(x86.Reg(e.k.Bufs[p.Buf].Reg - 1))
}

func (e *a64Lowerer) ptrReg(b, ptr int) cpu.XReg {
	r := e.k.Bufs[b].Reg
	if p := e.k.Ptrs; ptr > 0 && p[ptr-1].Off > 0 {
		e.a.ADDlsl(e.gp(reloadReg), e.offBase(ptr), e.gp(x86.Reg(p[ptr-1].Off-1)), uint8(bits.TrailingZeros(uint(p[ptr-1].Scale))))
		return e.gp(reloadReg)
	}
	if ptr > 0 {
		r = e.k.Ptrs[ptr-1].Reg
		if s := e.k.Ptrs[ptr-1].Spill; r == 0 && s > 0 {
			e.slot(cpu.A64LdrX, uint8(e.gp(reloadReg)), s)
			return e.gp(reloadReg)
		}
	}
	if r == 0 && ptr == 0 && e.k.Bufs[b].Reload && e.k.Bufs[b].Base > 0 {
		e.access(cpu.A64LdrX, uint8(e.gp(reloadReg)), a64Mem{cpu.X0, int32(e.k.Nodes[e.k.Bufs[b].Base-1].Imm)})
		return e.gp(reloadReg)
	}
	if r == 0 {
		e.fail("%s has no register", e.k.Bufs[b].Operand.Name)
	}
	return e.gp(x86.Reg(max(r-1, 0)))
}

// temp is a vector register none of avoid is, for one operation's
// temporary: saved below SP until release.
func (e *a64Lowerer) temp(avoid ...cpu.VReg) cpu.VReg {
	for v := cpu.VReg(31); ; v-- {
		free := true
		for _, x := range avoid {
			if x == v {
				free = false
			}
		}
		if free {
			e.vaccess(cpu.A64StrQ, v, a64Mem{cpu.SP, a64Save - 16*int32(e.saved)})
			e.saved++
			return v
		}
	}
}

func (e *a64Lowerer) release(v cpu.VReg) {
	e.saved--
	e.vaccess(cpu.A64LdrQ, v, a64Mem{cpu.SP, a64Save - 16*int32(e.saved)})
}

func (e *a64Lowerer) vaccess(op cpu.A64Access, v cpu.VReg, m a64Mem) {
	e.access(op, uint8(v), m)
}

// pair is both halves of a vector register.
func pair(r x86.Reg) [2]cpu.VReg { return [2]cpu.VReg{lo(r), hi(r)} }

// each is f on each half of d, n and m.
func each(d, n, m x86.Reg, f func(d, n, m cpu.VReg)) {
	f(lo(d), lo(n), lo(m))
	f(hi(d), hi(n), hi(m))
}

func each2(d, n x86.Reg, f func(d, n cpu.VReg)) {
	f(lo(d), lo(n))
	f(hi(d), hi(n))
}

func (e *a64Lowerer) mov(d, n x86.Reg) {
	if d != n {
		each2(d, n, e.a.MOVvec)
	}
}

func (e *a64Lowerer) zero(d x86.Reg) {
	e.a.MOVIzero(lo(d))
	e.a.MOVIzero(hi(d))
}

// load8 is the eight words at m into r.
func (e *a64Lowerer) load8(r x86.Reg, m a64Mem) {
	e.vaccess(cpu.A64LdrQ, lo(r), m)
	e.vaccess(cpu.A64LdrQ, hi(r), m.plus(16))
}

// splat is the word at m in every lane of r.
func (e *a64Lowerer) splat(r x86.Reg, m a64Mem) {
	e.a.LD1R4s(lo(r), e.bare(m))
	e.a.MOVvec(hi(r), lo(r))
}

// withMem runs f on each half h of the operation, node b's half in v: its
// register's, or, for a folded load, a temporary none of avoid is that the
// half is loaded into.
func (e *a64Lowerer) withMem(b int, avoid []cpu.VReg, f func(h int, v cpu.VReg)) {
	nd := e.k.Nodes[b]
	if !nd.Mem {
		r := e.reg(b)
		f(0, lo(r))
		f(1, hi(r))
		return
	}
	m := e.addr(nd.Buf, nd.A, nd.Ptr)
	t := e.temp(avoid...)
	for h := range 2 {
		e.vaccess(cpu.A64LdrQ, t, m.plus(16*int32(h)))
		f(h, t)
	}
	e.release(t)
}

// a64Halves is both halves of each register of rs, for a temporary to avoid.
func a64Halves(rs ...x86.Reg) []cpu.VReg {
	var out []cpu.VReg
	for _, r := range rs {
		out = append(out, lo(r), hi(r))
	}
	return out
}
