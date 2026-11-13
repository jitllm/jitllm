package composer

import (
	"fmt"
	"math/bits"
	"runtime"

	"github.com/jitllm/jitllm/composer/a64"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/composer/x86"
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
	a        *a64.A64
	top, end map[int]int
	preset   map[int]bool
	err      error
	cur      int
	reloads  int
	dot      bool
	saved    int
}

// a64GPR is the X register each of the profile's general registers is.
func a64GPR(r x86.Reg) a64.XReg {
	switch r {
	case x86.RAX:
		return a64.X1
	case x86.RCX:
		return a64.X2
	case x86.RDX:
		return a64.X3
	case x86.RBX:
		return a64.X4
	case x86.RSP:
		return a64.SP
	case x86.RBP:
		return a64.X5
	case x86.RSI:
		return a64.X6
	case x86.RDI:
		return a64.X0
	case x86.R8:
		return a64.X7
	case x86.R9:
		return a64.X8
	case x86.R10:
		return a64.X9
	case x86.R11:
		return a64.X10
	case x86.R12:
		return a64.X11
	case x86.R13:
		return a64.X12
	case x86.R14:
		return a64.X13
	case x86.R15:
		return a64.X14
	}
	return a64.X0
}

const (
	a64T    = a64.X15
	a64Imm  = a64.X16
	a64Addr = a64.X17
	a64Q    = a64.X19
	// a64Save is the bytes below SP a saved vector temporary takes, past
	// the spill slots and the shift-count word.
	a64Save = -160
)

// LowersCPU reports whether the composer's CPU lowering runs on a CPU of
// this architecture stating isa: arm64's, or x86's at the AVX2 tier or at
// the SSE tier below it (SSE4.1 and SSSE3: the narrow passes in the legacy
// encodings).
func LowersCPU(isa uint64) bool {
	avx2, sse := uint64(x86.ISATierAVX2), uint64(x86.ISATierSSE)
	return runtime.GOARCH == "arm64" || isa&avx2 == avx2 || isa&sse == sse
}

func lowerA64(k *Kernel, d Device) (lw Lowered, err error) {
	if k.Seq == nil {
		return Lowered{}, fmt.Errorf("the arm64 lowering of %s: the kernel is not ordered", k.Name)
	}
	e := &a64Lowerer{k: k, a: &a64.A64{}, top: map[int]int{}, end: map[int]int{}, preset: map[int]bool{},
		x:   &x86Lowerer{k: k, home: k.Homes(), words: k.VectorWords()},
		dot: d.ISA&a64.ISADotProd != 0}
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
				e.slot(a64.A64StrX, uint8(e.gp(x86.Reg(k.Nodes[en.I].Reg-1))), s)
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
	code, err := e.a.Bytes()
	if err != nil {
		return Lowered{}, fmt.Errorf("the arm64 lowering of %s: %w", k.Name, err)
	}
	return Lowered{CPU: code, Regs: RegUse{Vec: -1, File: 32, Stack: a64.StackRefs(code)}}, nil
}

func (e *a64Lowerer) fail(format string, args ...any) {
	if e.err == nil {
		e.err = fmt.Errorf(format, args...)
	}
}

func (e *a64Lowerer) gp(r x86.Reg) a64.XReg { return a64GPR(r & 15) }

func lo(r x86.Reg) a64.VReg { return a64.VReg(2 * r) }
func hi(r x86.Reg) a64.VReg { return a64.VReg(2*r + 1) }

// slot is op of register rt at spill slot s, 1-based, below SP
// (spillOffset): one past the unscaled offset's reach is addressed from
// the register far names.
func (e *a64Lowerer) slot(op a64.A64Access, rt uint8, s int) {
	off := spillOffset(s)
	if e.a.Access(op, rt, a64.SP, off) {
		return
	}
	base := e.far(op, rt, a64.SP)
	e.addConstTo(base, a64.SP, int64(off))
	if !e.a.Access(op, rt, base, 0) {
		e.fail("spill slot %d is out of reach", s)
	}
}

// far is the register an access of rt past every addressing mode's reach
// from base makes its address in: a general register's load its own
// destination, which it writes anyway, so a constant the lowering made in
// a64Imm before the load -- an operand it reloads read after it -- is still
// there; a64Imm for the rest.
func (e *a64Lowerer) far(op a64.A64Access, rt uint8, base a64.XReg) a64.XReg {
	if (op == a64.A64LdrX || op == a64.A64LdrW) && a64.XReg(rt) != base {
		return a64.XReg(rt)
	}
	return a64Imm
}

// reg is node i's register, a spilled value loaded into a reload register.
func (e *a64Lowerer) reg(i int) x86.Reg {
	nd := e.k.Nodes[i]
	if nd.Spill < 0 {
		r := spillReg(e.reloads)
		e.reloads++
		e.access(a64.A64LdrX, uint8(e.gp(r)), a64Mem{a64.X0, int32(nd.Imm)})
		return r
	}
	if s := nd.Spill; s > 0 && i != e.cur {
		r := spillReg(e.reloads)
		e.reloads++
		e.slot(a64.A64LdrX, uint8(e.gp(r)), s)
		return r
	}
	if nd.Reg == 0 {
		e.fail("node %d (%s) has no register", i, nd.Op)
	}
	return x86.Reg(max(nd.Reg-1, 0))
}

func (e *a64Lowerer) xr(i int) a64.XReg { return e.gp(e.reg(i)) }

// a64Mem is an address: a base register and a byte offset.
type a64Mem struct {
	base a64.XReg
	off  int32
}

func (m a64Mem) plus(n int32) a64Mem { return a64Mem{m.base, m.off + n} }

// access is op of register rt at m, the address made in the register far
// names where no addressing mode reaches it.
func (e *a64Lowerer) access(op a64.A64Access, rt uint8, m a64Mem) {
	if e.a.Access(op, rt, m.base, m.off) {
		return
	}
	base := e.far(op, rt, m.base)
	e.a.MOVconst(base, uint64(int64(m.off)))
	e.a.ADDreg(base, m.base, base)
	e.a.Access(op, rt, base, 0)
}

// bare is m's address in a register, for the forms that take no offset.
func (e *a64Lowerer) bare(m a64Mem) a64.XReg {
	if m.off == 0 && m.base != a64.SP {
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
func (e *a64Lowerer) addConstTo(dst, src a64.XReg, c int64) {
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
		return a64Mem{a64.X0, int32(w-1) + int32(nd.Imm)*int32(ElemBytes(k.Bufs[b]))}
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
		e.slot(a64.A64LdrX, uint8(a64Imm), k.Ptrs[ptr-1].Spill)
		a.ADDreg(a64Addr, a64Addr, a64Imm)
	case ptr > 0 && k.Ptrs[ptr-1].Reg > 0:
		a.ADDreg(a64Addr, a64Addr, e.gp(x86.Reg(k.Ptrs[ptr-1].Reg-1)))
	case ptr == 0 && k.Bufs[b].Reg == 0 && k.Bufs[b].Reload && k.Bufs[b].Base > 0:
		e.access(a64.A64LdrX, uint8(a64Imm), a64Mem{a64.X0, int32(k.Nodes[k.Bufs[b].Base-1].Imm)})
		a.ADDreg(a64Addr, a64Addr, a64Imm)
	case ptr == 0 && k.Bufs[b].Reg > 0:
		a.ADDreg(a64Addr, a64Addr, e.gp(x86.Reg(k.Bufs[b].Reg-1)))
	default:
		e.fail("%s has no register", k.Bufs[b].Operand.Name)
	}
	return a64Mem{a64Addr, 0}
}

func (e *a64Lowerer) offBase(ptr int) a64.XReg {
	p := e.k.Ptrs[ptr-1]
	if p.From > 0 {
		return e.gp(x86.Reg(e.k.Ptrs[p.From-1].Reg - 1))
	}
	return e.gp(x86.Reg(e.k.Bufs[p.Buf].Reg - 1))
}

func (e *a64Lowerer) ptrReg(b, ptr int) a64.XReg {
	r := e.k.Bufs[b].Reg
	if p := e.k.Ptrs; ptr > 0 && p[ptr-1].Off > 0 {
		e.a.ADDlsl(e.gp(reloadReg), e.offBase(ptr), e.gp(x86.Reg(p[ptr-1].Off-1)), uint8(bits.TrailingZeros(uint(p[ptr-1].Scale))))
		return e.gp(reloadReg)
	}
	if ptr > 0 {
		r = e.k.Ptrs[ptr-1].Reg
		if s := e.k.Ptrs[ptr-1].Spill; r == 0 && s > 0 {
			e.slot(a64.A64LdrX, uint8(e.gp(reloadReg)), s)
			return e.gp(reloadReg)
		}
	}
	if r == 0 && ptr == 0 && e.k.Bufs[b].Reload && e.k.Bufs[b].Base > 0 {
		e.access(a64.A64LdrX, uint8(e.gp(reloadReg)), a64Mem{a64.X0, int32(e.k.Nodes[e.k.Bufs[b].Base-1].Imm)})
		return e.gp(reloadReg)
	}
	if r == 0 {
		e.fail("%s has no register", e.k.Bufs[b].Operand.Name)
	}
	return e.gp(x86.Reg(max(r-1, 0)))
}

// temp is a vector register none of avoid is, for one operation's
// temporary: saved below SP until release.
func (e *a64Lowerer) temp(avoid ...a64.VReg) a64.VReg {
	for v := a64.VReg(31); ; v-- {
		free := true
		for _, x := range avoid {
			if x == v {
				free = false
			}
		}
		if free {
			e.vaccess(a64.A64StrQ, v, a64Mem{a64.SP, a64Save - 16*int32(e.saved)})
			e.saved++
			return v
		}
	}
}

func (e *a64Lowerer) release(v a64.VReg) {
	e.saved--
	e.vaccess(a64.A64LdrQ, v, a64Mem{a64.SP, a64Save - 16*int32(e.saved)})
}

func (e *a64Lowerer) vaccess(op a64.A64Access, v a64.VReg, m a64Mem) {
	e.access(op, uint8(v), m)
}

// pair is both halves of a vector register.
func pair(r x86.Reg) [2]a64.VReg { return [2]a64.VReg{lo(r), hi(r)} }

// each is f on each half of d, n and m.
func each(d, n, m x86.Reg, f func(d, n, m a64.VReg)) {
	f(lo(d), lo(n), lo(m))
	f(hi(d), hi(n), hi(m))
}

func each2(d, n x86.Reg, f func(d, n a64.VReg)) {
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
	e.vaccess(a64.A64LdrQ, lo(r), m)
	e.vaccess(a64.A64LdrQ, hi(r), m.plus(16))
}

// splat is the word at m in every lane of r.
func (e *a64Lowerer) splat(r x86.Reg, m a64Mem) {
	e.a.LD1R4s(lo(r), e.bare(m))
	e.a.MOVvec(hi(r), lo(r))
}

// withMem runs f on each half h of the operation, node b's half in v: its
// register's, or, for a folded load, a temporary none of avoid is that the
// half is loaded into.
func (e *a64Lowerer) withMem(b int, avoid []a64.VReg, f func(h int, v a64.VReg)) {
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
		e.vaccess(a64.A64LdrQ, t, m.plus(16*int32(h)))
		f(h, t)
	}
	e.release(t)
}

// a64Halves is both halves of each register of rs, for a temporary to avoid.
func a64Halves(rs ...x86.Reg) []a64.VReg {
	var out []a64.VReg
	for _, r := range rs {
		out = append(out, lo(r), hi(r))
	}
	return out
}
