package probe

import (
	"fmt"

	"github.com/jitllm/jitllm/composer/x86"
	"github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/engine/device/cpu"
)

// sse is whether this host runs the SSE tier: its measuring kernels are the
// AVX2 ones in SSE's encodings, with an add where AVX2 has a fused
// multiply-add and four floats a vector rather than eight.

// hostLanes is a vector's floats.
func (e emitter) hostLanes() int {
	if e.sse {
		return 4
	}
	return 8
}

// hostCheck refuses an x86 host below the SSE tier.
func hostCheck(h *cpu.Target) error {
	if t := h.Tier(); t != cpu.TierAVX2 && t != cpu.TierSSE {
		return fmt.Errorf("probe: the host's measuring kernels need SSE4.1 and SSSE3 at least; this host is %v", t)
	}
	return nil
}

// emptyKernel returns at once.
func (e emitter) emptyKernel() ([]byte, error) {
	a := e.buf()
	a.RET()
	return a.Bytes()
}

// buf is a kernel's buffer, declared SSE-tier on a host that runs that
// tier, so Map lets it run there.
func (e emitter) buf() *x86.Buf {
	var a x86.Buf
	if e.sse {
		a.DeclareISA(x86.ISATierSSE)
	}
	return &a
}

// zero clears vector register r.
func (e emitter) zero(a *x86.Buf, r x86.Reg) {
	if e.sse {
		a.XORPS(r, r, r)
		return
	}
	a.VPXOR(r, r, r)
}

// done ends a kernel that used vector registers.
func (e emitter) done(a *x86.Buf) {
	if !e.sse {
		a.VZEROUPPER()
	}
	a.RET()
}

// readKernel sums words [0] for [1] blocks of 256 bytes: eight loads a
// trip into eight accumulators (sixteen on SSE, whose vectors are half as
// wide), so the loads, not the adds, bind it.
func (e emitter) readKernel() ([]byte, error) {
	a := e.buf()
	a.MOVLoad(x86.RSI, x86.At(x86.RDI, 0))
	a.MOVLoad(x86.RCX, x86.At(x86.RDI, 8))
	for r := range 8 {
		e.zero(a, x86.Y0+x86.Reg(r))
	}
	top := a.Label()
	a.Bind(top)
	if e.sse {
		for r := range 16 {
			acc, tmp := x86.Y0+x86.Reg(r%8), x86.Y8+x86.Reg(r%8)
			a.LDDQU(tmp, x86.At(x86.RSI, int32(16*r)))
			a.ADDPS(acc, acc, tmp)
		}
	} else {
		for r := range 8 {
			a.VADDPSMem(x86.Y0+x86.Reg(r), x86.Y0+x86.Reg(r), x86.At(x86.RSI, int32(32*r)))
		}
	}
	a.ADDimm(x86.RSI, 256)
	a.DEC(x86.RCX)
	a.JNZ(top)
	e.done(a)
	return a.Bytes()
}

// chaseKernel follows [1] pointers from the address in [0], each read
// waiting for the one before.
func (e emitter) chaseKernel() ([]byte, error) {
	a := e.buf()
	a.MOVLoad(x86.RSI, x86.At(x86.RDI, 0))
	a.MOVLoad(x86.RCX, x86.At(x86.RDI, 8))
	top := a.Label()
	a.Bind(top)
	a.MOVLoad(x86.RSI, x86.At(x86.RSI, 0))
	a.DEC(x86.RCX)
	a.JNZ(top)
	a.RET()
	return a.Bytes()
}

// vectorOp is class cl's instruction on vector register r: independent of
// the other registers' where indep is set (r's own value carried from the
// trip before), else r = op(r).
func (e emitter) vectorOp(a *x86.Buf, cl cost.Class, r x86.Reg) bool {
	switch {
	case cl == cost.ClassFloat && e.sse:
		a.ADDPS(r, r, x86.Y15)
	case cl == cost.ClassFloat:
		a.VFMADD231PS(r, x86.Y14, x86.Y15)
	case cl == cost.ClassSpecial && e.sse:
		a.DIVPS(r, r, x86.Y15)
	case cl == cost.ClassSpecial:
		a.VDIVPS(r, r, x86.Y15)
	case cl == cost.ClassInt && e.sse:
		a.PADDD(r, r, x86.Y15)
	case cl == cost.ClassInt:
		a.VPADDD(r, r, x86.Y15)
	case cl == cost.ClassLanes && e.sse:
		a.SHUFPS(r, r, r, 0xB1)
	case cl == cost.ClassLanes:
		a.VPERMILPS(r, r, 0xB1)
	default:
		return false
	}
	return true
}

// rateKernel runs [0] trips of eight instructions of class cl, each on a
// register of its own, nil for a class with no instruction measured alone.
// The index class adds to four of the call-clobbered general registers.
func (e emitter) rateKernel(cl cost.Class) ([]byte, error) {
	regs := [4]x86.Reg{x86.R8, x86.R9, x86.R10, x86.R11}
	return e.trips(cl, func(a *x86.Buf, j int) bool {
		if cl == cost.ClassIndex {
			a.ADDimm(regs[j%4], 1)
			return true
		}
		return e.vectorOp(a, cl, x86.Y0+x86.Reg(j))
	})
}

// chainKernel runs [0] trips of eight instructions of class cl, each using
// the one before's result, nil for a class rateKernel has none of. The index
// chain subtracts a register: a core may fold a chain of immediate adds in
// its renamer, which an index computed from registers does not get.
func (e emitter) chainKernel(cl cost.Class) ([]byte, error) {
	return e.trips(cl, func(a *x86.Buf, j int) bool {
		if cl == cost.ClassIndex {
			a.SUBQ(x86.R8, x86.R9)
			return true
		}
		if cl == cost.ClassFloat && !e.sse {
			a.VFMADD231PS(x86.Y0, x86.Y0, x86.Y15)
			return true
		}
		return e.vectorOp(a, cl, x86.Y0)
	})
}

// trips is a kernel running [0] trips of op's eight instructions, the
// vector registers cleared first; nil where op has none for the class.
func (e emitter) trips(cl cost.Class, op func(a *x86.Buf, j int) bool) ([]byte, error) {
	a := e.buf()
	a.MOVLoad(x86.RCX, x86.At(x86.RDI, 0))
	for r := range 16 {
		e.zero(a, x86.Y0+x86.Reg(r))
	}
	top := a.Label()
	a.Bind(top)
	for j := range 8 {
		if !op(a, j) {
			return nil, nil
		}
	}
	a.DEC(x86.RCX)
	a.JNZ(top)
	e.done(a)
	return a.Bytes()
}

// mixKernel runs [0] trips of twenty-four instructions: eight float
// chains, as many as a float rate needs to hide its latency, and eight
// each of the index and memory classes, which take ports of their own, so
// that the rate is what the core renames a second rather than any class's
// units.
func (e emitter) mixKernel() ([]byte, error) {
	regs := [4]x86.Reg{x86.R8, x86.R9, x86.R10, x86.R11}
	a := e.buf()
	a.MOVLoad(x86.RCX, x86.At(x86.RDI, 0))
	for r := range 16 {
		e.zero(a, x86.Y0+x86.Reg(r))
	}
	top := a.Label()
	a.Bind(top)
	for j := range 8 {
		e.vectorOp(a, cost.ClassFloat, x86.Y0+x86.Reg(j))
		a.ADDimm(regs[j%4], 1)
		if e.sse {
			a.LDDQU(x86.Y8+x86.Reg(j%4), x86.At(x86.RDI, int32(8*(j%4))))
		} else {
			a.VMOVDQULoad(x86.Y8+x86.Reg(j%4), x86.At(x86.RDI, int32(8*(j%4))))
		}
	}
	a.DEC(x86.RCX)
	a.JNZ(top)
	e.done(a)
	return a.Bytes()
}
