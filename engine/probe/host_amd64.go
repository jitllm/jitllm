package probe

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/jit/x86"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// hostLanes is an AVX2 vector's floats.
const hostLanes = 8

// hostCheck refuses an x86 host below AVX2: the measuring kernels use FMA.
func hostCheck() error {
	if cpu.HostTier() != cpu.TierAVX2 {
		return fmt.Errorf("probe: the host's measuring kernels need AVX2 and FMA; this host is %v", cpu.HostTier())
	}
	return nil
}

// emptyKernel returns at once.
func emptyKernel() ([]byte, error) {
	var a x86.Buf
	a.RET()
	return a.Bytes()
}

// readKernel sums words [0] for [1] vectors of 256 bytes: eight loads a
// trip into eight accumulators, so the loads, not the adds, bind it.
func readKernel() ([]byte, error) {
	var a x86.Buf
	a.MOVLoad(x86.RSI, x86.At(x86.RDI, 0))
	a.MOVLoad(x86.RCX, x86.At(x86.RDI, 8))
	for r := range 8 {
		a.VPXOR(x86.Y0+x86.Reg(r), x86.Y0+x86.Reg(r), x86.Y0+x86.Reg(r))
	}
	top := a.Label()
	a.Bind(top)
	for r := range 8 {
		a.VADDPSMem(x86.Y0+x86.Reg(r), x86.Y0+x86.Reg(r), x86.At(x86.RSI, int32(32*r)))
	}
	a.ADDimm(x86.RSI, 256)
	a.DEC(x86.RCX)
	a.JNZ(top)
	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// chaseKernel follows [1] pointers from the address in [0], each read
// waiting for the one before.
func chaseKernel() ([]byte, error) {
	var a x86.Buf
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

// rateKernel runs [0] trips of eight independent instructions of class cl
// on registers, nil for a class with no instruction measured alone. The
// index class adds to four of the call-clobbered general registers.
func rateKernel(cl cost.Class) ([]byte, error) {
	var a x86.Buf
	var op func(r x86.Reg)
	switch cl {
	case cost.ClassFloat:
		op = func(r x86.Reg) { a.VFMADD231PS(r, x86.Y14, x86.Y15) }
	case cost.ClassSpecial:
		op = func(r x86.Reg) { a.VDIVPS(r, x86.Y14, x86.Y15) }
	case cost.ClassInt:
		op = func(r x86.Reg) { a.VPADDD(r, r, x86.Y15) }
	case cost.ClassLanes:
		op = func(r x86.Reg) { a.VPERMILPS(r, r, 0xB1) }
	case cost.ClassIndex:
		regs := [4]x86.Reg{x86.R8, x86.R9, x86.R10, x86.R11}
		op = func(r x86.Reg) { a.ADDimm(regs[(r-x86.Y0)%4], 1) }
	default:
		return nil, nil
	}
	a.MOVLoad(x86.RCX, x86.At(x86.RDI, 0))
	for r := range 16 {
		a.VPXOR(x86.Y0+x86.Reg(r), x86.Y0+x86.Reg(r), x86.Y0+x86.Reg(r))
	}
	top := a.Label()
	a.Bind(top)
	for r := range 8 {
		op(x86.Y0 + x86.Reg(r))
	}
	a.DEC(x86.RCX)
	a.JNZ(top)
	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// chainKernel runs [0] trips of eight instructions of class cl, each
// using the one before's result, nil for a class rateKernel has none of.
// The index chain subtracts a register: a core may fold a chain of
// immediate adds in its renamer, which an index computed from registers
// does not get.
func chainKernel(cl cost.Class) ([]byte, error) {
	var a x86.Buf
	var op func()
	switch cl {
	case cost.ClassFloat:
		op = func() { a.VFMADD231PS(x86.Y0, x86.Y14, x86.Y15) }
	case cost.ClassSpecial:
		op = func() { a.VDIVPS(x86.Y0, x86.Y0, x86.Y15) }
	case cost.ClassInt:
		op = func() { a.VPADDD(x86.Y0, x86.Y0, x86.Y15) }
	case cost.ClassLanes:
		op = func() { a.VPERMILPS(x86.Y0, x86.Y0, 0xB1) }
	case cost.ClassIndex:
		op = func() { a.SUBQ(x86.R8, x86.R9) }
	default:
		return nil, nil
	}
	a.MOVLoad(x86.RCX, x86.At(x86.RDI, 0))
	for r := range 16 {
		a.VPXOR(x86.Y0+x86.Reg(r), x86.Y0+x86.Reg(r), x86.Y0+x86.Reg(r))
	}
	top := a.Label()
	a.Bind(top)
	for range 8 {
		op()
	}
	a.DEC(x86.RCX)
	a.JNZ(top)
	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// mixKernel runs [0] trips of twenty-four instructions: eight float
// chains, as many as a float rate needs to hide its latency, and eight
// each of the index and memory classes, which take ports of their own, so
// that the rate is what the core renames a second rather than any class's
// units.
func mixKernel() ([]byte, error) {
	var a x86.Buf
	a.MOVLoad(x86.RCX, x86.At(x86.RDI, 0))
	for r := range 16 {
		a.VPXOR(x86.Y0+x86.Reg(r), x86.Y0+x86.Reg(r), x86.Y0+x86.Reg(r))
	}
	regs := [4]x86.Reg{x86.R8, x86.R9, x86.R10, x86.R11}
	top := a.Label()
	a.Bind(top)
	for j := range 8 {
		a.VFMADD231PS(x86.Y0+x86.Reg(j), x86.Y14, x86.Y15)
		a.ADDimm(regs[j%4], 1)
		a.VMOVDQULoad(x86.Y8+x86.Reg(j%4), x86.At(x86.RDI, int32(8*(j%4))))
	}
	a.DEC(x86.RCX)
	a.JNZ(top)
	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}
