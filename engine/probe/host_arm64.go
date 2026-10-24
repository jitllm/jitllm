package probe

import (
	"github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/jit/cpu"
)

// hostLanes is a NEON vector's floats.
func hostLanes() int { return 4 }

// hostCheck accepts every arm64 host: the measuring kernels are base NEON.
func hostCheck() error { return nil }

// emptyKernel returns at once.
func emptyKernel() ([]byte, error) {
	var a cpu.A64
	a.RET()
	return a.Bytes(), nil
}

// readKernel sums words [0] for [1] blocks of 256 bytes: sixteen loads a
// trip into sixteen accumulators, so the loads, not the adds, bind it.
func readKernel() ([]byte, error) {
	var a cpu.A64
	a.LDRx(cpu.X1, cpu.X0, 0)
	a.LDRx(cpu.X2, cpu.X0, 8)
	for r := range 16 {
		a.MOVIzero(cpu.V0 + cpu.VReg(r))
	}
	top := a.Label()
	a.Bind(top)
	for r := range 16 {
		a.LDRq(cpu.V16+cpu.VReg(r), cpu.X1, int32(16*r))
	}
	for r := range 16 {
		a.FADD4s(cpu.V0+cpu.VReg(r), cpu.V0+cpu.VReg(r), cpu.V16+cpu.VReg(r))
	}
	a.ADDimm(cpu.X1, cpu.X1, 256)
	a.SUBimm(cpu.X2, cpu.X2, 1)
	a.CBNZ(cpu.X2, top)
	a.RET()
	return a.Bytes(), nil
}

// chaseKernel follows [1] pointers from the address in [0], each read
// waiting for the one before.
func chaseKernel() ([]byte, error) {
	var a cpu.A64
	a.LDRx(cpu.X1, cpu.X0, 0)
	a.LDRx(cpu.X2, cpu.X0, 8)
	top := a.Label()
	a.Bind(top)
	a.LDRx(cpu.X1, cpu.X1, 0)
	a.SUBimm(cpu.X2, cpu.X2, 1)
	a.CBNZ(cpu.X2, top)
	a.RET()
	return a.Bytes(), nil
}

// rateKernel runs [0] trips of eight independent instructions of class cl
// on registers, nil for a class with no instruction measured alone. The
// index class adds a register into four general registers.
func rateKernel(cl cost.Class) ([]byte, error) {
	var a cpu.A64
	var op func(r int)
	switch cl {
	case cost.ClassFloat:
		op = func(r int) { a.FMLA4s(cpu.V0+cpu.VReg(r), cpu.V14, cpu.V15) }
	case cost.ClassSpecial:
		op = func(r int) { a.FDIV4s(cpu.V0+cpu.VReg(r), cpu.V14, cpu.V15) }
	case cost.ClassInt:
		op = func(r int) { a.ADD4s(cpu.V0+cpu.VReg(r), cpu.V0+cpu.VReg(r), cpu.V15) }
	case cost.ClassLanes:
		op = func(r int) { a.FADDP4s(cpu.V0+cpu.VReg(r), cpu.V0+cpu.VReg(r), cpu.V0+cpu.VReg(r)) }
	case cost.ClassIndex:
		op = func(r int) { a.ADDreg(cpu.X3+cpu.XReg(r%4), cpu.X3+cpu.XReg(r%4), cpu.X7) }
	default:
		return nil, nil
	}
	return loop(&a, func() {
		for r := range 8 {
			op(r)
		}
	}), nil
}

// chainKernel runs [0] trips of eight instructions of class cl, each using
// the one before's result, nil for a class rateKernel has none of.
func chainKernel(cl cost.Class) ([]byte, error) {
	var a cpu.A64
	var op func()
	switch cl {
	case cost.ClassFloat:
		op = func() { a.FMLA4s(cpu.V0, cpu.V14, cpu.V15) }
	case cost.ClassSpecial:
		op = func() { a.FDIV4s(cpu.V0, cpu.V0, cpu.V15) }
	case cost.ClassInt:
		op = func() { a.ADD4s(cpu.V0, cpu.V0, cpu.V15) }
	case cost.ClassLanes:
		op = func() { a.FADDP4s(cpu.V0, cpu.V0, cpu.V0) }
	case cost.ClassIndex:
		op = func() { a.ADDreg(cpu.X3, cpu.X3, cpu.X7) }
	default:
		return nil, nil
	}
	return loop(&a, func() {
		for range 8 {
			op()
		}
	}), nil
}

// mixKernel runs [0] trips of twenty-four instructions: eight float
// chains, eight index adds and eight loads from the argument block, so that
// the rate is what the core issues a second rather than any class's units.
func mixKernel() ([]byte, error) {
	var a cpu.A64
	return loop(&a, func() {
		for j := range 8 {
			a.FMLA4s(cpu.V0+cpu.VReg(j), cpu.V14, cpu.V15)
			a.ADDreg(cpu.X3+cpu.XReg(j%4), cpu.X3+cpu.XReg(j%4), cpu.X7)
			a.LDRq(cpu.V16+cpu.VReg(j%4), cpu.X0, int32(16*(j%4)))
		}
	}), nil
}

// loop is body run [0] times, the vector and general registers it uses
// cleared first.
func loop(a *cpu.A64, body func()) []byte {
	a.LDRx(cpu.X2, cpu.X0, 0)
	for r := range 16 {
		a.MOVIzero(cpu.V0 + cpu.VReg(r))
	}
	for r := cpu.X3; r <= cpu.X7; r++ {
		a.MOVimm(r, 0)
	}
	top := a.Label()
	a.Bind(top)
	body()
	a.SUBimm(cpu.X2, cpu.X2, 1)
	a.CBNZ(cpu.X2, top)
	a.RET()
	return a.Bytes()
}
