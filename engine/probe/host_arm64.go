package probe

import (
	"github.com/jitllm/jitllm/composer/a64"
	"github.com/jitllm/jitllm/engine/cost"
)

// hostLanes is a NEON vector's floats.
func hostLanes() int { return 4 }

// hostCheck accepts every arm64 host: the measuring kernels are base NEON.
func hostCheck() error { return nil }

// emptyKernel returns at once.
func emptyKernel() ([]byte, error) {
	var a a64.A64
	a.RET()
	return a.Bytes()
}

// readKernel sums words [0] for [1] blocks of 256 bytes: sixteen loads a
// trip into sixteen accumulators, so the loads, not the adds, bind it.
func readKernel() ([]byte, error) {
	var a a64.A64
	a.LDRx(a64.X1, a64.X0, 0)
	a.LDRx(a64.X2, a64.X0, 8)
	for r := range 16 {
		a.MOVIzero(a64.V0 + a64.VReg(r))
	}
	top := a.Label()
	a.Bind(top)
	for r := range 16 {
		a.LDRq(a64.V16+a64.VReg(r), a64.X1, int32(16*r))
	}
	for r := range 16 {
		a.FADD4s(a64.V0+a64.VReg(r), a64.V0+a64.VReg(r), a64.V16+a64.VReg(r))
	}
	a.ADDimm(a64.X1, a64.X1, 256)
	a.SUBimm(a64.X2, a64.X2, 1)
	a.CBNZ(a64.X2, top)
	a.RET()
	return a.Bytes()
}

// chaseKernel follows [1] pointers from the address in [0], each read
// waiting for the one before.
func chaseKernel() ([]byte, error) {
	var a a64.A64
	a.LDRx(a64.X1, a64.X0, 0)
	a.LDRx(a64.X2, a64.X0, 8)
	top := a.Label()
	a.Bind(top)
	a.LDRx(a64.X1, a64.X1, 0)
	a.SUBimm(a64.X2, a64.X2, 1)
	a.CBNZ(a64.X2, top)
	a.RET()
	return a.Bytes()
}

// rateKernel runs [0] trips of eight independent instructions of class cl
// on registers, nil for a class with no instruction measured alone. The
// index class adds a register into four general registers.
func rateKernel(cl cost.Class) ([]byte, error) {
	var a a64.A64
	var op func(r int)
	switch cl {
	case cost.ClassFloat:
		op = func(r int) { a.FMLA4s(a64.V0+a64.VReg(r), a64.V14, a64.V15) }
	case cost.ClassSpecial:
		op = func(r int) { a.FDIV4s(a64.V0+a64.VReg(r), a64.V14, a64.V15) }
	case cost.ClassInt:
		op = func(r int) { a.ADD4s(a64.V0+a64.VReg(r), a64.V0+a64.VReg(r), a64.V15) }
	case cost.ClassLanes:
		op = func(r int) { a.FADDP4s(a64.V0+a64.VReg(r), a64.V0+a64.VReg(r), a64.V0+a64.VReg(r)) }
	case cost.ClassIndex:
		op = func(r int) { a.ADDreg(a64.X3+a64.XReg(r%4), a64.X3+a64.XReg(r%4), a64.X7) }
	default:
		return nil, nil
	}
	return loop(&a, func() {
		for r := range 8 {
			op(r)
		}
	})
}

// chainKernel runs [0] trips of eight instructions of class cl, each using
// the one before's result, nil for a class rateKernel has none of.
func chainKernel(cl cost.Class) ([]byte, error) {
	var a a64.A64
	var op func()
	switch cl {
	case cost.ClassFloat:
		op = func() { a.FMLA4s(a64.V0, a64.V14, a64.V15) }
	case cost.ClassSpecial:
		op = func() { a.FDIV4s(a64.V0, a64.V0, a64.V15) }
	case cost.ClassInt:
		op = func() { a.ADD4s(a64.V0, a64.V0, a64.V15) }
	case cost.ClassLanes:
		op = func() { a.FADDP4s(a64.V0, a64.V0, a64.V0) }
	case cost.ClassIndex:
		op = func() { a.ADDreg(a64.X3, a64.X3, a64.X7) }
	default:
		return nil, nil
	}
	return loop(&a, func() {
		for range 8 {
			op()
		}
	})
}

// mixKernel runs [0] trips of twenty-four instructions: eight float
// chains, eight index adds and eight loads from the argument block, so that
// the rate is what the core issues a second rather than any class's units.
func mixKernel() ([]byte, error) {
	var a a64.A64
	return loop(&a, func() {
		for j := range 8 {
			a.FMLA4s(a64.V0+a64.VReg(j), a64.V14, a64.V15)
			a.ADDreg(a64.X3+a64.XReg(j%4), a64.X3+a64.XReg(j%4), a64.X7)
			a.LDRq(a64.V16+a64.VReg(j%4), a64.X0, int32(16*(j%4)))
		}
	})
}

// loop is body run [0] times, the vector and general registers it uses
// cleared first.
func loop(a *a64.A64, body func()) ([]byte, error) {
	a.LDRx(a64.X2, a64.X0, 0)
	for r := range 16 {
		a.MOVIzero(a64.V0 + a64.VReg(r))
	}
	for r := a64.X3; r <= a64.X7; r++ {
		a.MOVimm(r, 0)
	}
	top := a.Label()
	a.Bind(top)
	body()
	a.SUBimm(a64.X2, a64.X2, 1)
	a.CBNZ(a64.X2, top)
	a.RET()
	return a.Bytes()
}
