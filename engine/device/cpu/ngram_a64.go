//go:build arm64

package cpu

// EmitNGramHash is the n-gram hash on arm64 (ngram_const.go has the
// contract), n and per baked and every loop unrolled.
//
// Registers: x1 the rows, x2 the window, x3 the next predecessors, x4 the
// constants, x5 the window's ids (Scratch); x6 an id, x7 eos, x8 the cut
// flag, x9..x12 temporaries.
func EmitNGramHash(n, per int, reset bool) ([]byte, error) {
	if err := ngramShape(n, per); err != nil {
		return nil, err
	}
	var a A64
	a.LDRx(X1, X0, 0)   // Out: the rows
	a.LDRx(X2, X0, 8)   // W: the window
	a.LDRx(X4, X0, 56)  // Scr: the constants
	a.LDRx(X5, X0, 72)  // Scratch: the ids
	a.LDRx(X3, X0, 120) // Out2: the next predecessors
	a.LDRx(X7, X4, ngramConstEOS)

	a.LDRw(X6, X2, 0)
	a.STRx(X6, X5, 0)
	a.ADDimm(X9, X6, 1)
	a.STRw(X9, X3, 0)
	a.MOVreg(X8, XZR) // not cut
	for s := 1; s < n; s++ {
		a.LDRw(X6, X2, int32(4*s))
		if s < n-1 {
			a.STRw(X6, X3, int32(4*s))
		}
		// none (zero) reads as eos, anything else is its id+1.
		a.MOVreg(X9, X6)
		a.SUBimm(X6, X6, 1)
		some := a.Label()
		a.CBNZ(X9, some)
		a.MOVreg(X6, X7)
		a.Bind(some)
		live := a.Label()
		a.CBZ(X8, live)
		a.MOVreg(X6, X7)
		a.Bind(live)
		if reset {
			other := a.Label()
			a.EORx(X9, X6, X7)
			a.CBNZ(X9, other)
			a.MOVimm(X8, 1)
			a.Bind(other)
		}
		a.STRx(X6, X5, int32(8*s))
	}
	for g := 2; g <= n; g++ {
		a.LDRx(X10, X5, 0)
		a.LDRx(X9, X4, ngramConstMul(0))
		a.MULx(X10, X10, X9)
		for j := 1; j < g; j++ {
			a.LDRx(X6, X5, int32(8*j))
			a.LDRx(X9, X4, ngramConstMul(j))
			a.MULx(X6, X6, X9)
			a.EORx(X10, X10, X6)
		}
		for k := 0; k < per; k++ {
			h := (g-2)*per + k
			a.LDRx(X9, X4, ngramConstVoc(n, h))
			a.UDIVx(X11, X10, X9)
			a.MSUBx(X11, X11, X9, X10) // mixed - q*voc
			a.LDRx(X12, X4, ngramConstOff(n, h))
			a.ADDreg(X11, X11, X12)
			a.STRx(X11, X1, int32(8*h))
		}
	}
	a.RET()
	return a.Bytes(), nil
}
