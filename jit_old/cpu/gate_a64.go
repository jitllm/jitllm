//go:build arm64

package cpu

// The arm64 twin of jit_old/cpu/gate.go.
//
// v16..v26 are exp's constants (loadA64ExpConsts); v0..v7 are free, more than
// the four this needs, so the coefficients are held rather than re-broadcast
// per Horner step as on amd64.
func EmitDeltaGate() []byte {
	var a A64
	a.LDRx(X1, X0, 0)   // Out:     decay
	a.LDRx(X2, X0, 120) // Out2:    beta
	a.LDRx(X3, X0, 8)   // W:       a
	a.LDRx(X4, X0, 24)  // AScale:  dt
	a.LDRx(X6, X0, 136) // AScale2: b
	a.LDRx(X7, X0, 112) // Q32:     A
	a.LDRx(X5, X0, 56)  // Scr:     consts
	a.LDRx(X8, X0, 40)  // K:       whole vectors
	a.LDRx(X10, X0, 32) // Rows:    tail elements
	loadA64ExpConsts(&a)

	// The atanh coefficients and the two masks, held for the whole loop.
	for _, c := range []struct {
		reg VReg
		off int32
	}{{27, 44}, {28, 48}, {29, 52}, {30, 56}, {31, 60}, {15, 64}} {
		a.LDRs(c.reg, X5, c.off)
		a.DUPs4(c.reg, c.reg)
	}

	a64ElemLoop(&a, X8, X10, []XReg{X1, X2, X3, X4, X6, X7}, func(ld func(VReg, XReg), st func(XReg, VReg)) {
		ld(V0, X3)
		ld(V1, X4)
		a.FADD4s(V0, V0, V1) // z = a + dt
		a.MOVIzero(V4)
		a.FMAX4s(V4, V0, V4) // p = max(z, 0)
		a.AND16b(V0, V0, V15)
		a.FNEG4s(V0, V0) // -|z|
		emitA64Exp(&a, V0, V1, V2)

		a.FADD4s(V1, V0, V27) // 2 + u
		a.FDIV4s(V2, V0, V1)  // s
		a.FMUL4s(V0, V2, V2)  // s^2
		a.MOVvec(V1, V28)     // 1/9
		for _, c := range []VReg{V29, V30, V31} {
			a.FMUL4s(V1, V1, V0)
			a.FADD4s(V1, V1, c)
		}
		a.FMUL4s(V1, V1, V0)
		a.FADD4s(V1, V1, V23) // ... + 1
		a.FMUL4s(V1, V1, V2)
		a.FMUL4s(V1, V1, V27) // 2*s*h
		a.FADD4s(V0, V1, V4)  // softplus
		ld(V3, X7)
		a.FMUL4s(V0, V0, V3) // * A
		emitA64Exp(&a, V0, V1, V2)
		st(X1, V0)

		ld(V0, X6)
		a.FNEG4s(V0, V0)
		emitA64Exp(&a, V0, V1, V2)
		a.FADD4s(V0, V0, V23)
		a.MOVvec(V1, V23)
		a.FDIV4s(V0, V1, V0)
		st(X2, V0)
	})

	a.RET()
	return a.Bytes()
}

// EmitDeltaDecayBound is the arm64 twin of gate.go's: Kimi-K3's bounded KDA
// decay, exp(lb / (1 + exp(A*(a + dt)))).
func EmitDeltaDecayBound() []byte {
	var a A64
	a.LDRx(X1, X0, 0)   // Out:     decay
	a.LDRx(X3, X0, 8)   // W:       a
	a.LDRx(X4, X0, 24)  // AScale:  dt
	a.LDRx(X6, X0, 136) // AScale2: lb
	a.LDRx(X7, X0, 112) // Q32:     A
	a.LDRx(X5, X0, 56)  // Scr:     consts
	a.LDRx(X8, X0, 40)  // K:       whole vectors
	a.LDRx(X10, X0, 32) // Rows:    tail elements
	loadA64ExpConsts(&a)
	a.LDRs(V6, X6, 0) // lb, held
	a.DUPs4(V6, V6)

	a64ElemLoop(&a, X8, X10, []XReg{X1, X3, X4, X7}, func(ld func(VReg, XReg), st func(XReg, VReg)) {
		ld(V0, X3)
		ld(V1, X4)
		a.FADD4s(V0, V0, V1) // z = a + dt
		ld(V1, X7)
		a.FMUL4s(V0, V0, V1) // A*z
		emitA64Exp(&a, V0, V1, V2)
		a.FADD4s(V0, V0, V23) // 1 + exp(A*z)
		a.FDIV4s(V0, V6, V0)  // lb * sigma(-A*z)
		emitA64Exp(&a, V0, V1, V2)
		st(X1, V0)
	})

	a.RET()
	return a.Bytes()
}

// EmitSSDGate is the arm64 twin of gate.go's: Mamba-2's dt = softplus(a +
// bias) into Out2 and decay = exp(A*dt) into Out.
func EmitSSDGate() []byte {
	var a A64
	a.LDRx(X1, X0, 0)   // Out:    decay
	a.LDRx(X2, X0, 120) // Out2:   dt
	a.LDRx(X3, X0, 8)   // W:      a
	a.LDRx(X4, X0, 24)  // AScale: bias
	a.LDRx(X7, X0, 112) // Q32:    A
	a.LDRx(X5, X0, 56)  // Scr:    consts
	a.LDRx(X8, X0, 40)  // K:      whole vectors
	a.LDRx(X10, X0, 32) // Rows:   tail elements
	loadA64ExpConsts(&a)
	for _, c := range []struct {
		reg VReg
		off int32
	}{{27, 44}, {28, 48}, {29, 52}, {30, 56}, {31, 60}, {15, 64}} {
		a.LDRs(c.reg, X5, c.off)
		a.DUPs4(c.reg, c.reg)
	}

	a64ElemLoop(&a, X8, X10, []XReg{X1, X2, X3, X4, X7}, func(ld func(VReg, XReg), st func(XReg, VReg)) {
		ld(V0, X3)
		ld(V1, X4)
		a.FADD4s(V0, V0, V1) // z = a + bias
		a.MOVIzero(V4)
		a.FMAX4s(V4, V0, V4) // p = max(z, 0)
		a.AND16b(V0, V0, V15)
		a.FNEG4s(V0, V0) // -|z|
		emitA64Exp(&a, V0, V1, V2)

		a.FADD4s(V1, V0, V27) // 2 + u
		a.FDIV4s(V2, V0, V1)  // s
		a.FMUL4s(V0, V2, V2)  // s^2
		a.MOVvec(V1, V28)     // 1/9
		for _, c := range []VReg{V29, V30, V31} {
			a.FMUL4s(V1, V1, V0)
			a.FADD4s(V1, V1, c)
		}
		a.FMUL4s(V1, V1, V0)
		a.FADD4s(V1, V1, V23) // ... + 1
		a.FMUL4s(V1, V1, V2)
		a.FMUL4s(V1, V1, V27) // 2*s*h
		a.FADD4s(V0, V1, V4)  // softplus
		st(X2, V0)
		ld(V3, X7)
		a.FMUL4s(V0, V0, V3) // * A
		emitA64Exp(&a, V0, V1, V2)
		st(X1, V0)
	})

	a.RET()
	return a.Bytes()
}
