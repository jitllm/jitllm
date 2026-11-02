//go:build arm64

package cpu

// EmitXIELU is the NEON twin of the AVX2 kernel in xielu.go;
// xielu_const.go has the contract.
//
//	x1 x   x3 the layer's four numbers   x5 consts   x4/x6 counters
//	v16..v26 exp's constants (v23 is 1.0), v0..v7 the body's
//
// FMIN propagates a NaN where x86's MINPS returns an operand, which is what
// torch.min does to a NaN x; FCMGT is false for a NaN, so a NaN takes the
// x <= 0 branch, as torch.where's `x > 0` sends it.
func EmitXIELU() []byte {
	var a A64
	a.LDRx(X1, X0, 0)  // Out -> x
	a.LDRx(X3, X0, 24) // AScale -> the four numbers
	a.LDRx(X5, X0, 56) // Scr -> consts
	a.LDRx(X4, X0, 40) // K -> whole vectors
	a.LDRx(X6, X0, 32) // Rows -> tail elements
	loadA64ExpConsts(&a)
	bcast := func(v VReg, base XReg, off int32) {
		a.LDRs(v, base, off)
		a.DUPs4(v, v)
	}
	a64ElemLoop(&a, X4, X6, []XReg{X1}, func(ld func(VReg, XReg), st func(XReg, VReg)) {
		ld(3, X1) // x
		bcast(4, X3, xieluParamE)
		a.FMIN4s(1, 3, 4) // m = min(x, eps)

		bcast(2, X5, xieluC8Off)
		for i := 1; i < xieluNCoef; i++ {
			a.FMUL4s(2, 2, 1)
			bcast(4, X5, int32(xieluC8Off+4*i))
			a.FADD4s(2, 2, 4)
		}
		a.FMUL4s(2, 2, 1)
		a.FMUL4s(2, 2, 1) // g(m) for m >= -1/2
		a.MOVvec(5, 1)
		emitA64Exp(&a, 5, 4, 6)
		a.FSUB4s(5, 5, 23)
		a.FSUB4s(5, 5, 1) // exp(m) - 1 - m below it
		bcast(6, X5, xieluLoOff)
		a.FCMGE4s(6, 1, 6) // m >= -1/2
		a.BIT16b(5, 2, 6)  // g(m)

		a.FSUB4s(1, 1, 3) // m - x
		a.FADD4s(5, 5, 1)
		bcast(4, X3, xieluParamAN)
		a.FMUL4s(5, 5, 4)
		bcast(4, X3, xieluParamB)
		a.FMUL4s(7, 4, 3) // beta*x
		a.FADD4s(5, 5, 7) // the x <= 0 branch
		bcast(4, X3, xieluParamAP)
		a.FMUL4s(2, 4, 3)
		a.FMUL4s(2, 2, 3)
		a.FADD4s(2, 2, 7) // the x > 0 branch

		a.MOVIzero(4)
		a.FCMGT4s(6, 3, 4) // x > 0
		a.BIT16b(5, 2, 6)
		st(X1, 5)
	})
	a.RET()
	return a.Bytes()
}
