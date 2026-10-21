//go:build amd64

package cpu

// EmitXIELU generates Apertus's xIELU over a runtime length, in place;
// xielu_const.go has the contract and the arithmetic.
//
// Registers: RCX x, RDX the layer's four numbers, RBX consts, R10/R11
// counters. Y7..Y15 are exp's constants (Y8 is 1.0); Y0..Y6 are the body's.
// The products and sums are separate instructions, not FMAs, in the
// reference's order -- (alpha_p*x)*x + beta*x and A*alpha_n + beta*x -- except
// g's Horner, which is this engine's own polynomial.
func EmitXIELU() []byte {
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out -> x
	a.MOVLoad(RDX, At(RDI, 24)) // AScale -> the four numbers
	a.MOVLoad(RBX, At(RDI, 56)) // Scr -> consts
	a.MOVLoad(R10, At(RDI, 40)) // K -> whole vectors
	a.MOVLoad(R11, At(RDI, 32)) // Rows -> tail elements
	loadExpConsts(&a)
	elemLoop(&a, R10, R11, []Reg{RCX}, func(ld func(Reg, Reg), st func(Reg, Reg)) {
		ld(Y3, RCX) // x
		a.VBROADCASTSS(Y4, At(RDX, xieluParamE))
		a.VMINPS(Y1, Y3, Y4) // m = min(x, eps)

		// g(m) = m^2 * (1/2 + m/6 + ... + m^6/8!), for m >= -1/2.
		a.VBROADCASTSS(Y2, At(RBX, xieluC8Off))
		for i := 1; i < xieluNCoef; i++ {
			a.VBROADCASTSS(Y4, At(RBX, int32(xieluC8Off+4*i)))
			a.VFMADD213PS(Y2, Y1, Y4) // Y2 = Y2*m + c
		}
		a.VMULPS(Y2, Y2, Y1)
		a.VMULPS(Y2, Y2, Y1)
		// and exp(m) - 1 - m below it.
		a.VMOVAPSReg(Y5, Y1)
		emitExpPS(&a, Y5, Y4, Y6)
		a.VSUBPS(Y5, Y5, Y8)
		a.VSUBPS(Y5, Y5, Y1)
		a.VBROADCASTSS(Y4, At(RBX, xieluLoOff))
		a.VCMPPS(Y6, Y1, Y4, 13) // m >= -1/2: the polynomial
		a.VPAND(Y2, Y2, Y6)
		a.VPANDN(Y6, Y6, Y5)
		a.VPOR(Y2, Y2, Y6) // g(m)

		// x <= 0: (g(m) + (m - x))*alpha_n + beta*x
		a.VSUBPS(Y1, Y1, Y3)
		a.VADDPS(Y2, Y2, Y1)
		a.VBROADCASTSS(Y4, At(RDX, xieluParamAN))
		a.VMULPS(Y2, Y2, Y4)
		a.VBROADCASTSS(Y4, At(RDX, xieluParamB))
		a.VMULPS(Y5, Y4, Y3) // beta*x
		a.VADDPS(Y2, Y2, Y5)
		// x > 0: alpha_p*x*x + beta*x
		a.VBROADCASTSS(Y4, At(RDX, xieluParamAP))
		a.VMULPS(Y1, Y4, Y3)
		a.VMULPS(Y1, Y1, Y3)
		a.VADDPS(Y1, Y1, Y5)

		a.VPXOR(Y4, Y4, Y4)
		a.VCMPPS(Y6, Y3, Y4, 14) // x > 0, ordered
		a.VPAND(Y1, Y1, Y6)
		a.VPANDN(Y6, Y6, Y2)
		a.VPOR(Y0, Y1, Y6)
		st(RCX, Y0)
	})
	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}
