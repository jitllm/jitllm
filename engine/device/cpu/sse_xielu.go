package cpu

// EmitXIELUSSE is EmitXIELU (xielu.go) on the SSE tier: the same sequence on
// XMM halves, Horner as a multiply and an add (there is no FMA here).
//
// Registers: RCX x, RDX the layer's four numbers, RBX consts, R10/R11
// counters; XMM7..XMM15 exp's constants (XMM8 is 1.0), XMM0..XMM6 the body's.
func EmitXIELUSSE() ([]byte, error) {
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))  // Out -> x
	a.MOVLoad(RDX, At(RDI, 24)) // AScale -> the four numbers
	a.MOVLoad(RBX, At(RDI, 56)) // Scr -> consts
	a.MOVLoad(R10, At(RDI, 40)) // K -> whole 8-float units
	a.MOVLoad(R11, At(RDI, 32)) // Rows -> tail elements
	loadExpConstsSSE(&a)
	elemLoopSSE(&a, R10, R11, []Reg{RCX}, func(ld func(Reg, Reg), st func(Reg, Reg)) {
		ld(XMM3, RCX) // x
		bcastSS(&a, XMM4, At(RDX, xieluParamE))
		a.MOVAPS(XMM1, XMM3)
		a.MINPS(XMM1, XMM1, XMM4) // m = min(x, eps), x first as on AVX2

		bcastSS(&a, XMM2, At(RBX, xieluC8Off))
		for i := 1; i < xieluNCoef; i++ {
			a.MULPS(XMM2, XMM2, XMM1)
			bcastSS(&a, XMM4, At(RBX, int32(xieluC8Off+4*i)))
			a.ADDPS(XMM2, XMM2, XMM4)
		}
		a.MULPS(XMM2, XMM2, XMM1)
		a.MULPS(XMM2, XMM2, XMM1) // g(m) for m >= -1/2
		a.MOVAPS(XMM5, XMM1)
		emitExpSSE(&a, XMM5, XMM4, XMM6)
		a.SUBPS(XMM5, XMM5, XMM8)
		a.SUBPS(XMM5, XMM5, XMM1) // exp(m) - 1 - m below it
		// -1/2 <= m (legacy CMPPS has LE, not GE: the operands swap).
		bcastSS(&a, XMM6, At(RBX, xieluLoOff))
		a.CMPPS(XMM6, XMM6, XMM1, 2)
		a.PAND(XMM2, XMM2, XMM6)
		a.PANDN(XMM6, XMM6, XMM5)
		a.POR(XMM2, XMM2, XMM6) // g(m)

		a.SUBPS(XMM1, XMM1, XMM3) // m - x
		a.ADDPS(XMM2, XMM2, XMM1)
		bcastSS(&a, XMM4, At(RDX, xieluParamAN))
		a.MULPS(XMM2, XMM2, XMM4)
		bcastSS(&a, XMM5, At(RDX, xieluParamB))
		a.MULPS(XMM5, XMM5, XMM3) // beta*x
		a.ADDPS(XMM2, XMM2, XMM5) // the x <= 0 branch
		bcastSS(&a, XMM1, At(RDX, xieluParamAP))
		a.MULPS(XMM1, XMM1, XMM3)
		a.MULPS(XMM1, XMM1, XMM3)
		a.ADDPS(XMM1, XMM1, XMM5) // the x > 0 branch

		a.XORPS(XMM4, XMM4, XMM4)
		a.CMPPS(XMM4, XMM4, XMM3, 1) // 0 < x, ordered
		a.PAND(XMM1, XMM1, XMM4)
		a.PANDN(XMM4, XMM4, XMM2)
		a.POR(XMM1, XMM1, XMM4)
		st(RCX, XMM1)
	})
	a.RET()
	return a.Bytes(), nil
}
