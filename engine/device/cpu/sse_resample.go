package cpu

// EmitResampleSSE is EmitResample for the SSE tier (SSE4.1's PMULLD, PMINSD,
// PMOVZXBD and PEXTRB): the same arithmetic, four lanes a half-unit, the four
// samples packed back to bytes with PACKSSDW and PACKUSWB (every lane is
// already 0..255).
func EmitResampleSSE(prec int) ([]byte, error) {
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RDX, At(RDI, 24)) // AScale -> in_0
	a.MOVLoad(R9, At(RDI, 8))   // W -> taps
	a.MOVLoad(R12, At(RDI, 80)) // Cols -> taps
	a.MOVLoad(R13, At(RDI, 48)) // RowStr
	a.MOVLoad(RBX, At(RDI, 56)) // Scr
	a.MOVLoad(R10, At(RDI, 40)) // K
	a.MOVLoad(R11, At(RDI, 32)) // Rows
	bcastD(&a, XMM0, At(RBX, resampleBiasOff))
	bcastD(&a, XMM2, At(RBX, resampleMaxOff))
	a.PXOR(XMM7, XMM7, XMM7)

	body := func(off int32, ld func(dst, p Reg), st func(p, src Reg)) {
		a.PXOR(XMM1, XMM1, XMM1)
		a.MOVQ(R8, RDX)
		if off != 0 {
			a.ADDimm(R8, off)
		}
		a.MOVQ(RAX, R9)
		a.MOVQ(RSI, R12)
		taps := a.Label()
		a.Bind(taps)
		ld(XMM3, R8)
		bcastD(&a, XMM4, At(RAX, 0))
		a.PMULLD(XMM3, XMM3, XMM4)
		a.PADDD(XMM1, XMM1, XMM3)
		a.ADDQ(R8, R13)
		a.ADDimm(RAX, 4)
		a.DEC(RSI)
		a.JNZ(taps)
		a.PADDD(XMM1, XMM1, XMM0)
		a.PSRAD(XMM1, XMM1, byte(prec))
		a.PSRAD(XMM3, XMM1, 31)
		a.PANDN(XMM3, XMM3, XMM1)
		a.PMINSD(XMM3, XMM3, XMM2)
		st(RCX, XMM3)
	}
	byteLoops(&a, R10, R11, []Reg{RCX, RDX}, 8, func() {
		for _, off := range []int32{0, 4} {
			body(off, func(dst, p Reg) { a.PMOVZXBDLoad(dst, At(p, 0)) }, func(p, src Reg) {
				a.PACKSSDW(src, src, src)
				a.PACKUSWB(src, src, src)
				a.MOVDStore(At(p, off), src)
			})
		}
	}, func() {
		body(0, func(dst, p Reg) { a.PINSRBLoad(dst, XMM7, At(p, 0), 0) }, func(p, src Reg) {
			a.PEXTRBStore(At(p, 0), src, 0)
		})
	})
	a.RET()
	return a.Bytes(), nil
}
