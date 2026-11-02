package cpu

import "fmt"

// EmitHCMixSSE is EmitHCMix on the SSE tier (hc_const.go has the contract):
// pre and post one XMM each, the mixer's four rows XMM0..XMM3. A row's
// maximum and sum are two PSHUFD rounds (pairs, then halves); a column's sum
// is the four rows added. No FMA on this tier, which the mixer does not use.
//
// Registers: RCX out, RDX mixes, R8 scales, R9 bases, RBX consts, R10 the
// round counter. XMM7..XMM15 are exp's; XMM4..XMM6 temporaries.
func EmitHCMixSSE(iters int, head bool) ([]byte, error) {
	if iters < 0 {
		return nil, fmt.Errorf("jit: hcmix: %d Sinkhorn rounds", iters)
	}
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))   // Out
	a.MOVLoad(RDX, At(RDI, 112)) // Q32: the mixes
	a.MOVLoad(R8, At(RDI, 24))   // AScale: the scales
	a.MOVLoad(R9, At(RDI, 128))  // Q2: the bases
	a.MOVLoad(RBX, At(RDI, 56))  // Scr
	loadExpConstsSSE(&a)

	// pre, then post: sigma(m*s + b), times (1 or 2), plus (eps or 0).
	for h := int32(0); h < 2; h++ {
		off := 16 * h
		a.MOVUPSLoad(XMM0, At(RDX, off))
		a.MOVUPSLoad(XMM1, At(R8, off))
		a.MULPS(XMM0, XMM0, XMM1)
		a.MOVUPSLoad(XMM1, At(R9, off))
		a.ADDPS(XMM0, XMM0, XMM1)
		a.XORPS(XMM4, XMM4, XMM4)
		a.SUBPS(XMM4, XMM4, XMM0) // -x
		emitExpSSE(&a, XMM4, XMM1, XMM2)
		a.ADDPS(XMM4, XMM4, XMM8) // 1 + exp(-x)
		a.MOVAPS(XMM0, XMM8)
		a.DIVPS(XMM0, XMM0, XMM4)
		a.MOVUPSLoad(XMM1, At(RBX, hcMulOff+off))
		a.MULPS(XMM0, XMM0, XMM1)
		a.MOVUPSLoad(XMM1, At(RBX, hcAddOff+off))
		a.ADDPS(XMM0, XMM0, XMM1)
		a.MOVUPSStore(At(RCX, off), XMM0)
	}
	if head {
		a.RET()
		return a.Bytes(), nil
	}

	rows := []Reg{XMM0, XMM1, XMM2, XMM3}
	for i, y := range rows {
		off := int32(32 + 16*i)
		a.MOVUPSLoad(y, At(RDX, off))
		a.MOVUPSLoad(XMM4, At(R8, off))
		a.MULPS(y, y, XMM4)
		a.MOVUPSLoad(XMM4, At(R9, off))
		a.ADDPS(y, y, XMM4)
	}
	// rowRed leaves op over row y in XMM4, broadcast across it.
	rowRed := func(y Reg, op func(d, s1, s2 Reg)) {
		a.PSHUFD(XMM4, y, 0xB1)
		op(XMM4, XMM4, y)
		a.PSHUFD(XMM5, XMM4, 0x4E)
		op(XMM4, XMM4, XMM5)
	}
	eps := func(t Reg) {
		bcastSS(&a, XMM5, At(RBX, hcEpsOff))
		a.ADDPS(t, t, XMM5)
	}
	for _, y := range rows {
		rowRed(y, a.MAXPS)
		a.SUBPS(y, y, XMM4)
		emitExpSSE(&a, y, XMM4, XMM5)
		rowRed(y, a.ADDPS)
		a.DIVPS(y, y, XMM4)
		eps(y)
	}
	colNorm := func() {
		a.MOVAPS(XMM4, XMM0)
		a.ADDPS(XMM4, XMM4, XMM1)
		a.ADDPS(XMM4, XMM4, XMM2)
		a.ADDPS(XMM4, XMM4, XMM3)
		eps(XMM4)
		for _, y := range rows {
			a.DIVPS(y, y, XMM4)
		}
	}
	if iters > 0 {
		colNorm()
		if iters > 1 {
			a.MOVimm(R10, int64(iters-1))
			lp := a.Label()
			a.Bind(lp)
			for _, y := range rows {
				rowRed(y, a.ADDPS)
				eps(XMM4)
				a.DIVPS(y, y, XMM4)
			}
			colNorm()
			a.DEC(R10)
			a.JNZ(lp)
		}
	}
	for i, y := range rows {
		a.MOVUPSStore(At(RCX, int32(32+16*i)), y)
	}
	a.RET()
	return a.Bytes(), nil
}

// EmitColPoolSSE is EmitColPool on the SSE tier: each unit's two XMM halves
// run the same slot loops, the AVX2 body's order with its FMA split in two.
//
// Registers: RCX out, RDX kv, R8 gate, RBX consts, R10/R11 the counts, R12
// S, R13 the stride, RSI/R9 the slot cursors, RAX the slot counter. XMM0 the
// maximum, XMM3 the sum, XMM4 the accumulator, XMM1/XMM2/XMM5 exp's,
// XMM6 kv.
func EmitColPoolSSE() ([]byte, error) {
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))   // Out
	a.MOVLoad(RDX, At(RDI, 112)) // Q32: kv
	a.MOVLoad(R8, At(RDI, 128))  // Q2: gate
	a.MOVLoad(RBX, At(RDI, 56))  // Scr
	a.MOVLoad(R10, At(RDI, 40))  // K
	a.MOVLoad(R11, At(RDI, 32))  // Rows
	a.MOVLoad(R12, At(RDI, 80))  // Cols: S
	a.MOVLoad(R13, At(RDI, 48))  // RowStr
	loadExpConstsSSE(&a)
	elemLoopSSE(&a, R10, R11, []Reg{RCX, RDX, R8}, func(ld func(Reg, Reg), st func(Reg, Reg)) {
		a.MOVQ(RSI, R8)
		ld(XMM0, RSI)
		a.MOVQ(RAX, R12)
		a.DEC(RAX)
		done := a.Label()
		a.JZ(done)
		lp := a.Label()
		a.Bind(lp)
		a.ADDQ(RSI, R13)
		ld(XMM1, RSI)
		a.MAXPS(XMM0, XMM0, XMM1)
		a.DEC(RAX)
		a.JNZ(lp)
		a.Bind(done)
		a.MOVQ(RSI, R8)
		a.MOVQ(R9, RDX)
		a.MOVQ(RAX, R12)
		a.XORPS(XMM3, XMM3, XMM3)
		a.XORPS(XMM4, XMM4, XMM4)
		lp2 := a.Label()
		a.Bind(lp2)
		ld(XMM1, RSI)
		a.SUBPS(XMM1, XMM1, XMM0)
		emitExpSSE(&a, XMM1, XMM2, XMM5)
		a.ADDPS(XMM3, XMM3, XMM1)
		ld(XMM6, R9)
		a.MULPS(XMM6, XMM6, XMM1)
		a.ADDPS(XMM4, XMM4, XMM6)
		a.ADDQ(RSI, R13)
		a.ADDQ(R9, R13)
		a.DEC(RAX)
		a.JNZ(lp2)
		a.DIVPS(XMM4, XMM4, XMM3)
		st(RCX, XMM4)
	})
	a.RET()
	return a.Bytes(), nil
}
