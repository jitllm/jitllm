//go:build arm64

package cpu

import "fmt"

// EmitHCMix is the hyper-connection mixer in NEON (hc_const.go has the
// contract): pre and post one Q register each, the mixer's four rows v4..v7.
// A row's maximum is FMAXV then a duplicate, its sum two pairwise adds; a
// column's sum is the four rows added.
//
// Registers: x1 out, x2 mixes, x3 scales, x4 bases, x5 consts, x6 the round
// counter. v16..v26 are exp's (v23 is one); v0..v3 and v8..v10 temporaries.
func EmitHCMix(iters int, head bool) ([]byte, error) {
	if iters < 0 {
		return nil, fmt.Errorf("jit: hcmix: %d Sinkhorn rounds", iters)
	}
	var a A64
	a.LDRx(X1, X0, 0)   // Out
	a.LDRx(X2, X0, 112) // Q32: the mixes
	a.LDRx(X3, X0, 24)  // AScale: the scales
	a.LDRx(X4, X0, 128) // Q2: the bases
	a.LDRx(X5, X0, 56)  // Scr
	loadA64ExpConsts(&a)

	for h := int32(0); h < 2; h++ {
		off := 16 * h
		a.LDRq(0, X2, off)
		a.LDRq(1, X3, off)
		a.FMUL4s(0, 0, 1)
		a.LDRq(1, X4, off)
		a.FADD4s(0, 0, 1)
		a.FNEG4s(0, 0)
		emitA64Exp(&a, 0, 1, 2)
		a.FADD4s(0, 0, 23) // 1 + exp(-x)
		a.FDIV4s(0, 23, 0)
		a.LDRq(1, X5, hcMulOff+off)
		a.FMUL4s(0, 0, 1)
		a.LDRq(1, X5, hcAddOff+off)
		a.FADD4s(0, 0, 1)
		a.STRq(0, X1, off)
	}
	if head {
		a.RET()
		return a.Bytes(), nil
	}

	rows := []VReg{4, 5, 6, 7}
	for i, y := range rows {
		off := int32(32 + 16*i)
		a.LDRq(y, X2, off)
		a.LDRq(0, X3, off)
		a.FMUL4s(y, y, 0)
		a.LDRq(0, X4, off)
		a.FADD4s(y, y, 0)
	}
	// The eps broadcast, held in v10 for the whole kernel.
	a.LDRs(10, X5, hcEpsOff)
	a.DUPs4(10, 10)
	rowSum := func(t, y VReg) {
		a.FADDP4s(t, y, y)
		a.FADDP4s(t, t, t)
	}
	for _, y := range rows {
		a.FMAXV(0, y)
		a.DUPs4(0, 0)
		a.FSUB4s(y, y, 0)
		emitA64Exp(&a, y, 0, 1)
		rowSum(0, y)
		a.FDIV4s(y, y, 0)
		a.FADD4s(y, y, 10)
	}
	colNorm := func() {
		a.FADD4s(0, 4, 5)
		a.FADD4s(0, 0, 6)
		a.FADD4s(0, 0, 7)
		a.FADD4s(0, 0, 10)
		for _, y := range rows {
			a.FDIV4s(y, y, 0)
		}
	}
	if iters > 0 {
		colNorm()
		if iters > 1 {
			a.MOVimm(X6, int64(iters-1))
			lp := a.Label()
			a.Bind(lp)
			for _, y := range rows {
				rowSum(0, y)
				a.FADD4s(0, 0, 10)
				a.FDIV4s(y, y, 0)
			}
			colNorm()
			a.SUBimm(X6, X6, 1)
			a.CBNZ(X6, lp)
		}
	}
	for i, y := range rows {
		a.STRq(y, X1, int32(32+16*i))
	}
	a.RET()
	return a.Bytes(), nil
}

// EmitColPool is the compressor's pool in NEON: the AVX2 body's loops, its
// FMA an FMLA.
//
// Registers: x1 out, x2 kv, x3 gate (the channel cursors), x5 consts, x4 and
// x6 the vector and tail counts, x7 S, x8 the stride, x9/x10 the slot
// cursors, x11 the slot counter. v0 the maximum, v3 the sum, v4 the
// accumulator, v1/v2/v5 exp's, v6 kv.
func EmitColPool() []byte {
	var a A64
	a.LDRx(X1, X0, 0)   // Out
	a.LDRx(X2, X0, 112) // Q32: kv
	a.LDRx(X3, X0, 128) // Q2: gate
	a.LDRx(X5, X0, 56)  // Scr
	a.LDRx(X4, X0, 40)  // K
	a.LDRx(X6, X0, 32)  // Rows
	a.LDRx(X7, X0, 80)  // Cols: S
	a.LDRx(X8, X0, 48)  // RowStr
	loadA64ExpConsts(&a)
	a64ElemLoop(&a, X4, X6, []XReg{X1, X2, X3}, func(ld func(VReg, XReg), st func(XReg, VReg)) {
		a.MOVreg(X9, X3)
		ld(0, X9)
		a.SUBimm(X11, X7, 1)
		done := a.Label()
		a.CBZ(X11, done)
		lp := a.Label()
		a.Bind(lp)
		a.ADDreg(X9, X9, X8)
		ld(1, X9)
		a.FMAX4s(0, 0, 1)
		a.SUBimm(X11, X11, 1)
		a.CBNZ(X11, lp)
		a.Bind(done)
		a.MOVreg(X9, X3)
		a.MOVreg(X10, X2)
		a.MOVreg(X11, X7)
		a.MOVIzero(3)
		a.MOVIzero(4)
		lp2 := a.Label()
		a.Bind(lp2)
		ld(1, X9)
		a.FSUB4s(1, 1, 0)
		emitA64Exp(&a, 1, 2, 5)
		a.FADD4s(3, 3, 1)
		ld(6, X10)
		a.FMLA4s(4, 1, 6)
		a.ADDreg(X9, X9, X8)
		a.ADDreg(X10, X10, X8)
		a.SUBimm(X11, X11, 1)
		a.CBNZ(X11, lp2)
		a.FDIV4s(4, 4, 3)
		st(X1, 4)
	})
	a.RET()
	return a.Bytes()
}
