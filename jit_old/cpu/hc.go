//go:build amd64

package cpu

import "fmt"

// EmitHCMix is the hyper-connection mixer on the AVX2 tier (hc_const.go has
// the contract). pre and post are one YMM; the mixer is two, rows 0-1 and
// 2-3, one row per 128-bit lane. A row's maximum and sum are two in-lane
// permutes (pairs, then halves); a column's sum adds the two registers and
// then their swapped halves (VPERMQ), which broadcasts it to both lanes.
//
// Registers: RCX out, RDX mixes, R8 scales, R9 bases, RBX consts, R10 the
// round counter. Y7..Y15 are exp's; Y0 is pre/post, Y2 and Y3 the mixer,
// Y4..Y6 temporaries.
func EmitHCMix(iters int, head bool) ([]byte, error) {
	if iters < 0 {
		return nil, fmt.Errorf("jit: hcmix: %d Sinkhorn rounds", iters)
	}
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))   // Out
	a.MOVLoad(RDX, At(RDI, 112)) // Q32: the mixes
	a.MOVLoad(R8, At(RDI, 24))   // AScale: the scales
	a.MOVLoad(R9, At(RDI, 128))  // Q2: the bases
	a.MOVLoad(RBX, At(RDI, 56))  // Scr
	loadExpConsts(&a)

	// pre and post: sigma(m*s + b), times (1, 2), plus (eps, 0).
	a.VMOVDQULoad(Y0, At(RDX, 0))
	a.VMOVDQULoad(Y1, At(R8, 0))
	a.VMULPS(Y0, Y0, Y1)
	a.VADDPSMem(Y0, Y0, At(R9, 0))
	a.VPXOR(Y4, Y4, Y4)
	a.VSUBPS(Y0, Y4, Y0)
	emitExpPS(&a, Y0, Y1, Y5)
	a.VADDPS(Y0, Y0, Y8)
	a.VDIVPS(Y0, Y8, Y0)
	a.VMULPSMem(Y0, Y0, At(RBX, hcMulOff))
	a.VADDPSMem(Y0, Y0, At(RBX, hcAddOff))
	a.VMOVDQUStore(At(RCX, 0), Y0)
	if head {
		a.VZEROUPPER()
		a.RET()
		return a.Bytes(), nil
	}

	// The mixer's logits, rows 0-1 in Y2 and 2-3 in Y3.
	for i, y := range []Reg{Y2, Y3} {
		off := int32(32 + 32*i)
		a.VMOVDQULoad(y, At(RDX, off))
		a.VMOVDQULoad(Y4, At(R8, off))
		a.VMULPS(y, y, Y4)
		a.VADDPSMem(y, y, At(R9, off))
	}
	// rowRed leaves op over each row of y in t, broadcast across the row.
	rowRed := func(t, y Reg, op func(d, s1, s2 Reg)) {
		a.VPERMILPS(t, y, 0xB1)
		op(t, y, t)
		a.VPERMILPS(Y6, t, 0x4E)
		op(t, t, Y6)
	}
	eps := func(t Reg) {
		a.VBROADCASTSS(Y6, At(RBX, hcEpsOff))
		a.VADDPS(t, t, Y6)
	}
	// The softmax of each row, plus eps.
	for _, y := range []Reg{Y2, Y3} {
		rowRed(Y4, y, a.VMAXPS)
		a.VSUBPS(y, y, Y4)
		emitExpPS(&a, y, Y4, Y5)
		rowRed(Y4, y, a.VADDPS)
		a.VDIVPS(y, y, Y4)
		eps(y)
	}
	colNorm := func() {
		a.VADDPS(Y4, Y2, Y3)
		a.VPERMQ(Y5, Y4, 0x4E)
		a.VADDPS(Y4, Y4, Y5)
		eps(Y4)
		a.VDIVPS(Y2, Y2, Y4)
		a.VDIVPS(Y3, Y3, Y4)
	}
	rowNorm := func(y Reg) {
		rowRed(Y4, y, a.VADDPS)
		eps(Y4)
		a.VDIVPS(y, y, Y4)
	}
	if iters > 0 {
		colNorm()
		if iters > 1 {
			a.MOVimm(R10, int64(iters-1))
			lp := a.Label()
			a.Bind(lp)
			rowNorm(Y2)
			rowNorm(Y3)
			colNorm()
			a.DEC(R10)
			a.JNZ(lp)
		}
	}
	a.VMOVDQUStore(At(RCX, 32), Y2)
	a.VMOVDQUStore(At(RCX, 64), Y3)
	a.VZEROUPPER()
	a.RET()
	return a.Bytes(), nil
}

// EmitColPool is the compressor's pool on the AVX2 tier (hc_const.go has the
// contract): per channel vector, the maximum over the slots, then the sum of
// the exponentials and of their products with kv, then the quotient.
//
// Registers: RCX out, RDX kv, R8 gate (the channel cursors), RBX consts, R10
// and R11 the vector and tail counts, R12 S, R13 the slot stride, RSI/R9 the
// slot cursors, RAX the slot counter. Y0 the maximum, Y3 the sum, Y4 the
// accumulator, Y1/Y2/Y5 exp's argument and temporaries, Y6 kv.
func EmitColPool() []byte {
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))   // Out
	a.MOVLoad(RDX, At(RDI, 112)) // Q32: kv
	a.MOVLoad(R8, At(RDI, 128))  // Q2: gate
	a.MOVLoad(RBX, At(RDI, 56))  // Scr
	a.MOVLoad(R10, At(RDI, 40))  // K
	a.MOVLoad(R11, At(RDI, 32))  // Rows
	a.MOVLoad(R12, At(RDI, 80))  // Cols: S
	a.MOVLoad(R13, At(RDI, 48))  // RowStr
	loadExpConsts(&a)
	elemLoop(&a, R10, R11, []Reg{RCX, RDX, R8}, func(ld func(Reg, Reg), st func(Reg, Reg)) {
		// The maximum over the slots.
		a.MOVQ(RSI, R8)
		ld(Y0, RSI)
		a.MOVQ(RAX, R12)
		a.DEC(RAX)
		done := a.Label()
		a.JZ(done)
		lp := a.Label()
		a.Bind(lp)
		a.ADDQ(RSI, R13)
		ld(Y1, RSI)
		a.VMAXPS(Y0, Y0, Y1)
		a.DEC(RAX)
		a.JNZ(lp)
		a.Bind(done)
		// The exponentials' sum and the weighted sum.
		a.MOVQ(RSI, R8)
		a.MOVQ(R9, RDX)
		a.MOVQ(RAX, R12)
		a.VPXOR(Y3, Y3, Y3)
		a.VPXOR(Y4, Y4, Y4)
		lp2 := a.Label()
		a.Bind(lp2)
		ld(Y1, RSI)
		a.VSUBPS(Y1, Y1, Y0)
		emitExpPS(&a, Y1, Y2, Y5)
		a.VADDPS(Y3, Y3, Y1)
		ld(Y6, R9)
		a.VFMADD231PS(Y4, Y1, Y6)
		a.ADDQ(RSI, R13)
		a.ADDQ(R9, R13)
		a.DEC(RAX)
		a.JNZ(lp2)
		a.VDIVPS(Y4, Y4, Y3)
		st(RCX, Y4)
	})
	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}
