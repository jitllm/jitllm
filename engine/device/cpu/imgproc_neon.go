package cpu

// The image path's kernels on NEON; imgproc_const.go has the contracts and
// imgproc_sse.go the x86 bodies these transcribe. The file carries no build
// tag, as a64.go does not: any host emits these bytes, and the amd64 host
// disassembles them (imgproc_test.go); imgproc_arm64.go binds the names nn
// calls.

// EmitA64ResampleH is the horizontal resampling pass: per pixel, its three
// channels as lanes of one register, read as a halfword and a byte lane so
// nothing past the pixel is touched, and written back the same way.
//
//	x1 out row  x2 in row  x3 rows left  x4 RowStr  x5 OutStr
//	x6 out  x7 weights  x8 window offsets  x9 pixels left (per row)
//	x10 the window  x11 taps left  x12, x13 scratch (per pixel)
//	v0 bias  v2 255  v1 the sum  v3 a pixel  v6 a weight
func EmitA64ResampleH(prec int) []byte {
	var a A64
	a.LDRx(X1, X0, 0)  // Out
	a.LDRx(X2, X0, 24) // AScale -> in
	a.LDRx(X3, X0, 32) // Rows
	a.LDRx(X4, X0, 48) // RowStr
	a.LDRx(X5, X0, 88) // OutStr
	a.LDRx(X12, X0, 56)
	a.LDRs(0, X12, resampleBiasOff)
	a.DUPs4(0, 0)
	a.LDRs(2, X12, resampleMaxOff)
	a.DUPs4(2, 2)

	done := a.Label()
	a.CBZ(X3, done)
	row := a.Label()
	a.Bind(row)
	a.MOVreg(X6, X1)
	a.LDRx(X7, X0, 8)   // W -> weights
	a.LDRx(X8, X0, 136) // AScale2 -> window offsets
	a.LDRx(X9, X0, 40)  // K -> pixels
	rowEnd := a.Label()
	a.CBZ(X9, rowEnd)
	px := a.Label()
	a.Bind(px)
	a.LDRx(X12, X8, 0)
	a.ADDreg(X10, X2, X12)
	a.MOVvec(1, 0)
	a.LDRx(X11, X0, 80) // Cols -> taps
	tap := a.Label()
	a.Bind(tap)
	a.LDRh(3, X10, 0)
	a.ADDimm(X13, X10, 2)
	a.LD1b(3, 2, X13)
	a.UXTL8h(3, 3)
	a.UXTL4s(3, 3)
	a.LDRs(6, X7, 0)
	a.MLAelem(1, 3, 6, 0)
	a.ADDimm(X10, X10, 3)
	a.ADDimm(X7, X7, 4)
	a.SUBimm(X11, X11, 1)
	a.CBNZ(X11, tap)
	a.SSHR4s(1, 1, uint8(prec))
	a.SSHR4s(3, 1, 31)
	a.BIC16b(1, 1, 3)
	a.SMIN4s(1, 1, 2)
	a.XTN4h(1, 1)
	a.XTN8b(1, 1)
	a.STRh(1, X6, 0)
	a.ADDimm(X13, X6, 2)
	a.ST1b(1, 2, X13)
	a.ADDimm(X6, X6, 3)
	a.ADDimm(X8, X8, 8)
	a.SUBimm(X9, X9, 1)
	a.CBNZ(X9, px)
	a.Bind(rowEnd)
	a.ADDreg(X1, X1, X5)
	a.ADDreg(X2, X2, X4)
	a.SUBimm(X3, X3, 1)
	a.CBNZ(X3, row)
	a.Bind(done)
	a.RET()
	return a.Bytes()
}

// EmitA64PixLUT is the per-channel table lookup.
//
//	x1 out  x2 in  x3, x10, x11 the three channels' tables  x4 pixels
//	x6 samples after them  x9 the index
func EmitA64PixLUT() []byte {
	var a A64
	a.LDRx(X1, X0, 0)  // Out
	a.LDRx(X2, X0, 8)  // W -> the samples
	a.LDRx(X3, X0, 24) // AScale -> the tables
	a.LDRx(X4, X0, 40) // K -> pixels
	a.LDRx(X6, X0, 32) // Rows -> samples after them
	a.ADDimm(X10, X3, 4*pixLUTSize)
	a.ADDimm(X11, X3, 8*pixLUTSize)
	tabs := [3]XReg{X3, X10, X11}
	one := func(ch int32) {
		a.LDRBw(X9, X2, ch)
		a.LDRsIdx(0, tabs[ch], X9)
		a.STRs(0, X1, 4*ch)
	}
	tail, done := a.Label(), a.Label()
	a.CBZ(X4, tail)
	lp := a.Label()
	a.Bind(lp)
	for ch := int32(0); ch < 3; ch++ {
		one(ch)
	}
	a.ADDimm(X2, X2, 3)
	a.ADDimm(X1, X1, 12)
	a.SUBimm(X4, X4, 1)
	a.CBNZ(X4, lp)
	a.Bind(tail)
	for ch := int32(0); ch < 2; ch++ {
		a.CBZ(X6, done)
		one(ch)
		a.SUBimm(X6, X6, 1)
	}
	a.Bind(done)
	a.RET()
	return a.Bytes()
}

// EmitA64PixelRow is one row of a decoded picture (PixelRow): a pixel's 16-bit
// (r, g, b, a) in the four lanes of v1, from the format, then out.
//
//	x1 out (or the first plane)  x2, x3 the other planes  x4 the source
//	x5, x6 the chroma rows  x7 pixels left  x8 the pixel's x  x9 the row's
//	first x shifted by sub  x10..x13 scratch
//	v31 the opaque alpha lane  v30 0xffff  v29 0xff  v28 257 or 0x10101
//	v27 255.0 or the Cb factors  v26 the Cr factors  v25 128  v24 0xffffff
//	v23 zero
func EmitA64PixelRow(f PixFmt, sub int, o PixOut) ([]byte, error) {
	if err := pixelShape(f, sub, o); err != nil {
		return nil, err
	}
	var a A64
	a.LDRx(X1, X0, 0)   // Out
	a.LDRx(X2, X0, 120) // Out2
	a.LDRx(X3, X0, 136) // AScale2
	a.LDRx(X4, X0, 8)   // W -> the source
	a.LDRx(X5, X0, 24)  // AScale -> Cb
	a.LDRx(X6, X0, 112) // Q32 -> Cr
	a.LDRx(X7, X0, 40)  // K -> pixels
	a.LDRx(X8, X0, 80)  // Cols -> the first x
	a.LSRimm(X9, X8, uint32(sub))
	a.LDRx(X10, X0, 56) // Scr
	a.LDRq(31, X10, pixAlphaOff)
	a.LDRq(30, X10, pixFFFFOff)
	a.LDRq(29, X10, pixFFOff)
	a.MOVIzero(23)
	if f == PixYCbCr {
		a.LDRq(28, X10, pixK10101Off)
		a.LDRq(27, X10, pixKCbOff)
		a.LDRq(26, X10, pixKCrOff)
		a.LDRq(25, X10, pixK128Off)
		a.LDRq(24, X10, pixClampOff)
	} else {
		a.LDRq(28, X10, pixK257Off)
		a.LDRq(27, X10, pixF255Off)
	}

	done := a.Label()
	a.CBZ(X7, done)
	lp := a.Label()
	a.Bind(lp)
	switch f {
	case PixRGBA, PixNRGBA:
		a.LDRs(1, X4, 0)
		a.UXTL8h(1, 1)
		a.UXTL4s(1, 1)
		if f == PixRGBA {
			a.MUL4s(1, 1, 28)
			break
		}
		a.DUPs4lane(3, 1, 3)
		a.MUL4s(1, 1, 28)
		a.MOVvec(4, 1)
		a.MUL4s(1, 1, 3)
		a.SCVTF4s(1, 1)
		a.FDIV4s(1, 1, 27)
		a.FCVTZS4s(1, 1)
		a.INSs(1, 3, 4, 3)
	case PixGray:
		a.LDRBw(X10, X4, 0)
		a.DUPgp(1, X10)
		a.MUL4s(1, 1, 28)
		a.INSs(1, 3, 31, 3)
	case PixRGBA64:
		a.LDRd(1, X4, 0)
		a.REV16_8b(1, 1)
		a.UXTL4s(1, 1)
	case PixYCbCr:
		a.LDRBw(X10, X4, 0)
		a.DUPgp(1, X10)
		a.MUL4s(1, 1, 28)
		a.LSRimm(X11, X8, uint32(sub))
		a.SUBreg(X11, X11, X9)
		a.ADDreg(X12, X5, X11)
		a.LDRBw(X12, X12, 0)
		a.ADDreg(X13, X6, X11)
		a.LDRBw(X13, X13, 0)
		for _, ch := range []struct {
			gp XReg
			k  VReg
		}{{X12, 27}, {X13, 26}} {
			a.DUPgp(2, ch.gp)
			a.SUB4s(2, 2, 25)
			a.MLA4s(1, 2, ch.k)
		}
		a.SMAX4s(1, 1, 23)
		a.SMIN4s(1, 1, 24)
		a.USHR4s(1, 1, 8)
		a.INSs(1, 3, 31, 3)
		a.ADDimm(X8, X8, 1)
	}
	a.ADDimm(X4, X4, pixBytes(f))
	switch o {
	case PixOver8:
		a.DUPs4lane(3, 1, 3)
		a.SUB4s(4, 30, 3)
		a.ADD4s(1, 1, 4)
		a.USHR4s(1, 1, 8)
		a.AND16b(1, 1, 29)
		a.XTN4h(1, 1)
		a.XTN8b(1, 1)
		a.STRh(1, X1, 0)
		a.ADDimm(X10, X1, 2)
		a.ST1b(1, 2, X10)
		a.ADDimm(X1, X1, 3)
	case PixPlanes:
		a.UCVTF4s(1, 1)
		a.STRs(1, X1, 0)
		a.INSs(5, 0, 1, 1)
		a.STRs(5, X2, 0)
		a.INSs(5, 0, 1, 2)
		a.STRs(5, X3, 0)
		a.ADDimm(X1, X1, 4)
		a.ADDimm(X2, X2, 4)
		a.ADDimm(X3, X3, 4)
	}
	a.SUBimm(X7, X7, 1)
	a.CBNZ(X7, lp)
	a.Bind(done)
	a.RET()
	return a.Bytes(), nil
}

// EmitA64AxisTaps is one axis of a resampled position table (AxisTaps): four
// outputs a vector, the rest one at a time in lane 0, every product and sum
// its own instruction as on x86.
//
//	x1..x4 the weight planes  x11..x14 the index planes  x5 consts
//	x6 vectors  x7 the outputs after them  x8 K  x9 a plane's bytes
//	v31 the outputs' indices as floats  v30 1.0  v28 zero  v27 hi  v26 4.0
//	v17..v25 the coordinate's steps  v10..v16 the filter's constants
//	v7..v9 the tap offsets  v0..v6 the body's
func EmitA64AxisTaps(f AxisFilter) []byte {
	var a A64
	cubic := f == AxisCubic
	n := AxisTapCount(f)
	w := [4]XReg{X1, X2, X3, X4}
	ix := [4]XReg{X11, X12, X13, X14}
	a.LDRx(X1, X0, 0)    // Out -> weights
	a.LDRx(X11, X0, 120) // Out2 -> indices
	a.LDRx(X5, X0, 56)   // Scr
	a.LDRx(X8, X0, 40)   // K
	a.LSRimm(X6, X8, 2)
	a.ANDlow(X7, X8, 2)
	a.ADDreg(X9, X8, X8)
	a.ADDreg(X9, X9, X9)
	for k := 1; k < n; k++ {
		a.ADDreg(w[k], w[k-1], X9)
		a.ADDreg(ix[k], ix[k-1], X9)
	}
	bcast := func(v VReg, off int32) {
		a.LDRs(v, X5, off)
		a.DUPs4(v, v)
	}
	a.LDRq(31, X5, axisIotaOff)
	bcast(30, axisOneOff)
	a.MOVIzero(28)
	bcast(27, axisHiOff)
	bcast(26, axisFourOff)
	for k := 0; k < 9; k++ {
		bcast(VReg(17+k), int32(axisChainOff+4*k))
	}
	for k, off := range []int32{axisC1Off, axisC1Off + 4, axisC2Off, axisC2Off + 4, axisC2Off + 8, axisC2Off + 12, axisTwoOff} {
		bcast(VReg(10+k), off)
	}
	for k := 0; k < 3; k++ {
		bcast(VReg(7+k), int32(axisOffsOff+4*k))
	}
	offReg := map[int]VReg{-1: 7, 1: 8, 2: 9}

	body := func(st func(v VReg, p XReg)) {
		a.MOVvec(0, 31)
		for k, op := range []func(vd, vn, vm VReg){a.FADD4s, a.FMUL4s, a.FDIV4s, a.FMUL4s, a.FADD4s, a.FADD4s,
			a.FMUL4s, a.FADD4s, a.FDIV4s} {
			op(0, 0, VReg(17+k))
		}
		if f == AxisLinear {
			a.FMAX4s(0, 0, 28) // max(x, 0)
		}
		a.FRINTM4s(2, 0)
		a.FSUB4s(3, 0, 2)
		a.FCVTZS4s(4, 2)
		for k := 0; k < n; k++ {
			off := k
			if cubic {
				off--
			}
			if off != 0 {
				a.ADD4s(5, 4, offReg[off])
			} else {
				a.MOVvec(5, 4)
			}
			a.SMAX4s(5, 5, 28)
			a.SMIN4s(5, 5, 27)
			st(5, ix[k])
		}
		if f == AxisLinear {
			a.FSUB4s(6, 30, 3) // 1 - t
			st(6, w[0])
			st(3, w[1]) // t
			return
		}
		if !cubic {
			for k := 0; k < 2; k++ {
				if k == 1 {
					a.FSUB4s(5, 3, 30)
					a.FABS4s(5, 5)
				} else {
					a.FABS4s(5, 3)
				}
				a.FSUB4s(6, 30, 5)
				a.FMAX4s(6, 6, 28)
				st(6, w[k])
			}
			return
		}
		conv := func(two bool) {
			if !two {
				a.FMUL4s(6, 10, 5)
				a.FSUB4s(6, 6, 11)
				a.FMUL4s(6, 6, 5)
				a.FMUL4s(6, 6, 5)
				a.FADD4s(6, 6, 30)
				return
			}
			a.FMUL4s(6, 12, 5)
			a.FADD4s(6, 6, 13)
			a.FMUL4s(6, 6, 5)
			a.FADD4s(6, 6, 14)
			a.FMUL4s(6, 6, 5)
			a.FADD4s(6, 6, 15)
		}
		for k := 0; k < 4; k++ {
			switch k {
			case 0:
				a.FADD4s(5, 3, 30)
			case 1:
				a.MOVvec(5, 3)
			case 2:
				a.FSUB4s(5, 30, 3)
			case 3:
				a.FSUB4s(5, 16, 3)
			}
			conv(k == 0 || k == 3)
			st(6, w[k])
		}
	}
	step := func(by int32) {
		for k := 0; k < n; k++ {
			a.ADDimm(w[k], w[k], by)
			a.ADDimm(ix[k], ix[k], by)
		}
	}
	tail, done := a.Label(), a.Label()
	a.CBZ(X6, tail)
	lp := a.Label()
	a.Bind(lp)
	body(func(v VReg, p XReg) { a.STRq(v, p, 0) })
	step(16)
	a.FADD4s(31, 31, 26)
	a.SUBimm(X6, X6, 1)
	a.CBNZ(X6, lp)
	a.Bind(tail)
	a.CBZ(X7, done)
	lt := a.Label()
	a.Bind(lt)
	body(func(v VReg, p XReg) { a.STRs(v, p, 0) })
	step(4)
	a.FADD4s(31, 31, 30)
	a.SUBimm(X7, X7, 1)
	a.CBNZ(X7, lt)
	a.Bind(done)
	a.RET()
	return a.Bytes()
}

// EmitA64LerpGrid is one tap pair's plane of a 2-D table (LerpGrid).
//
//	x1 weights out  x2 indices out  x3 hw  x4 ht  x5 rows left
//	x7 ww  x8 wt  x9 vectors  x10 the columns after them  x11 K
//	v15 side  v0 hw[y]  v1 ht[y]*side
func EmitA64LerpGrid() []byte {
	var a A64
	a.LDRx(X1, X0, 0)   // Out -> weights
	a.LDRx(X2, X0, 120) // Out2 -> indices
	a.LDRx(X3, X0, 112) // Q32 -> hw
	a.LDRx(X4, X0, 128) // Q2 -> ht
	a.LDRx(X5, X0, 32)  // Rows
	a.LDRx(X6, X0, 80)  // Cols -> side
	a.DUPgp(15, X6)
	done := a.Label()
	a.CBZ(X5, done)
	row := a.Label()
	a.Bind(row)
	a.LDRs(0, X3, 0)
	a.DUPs4(0, 0)
	a.LDRs(1, X4, 0)
	a.DUPs4(1, 1)
	a.MUL4s(1, 1, 15)
	a.LDRx(X7, X0, 24)  // AScale -> ww
	a.LDRx(X8, X0, 136) // AScale2 -> wt
	a.LDRx(X11, X0, 40) // K
	a.LSRimm(X9, X11, 2)
	a.ANDlow(X10, X11, 2)
	body := func(ld func(v VReg, p XReg), st func(v VReg, p XReg)) {
		ld(2, X7)
		a.FMUL4s(2, 2, 0)
		st(2, X1)
		ld(3, X8)
		a.ADD4s(3, 3, 1)
		st(3, X2)
	}
	step := func(by int32) {
		for _, r := range []XReg{X1, X2, X7, X8} {
			a.ADDimm(r, r, by)
		}
	}
	tail, rowEnd := a.Label(), a.Label()
	a.CBZ(X9, tail)
	lp := a.Label()
	a.Bind(lp)
	body(func(v VReg, p XReg) { a.LDRq(v, p, 0) }, func(v VReg, p XReg) { a.STRq(v, p, 0) })
	step(16)
	a.SUBimm(X9, X9, 1)
	a.CBNZ(X9, lp)
	a.Bind(tail)
	a.CBZ(X10, rowEnd)
	lt := a.Label()
	a.Bind(lt)
	body(func(v VReg, p XReg) { a.LDRs(v, p, 0) }, func(v VReg, p XReg) { a.STRs(v, p, 0) })
	step(4)
	a.SUBimm(X10, X10, 1)
	a.CBNZ(X10, lt)
	a.Bind(rowEnd)
	a.ADDimm(X3, X3, 4)
	a.ADDimm(X4, X4, 4)
	a.SUBimm(X5, X5, 1)
	a.CBNZ(X5, row)
	a.Bind(done)
	a.RET()
	return a.Bytes()
}

// EmitA64SinCosTab is the resampler's sin/cos table along one axis
// (SinCosTab), instruction for instruction the x86 body's float64 sequence,
// so both architectures compute the same float64 and round it the same way.
//
//	x1, x2 the pair's sin and cos columns  x3 consts  x4 positions  x5 K
//	x6 a row's bytes  x7 pairs  x8 the frequency after them
//	x9, x10 the row cursors  x11 positions left
//	v31 the pair's frequency indices (float32)  v30 K (float32)
//	v29 the frequencies  v28 the position (float32)  v27 1.0f  v26 2.0f
//	v12 2/pi  v13..v15 pi/2's parts  v16..v21 S6..S1  v22 1.0  v23 0.5
//	v24 0.25  v25 4.0  v0..v11 the float64 work
func EmitA64SinCosTab() []byte {
	var a A64
	a.LDRx(X1, X0, 0)   // Out -> sin
	a.LDRx(X2, X0, 120) // Out2 -> cos
	a.LDRx(X3, X0, 56)  // Scr
	a.LDRx(X4, X0, 32)  // Rows -> positions
	a.LDRx(X5, X0, 40)  // K -> frequencies
	a.ADDreg(X6, X5, X5)
	a.ADDreg(X6, X6, X6)
	a.LSRimm(X7, X5, 1)
	a.ANDlow(X8, X5, 1)
	k := func(v VReg, off int32) {
		a.LDRd(v, X3, off)
		a.DUPd2(v, v)
	}
	a.LDRd(31, X3, scIotaF32Off)
	a.DUPgp(30, X5)
	a.SCVTF4s(30, 30)
	a.LDRs(27, X3, scOneF32Off)
	a.DUPs4(27, 27)
	a.LDRs(26, X3, scTwoF32Off)
	a.DUPs4(26, 26)
	k(12, scTwoPiOff)
	for i := int32(0); i < 3; i++ {
		k(VReg(13+i), scPio2Off+8*i)
	}
	for i := int32(0); i < 6; i++ {
		k(VReg(16+i), scSinOff+8*i)
	}
	k(22, scOneOff)
	k(23, scHalfOff)
	k(24, scQuarterOff)
	k(25, scFourOff)

	freqs := func() {
		a.FDIV4s(0, 31, 30)
		a.FCVTL2d(0, 0) // y
		k(1, scLnHiOff)
		a.FMUL2d(1, 1, 0)
		k(2, scLnLoOff)
		a.FMUL2d(2, 2, 0)
		a.FADD2d(3, 1, 2)
		k(4, scLog2eOff)
		a.FMUL2d(3, 3, 4)
		a.FRINTN2d(3, 3) // k
		k(4, scLn2HiOff)
		a.FMUL2d(4, 4, 3)
		a.FSUB2d(4, 4, 1)
		k(5, scLn2LoOff)
		a.FMUL2d(5, 5, 3)
		a.FSUB2d(5, 5, 2)
		a.FADD2d(4, 4, 5) // r
		k(5, scExpOff)
		for i := 1; i < scExpTerms; i++ {
			a.FMUL2d(5, 5, 4)
			k(6, int32(scExpOff+8*i))
			a.FADD2d(5, 5, 6)
		}
		k(6, scMagicOff)
		a.FSUB2d(6, 6, 3)
		a.SHL2d(6, 6, 52)
		a.FMUL2d(5, 5, 6)
		a.FCVTN2s(29, 5)
	}
	sincos := func() {
		a.FMUL2d(1, 12, 0)
		a.FRINTN2d(1, 1) // j
		a.MOVvec(3, 0)
		for i := VReg(0); i < 3; i++ {
			a.FMUL2d(2, 13+i, 1)
			a.FSUB2d(3, 3, 2)
		}
		a.FMUL2d(4, 3, 3) // z
		a.MOVvec(5, 16)
		for i := VReg(1); i < 6; i++ {
			a.FMUL2d(5, 5, 4)
			a.FADD2d(5, 5, 16+i)
		}
		a.FMUL2d(6, 3, 4)
		a.FMUL2d(5, 5, 6)
		a.FADD2d(5, 5, 3) // s
		k(6, scCosOff)
		for i := int32(1); i < 6; i++ {
			a.FMUL2d(6, 6, 4)
			k(7, scCosOff+8*i)
			a.FADD2d(6, 6, 7)
		}
		a.FMUL2d(7, 4, 4)
		a.FMUL2d(6, 6, 7)
		a.FMUL2d(7, 23, 4)
		a.FSUB2d(8, 22, 7)
		a.FADD2d(6, 6, 8) // c
		a.FMUL2d(7, 24, 1)
		a.FRINTM2d(7, 7)
		a.FMUL2d(7, 7, 25)
		a.FSUB2d(2, 1, 7) // q
		a.FMUL2d(7, 2, 23)
		a.FRINTM2d(7, 7) // h
		a.FADD2d(9, 7, 7)
		a.FSUB2d(3, 2, 9)  // sw
		a.FSUB2d(4, 22, 9) // sin's sign
		a.FADD2d(9, 22, 2)
		a.FMUL2d(9, 9, 23)
		a.FRINTM2d(9, 9)
		a.FMUL2d(7, 9, 23)
		a.FRINTM2d(7, 7)
		a.FADD2d(7, 7, 7)
		a.FSUB2d(9, 9, 7)
		a.FADD2d(9, 9, 9)
		a.FSUB2d(2, 22, 9) // cos's sign
		a.FSUB2d(7, 22, 3) // 1 - sw
		a.FMUL2d(8, 5, 7)
		a.FMUL2d(9, 6, 3)
		a.FADD2d(8, 8, 9)
		a.FMUL2d(8, 8, 4) // sin
		a.FMUL2d(6, 6, 7)
		a.FMUL2d(5, 5, 3)
		a.FADD2d(6, 6, 5)
		a.FMUL2d(6, 6, 2) // cos
	}
	column := func(st func(v VReg, p XReg)) {
		a.MOVreg(X9, X1)
		a.MOVreg(X10, X2)
		a.MOVIzero(28)
		a.MOVreg(X11, X4)
		end := a.Label()
		a.CBZ(X11, end)
		lp := a.Label()
		a.Bind(lp)
		a.FMUL4s(0, 28, 29)
		a.FCVTL2d(0, 0)
		sincos()
		a.FCVTN2s(8, 8)
		a.FCVTN2s(6, 6)
		st(8, X9)
		st(6, X10)
		a.ADDreg(X9, X9, X6)
		a.ADDreg(X10, X10, X6)
		a.FADD4s(28, 28, 27)
		a.SUBimm(X11, X11, 1)
		a.CBNZ(X11, lp)
		a.Bind(end)
	}
	tail, done := a.Label(), a.Label()
	a.CBZ(X7, tail)
	pair := a.Label()
	a.Bind(pair)
	freqs()
	column(func(v VReg, p XReg) { a.STRd(v, p, 0) })
	a.ADDimm(X1, X1, 8)
	a.ADDimm(X2, X2, 8)
	a.FADD4s(31, 31, 26)
	a.SUBimm(X7, X7, 1)
	a.CBNZ(X7, pair)
	a.Bind(tail)
	a.CBZ(X8, done)
	freqs()
	column(func(v VReg, p XReg) { a.STRs(v, p, 0) })
	a.Bind(done)
	a.RET()
	return a.Bytes()
}

// EmitA64Copy32 is the strided copy of 4-byte elements.
//
//	x1 out row  x2 in row  x3 rows left  x4 RowStr  x5 OutStr
//	x7 source element stride  x8 destination's  x6, x9 the cursors  x10 left
func EmitA64Copy32() []byte {
	var a A64
	a.LDRx(X1, X0, 0)   // Out
	a.LDRx(X2, X0, 24)  // AScale -> in
	a.LDRx(X3, X0, 32)  // Rows
	a.LDRx(X4, X0, 48)  // RowStr
	a.LDRx(X5, X0, 88)  // OutStr
	a.LDRx(X7, X0, 80)  // Cols -> source element stride
	a.LDRx(X8, X0, 160) // DStr -> destination element stride
	done := a.Label()
	a.CBZ(X3, done)
	row := a.Label()
	a.Bind(row)
	a.MOVreg(X6, X1)
	a.MOVreg(X9, X2)
	a.LDRx(X10, X0, 40) // K
	rowEnd := a.Label()
	a.CBZ(X10, rowEnd)
	el := a.Label()
	a.Bind(el)
	a.LDRs(0, X9, 0)
	a.STRs(0, X6, 0)
	a.ADDreg(X9, X9, X7)
	a.ADDreg(X6, X6, X8)
	a.SUBimm(X10, X10, 1)
	a.CBNZ(X10, el)
	a.Bind(rowEnd)
	a.ADDreg(X1, X1, X5)
	a.ADDreg(X2, X2, X4)
	a.SUBimm(X3, X3, 1)
	a.CBNZ(X3, row)
	a.Bind(done)
	a.RET()
	return a.Bytes()
}

// EmitA64Resample is the NEON twin of EmitResample: four samples a unit
// (ElemLanes on arm64), the bytes widened with UXTL and narrowed back with XTN,
// the taps multiply-accumulated by element, the arithmetic shift and the
// clamp of the amd64 kernel.
//
// Registers: X1 out, X2 in, X3 taps, X8 taps count, X7 RowStr, X4/X6
// counters; X9 and X10 walk one sample's taps, X11 counts them. v0 bias,
// v2 255, v1 the sums.
func EmitA64Resample(prec int) []byte {
	var a A64
	a.LDRx(X1, X0, 0)  // Out
	a.LDRx(X2, X0, 24) // AScale -> in_0
	a.LDRx(X3, X0, 8)  // W -> taps
	a.LDRx(X8, X0, 80) // Cols -> taps
	a.LDRx(X7, X0, 48) // RowStr
	a.LDRx(X5, X0, 56) // Scr
	a.LDRx(X4, X0, 40) // K
	a.LDRx(X6, X0, 32) // Rows
	a.LDRs(0, X5, resampleBiasOff)
	a.DUPs4(0, 0)
	a.LDRs(2, X5, resampleMaxOff)
	a.DUPs4(2, 2)

	// finish turns a half's sum into its clamped samples.
	finish := func(v VReg) {
		a.ADD4s(v, v, 0)
		a.SSHR4s(v, v, uint8(prec))
		a.SSHR4s(3, v, 31)
		a.BIC16b(v, v, 3)
		a.SMIN4s(v, v, 2)
	}
	walk := func(each func()) {
		a.MOVreg(X9, X2)
		a.MOVreg(X10, X3)
		a.MOVreg(X11, X8)
		taps := a.Label()
		a.Bind(taps)
		each()
		a.ADDreg(X9, X9, X7)
		a.ADDimm(X10, X10, 4)
		a.SUBimm(X11, X11, 1)
		a.CBNZ(X11, taps)
	}
	body := func(ld func(), st func()) {
		a.MOVIzero(1)
		walk(func() {
			ld()
			a.UXTL8h(3, 3)
			a.UXTL4s(3, 3)
			a.LDRs(6, X10, 0)
			a.MLAelem(1, 3, 6, 0)
		})
		finish(1)
		a.XTN4h(1, 1)
		a.XTN8b(1, 1)
		st()
	}
	a64ByteLoops(&a, X4, X6, []XReg{X1, X2}, 4, func() {
		body(func() { a.LDRs(3, X9, 0) }, func() { a.STRs(1, X1, 0) })
	}, func() {
		body(func() { a.LDRb(3, X9, 0) }, func() { a.STRb(1, X1, 0) })
	})
	a.RET()
	return a.Bytes()
}
