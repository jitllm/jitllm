package cpu

import "fmt"

// The image path's kernels on x86 (imgproc_const.go has every contract).
//
// Each body here is written once in legacy SSE (SSE4.1 at most) and serves
// both x86 tiers: the SSE tier's kernel declares ISATierSSE before it, and
// the AVX2 tier's (imgproc.go) issues VZEROUPPER first, so no legacy
// instruction meets a dirty upper half, and none of it leaves one. These are
// the BASIC generated kernels RULE 8 asks for -- a pixel, a few samples or two
// float64 lanes at a time -- and a 256-bit AVX2 form of each is still owed.
// The vertical resampling pass, which carries the bulk of a resize, has its
// own AVX2 kernel (resample.go).

// emitResampleHX86 is the horizontal resampling pass: per row, per output
// pixel, its three channels as three int32 lanes of one register.
//
// Registers: RDI Args (re-read for K, Cols, W and AScale2 at each row or
// pixel), RCX out row, RDX in row, R10 rows left, R13 RowStr, R15 OutStr;
// per row RSI out, RAX weights, RBX window offsets, R11 pixels left; per
// pixel R12 the window's first sample, R9 taps left, R8 the offset. XMM0
// bias, XMM2 255, XMM7 zero.
func emitResampleHX86(a *Buf, prec int) {
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RDX, At(RDI, 24)) // AScale -> in
	a.MOVLoad(R10, At(RDI, 32)) // Rows
	a.MOVLoad(R13, At(RDI, 48)) // RowStr
	a.MOVLoad(R15, At(RDI, 88)) // OutStr
	a.MOVLoad(RBX, At(RDI, 56)) // Scr
	bcastD(a, XMM0, At(RBX, resampleBiasOff))
	bcastD(a, XMM2, At(RBX, resampleMaxOff))
	a.PXOR(XMM7, XMM7, XMM7)

	done := a.Label()
	a.TESTQ(R10, R10)
	a.JZ(done)
	row := a.Label()
	a.Bind(row)
	a.MOVQ(RSI, RCX)
	a.MOVLoad(RAX, At(RDI, 8))   // W -> weights
	a.MOVLoad(RBX, At(RDI, 136)) // AScale2 -> window offsets
	a.MOVLoad(R11, At(RDI, 40))  // K -> pixels
	rowEnd := a.Label()
	a.TESTQ(R11, R11)
	a.JZ(rowEnd)
	px := a.Label()
	a.Bind(px)
	a.MOVLoad(R8, At(RBX, 0))
	a.MOVQ(R12, RDX)
	a.ADDQ(R12, R8)
	a.MOVAPS(XMM1, XMM0)
	a.MOVLoad(R9, At(RDI, 80)) // Cols -> taps
	tap := a.Label()
	a.Bind(tap)
	// The pixel's three bytes and not the next one's: a word and a byte.
	a.PINSRWLoad(XMM3, XMM7, At(R12, 0), 0)
	a.PINSRBLoad(XMM3, XMM3, At(R12, 2), 2)
	a.PMOVZXBD(XMM3, XMM3)
	bcastD(a, XMM4, At(RAX, 0))
	a.PMULLD(XMM3, XMM3, XMM4)
	a.PADDD(XMM1, XMM1, XMM3)
	a.ADDimm(R12, 3)
	a.ADDimm(RAX, 4)
	a.DEC(R9)
	a.JNZ(tap)
	a.PSRAD(XMM1, XMM1, byte(prec))
	a.PSRAD(XMM3, XMM1, 31)
	a.PANDN(XMM3, XMM3, XMM1)
	a.PMINSD(XMM3, XMM3, XMM2)
	a.PACKSSDW(XMM3, XMM3, XMM3)
	a.PACKUSWB(XMM3, XMM3, XMM3)
	a.PEXTRWStore(At(RSI, 0), XMM3, 0)
	a.PEXTRBStore(At(RSI, 2), XMM3, 2)
	a.ADDimm(RSI, 3)
	a.ADDimm(RBX, 8)
	a.DEC(R11)
	a.JNZ(px)
	a.Bind(rowEnd)
	a.ADDQ(RCX, R15)
	a.ADDQ(RDX, R13)
	a.DEC(R10)
	a.JNZ(row)
	a.Bind(done)
}

// emitPixLUTX86 is the per-channel table lookup: a pixel's three bytes, each
// the index of its channel's 256-float table.
//
// Registers: RCX out, RDX in, RBX the tables, R10 pixels, R11 the samples
// after them, RAX the index.
func emitPixLUTX86(a *Buf) {
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RDX, At(RDI, 8))  // W -> the samples
	a.MOVLoad(RBX, At(RDI, 24)) // AScale -> the tables
	a.MOVLoad(R10, At(RDI, 40)) // K -> pixels
	a.MOVLoad(R11, At(RDI, 32)) // Rows -> samples after them
	one := func(ch int32) {
		a.MOVZXBLoad(RAX, At(RDX, ch))
		a.MOVSSLoad(XMM0, Idx(RBX, RAX, 4, ch*4*pixLUTSize))
		a.MOVSSStore(At(RCX, ch*4), XMM0)
	}
	tail, done := a.Label(), a.Label()
	a.TESTQ(R10, R10)
	a.JZ(tail)
	lp := a.Label()
	a.Bind(lp)
	for ch := int32(0); ch < 3; ch++ {
		one(ch)
	}
	a.ADDimm(RDX, 3)
	a.ADDimm(RCX, 12)
	a.DEC(R10)
	a.JNZ(lp)
	a.Bind(tail)
	// At most two samples are left, channels 0 and 1 of a last pixel.
	for ch := int32(0); ch < 2; ch++ {
		a.TESTQ(R11, R11)
		a.JZ(done)
		one(ch)
		a.DEC(R11)
	}
	a.Bind(done)
}

// emitCopy32X86 is the strided copy of 4-byte elements.
//
// Registers: RCX out row, RDX in row, R10 rows left, R13 RowStr, R15 OutStr,
// R12 the source element stride, R9 the destination's; per row RSI out, RAX
// in, R11 elements left.
func emitCopy32X86(a *Buf) {
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RDX, At(RDI, 24)) // AScale -> in
	a.MOVLoad(R10, At(RDI, 32)) // Rows
	a.MOVLoad(R13, At(RDI, 48)) // RowStr
	a.MOVLoad(R15, At(RDI, 88)) // OutStr
	a.MOVLoad(R12, At(RDI, 80)) // Cols -> source element stride
	a.MOVLoad(R9, At(RDI, 160)) // DStr -> destination element stride
	done := a.Label()
	a.TESTQ(R10, R10)
	a.JZ(done)
	row := a.Label()
	a.Bind(row)
	a.MOVQ(RSI, RCX)
	a.MOVQ(RAX, RDX)
	a.MOVLoad(R11, At(RDI, 40)) // K
	rowEnd := a.Label()
	a.TESTQ(R11, R11)
	a.JZ(rowEnd)
	el := a.Label()
	a.Bind(el)
	a.MOVSSLoad(XMM0, At(RAX, 0))
	a.MOVSSStore(At(RSI, 0), XMM0)
	a.ADDQ(RAX, R12)
	a.ADDQ(RSI, R9)
	a.DEC(R11)
	a.JNZ(el)
	a.Bind(rowEnd)
	a.ADDQ(RCX, R15)
	a.ADDQ(RDX, R13)
	a.DEC(R10)
	a.JNZ(row)
	a.Bind(done)
}

// emitPixelRowX86 is one row of a decoded picture (PixelRow): each pixel's
// (r, g, b, a) at 16 bits in the four int32 lanes of XMM1, from the format,
// then out as three bytes or three floats.
//
// Registers: RCX out (or the first plane), RSI and R8 the other planes, RDX
// the source (the luma), R9 and R12 the chroma rows, R10 pixels left, R11
// the pixel's x and R13 the row's first x shifted by sub (PixYCbCr), RAX and
// RBX scratch. XMM15 the opaque alpha lane, XMM14 0xffff, XMM13 0xff, XMM12
// 257 or 0x10101, XMM11 255.0 or the Cb factors, XMM10 the byte swap or the
// Cr factors, XMM9 128, XMM8 0xffffff, XMM7 zero.
func emitPixelRowX86(a *Buf, f PixFmt, sub int, o PixOut) {
	a.MOVLoad(RCX, At(RDI, 0))   // Out
	a.MOVLoad(RSI, At(RDI, 120)) // Out2
	a.MOVLoad(R8, At(RDI, 136))  // AScale2
	a.MOVLoad(RDX, At(RDI, 8))   // W -> the source
	a.MOVLoad(R9, At(RDI, 24))   // AScale -> Cb
	a.MOVLoad(R12, At(RDI, 112)) // Q32 -> Cr
	a.MOVLoad(R10, At(RDI, 40))  // K -> pixels
	a.MOVLoad(R11, At(RDI, 80))  // Cols -> the first x
	a.MOVQ(R13, R11)
	a.SHRimm(R13, byte(sub))
	a.MOVLoad(RBX, At(RDI, 56)) // Scr
	ld := func(x Reg, off int32) { a.MOVDQULoad(x, At(RBX, off)) }
	ld(XMM15, pixAlphaOff)
	ld(XMM14, pixFFFFOff)
	ld(XMM13, pixFFOff)
	a.PXOR(XMM7, XMM7, XMM7)
	switch f {
	case PixYCbCr:
		ld(XMM12, pixK10101Off)
		ld(XMM11, pixKCbOff)
		ld(XMM10, pixKCrOff)
		ld(XMM9, pixK128Off)
		ld(XMM8, pixClampOff)
	default:
		ld(XMM12, pixK257Off)
		ld(XMM11, pixF255Off)
		ld(XMM10, pixBswapOff)
	}
	splatGP := func(x, gp Reg) {
		a.MOVDToX(x, gp)
		a.PSHUFD(x, x, 0)
	}

	done := a.Label()
	a.TESTQ(R10, R10)
	a.JZ(done)
	lp := a.Label()
	a.Bind(lp)
	switch f {
	case PixRGBA:
		a.PMOVZXBDLoad(XMM1, At(RDX, 0))
		a.PMULLD(XMM1, XMM1, XMM12)
		a.ADDimm(RDX, 4)
	case PixNRGBA:
		a.PMOVZXBDLoad(XMM1, At(RDX, 0))
		a.PSHUFD(XMM3, XMM1, 0xFF)
		a.PMULLD(XMM1, XMM1, XMM12)
		a.MOVAPS(XMM4, XMM1)
		a.PMULLD(XMM1, XMM1, XMM3)
		// c*257*a/255, truncated: below 2^24, so its float is exact and
		// the correctly rounded quotient never reaches the next integer.
		a.CVTDQ2PS(XMM1, XMM1)
		a.DIVPS(XMM1, XMM1, XMM11)
		a.CVTTPS2DQ(XMM1, XMM1)
		a.PBLENDW(XMM1, XMM1, XMM4, 0xC0)
		a.ADDimm(RDX, 4)
	case PixGray:
		a.MOVZXBLoad(RAX, At(RDX, 0))
		splatGP(XMM1, RAX)
		a.PMULLD(XMM1, XMM1, XMM12)
		a.PBLENDW(XMM1, XMM1, XMM15, 0xC0)
		a.ADDimm(RDX, 1)
	case PixRGBA64:
		a.MOVQLoad(XMM1, At(RDX, 0))
		a.PSHUFB(XMM1, XMM1, XMM10)
		a.PMOVZXWD(XMM1, XMM1)
		a.ADDimm(RDX, 8)
	case PixYCbCr:
		a.MOVZXBLoad(RAX, At(RDX, 0))
		splatGP(XMM1, RAX)
		a.PMULLD(XMM1, XMM1, XMM12)
		a.MOVQ(RAX, R11)
		a.SHRimm(RAX, byte(sub))
		a.SUBQ(RAX, R13)
		a.MOVZXBLoad(RBX, Idx(R9, RAX, 1, 0))
		a.MOVZXBLoad(RAX, Idx(R12, RAX, 1, 0))
		for _, ch := range []struct{ gp, k Reg }{{RBX, XMM11}, {RAX, XMM10}} {
			splatGP(XMM2, ch.gp)
			a.PSUBD(XMM2, XMM2, XMM9)
			a.PMULLD(XMM2, XMM2, ch.k)
			a.PADDD(XMM1, XMM1, XMM2)
		}
		a.PMAXSD(XMM1, XMM1, XMM7)
		a.PMINSD(XMM1, XMM1, XMM8)
		a.PSRLD(XMM1, XMM1, 8)
		a.PBLENDW(XMM1, XMM1, XMM15, 0xC0)
		a.ADDimm(RDX, 1)
		a.ADDimm(R11, 1)
	}
	switch o {
	case PixOver8:
		a.PSHUFD(XMM3, XMM1, 0xFF)
		a.MOVAPS(XMM4, XMM14)
		a.PSUBD(XMM4, XMM4, XMM3)
		a.PADDD(XMM1, XMM1, XMM4)
		a.PSRLD(XMM1, XMM1, 8)
		a.PAND(XMM1, XMM1, XMM13)
		a.PACKSSDW(XMM1, XMM1, XMM1)
		a.PACKUSWB(XMM1, XMM1, XMM1)
		a.PEXTRWStore(At(RCX, 0), XMM1, 0)
		a.PEXTRBStore(At(RCX, 2), XMM1, 2)
		a.ADDimm(RCX, 3)
	case PixPlanes:
		a.CVTDQ2PS(XMM1, XMM1)
		a.PEXTRDStore(At(RCX, 0), XMM1, 0)
		a.PEXTRDStore(At(RSI, 0), XMM1, 1)
		a.PEXTRDStore(At(R8, 0), XMM1, 2)
		a.ADDimm(RCX, 4)
		a.ADDimm(RSI, 4)
		a.ADDimm(R8, 4)
	}
	a.DEC(R10)
	a.JNZ(lp)
	a.Bind(done)
}

// emitAxisTapsX86 is one axis of a resampled position table (AxisTaps), four
// outputs a vector and the rest one at a time in lane 0.
//
// Registers: RCX the weight planes, RSI the index planes, R9 a plane's bytes,
// R13 three planes', RBX consts, R10 vectors, R11 the outputs after them.
// XMM15 the outputs' indices as floats, XMM14 1.0, XMM13 the absolute mask,
// XMM12 zero, XMM11 hi, XMM10 4.0; XMM0..XMM7 the body's.
func emitAxisTapsX86(a *Buf, f AxisFilter) {
	cubic := f == AxisCubic
	a.MOVLoad(RCX, At(RDI, 0))   // Out -> weights
	a.MOVLoad(RSI, At(RDI, 120)) // Out2 -> indices
	a.MOVLoad(RBX, At(RDI, 56))  // Scr
	a.MOVLoad(RAX, At(RDI, 40))  // K
	a.MOVQ(R10, RAX)
	a.SHRimm(R10, 2)
	a.MOVQ(R11, RAX)
	a.ANDimm8(R11, 3)
	a.MOVQ(R9, RAX)
	a.SHLimm(R9, 2)
	a.MOVQ(R13, R9)
	a.ADDQ(R13, R9)
	a.ADDQ(R13, R9)
	a.MOVUPSLoad(XMM15, At(RBX, axisIotaOff))
	bcastSS(a, XMM14, At(RBX, axisOneOff))
	bcastSS(a, XMM13, At(RBX, axisAbsOff))
	a.XORPS(XMM12, XMM12, XMM12)
	bcastD(a, XMM11, At(RBX, axisHiOff))
	bcastSS(a, XMM10, At(RBX, axisFourOff))
	plane := func(base Reg, k int) Mem {
		switch k {
		case 1:
			return Idx(base, R9, 1, 0)
		case 2:
			return Idx(base, R9, 2, 0)
		case 3:
			return Idx(base, R13, 1, 0)
		}
		return At(base, 0)
	}
	cnst := func(x Reg, off int32) { bcastSS(a, x, At(RBX, off)) }

	body := func(st func(m Mem, x Reg)) {
		a.MOVAPS(XMM0, XMM15)
		for k, op := range []func(dst, s1, s2 Reg){a.ADDPS, a.MULPS, a.DIVPS, a.MULPS, a.ADDPS, a.ADDPS,
			a.MULPS, a.ADDPS, a.DIVPS} {
			cnst(XMM1, int32(axisChainOff+4*k))
			op(XMM0, XMM0, XMM1)
		}
		if f == AxisLinear {
			a.MAXPS(XMM0, XMM0, XMM12) // max(x, 0)
		}
		a.ROUNDPS(XMM2, XMM0, 1) // floor
		a.MOVAPS(XMM3, XMM0)
		a.SUBPS(XMM3, XMM3, XMM2) // t
		a.CVTTPS2DQ(XMM4, XMM2)   // int(floor)
		n := AxisTapCount(f)
		for k := 0; k < n; k++ {
			off := k
			if cubic {
				off--
			}
			a.MOVAPS(XMM5, XMM4)
			if off != 0 {
				slot := map[int]int32{-1: 0, 1: 4, 2: 8}[off]
				bcastD(a, XMM6, At(RBX, axisOffsOff+slot))
				a.PADDD(XMM5, XMM5, XMM6)
			}
			a.PMAXSD(XMM5, XMM5, XMM12)
			a.PMINSD(XMM5, XMM5, XMM11)
			st(plane(RSI, k), XMM5)
		}
		if f == AxisLinear {
			a.MOVAPS(XMM6, XMM14)
			a.SUBPS(XMM6, XMM6, XMM3) // 1 - t
			st(plane(RCX, 0), XMM6)
			st(plane(RCX, 1), XMM3) // t
			return
		}
		if !cubic {
			for k := 0; k < 2; k++ {
				a.MOVAPS(XMM5, XMM3)
				if k == 1 {
					a.SUBPS(XMM5, XMM5, XMM14) // t - 1
				}
				a.ANDPS(XMM5, XMM5, XMM13)
				a.MOVAPS(XMM6, XMM14)
				a.SUBPS(XMM6, XMM6, XMM5)
				a.MAXPS(XMM6, XMM6, XMM12)
				st(plane(RCX, k), XMM6)
			}
			return
		}
		// conv1(x) = ((1.25x - 2.25)x)x + 1 and conv2(x) = ((-0.75x + 3.75)x
		// - 6)x + 3 of x in XMM5, into XMM6.
		conv := func(two bool) {
			if !two {
				cnst(XMM6, axisC1Off)
				a.MULPS(XMM6, XMM6, XMM5)
				cnst(XMM7, axisC1Off+4)
				a.SUBPS(XMM6, XMM6, XMM7)
				a.MULPS(XMM6, XMM6, XMM5)
				a.MULPS(XMM6, XMM6, XMM5)
				a.ADDPS(XMM6, XMM6, XMM14)
				return
			}
			cnst(XMM6, axisC2Off)
			a.MULPS(XMM6, XMM6, XMM5)
			cnst(XMM7, axisC2Off+4)
			a.ADDPS(XMM6, XMM6, XMM7)
			a.MULPS(XMM6, XMM6, XMM5)
			cnst(XMM7, axisC2Off+8)
			a.ADDPS(XMM6, XMM6, XMM7)
			a.MULPS(XMM6, XMM6, XMM5)
			cnst(XMM7, axisC2Off+12)
			a.ADDPS(XMM6, XMM6, XMM7)
		}
		for k := 0; k < 4; k++ {
			switch k {
			case 0: // t + 1
				a.MOVAPS(XMM5, XMM3)
				a.ADDPS(XMM5, XMM5, XMM14)
			case 1: // t
				a.MOVAPS(XMM5, XMM3)
			case 2: // 1 - t
				a.MOVAPS(XMM5, XMM14)
				a.SUBPS(XMM5, XMM5, XMM3)
			case 3: // 2 - t
				cnst(XMM5, axisTwoOff)
				a.SUBPS(XMM5, XMM5, XMM3)
			}
			conv(k == 0 || k == 3)
			st(plane(RCX, k), XMM6)
		}
	}

	tail, done := a.Label(), a.Label()
	a.TESTQ(R10, R10)
	a.JZ(tail)
	lp := a.Label()
	a.Bind(lp)
	body(func(m Mem, x Reg) { a.MOVUPSStore(m, x) })
	a.ADDimm(RCX, 16)
	a.ADDimm(RSI, 16)
	a.ADDPS(XMM15, XMM15, XMM10)
	a.DEC(R10)
	a.JNZ(lp)
	a.Bind(tail)
	a.TESTQ(R11, R11)
	a.JZ(done)
	lt := a.Label()
	a.Bind(lt)
	body(func(m Mem, x Reg) { a.MOVSSStore(m, x) })
	a.ADDimm(RCX, 4)
	a.ADDimm(RSI, 4)
	a.ADDPS(XMM15, XMM15, XMM14)
	a.DEC(R11)
	a.JNZ(lt)
	a.Bind(done)
}

// emitLerpGridX86 is one tap pair's plane of a 2-D table (LerpGrid): per row
// y, hw[y] and ht[y]*side broadcast against the column axis, four columns a
// vector and the rest one at a time.
//
// Registers: RCX weights out, RSI indices out, RDX hw, R8 ht, R10 rows left;
// per row RAX ww, R9 wt, R11 vectors, R12 the columns after them. XMM15
// side, XMM0 hw[y], XMM1 ht[y]*side.
func emitLerpGridX86(a *Buf) {
	a.MOVLoad(RCX, At(RDI, 0))   // Out -> weights
	a.MOVLoad(RSI, At(RDI, 120)) // Out2 -> indices
	a.MOVLoad(RDX, At(RDI, 112)) // Q32 -> hw
	a.MOVLoad(R8, At(RDI, 128))  // Q2 -> ht
	a.MOVLoad(R10, At(RDI, 32))  // Rows
	a.MOVLoad(RAX, At(RDI, 80))  // Cols -> side
	a.MOVDToX(XMM15, RAX)
	a.PSHUFD(XMM15, XMM15, 0)
	done := a.Label()
	a.TESTQ(R10, R10)
	a.JZ(done)
	row := a.Label()
	a.Bind(row)
	bcastSS(a, XMM0, At(RDX, 0))
	bcastD(a, XMM1, At(R8, 0))
	a.PMULLD(XMM1, XMM1, XMM15)
	a.MOVLoad(RAX, At(RDI, 24)) // AScale -> ww
	a.MOVLoad(R9, At(RDI, 136)) // AScale2 -> wt
	a.MOVLoad(R11, At(RDI, 40)) // K
	a.MOVQ(R12, R11)
	a.SHRimm(R11, 2)
	a.ANDimm8(R12, 3)
	body := func(ld func(x Reg, m Mem), st func(m Mem, x Reg)) {
		ld(XMM2, At(RAX, 0))
		a.MULPS(XMM2, XMM2, XMM0)
		st(At(RCX, 0), XMM2)
		ld(XMM3, At(R9, 0))
		a.PADDD(XMM3, XMM3, XMM1)
		st(At(RSI, 0), XMM3)
	}
	step := func(n int32) {
		for _, r := range []Reg{RCX, RSI, RAX, R9} {
			a.ADDimm(r, n)
		}
	}
	tail, rowEnd := a.Label(), a.Label()
	a.TESTQ(R11, R11)
	a.JZ(tail)
	lp := a.Label()
	a.Bind(lp)
	body(a.MOVUPSLoad, a.MOVUPSStore)
	step(16)
	a.DEC(R11)
	a.JNZ(lp)
	a.Bind(tail)
	a.TESTQ(R12, R12)
	a.JZ(rowEnd)
	lt := a.Label()
	a.Bind(lt)
	body(a.MOVSSLoad, a.MOVSSStore)
	step(4)
	a.DEC(R12)
	a.JNZ(lt)
	a.Bind(rowEnd)
	a.ADDimm(RDX, 4)
	a.ADDimm(R8, 4)
	a.DEC(R10)
	a.JNZ(row)
	a.Bind(done)
}

// emitSinCosTabX86 is the resampler's sin/cos table along one axis
// (SinCosTab): two frequencies at a time as float64 lanes, their positions
// down the columns.
//
// Registers: RCX and RDX the pair's sin and cos columns, RBX consts, R9 a
// row's bytes, R10 pairs, R11 the frequency after them, R13 positions; per
// pair RSI and RAX the row cursors, R12 positions left. XMM15 the pair's
// frequency indices (float32), XMM14 K (float32), XMM13 the frequencies,
// XMM12 the position (float32), XMM11 1.0f, XMM10 2.0f; XMM0..XMM9 the
// float64 work.
func emitSinCosTabX86(a *Buf) {
	a.MOVLoad(RCX, At(RDI, 0))   // Out -> sin
	a.MOVLoad(RDX, At(RDI, 120)) // Out2 -> cos
	a.MOVLoad(RBX, At(RDI, 56))  // Scr
	a.MOVLoad(R13, At(RDI, 32))  // Rows -> positions
	a.MOVLoad(RAX, At(RDI, 40))  // K -> frequencies
	a.MOVQ(R9, RAX)
	a.SHLimm(R9, 2)
	a.MOVQ(R10, RAX)
	a.SHRimm(R10, 1)
	a.MOVQ(R11, RAX)
	a.ANDimm8(R11, 1)
	a.CVTSI2SS(XMM14, XMM14, RAX)
	a.SHUFPS(XMM14, XMM14, XMM14, 0)
	a.MOVQLoad(XMM15, At(RBX, scIotaF32Off))
	bcastSS(a, XMM11, At(RBX, scOneF32Off))
	bcastSS(a, XMM10, At(RBX, scTwoF32Off))
	k := func(x Reg, off int32) { a.MOVDDUPLoad(x, At(RBX, off)) }

	// The pair's frequencies, into XMM13: om = 2^-k * e^r of s = y*ln(base).
	freqs := func() {
		a.MOVAPS(XMM0, XMM15)
		a.DIVPS(XMM0, XMM0, XMM14)
		a.CVTPS2PD(XMM0, XMM0) // y
		k(XMM1, scLnHiOff)
		a.MULPD(XMM1, XMM1, XMM0) // t1, exact
		k(XMM2, scLnLoOff)
		a.MULPD(XMM2, XMM2, XMM0) // t2
		a.MOVAPS(XMM3, XMM1)
		a.ADDPD(XMM3, XMM3, XMM2)
		k(XMM4, scLog2eOff)
		a.MULPD(XMM3, XMM3, XMM4)
		a.ROUNDPD(XMM3, XMM3, 0) // k
		k(XMM4, scLn2HiOff)
		a.MULPD(XMM4, XMM4, XMM3)
		a.SUBPD(XMM4, XMM4, XMM1)
		k(XMM5, scLn2LoOff)
		a.MULPD(XMM5, XMM5, XMM3)
		a.SUBPD(XMM5, XMM5, XMM2)
		a.ADDPD(XMM4, XMM4, XMM5) // r = k*ln2 - s
		k(XMM5, scExpOff)
		for i := 1; i < scExpTerms; i++ {
			a.MULPD(XMM5, XMM5, XMM4)
			k(XMM6, int32(scExpOff+8*i))
			a.ADDPD(XMM5, XMM5, XMM6)
		}
		k(XMM6, scMagicOff)
		a.SUBPD(XMM6, XMM6, XMM3)
		a.PSLLQ(XMM6, XMM6, 52) // 2^-k
		a.MULPD(XMM5, XMM5, XMM6)
		a.CVTPD2PS(XMM13, XMM5)
	}
	// sin and cos of the float64 lanes of XMM0, into XMM8 and XMM6.
	sincos := func() {
		k(XMM1, scTwoPiOff)
		a.MULPD(XMM1, XMM1, XMM0)
		a.ROUNDPD(XMM1, XMM1, 0) // j, the nearest quarter-turn
		a.MOVAPS(XMM3, XMM0)
		for i := int32(0); i < 3; i++ {
			k(XMM2, scPio2Off+8*i)
			a.MULPD(XMM2, XMM2, XMM1)
			a.SUBPD(XMM3, XMM3, XMM2)
		}
		a.MOVAPS(XMM4, XMM3)
		a.MULPD(XMM4, XMM4, XMM3) // z = r*r
		k(XMM5, scSinOff)
		for i := int32(1); i < 6; i++ {
			a.MULPD(XMM5, XMM5, XMM4)
			k(XMM6, scSinOff+8*i)
			a.ADDPD(XMM5, XMM5, XMM6)
		}
		a.MOVAPS(XMM6, XMM3)
		a.MULPD(XMM6, XMM6, XMM4)
		a.MULPD(XMM5, XMM5, XMM6)
		a.ADDPD(XMM5, XMM5, XMM3) // s = r + (r*z)*poly
		k(XMM6, scCosOff)
		for i := int32(1); i < 6; i++ {
			a.MULPD(XMM6, XMM6, XMM4)
			k(XMM7, scCosOff+8*i)
			a.ADDPD(XMM6, XMM6, XMM7)
		}
		a.MOVAPS(XMM7, XMM4)
		a.MULPD(XMM7, XMM7, XMM4)
		a.MULPD(XMM6, XMM6, XMM7)
		k(XMM7, scHalfOff)
		a.MULPD(XMM7, XMM7, XMM4)
		k(XMM8, scOneOff)
		a.SUBPD(XMM8, XMM8, XMM7)
		a.ADDPD(XMM6, XMM6, XMM8) // c = (1 - z/2) + z*z*poly
		// The quadrant q = j mod 4, in exact float64 arithmetic: sw = q mod 2
		// swaps the polynomials, 1 - 2*floor(q/2) signs sin and
		// 1 - 2*(floor((q+1)/2) mod 2) signs cos.
		k(XMM7, scQuarterOff)
		a.MULPD(XMM7, XMM7, XMM1)
		a.ROUNDPD(XMM7, XMM7, 1)
		k(XMM8, scFourOff)
		a.MULPD(XMM7, XMM7, XMM8)
		a.MOVAPS(XMM2, XMM1)
		a.SUBPD(XMM2, XMM2, XMM7) // q
		k(XMM8, scHalfOff)
		a.MOVAPS(XMM7, XMM2)
		a.MULPD(XMM7, XMM7, XMM8)
		a.ROUNDPD(XMM7, XMM7, 1) // h = floor(q/2)
		a.MOVAPS(XMM9, XMM7)
		a.ADDPD(XMM9, XMM9, XMM7)
		a.MOVAPS(XMM3, XMM2)
		a.SUBPD(XMM3, XMM3, XMM9) // sw
		k(XMM4, scOneOff)
		a.SUBPD(XMM4, XMM4, XMM9) // sin's sign
		k(XMM9, scOneOff)
		a.ADDPD(XMM9, XMM9, XMM2)
		a.MULPD(XMM9, XMM9, XMM8)
		a.ROUNDPD(XMM9, XMM9, 1) // floor((q+1)/2)
		a.MOVAPS(XMM7, XMM9)
		a.MULPD(XMM7, XMM7, XMM8)
		a.ROUNDPD(XMM7, XMM7, 1)
		a.ADDPD(XMM7, XMM7, XMM7)
		a.SUBPD(XMM9, XMM9, XMM7)
		a.ADDPD(XMM9, XMM9, XMM9)
		k(XMM2, scOneOff)
		a.SUBPD(XMM2, XMM2, XMM9) // cos's sign
		k(XMM7, scOneOff)
		a.SUBPD(XMM7, XMM7, XMM3) // 1 - sw
		a.MOVAPS(XMM8, XMM5)
		a.MULPD(XMM8, XMM8, XMM7)
		a.MOVAPS(XMM9, XMM6)
		a.MULPD(XMM9, XMM9, XMM3)
		a.ADDPD(XMM8, XMM8, XMM9)
		a.MULPD(XMM8, XMM8, XMM4) // sin
		a.MULPD(XMM6, XMM6, XMM7)
		a.MULPD(XMM5, XMM5, XMM3)
		a.ADDPD(XMM6, XMM6, XMM5)
		a.MULPD(XMM6, XMM6, XMM2) // cos
	}
	// column runs the positions down the current columns.
	column := func(st func(m Mem, x Reg)) {
		a.MOVQ(RSI, RCX)
		a.MOVQ(RAX, RDX)
		a.XORPS(XMM12, XMM12, XMM12)
		a.MOVQ(R12, R13)
		end := a.Label()
		a.TESTQ(R12, R12)
		a.JZ(end)
		lp := a.Label()
		a.Bind(lp)
		a.MOVAPS(XMM0, XMM12)
		a.MULPS(XMM0, XMM0, XMM13)
		a.CVTPS2PD(XMM0, XMM0)
		sincos()
		a.CVTPD2PS(XMM8, XMM8)
		a.CVTPD2PS(XMM6, XMM6)
		st(At(RSI, 0), XMM8)
		st(At(RAX, 0), XMM6)
		a.ADDQ(RSI, R9)
		a.ADDQ(RAX, R9)
		a.ADDPS(XMM12, XMM12, XMM11)
		a.DEC(R12)
		a.JNZ(lp)
		a.Bind(end)
	}

	tail, done := a.Label(), a.Label()
	a.TESTQ(R10, R10)
	a.JZ(tail)
	pair := a.Label()
	a.Bind(pair)
	freqs()
	column(a.MOVQStore)
	a.ADDimm(RCX, 8)
	a.ADDimm(RDX, 8)
	a.ADDPS(XMM15, XMM15, XMM10)
	a.DEC(R10)
	a.JNZ(pair)
	a.Bind(tail)
	a.TESTQ(R11, R11)
	a.JZ(done)
	freqs()
	column(a.MOVSSStore)
	a.Bind(done)
}

// sseKernel wraps an x86 body as the SSE tier's kernel.
func sseKernel(body func(a *Buf)) ([]byte, error) {
	var a Buf
	a.DeclareISA(ISATierSSE)
	body(&a)
	a.RET()
	return a.Bytes(), nil
}

// EmitResampleHSSE, EmitPixLUTSSE and EmitCopy32SSE are the SSE tier's.
func EmitResampleHSSE(prec int) ([]byte, error) {
	return sseKernel(func(a *Buf) { emitResampleHX86(a, prec) })
}
func EmitPixLUTSSE() ([]byte, error) { return sseKernel(emitPixLUTX86) }
func EmitCopy32SSE() ([]byte, error) { return sseKernel(emitCopy32X86) }

// EmitAxisTapsSSE and EmitLerpGridSSE are the SSE tier's.
func EmitAxisTapsSSE(f AxisFilter) ([]byte, error) {
	if f > AxisCubic {
		return nil, fmt.Errorf("jit: AxisTaps: filter %d", f)
	}
	return sseKernel(func(a *Buf) { emitAxisTapsX86(a, f) })
}
func EmitLerpGridSSE() ([]byte, error)  { return sseKernel(emitLerpGridX86) }
func EmitSinCosTabSSE() ([]byte, error) { return sseKernel(emitSinCosTabX86) }

// EmitPixelRowSSE is the SSE tier's PixelRow.
func EmitPixelRowSSE(f PixFmt, sub int, o PixOut) ([]byte, error) {
	if err := pixelShape(f, sub, o); err != nil {
		return nil, err
	}
	return sseKernel(func(a *Buf) { emitPixelRowX86(a, f, sub, o) })
}
