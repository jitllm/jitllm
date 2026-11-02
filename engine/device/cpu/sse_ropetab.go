package cpu

// The SSE tier's rotary table: EmitRopeTable (ropetab.go) on legacy SSE.
//
// It takes the AVX2 twin's ABI and constant blocks unchanged (Out the table,
// AScale the per-model planes, Scr = RopeTabConsts(), K the position); the
// per-model block is padded to eight lanes on every tier so the same bytes
// serve a four-lane kernel.
//
// It is not bit-identical to the AVX2 kernel because of FMA, not width: no
// lanes meet, but the AVX2 tier fuses the pi/2 tail and each Horner step, so
// the tables land a few ulps apart and the gate holds this one to
// math.Sin/math.Cos. The reduction's digit products, sum and mod-4 fold are
// exact on both tiers, so the large-angle behaviour is identical.
//
// Registers: RCX cs, RDX the plane cursor, RBX Scr, RAX the vector count;
// XMM0..XMM3 the position digits, XMM4 the mod-4 magic, XMM5 mscale, XMM6/XMM7
// pi/2's two words, XMM8..XMM15 the working set.
func EmitRopeTableSSE(npairs int) ([]byte, error) {
	const lanes = 4
	if npairs <= 0 {
		return nil, ropeTabShapeErr(npairs)
	}
	stride := int32(4 * RopeTabStride(npairs))
	nv, tail := npairs/lanes, npairs%lanes

	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))  // Out    -> cs
	a.MOVLoad(RDX, At(RDI, 24)) // AScale -> the planes
	a.MOVLoad(RBX, At(RDI, 56)) // Scr    -> RopeTabConsts()

	a.MOVDLoad(XMM8, At(RDI, 40)) // the position, low dword of Args.K
	a.PSHUFD(XMM8, XMM8, 0)
	a.MOVDLoad(XMM9, At(RBX, rtDigitOff))
	a.PSHUFD(XMM9, XMM9, 0)
	for d := 0; d < RopeTabDigits; d++ {
		if d > 0 {
			a.PSRLD(XMM8, XMM8, RopeTabDigitBits)
		}
		a.PAND(XMM0+Reg(d), XMM8, XMM9)
		a.CVTDQ2PS(XMM0+Reg(d), XMM0+Reg(d))
	}
	bcastSS(&a, XMM4, At(RBX, rtMagicOff))
	bcastSS(&a, XMM6, At(RBX, rtPio2HiOff))
	bcastSS(&a, XMM7, At(RBX, rtPio2LoOff))
	bcastSS(&a, XMM5, At(RDX, int32(4*RopeTabScaleOff(npairs))))

	// horner is p = z*p + c with c read from the constant block. Two
	// instructions and a broadcast where the AVX2 tier has one fused one.
	horner := func(p, z Reg, cOff int32) {
		a.MULPS(p, p, z)
		bcastSS(&a, XMM13, At(RBX, cOff))
		a.ADDPS(p, p, XMM13)
	}

	body := func(n int) {
		// s = sum of the four digit-by-head products, every one exact.
		a.MOVUPSLoad(XMM8, At(RDX, 0))
		a.MULPS(XMM8, XMM8, XMM0)
		for d := 1; d < RopeTabDigits; d++ {
			a.MOVUPSLoad(XMM9, At(RDX, int32(d)*stride))
			a.MULPS(XMM9, XMM9, XMM0+Reg(d))
			a.ADDPS(XMM8, XMM8, XMM9)
		}
		// s mod 4, exactly.
		a.ADDPS(XMM9, XMM8, XMM4)
		a.SUBPS(XMM9, XMM9, XMM4)
		a.SUBPS(XMM8, XMM8, XMM9)

		// c, the residual planes' contribution.
		a.MOVUPSLoad(XMM10, At(RDX, int32(RopeTabDigits)*stride))
		a.MULPS(XMM10, XMM10, XMM0)
		for d := 1; d < RopeTabDigits; d++ {
			a.MOVUPSLoad(XMM9, At(RDX, int32(RopeTabDigits+d)*stride))
			a.MULPS(XMM9, XMM9, XMM0+Reg(d))
			a.ADDPS(XMM10, XMM10, XMM9)
		}

		// The quadrant from s+c, the remainder from s alone. See ropetab.go.
		a.ADDPS(XMM9, XMM8, XMM10)
		a.CVTPS2DQ(XMM12, XMM9)
		a.CVTDQ2PS(XMM9, XMM12)
		a.SUBPS(XMM8, XMM8, XMM9) // f, exact

		// r = (f + c)*pi/2 with the large term added last; see ropetab.go.
		// Without FMA the f*pi/2hi product rounds too, so this tier carries
		// two roundings here where the AVX2 one carries a single fused add.
		a.MULPS(XMM9, XMM10, XMM6)
		a.MULPS(XMM13, XMM8, XMM7)
		a.ADDPS(XMM9, XMM9, XMM13)
		a.MULPS(XMM13, XMM8, XMM6)
		a.ADDPS(XMM9, XMM9, XMM13)
		a.MULPS(XMM10, XMM9, XMM9) // z

		// sin(r) = r + (r*z)*(c3 + z*(c5 + z*c7))
		bcastSS(&a, XMM11, At(RBX, rtSinC7Off))
		horner(XMM11, XMM10, rtSinC5Off)
		horner(XMM11, XMM10, rtSinC3Off)
		a.MOVAPS(XMM15, XMM9)
		a.MULPS(XMM15, XMM15, XMM10) // r*z
		a.MULPS(XMM11, XMM11, XMM15)
		a.ADDPS(XMM11, XMM11, XMM9)

		// cos(r) = 1 + z*(-1/2 + z*(c4 + z*(c6 + z*c8)))
		bcastSS(&a, XMM14, At(RBX, rtCosC8Off))
		for _, off := range []int32{rtCosC6Off, rtCosC4Off, rtNegHalfOff, rtOneOff} {
			horner(XMM14, XMM10, off)
		}

		// Bit 0 of n swaps the two polynomials.
		a.PSLLD(XMM15, XMM12, 31)
		a.PSRAD(XMM15, XMM15, 31)
		a.PXOR(XMM13, XMM11, XMM14)
		a.PAND(XMM13, XMM13, XMM15)
		a.PXOR(XMM11, XMM11, XMM13)
		a.PXOR(XMM14, XMM14, XMM13)

		// Bit 1 of n negates sin; bit 1 of n+1 negates cos.
		a.MOVUPSLoad(XMM13, At(RBX, rtSignVecOff))
		a.PSLLD(XMM15, XMM12, 30)
		a.PAND(XMM15, XMM15, XMM13)
		a.PXOR(XMM11, XMM11, XMM15)
		a.MOVUPSLoad(XMM15, At(RBX, rtOneVecOff))
		a.PADDD(XMM12, XMM12, XMM15)
		a.PSLLD(XMM15, XMM12, 30)
		a.PAND(XMM15, XMM15, XMM13)
		a.PXOR(XMM14, XMM14, XMM15)

		a.MULPS(XMM11, XMM11, XMM5)
		a.MULPS(XMM14, XMM14, XMM5)

		// Interleave to {cos, sin} per pair and store n of the four; with one
		// 128-bit lane the two shuffles are the whole job.
		a.SHUFPS(XMM13, XMM14, XMM11, 0x44)
		a.SHUFPS(XMM13, XMM13, XMM13, 0xD8)
		a.SHUFPS(XMM15, XMM14, XMM11, 0xEE)
		a.SHUFPS(XMM15, XMM15, XMM15, 0xD8)
		chunks := []Reg{XMM13, XMM15}
		for j := 0; j < n/2; j++ {
			a.MOVUPSStore(At(RCX, int32(16*j)), chunks[j])
		}
		if n%2 != 0 {
			c := chunks[n/2]
			a.MOVSSStore(At(RCX, int32(16*(n/2))), c)
			a.SHUFPS(XMM9, c, c, 0x01)
			a.MOVSSStore(At(RCX, int32(16*(n/2)+4)), XMM9)
		}
	}

	if nv > 0 {
		a.MOVimm32(RAX, int32(nv))
		lp := a.Label()
		a.Bind(lp)
		body(lanes)
		a.ADDimm(RDX, 4*lanes)
		a.ADDimm(RCX, 8*lanes)
		a.DEC(RAX)
		a.JNZ(lp)
	}

	// The ragged tail runs the whole body and stores part of it; see ropetab.go.
	if tail != 0 {
		body(tail)
	}
	a.RET()
	return a.Bytes(), nil
}
