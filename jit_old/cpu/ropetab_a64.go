//go:build arm64

package cpu

// EmitRopeTable is the NEON twin of the amd64 rotary-table kernel; ropetab.go
// carries the contract, the quadrant argument and the register discipline, and
// ropetab_const.go the constant blocks and the exactness of the position
// decomposition. The two differ in width and in three instructions:
//
//	ST2 stores {cos, sin} interleaved from two registers, so the table's layout
//	    falls out of the store and the four shuffles amd64 needs do not exist
//	    here.
//	BIT selects under a mask in one instruction, so the quadrant swap is two
//	    of them rather than an XOR-mask-XOR-XOR.
//	FNEG + BIT is the conditional negate, because the sign flip is a select
//	    here rather than an XOR with a mask constant.
//
// Thirty-two registers mean every polynomial coefficient is loaded once, before
// the loop, where the amd64 kernel broadcasts each from its block per iteration.
//
//	x0  Args     x1  cs        x2  the plane cursor   x3  Scr   x4  the count
//	v0..v3   the position digits      v4  the mod-4 magic   v5  mscale
//	v6,v7    pi/2's two words         v8..v10  sin's coefficients
//	v11..v15 cos's                    v16 the integer 1, v17/v27 the block's 2 and 1
//	v18..v25 the working set          v28,v29  the interleaved store pair
func EmitRopeTable(npairs int) ([]byte, error) {
	const lanes = 4
	if npairs <= 0 {
		return nil, ropeTabShapeErr(npairs)
	}
	stride := int32(4 * RopeTabStride(npairs))
	// LDR q's immediate reaches 65520 bytes and the eighth plane sits at
	// 7*stride; past that (about 2340 pairs) the planes' offsets are held in
	// x5..x11 and read through the register form. Within reach the immediate
	// form is kept, so every shipped width emits the bytes it did.
	farPlanes := 7*stride > 65520
	nv, tail := npairs/lanes, npairs%lanes

	var a A64
	a.LDRx(X1, X0, 0)  // Out    -> cs
	a.LDRx(X2, X0, 24) // AScale -> the planes
	a.LDRx(X3, X0, 56) // Scr    -> RopeTabConsts()

	bcast := func(v VReg, base XReg, off int32) {
		a.LDRs(v, base, off)
		a.DUPs4(v, v)
	}
	// The position's four base-128 digits, taken apart in the vector unit.
	bcast(V18, X0, 40)
	bcast(V19, X3, rtDigitOff)
	for d := 0; d < RopeTabDigits; d++ {
		if d > 0 {
			a.USHR4s(V18, V18, RopeTabDigitBits)
		}
		a.AND16b(V0+VReg(d), V18, V19)
		a.SCVTF4s(V0+VReg(d), V0+VReg(d))
	}
	bcast(V4, X3, rtMagicOff)
	bcast(V6, X3, rtPio2HiOff)
	bcast(V7, X3, rtPio2LoOff)
	for i, off := range []int32{rtSinC7Off, rtSinC5Off, rtSinC3Off,
		rtCosC8Off, rtCosC6Off, rtCosC4Off, rtNegHalfOff, rtOneOff} {
		bcast(V8+VReg(i), X3, off)
	}
	// The scale word sits past the eight planes, beyond LDR s's 16380-byte
	// reach from 512 pairs up; there its word index goes through x5, which the
	// plane offsets below take over afterwards.
	if off := int32(4 * RopeTabScaleOff(npairs)); off <= 16380 {
		bcast(V5, X2, off)
	} else {
		a.MOVimm(X5, int64(RopeTabScaleOff(npairs)))
		a.LDRsIdx(V5, X2, X5)
		a.DUPs4(V5, V5)
	}
	plane := func(v VReg, p int) {
		if p == 0 || !farPlanes {
			a.LDRq(v, X2, int32(p)*stride)
			return
		}
		a.LDRqIdx(v, X2, X4+XReg(p))
	}
	if farPlanes {
		for p := 1; p < 2*RopeTabDigits; p++ {
			a.MOVimm(X4+XReg(p), int64(int32(p)*stride))
		}
	}
	a.MOVI4s(V16, 1)
	// The other two quadrant constants come from the block rather than MOVI
	// so the gate's violations, which zero words of Scr (ropetab_const.go),
	// reach them. Two loads per call, outside the loop.
	a.LDRq(V17, X3, rtTwoVecOff) // the integer two, for bit 1 of the quadrant
	a.LDRq(V27, X3, rtOneVecOff) // the integer one, for cosine's n+1

	body := func(n int) {
		// s = sum of the four digit-by-head products, every one exact, and so
		// is the sum (ropetab_const.go).
		plane(V18, 0)
		a.FMUL4s(V18, V18, V0)
		for d := 1; d < RopeTabDigits; d++ {
			plane(V19, d)
			a.FMUL4s(V19, V19, V0+VReg(d))
			a.FADD4s(V18, V18, V19)
		}
		// s mod 4, exactly: the magic add rounds to a multiple of four.
		a.FADD4s(V19, V18, V4)
		a.FSUB4s(V19, V19, V4)
		a.FSUB4s(V18, V18, V19)

		// c, the residual planes' contribution.
		plane(V20, RopeTabDigits)
		a.FMUL4s(V20, V20, V0)
		for d := 1; d < RopeTabDigits; d++ {
			plane(V19, RopeTabDigits+d)
			a.FMUL4s(V19, V19, V0+VReg(d))
			a.FADD4s(V20, V20, V19)
		}

		// The quadrant from s+c, the remainder from s alone. See ropetab.go.
		a.FADD4s(V19, V18, V20)
		a.FCVTNS4s(V21, V19)
		a.SCVTF4s(V19, V21)
		a.FSUB4s(V18, V18, V19) // f, exact

		// r = (f + c)*pi/2 with the large term added last; see ropetab.go.
		a.FMUL4s(V19, V20, V6)
		a.FMLA4s(V19, V18, V7)
		a.FMLA4s(V19, V18, V6)
		a.FMUL4s(V20, V19, V19) // z

		// sin(r) = r + (r*z)*(c3 + z*(c5 + z*c7))
		a.MOVvec(V22, V9)
		a.FMLA4s(V22, V20, V8)
		a.MOVvec(V23, V10)
		a.FMLA4s(V23, V20, V22)
		a.FMUL4s(V24, V19, V20)
		a.MOVvec(V22, V19)
		a.FMLA4s(V22, V24, V23) // sin

		// cos(r) = 1 + z*(-1/2 + z*(c4 + z*(c6 + z*c8)))
		a.MOVvec(V23, V12)
		a.FMLA4s(V23, V20, V11)
		a.MOVvec(V24, V13)
		a.FMLA4s(V24, V20, V23)
		a.MOVvec(V23, V14)
		a.FMLA4s(V23, V20, V24)
		a.MOVvec(V24, V15)
		a.FMLA4s(V24, V20, V23) // cos

		// Bit 0 of n swaps the two polynomials.
		a.AND16b(V25, V21, V16)
		a.CMEQ4s(V25, V25, V16)
		a.MOVvec(V18, V22)
		a.BIT16b(V22, V24, V25)
		a.BIT16b(V24, V18, V25)

		// Bit 1 of n negates sin; bit 1 of n+1 negates cos.
		a.AND16b(V25, V21, V17)
		a.CMEQ4s(V25, V25, V17)
		a.FNEG4s(V18, V22)
		a.BIT16b(V22, V18, V25)
		a.ADD4s(V21, V21, V27)
		a.AND16b(V25, V21, V17)
		a.CMEQ4s(V25, V25, V17)
		a.FNEG4s(V18, V24)
		a.BIT16b(V24, V18, V25)

		a.FMUL4s(V28, V24, V5) // cos
		a.FMUL4s(V29, V22, V5) // sin
		if n == lanes {
			a.ST2(V28, X1)
			return
		}
		// A partial tail: ST2 has no shortened form, so each pair is gathered
		// into lanes 0 and 1 of a scratch register and stored as one doubleword.
		for p := 0; p < n; p++ {
			a.INSs(V26, 0, V28, uint8(p))
			a.INSs(V26, 1, V29, uint8(p))
			a.STRd(V26, X1, int32(8*p))
		}
	}

	if nv > 0 {
		a.MOVimm(X4, int64(nv))
		lp := a.Label()
		a.Bind(lp)
		body(lanes)
		a.ADDimm(X2, X2, 4*lanes)
		a.ADDimm(X1, X1, 8*lanes)
		a.SUBimm(X4, X4, 1)
		a.CBNZ(X4, lp)
	}

	// The ragged tail runs the whole body and stores part of it; see ropetab.go.
	if tail != 0 {
		body(tail)
	}
	a.RET()
	return a.Bytes(), nil
}
