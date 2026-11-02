//go:build arm64

package cpu

// The sampler's three kernels on NEON. sample.go carries the contract, the ABI
// and the reasoning; these are the same kernels with arm64's masked select.
//
//	x0  the Args block          x1  the value cursor   x2  Scr
//	x3  the id cursor           x4  the output values  x5  the output ids
//	x6  the parameters          x7  the counters       x9  the mass
//	x10 the element count       x11, x12 loop counters x13 the segment count
//	x14 a scratch address
//
// A masked select is one instruction here: BIT (Bitwise Insert if True) is
// vd = (vn & vm) | (vd & ~vm), as in topk_a64.go.
//
// The compares are FCMGT/FCMGE/CMGT directly, with no operand swapping: NEON
// has an ordered greater-or-equal, so `acc >= TopP` is written as meant rather
// than as `!(acc < TopP)`, which would admit a NaN.

// emitSampleSegMaxA64 is EmitSampleSegMax (sample.go) on NEON.
func emitSampleSegMaxA64(first, idsMem bool) []byte {
	var a A64
	a.LDRx(X2, X0, 56)  // Scr
	a.LDRx(X1, X0, 112) // Q32      -> the values
	a.LDRx(X4, X0, 0)   // Out
	a.LDRx(X5, X0, 104) // AHalfSum -> the ids out
	a.LDRx(X10, X0, 40) // K
	a.LDRx(X13, X0, 32) // Rows
	// ANDlow takes a bit width, not a mask: two bits is `K & 3`, the tail of a
	// four-lane vector. (`K & 7` is the amd64 tail and read past every segment
	// whose count has bit 2 set.)
	a.ANDlow(X15, X10, 2)
	a.LSRimm(X10, X10, 2)

	a.LDRs(V13, X2, moeMaxIOff)
	a.DUPs4(V13, V13)
	a.LDRs(V14, X2, moeNaNOff)
	a.DUPs4(V14, V14)

	if idsMem {
		a.LDRx(X3, X0, 96)
	} else {
		// The index vector starts at Cols and then flows across the segments:
		// after a segment of K elements it already holds the next segment's base.
		a.LDRq(V2, X2, moeIdxOff) // 0..3
		a.LDRs(V4, X0, 80)        // the low half of Cols
		a.DUPs4(V4, V4)
		a.ADD4s(V2, V2, V4)
	}
	if !first {
		a.LDRx(X14, X0, 24)
		a.LDRs(V11, X14, 0)
		a.DUPs4(V11, V11)
		a.LDRs(V12, X14, 4)
		a.DUPs4(V12, V12)
	}

	body := func() {
		if first {
			a.FCMEQ4s(V5, V4, V4) // ordered: every element but a NaN
		} else {
			a.FCMGT4s(V5, V11, V4) // prevVal > v, i.e. v < prevVal
			a.FCMEQ4s(V6, V4, V11)
			a.CMGT4s(V7, V2, V12) // and a higher id than the tie it lost
			a.AND16b(V6, V6, V7)
			a.ORR16b(V5, V5, V6)
		}
		a.FCMGT4s(V8, V4, V0) // v > max, false for a NaN on either side
		a.CMEQ4s(V9, V1, V13) // unless this lane is still empty
		a.ORR16b(V8, V8, V9)
		a.AND16b(V5, V5, V8)
		a.BIT16b(V0, V4, V5)
		a.BIT16b(V1, V2, V5)
	}

	seg := a.Label()
	a.Bind(seg)
	a.LDRs(V0, X2, moeNegInfOff)
	a.DUPs4(V0, V0)
	a.MOVvec(V1, V13)
	a.MOVreg(X11, X10)
	a.MOVreg(X12, X15)

	vt, vd := a.Label(), a.Label()
	a.CBZ(X11, vt)
	if !idsMem {
		a.LDRs(V3, X2, moeStep4Off)
		a.DUPs4(V3, V3)
	}
	vl := a.Label()
	a.Bind(vl)
	a.LDRq(V4, X1, 0)
	if idsMem {
		a.LDRq(V2, X3, 0)
		a.ADDimm(X3, X3, 16)
	}
	body()
	if !idsMem {
		a.ADD4s(V2, V2, V3)
	}
	a.ADDimm(X1, X1, 16)
	a.SUBimm(X11, X11, 1)
	a.CBNZ(X11, vl)

	a.Bind(vt)
	a.CBZ(X12, vd)
	if !idsMem {
		a.LDRs(V3, X2, moeOneIOff)
		a.DUPs4(V3, V3)
	}
	tl := a.Label()
	a.Bind(tl)
	a.LDRs(V10, X1, 0)
	a.MOVvec(V4, V14)     // NaN in every lane
	a.INSs(V4, 0, V10, 0) // and the element in lane 0
	if idsMem {
		a.LDRs(V2, X3, 0)
		a.ADDimm(X3, X3, 4)
	}
	body()
	if !idsMem {
		a.ADD4s(V2, V2, V3)
	}
	a.ADDimm(X1, X1, 4)
	a.SUBimm(X12, X12, 1)
	a.CBNZ(X12, tl)
	a.Bind(vd)

	// The fold. The per-lane maxima survive in v15; no lane of v0 can hold a
	// NaN -- FCMGT never selected one -- so FMAXV's NaN rule is unreachable.
	a.MOVvec(V15, V0)
	a.FMAXV(V0, V0)
	a.DUPs4(V0, V0)
	a.FCMEQ4s(V5, V15, V0)
	a.MOVvec(V6, V13)
	a.BIT16b(V6, V1, V5) // this lane's id where it ties, INT_MAX where not
	a.SMINV(V6, V6)
	a.DUPs4(V9, V6) // the winner's id
	a.CMEQ4s(V5, V1, V9)
	a.MOVIzero(V6)
	a.BIT16b(V6, V15, V5)
	a.ADDV(V6, V6) // its value's BITS: one lane survives the mask (sample.go)

	// Except when the segment is exhausted: then every lane carries INT_MAX
	// and matches, so ADDV sums four -Inf bit patterns into a value above
	// -Inf, letting an exhausted segment outrank a live one holding a -Inf
	// logit. (The x86 tiers fold with OR, which is exact here.)
	a.CMEQ4s(V5, V9, V13) // the winner id IS the empty sentinel
	a.LDRs(V10, X2, moeNegInfOff)
	a.DUPs4(V10, V10)
	a.BIT16b(V6, V10, V5)

	a.STRs(V6, X4, 0)
	a.STRs(V9, X5, 0)
	a.ADDimm(X4, X4, 4)
	a.ADDimm(X5, X5, 4)
	a.SUBimm(X13, X13, 1)
	a.CBNZ(X13, seg)

	a.RET()
	return a.Bytes()
}

// emitSampleDrawA64 is EmitSampleDraw (sample.go) on NEON.
func emitSampleDrawA64() []byte {
	var a A64
	a.LDRx(X2, X0, 56)  // Scr
	a.LDRx(X1, X0, 112) // Q32    -> the probabilities
	a.LDRx(X6, X0, 24)  // AScale -> the parameters
	a.LDRx(X7, X0, 96)  // ASum   -> the counters
	a.LDRx(X9, X0, 0)   // Out    -> the surviving mass
	a.LDRx(X10, X0, 40) // K

	a.MOVIzero(V15) // zero, for every compare against it
	a.LDRs(V7, X2, moeOneIOff)
	a.DUPs4(V7, V7)
	a.LDRs(V2, X6, 4*SampleParTopP)
	a.DUPs4(V2, V2)
	a.LDRs(V1, X6, 4*SampleParMinP)
	a.LDRs(V4, X1, 0)
	a.FMUL4s(V1, V1, V4) // the min-p cut is MinP * p[0]
	a.DUPs4(V1, V1)
	a.MOVIzero(V0)  // acc
	a.MOVIzero(V3)  // m
	a.MOVIzero(V16) // sum
	a.MOVIzero(V5)  // cut, the mask that latches
	a.MOVIzero(V6)  // i

	a.MOVreg(X11, X10)
	p1 := a.Label()
	a.Bind(p1)
	a.LDRs(V17, X1, 0)
	a.FCMGT4s(V8, V1, V17) // cut > p, i.e. p < cut
	a.BIC16b(V8, V8, V5)   // and not already cut
	a.BIT16b(V16, V0, V8)  // the mass is the cumulative BEFORE this element
	a.BIT16b(V3, V6, V8)   // and the survivor count is i
	a.ORR16b(V5, V5, V8)
	a.FADD4s(V0, V0, V17)
	a.FCMGE4s(V8, V0, V2) // acc >= TopP
	a.BIC16b(V8, V8, V5)
	a.BIT16b(V16, V0, V8) // here the mass INCLUDES this element
	a.ADD4s(V18, V6, V7)
	a.BIT16b(V3, V18, V8) // and the count is i+1
	a.ORR16b(V5, V5, V8)
	a.ADD4s(V6, V6, V7)
	a.ADDimm(X1, X1, 4)
	a.SUBimm(X11, X11, 1)
	a.CBNZ(X11, p1)

	// Neither cut fired: every candidate survives, and the divisor is the whole
	// distribution's mass when the caller knows it and the accumulator when it
	// does not.
	a.LDRs(V9, X6, 4*SampleParSum)
	a.DUPs4(V9, V9)
	a.FCMGE4s(V10, V9, V15) // sumAll >= 0
	a.MOVvec(V18, V0)
	a.BIT16b(V18, V9, V10)
	a.BIT16b(V18, V16, V5) // keep the cut's mass when one fired
	a.MOVvec(V16, V18)
	a.LDRs(V19, X0, 40) // K, the candidate count
	a.DUPs4(V19, V19)
	a.BIT16b(V19, V3, V5)
	a.MOVvec(V3, V19)

	a.STRs(V3, X7, 4*SampleOutM)
	a.STRs(V5, X7, 4*SampleOutCut) // a mask, read as "nonzero"
	a.STRs(V16, X9, 0)

	// The walk. r = u * mass, then subtract in order until it is spent; the
	// answer is m-1 if it never is, which is the Go loop's fallthrough.
	a.LDRx(X1, X0, 112)
	a.LDRs(V0, X6, 4*SampleParU)
	a.DUPs4(V0, V0)
	a.FMUL4s(V0, V0, V16)
	a.SUB4s(V12, V3, V7) // res = m-1
	a.MOVIzero(V5)       // resolved
	a.MOVIzero(V6)       // i

	// RowStr != 0 asks for the mass alone, and skips the walk (sample.go).
	after := a.Label()
	a.LDRx(X14, X0, 48)
	a.CBNZ(X14, after)

	a.MOVreg(X11, X10)
	p2 := a.Label()
	a.Bind(p2)
	a.LDRs(V17, X1, 0)
	a.CMGT4s(V8, V3, V6) // i < m
	a.BIC16b(V8, V8, V5) // and not already resolved
	a.FSUB4s(V9, V0, V17)
	a.BIT16b(V0, V9, V8)
	a.FCMGE4s(V10, V15, V0) // 0 >= r, i.e. r <= 0
	a.AND16b(V10, V10, V8)
	a.BIT16b(V12, V6, V10)
	a.ORR16b(V5, V5, V10)
	a.ADD4s(V6, V6, V7)
	a.ADDimm(X1, X1, 4)
	a.SUBimm(X11, X11, 1)
	a.CBNZ(X11, p2)
	a.Bind(after)

	a.STRs(V12, X7, 4*SampleOutRes)
	a.STRs(V5, X7, 4*SampleOutResolved)
	a.RET()
	return a.Bytes()
}

// emitSamplePenaltyA64 is EmitSamplePenalty (sample.go) on NEON.
func emitSamplePenaltyA64() []byte {
	var a A64
	a.LDRx(X1, X0, 112) // Q32    -> the logits
	a.LDRx(X3, X0, 8)   // W      -> the byte offsets
	a.LDRx(X6, X0, 24)  // AScale -> &penalty
	a.LDRx(X10, X0, 40) // K

	a.LDRs(V1, X6, 0)
	a.DUPs4(V1, V1)
	a.MOVIzero(V15)

	done := a.Label()
	a.CBZ(X10, done)
	lp := a.Label()
	a.Bind(lp)
	a.LDRx(X14, X3, 0)
	a.ADDreg(X14, X1, X14)
	a.LDRs(V4, X14, 0)
	a.FDIV4s(V5, V4, V1)
	a.FMUL4s(V6, V4, V1)
	a.FCMGT4s(V7, V4, V15) // v > 0
	a.BIT16b(V6, V5, V7)   // the divide where it holds, the multiply where not
	a.STRs(V6, X14, 0)
	a.ADDimm(X3, X3, 8)
	a.SUBimm(X10, X10, 1)
	a.CBNZ(X10, lp)
	a.Bind(done)

	a.RET()
	return a.Bytes()
}

// EmitSampleSegMax, EmitSampleDraw and EmitSamplePenalty are this
// architecture's entries in the primary Emitters table.
func EmitSampleSegMax(first, idsMem bool) ([]byte, error) {
	return must("sample_segmax", emitSampleSegMaxA64(first, idsMem))
}

func EmitSampleDraw() ([]byte, error) { return must("sample_draw", emitSampleDrawA64()) }

func EmitSamplePenalty() ([]byte, error) { return must("sample_penalty", emitSamplePenaltyA64()) }
