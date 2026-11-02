//go:build arm64

package cpu

// emitArgmaxA64 is the NEON twin of EmitArgmax: the index of the largest
// element of Q32[:K], with the lowest index winning a tie. argmax.go carries
// the contract and the constant block; both kernels read one layout.
//
//	x1  Q32  the elements, 8 per iteration (two vectors)
//	x2  ASum the index of the maximum   (int32, written)
//	x3  Out  the maximum value          (f32, written)
//	x5  Scr  ArgmaxConsts()
//	x10 K    how many elements
//
// Strict > per lane keeps the first of a tie within a lane; the SMINV fold
// keeps the lowest lane index across them. The value is selected under the
// same FCMGT mask as the index (one BIT instruction) rather than taken from
// FMAX, which propagates a NaN: FCMGT is false for a NaN on either side, which
// is exactly the scan's `v > best`.
func emitArgmaxA64() []byte {
	var a A64
	a.LDRx(X1, X0, 112) // Q32
	a.LDRx(X5, X0, 56)  // Scr
	a.LDRx(X10, X0, 40) // K, in elements
	// K is the element count and the kernel splits it, as on amd64.
	a.ANDlow(X11, X10, 3) // the tail: 0..7 elements
	a.LSRimm(X10, X10, 3) // whole vectors of eight

	a.LDRs(V0, X5, 40) // vmax = -Inf
	a.DUPs4(V0, V0)
	a.MOVvec(V1, V0)
	a.LDRq(V2, X5, 0)  // best index, lanes 0..3 = 0,1,2,3
	a.LDRq(V3, X5, 16) // best index, lanes 4..7 = 4,5,6,7
	a.MOVvec(V4, V2)   // the current index vector, low half
	a.MOVvec(V5, V3)   // and high
	a.LDRs(V6, X5, 32) // +8 per iteration
	a.DUPs4(V6, V6)

	// The tail's accumulators are separate and lane 0 only: LDRs zeroes the
	// other three lanes, and zero is not neutral for a maximum. Only lane 0 of
	// v18 and v19 is read.
	a.LDRs(V18, X5, 40) // the tail's maximum = -Inf
	a.DUPs4(V18, V18)
	a.MOVIzero(V19)    // the tail's best index
	a.LDRs(V20, X5, 4) // +1 per tail element (the integer 1)
	a.DUPs4(V20, V20)

	// The tail's index cursor is the vector loop's own, v4, whose lane 0 holds
	// 8k after k iterations -- the index of the first tail element.
	skipVec := a.Label()
	a.CBZ(X10, skipVec)

	lp := a.Label()
	a.Bind(lp)
	a.LDRq(V7, X1, 0)
	a.LDRq(V16, X1, 16)
	a.FCMGT4s(V9, V7, V0)   // lanes where this element beats the running max
	a.FCMGT4s(V10, V16, V1) // strictly, so a tie leaves the earlier index in place
	a.BIT16b(V0, V7, V9)    // vmax = v where it improved
	a.BIT16b(V2, V4, V9)    // and the index with it
	a.BIT16b(V1, V16, V10)
	a.BIT16b(V3, V5, V10)
	a.ADD4s(V4, V4, V6)
	a.ADD4s(V5, V5, V6)
	a.ADDimm(X1, X1, 32)
	a.SUBimm(X10, X10, 1)
	a.CBNZ(X10, lp)
	a.Bind(skipVec)

	// The ragged tail, one element at a time into lane 0.
	skipTail := a.Label()
	a.CBZ(X11, skipTail)
	lpT := a.Label()
	a.Bind(lpT)
	a.LDRs(V21, X1, 0)       // one element, lanes 1..3 zeroed
	a.FCMGT4s(V22, V21, V18) // strictly, so a tie keeps the earlier index
	a.BIT16b(V18, V21, V22)
	a.BIT16b(V19, V4, V22)
	a.ADD4s(V4, V4, V20)
	a.ADDimm(X1, X1, 4)
	a.SUBimm(X11, X11, 1)
	a.CBNZ(X11, lpT)
	a.Bind(skipTail)

	// The per-lane maxima have to survive the fold, because the winner's index
	// is found by asking which lanes equal the overall maximum. v0 and v1 are
	// kept; v11 carries the reduction. No lane can hold a NaN (FCMGT never
	// selected one), so FMAX's NaN rule cannot be reached here.
	a.FMAX4s(V11, V0, V1)
	a.FMAXV(V11, V11)
	a.DUPs4(V11, V11) // the maximum, in every lane

	// Lanes that hold it keep their index; the rest contribute INT_MAX, so the
	// signed minimum over the lanes is the lowest index that ties. It must be
	// a signed integer min: 0x7FFFFFFF read as a float is a NaN.
	a.FCMEQ4s(V12, V0, V11)
	a.FCMEQ4s(V13, V1, V11)
	a.LDRs(V14, X5, 36) // 0x7FFFFFFF
	a.DUPs4(V14, V14)
	a.MOVvec(V15, V14)
	a.BIT16b(V15, V2, V12) // index where this lane ties, INT_MAX where it does not
	a.MOVvec(V17, V14)
	a.BIT16b(V17, V3, V13)
	a.SMIN4s(V15, V15, V17)
	a.SMINV(V15, V15)

	// The tail is merged under the same tie rule: v11/v15 hold the vector
	// part's maximum and lowest tying index, v18/v19 the tail's. A strictly
	// greater tail wins, an equal one contributes only its index to the
	// minimum. Only lane 0 is stored.
	a.FCMGT4s(V23, V18, V11) // the tail is strictly greater
	a.FCMEQ4s(V24, V18, V11) // or exactly equal
	a.SMIN4s(V25, V15, V19)
	a.BIT16b(V15, V25, V24) // on a tie, the lower index
	a.BIT16b(V15, V19, V23) // outright, the tail's index
	a.BIT16b(V11, V18, V23) // and the tail's value

	a.LDRx(X2, X0, 96) // ASum -> the index
	a.STRs(V15, X2, 0)
	a.LDRx(X3, X0, 0) // Out -> the value
	a.STRs(V11, X3, 0)
	a.RET()
	return a.Bytes()
}
