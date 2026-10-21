//go:build arm64

package cpu

// The NEON activation quantizer: the matvec's pair layout, in both window
// shapes. packact.go carries the contract, the argument block and the reasoning
// for every constant; packact_a64.go carries the GEMM's twin of the payload.
// This joins the two: the NEON payload writing the pair layout.
//
// The amax window, the two divisions, the copysign round, the saturating
// narrow and the two sums are where bit-identity with QuantizeQ8Window lives,
// so both shapes reach one payload (emitQuantBlockA64), as emitQuantBlock
// does on amd64.

// emitQuantActA64 is the wide-window form: one or more whole amax windows per
// call, the NEON twin of EmitQuantAct.
//
// The scan cursor is the window advance: after K/4 iterations x10 holds
// A + K*4, the next window's first element, so nothing is computed between
// windows (ADDimm tops out at 4095 bytes, below a 256-element window).
//
//	x1  Out       -> pairs, two float32 a block
//	x2  W         -> dst, 32 int8 a block
//	x3  A         -> the amax cursor, one window ahead per iteration
//	x4  the block counter, reloaded from x15 each window
//	x5  Scr       -> QuantActConsts
//	x6  K         -> elements per window
//	x8  AHalfSum  -> half, two float32 a block, only when half is baked
//	x9  Q32       -> the quantize cursor
//	x10 the scan cursor, x11 its counter
//	x12 Cols      -> windows left in this call
//	x15 Rows      -> blocks per window
//
// v2/v4/v11/v12/v18/v19 hold the six loop-invariant constants; v3 is d and v5
// is inv, live across the whole block loop.
func emitQuantActA64(half bool) []byte {
	var a A64
	a.LDRx(X5, X0, 56)  // Scr
	a.LDRx(X3, X0, 16)  // A
	a.LDRx(X6, X0, 40)  // K
	a.LDRx(X12, X0, 80) // Cols
	a.LDRx(X15, X0, 32) // Rows
	a.LDRx(X9, X0, 112) // Q32
	a.LDRx(X2, X0, 8)   // W
	a.LDRx(X1, X0, 0)   // Out
	if half {
		a.LDRx(X8, X0, 104) // AHalfSum
	}
	quantActConstsA64(&a, half)

	win := a.Label()
	a.Bind(win)

	// ---- the window's amax, over |v| ----
	//
	// FABS where amd64 needs an AND against a mask. The Go loop clears the
	// sign bit and compares as an integer, which orders magnitudes the same
	// for every finite value, and activations are never NaN.
	a.MOVIzero(V0)
	a.MOVreg(X10, X3)
	a.LSRimm(X11, X6, 2) // vectors of four
	mx := a.Label()
	a.Bind(mx)
	a.LDRq(V1, X10, 0)
	a.FABS4s(V1, V1)
	a.FMAX4s(V0, V0, V1)
	a.ADDimm(X10, X10, 16)
	a.SUBimm(X11, X11, 1)
	a.CBNZ(X11, mx)
	a.MOVreg(X3, X10) // the next window starts where this scan ended
	a.FMAXV(V0, V0)   // one instruction; amd64 needs a three-fold ladder
	a.DUPs4(V0, V0)

	// ---- d = amax/127 and inv = 1/d, with inv zeroed when d is zero ----
	//
	// Two divisions, as the Go loop does, not one multiply by 127/amax, which
	// differs in the last bit. The zero case is the padding rows of a ragged
	// tile (0*Inf is NaN), so it is masked rather than branched on.
	//
	// On arm64 the mask is parity rather than load-bearing: FCVTZS converts a
	// NaN to zero, so the NaN would quantize correctly by accident. It stays
	// because `inv = 0` is what the Go loop computes and correctness should not
	// rest on a conversion's NaN rule.
	a.FDIV4s(V3, V0, V2) // d
	a.FDIV4s(V5, V4, V3) // 1/d
	a.FCMEQzero(V6, V3)  // all ones where d == 0
	a.BIC16b(V5, V5, V6) // inv

	a.MOVreg(X4, X15)
	blk := a.Label()
	a.Bind(blk)
	emitQuantBlockA64(&a, half)
	a.SUBimm(X4, X4, 1)
	a.CBNZ(X4, blk)

	a.SUBimm(X12, X12, 1)
	a.CBNZ(X12, win)
	a.RET()
	return a.Bytes()
}

// emitQuantActNarrowA64 is the window-of-one-block form, the NEON twin of
// EmitQuantActNarrow.
//
// Batching the divisions is the lever: d and inv are two dependent divides
// that dominate a 32-element block, so the kernel folds eight blocks'
// maxima, divides once per vector (two vectors of four on NEON), spills d
// and inv, and reads lane j back from scratch for block j (as amd64's
// VBROADCASTSS-from-memory does). The group stays at QuantActNarrowBlocks so
// a caller sizes scratch the same on both architectures.
//
//	x1  Out      -> pairs        x2  W -> dst        x5  Scr -> QuantActConsts
//	x8  AHalfSum                 x9  Q32             x10 Cols -> groups left
//	x12 Scratch  -> eight amaxes, then the eight d and eight inv
func emitQuantActNarrowA64(half bool) []byte {
	var a A64
	a.LDRx(X5, X0, 56)  // Scr
	a.LDRx(X9, X0, 112) // Q32
	a.LDRx(X2, X0, 8)   // W
	a.LDRx(X1, X0, 0)   // Out
	a.LDRx(X12, X0, 72) // Scratch
	a.LDRx(X10, X0, 80) // Cols
	if half {
		a.LDRx(X8, X0, 104) // AHalfSum
	}
	quantActConstsA64(&a, half)

	grp := a.Label()
	a.Bind(grp)

	// ---- phase 1: eight blocks' amax, folded into scratch ----
	//
	// The block cursor is not advanced here: phase 2's payload walks it, so
	// every offset below is relative to the group's first block.
	for j := 0; j < QuantActNarrowBlocks; j++ {
		base := int32(j) * 128
		a.LDRq(V0, X9, base)
		a.FABS4s(V0, V0)
		for w := 1; w < 8; w++ {
			a.LDRq(V14, X9, base+int32(w)*16)
			a.FABS4s(V14, V14)
			a.FMAX4s(V0, V0, V14)
		}
		a.FMAXV(V0, V0)
		a.STRs(V0, X12, int32(j)*4)
	}

	// ---- two pairs of divisions for all eight ----
	a.LDRq(V0, X12, 0)     // amax 0..3
	a.LDRq(V20, X12, 16)   // amax 4..7
	a.FDIV4s(V3, V0, V2)   // d   0..3
	a.FDIV4s(V21, V20, V2) // d   4..7
	a.FDIV4s(V5, V4, V3)   // inv 0..3
	a.FDIV4s(V6, V4, V21)  // inv 4..7
	a.FCMEQzero(V7, V3)
	a.BIC16b(V5, V5, V7)
	a.FCMEQzero(V7, V21)
	a.BIC16b(V6, V6, V7)
	a.STRq(V3, X12, 0)
	a.STRq(V21, X12, 16)
	a.STRq(V5, X12, 32)
	a.STRq(V6, X12, 48)

	// ---- phase 2: the blocks, each with its own lane broadcast back ----
	for j := 0; j < QuantActNarrowBlocks; j++ {
		a.LDRs(V3, X12, int32(j)*4) // d
		a.DUPs4(V3, V3)
		a.LDRs(V5, X12, 32+int32(j)*4) // inv
		a.DUPs4(V5, V5)
		emitQuantBlockA64(&a, half)
	}

	a.SUBimm(X10, X10, 1)
	a.CBNZ(X10, grp)
	a.RET()
	return a.Bytes()
}

// quantActConstsA64 loads the six loop invariants both kernels multiply by.
// QuantActConsts appends its two float multipliers to PackActConsts and never
// moves a slot, so these byte offsets are the same block EmitPackAct reads.
func quantActConstsA64(a *A64, half bool) {
	a.LDRs(V2, X5, 0) // 127
	a.DUPs4(V2, V2)
	a.LDRs(V4, X5, 4) // 1.0
	a.DUPs4(V4, V4)
	a.LDRs(V12, X5, 8) // 0.5
	a.DUPs4(V12, V12)
	a.LDRs(V11, X5, 16) // 0x80000000
	a.DUPs4(V11, V11)
	a.LDRs(V18, X5, 24) // -biasC/8
	a.DUPs4(V18, V18)
	if half {
		a.LDRs(V19, X5, 28) // -biasC/4
		a.DUPs4(V19, V19)
	}
}

// emitQuantBlockA64 is one 32-element block in the pair layout: the payload,
// the sums and the two or three stores, with d in v3 and inv in v5 and the
// cursors x9/x2/x1/x8 advanced on the way out. It is the arm64 emitQuantBlock
// and the two must stay the same arithmetic in the same order.
func emitQuantBlockA64(a *A64, half bool) {
	a.MOVIzero(V8) // elements 0..15
	a.MOVIzero(V9) // elements 16..31
	for j := 0; j < 8; j++ {
		acc := V8
		if j >= 4 {
			acc = V9
		}
		a.LDRq(V1, X9, int32(j)*16)
		a.FMUL4s(V1, V1, V5)
		// copysign(0.5, x), so truncation rounds half away from zero, which is
		// math.Round's rule and therefore the Go loop's.
		a.AND16b(V13, V1, V11)
		a.ORR16b(V13, V13, V12)
		a.FADD4s(V1, V1, V13)
		a.FCVTZS4s(V1, V1)
		// Summing the unsaturated integers is exact: amax dominates its
		// window, so |v*inv| <= 127 and the clamp never fires. The saturating
		// narrow below is the clamp and a no-op.
		a.ADD4s(acc, acc, V1)
		a.SQXTN4h(V14, V1)
		a.SQXTN8b(V14, V14)
		a.STRs(V14, X2, int32(j)*4)
	}

	// The f32 multiply is exact (packact.go's argument, checked by
	// TestQuantActConstantsAreExact): |sum| <= 32*127 converts exactly and
	// the constant is m/2^e, so this matches the Go loop's f64 product
	// narrowed once, bit for bit.
	a.ADD4s(V15, V8, V9)
	a.ADDV(V15, V15) // the whole block, one instruction
	a.SCVTF4s(V15, V15)
	a.FMUL4s(V15, V15, V18)
	a.STRs(V3, X1, 0)  // pairs[2b]   = d
	a.STRs(V15, X1, 4) // pairs[2b+1] = -sum*biasC/8
	if half {
		// Each half is reduced from its own accumulator: the multiply is
		// float here, so deriving one half by subtraction would round twice.
		a.ADDV(V16, V8)
		a.SCVTF4s(V16, V16)
		a.FMUL4s(V16, V16, V19)
		a.STRs(V16, X8, 0)
		a.ADDV(V17, V9)
		a.SCVTF4s(V17, V17)
		a.FMUL4s(V17, V17, V19)
		a.STRs(V17, X8, 4)
	}

	a.ADDimm(X9, X9, 128) // 32 float32
	a.ADDimm(X2, X2, 32)  // 32 int8
	a.ADDimm(X1, X1, 8)   // one pair
	if half {
		a.ADDimm(X8, X8, 8) // two halves
	}
}
