package cpu

// The SSE tier's activation quantizer, both window shapes: EmitQuantAct
// (packact.go) and EmitQuantActNarrow (quantact_narrow.go) in legacy SSE.
//
// The arithmetic is bit-identical to the Go loop (cpu.QuantizeQ8Window,
// cpu.QuantizeHalfSums) and to the AVX2 twin:
//
//	Two divisions: d = amax/127 then inv = 1/d, not one multiply by 127/amax
//	(which differs in the last bit). RCPPS is useless for the same reason.
//
//	copysign(0.5, x) then truncate, which is round-half-away-from-zero like
//	math.Round. CVTPS2DQ rounds half to even under the default MXCSR.
//
//	inv masked to zero when d is zero: padded rows are all zero, and 0*Inf
//	is NaN.
//
//	The sums are taken on the unsaturated int32 values, so width cannot move
//	them, and the float side is one multiply by a constant that
//	TestQuantActConstantsAreExact proves exact.
//
// Neither tier fuses anything, so there is no product this tier rounds that
// the AVX2 tier does not; the amax fold is a maximum and exact.
//
// DeclareISA(ISATierSSE) makes vex() and VZEROUPPER panic, so no VEX prefix
// can reach a Goldmont Atom. The ABI (Args offsets, QuantActConsts,
// QuantActNarrowScratch, outputs) is the AVX2 kernels', so QuantActKernels.Run
// (quantdrive.go) drives both tiers.
//
// One register differs: AHalfSum lives in R9 here, where packact.go and
// quantact_narrow.go use R14. Go's register ABI pins R14 to the current
// goroutine, callKernel does not save it, and ssegate_test.go refuses a
// write to it; the AVX2 quantizer's R14 is the divergence.

// emitQuantBlockSSE is one 32-element block of the matvec's pair layout: the
// payload, the two half-block sums and the two or four stores, with d in XMM3
// and inv in XMM5 and the cursors RCX/RDX/R8/R9 advanced on the way out. Both
// shapes emit it, so the arithmetic has one transcription.
//
// The payload store is one MOVDQU per half-block: four XMM of int32 narrow
// into one register in order (PACKSSDW then PACKSSWB). The saturating packs
// never saturate, since |v*inv| <= 127 by construction.
//
// Registers: XMM0/XMM1/XMM2/XMM4 the four quantized vectors of a half-block,
// XMM6 the round temporary, XMM7/XMM13 the pack temporaries, XMM8/XMM9 the two
// half-block int32 accumulators, XMM10 the block sum. XMM3 (d), XMM5 (inv),
// XMM11 (signmask), XMM12 (0.5) and XMM15 (absmask) are live across this and
// must not be touched.
func emitQuantBlockSSE(a *Buf, half bool) {
	v := [4]Reg{XMM0, XMM1, XMM2, XMM4}
	for h := 0; h < 2; h++ {
		// Each half has its own accumulator whether or not half is baked; the
		// block sum is their sum.
		acc := XMM8 + Reg(h)
		a.PXOR(acc, acc, acc)
		for i := 0; i < 4; i++ {
			a.MOVUPSLoad(v[i], At(RCX, int32(16*(4*h+i))))
			a.MULPS(v[i], v[i], XMM5)
			// copysign(0.5, x), so the truncation below rounds half AWAY from
			// zero: the sign bit of x, or-ed onto the bits of 0.5.
			a.PAND(XMM6, v[i], XMM11)
			a.POR(XMM6, XMM6, XMM12)
			a.ADDPS(v[i], v[i], XMM6)
			a.CVTTPS2DQ(v[i], v[i])
			// Summing the unsaturated integers is exact: the Go loop sums
			// after the int8 conversion, and the clamp cannot fire because
			// amax dominates its window, so |v*inv| <= 127.
			a.PADDD(acc, acc, v[i])
		}
		a.PACKSSDW(XMM7, v[0], v[1])  // words  0..7  of the half
		a.PACKSSDW(XMM13, v[2], v[3]) // words  8..15
		a.PACKSSWB(XMM7, XMM7, XMM13) // the sixteen int8, in order
		a.MOVDQUStore(At(RDX, int32(16*h)), XMM7)
	}

	// hsumD leaves the sum of x's four int32 lanes in every lane of x.
	hsumD := func(x Reg) {
		a.PHADDD(x, x, x)
		a.PHADDD(x, x, x)
	}
	a.PADDD(XMM10, XMM8, XMM9)
	hsumD(XMM10)
	// The scalar f32 multiply is exact: |sum| <= 32*127 and every biasC/8 is
	// m/2^e with m 1 or 3, so it matches the Go loop's f64 product narrowed
	// once. TestQuantActConstantsAreExact fails if a future format breaks
	// that. MULSS from m32 needs no broadcast and no alignment.
	a.CVTDQ2PS(XMM10, XMM10)
	a.MULSSMem(XMM10, XMM10, At(RBX, 24)) // -biasC/8
	a.MOVSSStore(At(R8, 0), XMM3)         // pairs[2b]   = d
	a.MOVSSStore(At(R8, 4), XMM10)        // pairs[2b+1] = -sum*biasC/8
	if half {
		// Each half is reduced from its own accumulator, not as block minus
		// lo, which would round twice.
		for i, dst := range [2]int32{0, 4} {
			x := XMM8 + Reg(i)
			hsumD(x)
			a.CVTDQ2PS(x, x)
			a.MULSSMem(x, x, At(RBX, 28)) // -biasC/4
			a.MOVSSStore(At(R9, dst), x)
		}
	}

	a.ADDimm(RCX, 128) // 32 float32 of source
	a.ADDimm(RDX, 32)  // 32 int8 of payload
	a.ADDimm(R8, 8)    // one {d, correction} pair
	if half {
		a.ADDimm(R9, 8) // two half sums
	}
}

// quantScalesSSE derives d = amax/127 and inv = 1/d from the amaxes in amax,
// lane by lane, with inv zeroed where d is zero. t0 and t1 are clobbered; RBX
// points at QuantActConsts. Both shapes use it so the two roundings stay the
// same.
func quantScalesSSE(a *Buf, amax, d, inv, t0, t1 Reg) {
	bcastSS(a, t0, At(RBX, 0)) // 127.0
	a.DIVPS(d, amax, t0)       // d
	bcastSS(a, t1, At(RBX, 4)) // 1.0
	a.DIVPS(inv, t1, d)        // 1/d
	// A ragged tile's padding rows are zero, so an all-zero window is real;
	// without the mask 0*Inf would make them NaN.
	a.PXOR(t0, t0, t0)
	a.CMPPS(t1, d, t0, 4) // d != 0; legacy CMPPS has predicates 0..7, and 4 is NEQ
	a.PAND(inv, inv, t1)
}

// emitQuantActSSEBody is EmitQuantAct (packact.go) on legacy SSE: one or more
// whole amax WINDOWS per call, each scanned once for its maximum and then
// quantized block by block into the matvec's pair layout.
//
//	A        the first window's first element, for the amax scan   (f32)
//	K        elements per window
//	Cols     how many windows this call covers
//	Q32      the first block's first element, to quantize          (f32)
//	Rows     blocks per window
//	W        &dst[b0*32]
//	Out      &pairs[2*b0]
//	AHalfSum &half[2*b0], written only when half is baked true
//	Scr      QuantActConsts()
//
// Many windows per call, since one call per 32-element window costs more in
// calls than the Go loop did. The scan advances RSI by exactly K*4 bytes and
// the block cursors run continuously across windows.
//
// Registers: RBX consts, RSI the scan cursor, RCX/RDX/R8/R9 the block cursors,
// R13 K, R12 the window count, R15 blocks per window, R10 the running counter.
// XMM15 absmask, XMM11 signmask and XMM12 0.5 are loop invariants; XMM3 holds d
// and XMM5 inv across a window's blocks.
func emitQuantActSSEBody(half bool) []byte {
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RBX, At(RDI, 56))  // Scr  -> constants
	a.MOVLoad(RSI, At(RDI, 16))  // A    -> amax source
	a.MOVLoad(R13, At(RDI, 40))  // K    -> elements per window
	a.MOVLoad(R12, At(RDI, 80))  // Cols -> windows in this call
	a.MOVLoad(R15, At(RDI, 32))  // Rows -> blocks per window
	a.MOVLoad(RCX, At(RDI, 112)) // Q32  -> quantize source
	a.MOVLoad(RDX, At(RDI, 8))   // W    -> dst
	a.MOVLoad(R8, At(RDI, 0))    // Out  -> pairs
	if half {
		a.MOVLoad(R9, At(RDI, 104)) // AHalfSum -- R9, never R14 (see the file comment)
	}
	// The three payload constants are loop invariant, and nothing between here
	// and the last block touches their registers.
	bcastD(&a, XMM15, At(RBX, 12)) // 0x7FFFFFFF
	bcastD(&a, XMM11, At(RBX, 16)) // 0x80000000
	bcastSS(&a, XMM12, At(RBX, 8)) // 0.5

	win := a.Label()
	a.Bind(win)
	a.MOVQ(R10, R13)

	// ---- the window's amax, over |v| ----
	//
	// Eight elements an iteration as two halves (XMM0 lanes 0..3, XMM1 lanes
	// 4..7), mirroring the AVX2 kernel's vector so the fold is hmax8SSE. |v| is
	// a mask, as the Go loop's Float32bits &^ (1<<31) is.
	a.PXOR(XMM0, XMM0, XMM0)
	a.PXOR(XMM1, XMM1, XMM1)
	a.SHRimm(R10, 3)
	mx := a.Label()
	a.Bind(mx)
	a.MOVDQULoad(XMM13, At(RSI, 0))
	a.PAND(XMM13, XMM13, XMM15)
	a.MAXPS(XMM0, XMM0, XMM13)
	a.MOVDQULoad(XMM14, At(RSI, 16))
	a.PAND(XMM14, XMM14, XMM15)
	a.MAXPS(XMM1, XMM1, XMM14)
	a.ADDimm(RSI, 32)
	a.DEC(R10)
	a.JNZ(mx)
	hmax8SSE(&a, XMM0, XMM1, XMM13)

	quantScalesSSE(&a, XMM0, XMM3, XMM5, XMM2, XMM4)

	// ---- the blocks; the cursors are already live ----
	a.MOVQ(R10, R15)
	blk := a.Label()
	a.Bind(blk)
	emitQuantBlockSSE(&a, half)
	a.DEC(R10)
	a.JNZ(blk)

	a.DEC(R12)
	a.JNZ(win)
	a.RET()
	return a.Bytes()
}

// emitQuantActNarrowSSEBody is EmitQuantActNarrow (quantact_narrow.go) on
// legacy SSE: the quantizer for a window of ONE block -- Q4_0, Q5_0, Q8_0 and
// MXFP4, whose accuracy depends on a scale every 32 elements.
//
//	Q32      the first block's first element        (f32)
//	Cols     how many GROUPS of eight blocks
//	W        &dst[b0*32]
//	Out      &pairs[2*b0]
//	AHalfSum &half[2*b0], written only when half is baked true
//	Scratch  QuantActNarrowScratch float32 of per-worker space
//	Scr      QuantActConsts()
//
// A second shape because a 32-element block cannot amortise a fold, a
// broadcast and two dependent divisions; batching eight blocks' divisions as
// two vectors of four keeps the same two roundings per lane. The group stays
// eight blocks because the driver's unit is QuantActNarrowBlocks: Run hands
// whole groups of eight and leaves the remainder to the wide kernel.
//
// Registers: RBX consts, RCX/RDX/R8/R9 the block cursors, R12 the scratch, R10
// the group count. Scratch holds the eight amaxes at [0,32), then d over them
// and inv at [32,64) -- the AVX2 kernel's spill layout exactly.
func emitQuantActNarrowSSEBody(half bool) []byte {
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RBX, At(RDI, 56))  // Scr     -> constants
	a.MOVLoad(RCX, At(RDI, 112)) // Q32     -> the elements
	a.MOVLoad(RDX, At(RDI, 8))   // W       -> dst
	a.MOVLoad(R8, At(RDI, 0))    // Out     -> pairs
	a.MOVLoad(R12, At(RDI, 72))  // Scratch -> the eight scales
	a.MOVLoad(R10, At(RDI, 80))  // Cols    -> groups of eight blocks
	if half {
		a.MOVLoad(R9, At(RDI, 104)) // AHalfSum -- R9, never R14 (see the file comment)
	}
	bcastD(&a, XMM15, At(RBX, 12)) // 0x7FFFFFFF, the absolute-value mask
	bcastD(&a, XMM11, At(RBX, 16)) // 0x80000000
	bcastSS(&a, XMM12, At(RBX, 8)) // 0.5

	grp := a.Label()
	a.Bind(grp)

	// ---- phase 1: eight blocks' amax, folded into scratch ----
	//
	// Each block is four eight-element groups, mirrored as two XMM halves
	// (lanes 0..3 into XMM0, 4..7 into XMM1), then hmax8SSE.
	for j := 0; j < QuantActNarrowBlocks; j++ {
		base := int32(j) * 128
		a.MOVDQULoad(XMM0, At(RCX, base))
		a.PAND(XMM0, XMM0, XMM15)
		a.MOVDQULoad(XMM1, At(RCX, base+16))
		a.PAND(XMM1, XMM1, XMM15)
		for w := 1; w < 4; w++ {
			a.MOVDQULoad(XMM13, At(RCX, base+int32(w)*32))
			a.PAND(XMM13, XMM13, XMM15)
			a.MAXPS(XMM0, XMM0, XMM13)
			a.MOVDQULoad(XMM14, At(RCX, base+int32(w)*32+16))
			a.PAND(XMM14, XMM14, XMM15)
			a.MAXPS(XMM1, XMM1, XMM14)
		}
		hmax8SSE(&a, XMM0, XMM1, XMM13)
		a.MOVSSStore(At(R12, int32(j)*4), XMM0)
	}

	// ---- two pairs of divisions for all eight ----
	for g := 0; g < QuantActNarrowBlocks/4; g++ {
		a.MOVUPSLoad(XMM0, At(R12, int32(16*g))) // four amaxes
		quantScalesSSE(&a, XMM0, XMM3, XMM5, XMM2, XMM4)
		a.MOVUPSStore(At(R12, int32(16*g)), XMM3)
		a.MOVUPSStore(At(R12, int32(32+16*g)), XMM5)
	}

	// ---- phase 2: the blocks, each with its own lane broadcast back ----
	for j := 0; j < QuantActNarrowBlocks; j++ {
		bcastSS(&a, XMM3, At(R12, int32(j)*4))    // d
		bcastSS(&a, XMM5, At(R12, 32+int32(j)*4)) // inv
		emitQuantBlockSSE(&a, half)
	}

	a.DEC(R10)
	a.JNZ(grp)
	a.RET()
	return a.Bytes()
}
