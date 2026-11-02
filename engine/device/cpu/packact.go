//go:build amd64

package cpu

// EmitPackAct generates the GEMM's activation packer for one amax window.
//
// The unit is a window, not a block, because the scale is: every block in an
// amax window shares one maximum, so the kernel scans once, derives inv once
// and quantizes the blocks it was given, needing no caller state.
//
// tok is baked: it is the batch width, and every output stride is a multiple of
// it. half is baked too, because whether a format needs 16-element sub-sums is a
// property of the quant type (NeedsHalfSums), constant from model.Open.
//
//	A        the window's first element, for the amax scan     (f32)
//	K        how many elements that scan covers
//	Q32      the first block's first element, to quantize      (f32)
//	Rows     how many 32-element blocks to quantize
//	W        &dst[(g0*tok + n)*4], the first group's dword
//	Out      &scale[b0*tok + n]
//	ASum     &sum[b0*tok + n]
//	AHalfSum &half[2*b0*tok + n], read only when half is baked true
//	Scr      [127.0, 1.0, 0.5, absmask, signmask, bias, -biasC/8, -biasC/4]
//
// Registers: RSI/RCX sources, RDX dst, R8 scale, R9 sum, R11 half, RBX consts,
// R10 counter. Y3 holds d and Y5 inv across the whole block loop. Never R14:
// Go's register ABI pins it to the current goroutine
// (TestGeneratedKernelsLeaveTheGoroutineAlone).
func EmitPackAct(tok int, half bool) []byte { return emitPackAct(tok, half, false) }

// EmitQuantAct is the same packer writing the matvec's pair layout instead of
// the GEMM's two arrays: `Out` is &pairs[2*b0] and each block stores
// {d, -sum*biasC/8} as two float32, which is what QuantizeQ8Window produces and
// what every packed kernel reads at +8 bytes a block. `AHalfSum` is float32
// too, -sum16*biasC/4 per half, matching QuantizeHalfSums.
//
// It shares the emitter so the payload, where the bit-identity lives, cannot
// drift between two copies; only the epilogue's stores differ. It scans each
// window once and writes that scale into all of its pairs.
//
// tok is 1: a matvec quantizes one activation, so every stride is its natural
// one and the dword store lands contiguous int8.
func EmitQuantAct(half bool) []byte { return emitPackAct(1, half, true) }

func emitPackAct(tok int, half, pairs bool) []byte {
	var a Buf
	a.MOVLoad(RBX, At(RDI, 56)) // Scr -> constants
	a.MOVLoad(RSI, At(RDI, 16)) // A   -> amax source

	// The pair form walks many windows per call (Cols of them): at the narrow
	// window a call per window would be one call per 32 elements, which costs
	// far more than it saves.
	//
	// The scan advances RSI by exactly K*4 bytes, so consecutive windows need
	// no cursor arithmetic, and the block cursors advance through the whole
	// range for the same reason.
	win := a.Label()
	if pairs {
		a.MOVLoad(R13, At(RDI, 40))  // K    -> elements per window
		a.MOVLoad(R12, At(RDI, 80))  // Cols -> windows in this call
		a.MOVLoad(R15, At(RDI, 32))  // Rows -> blocks per window
		a.MOVLoad(RCX, At(RDI, 112)) // Q32  -> quantize source
		a.MOVLoad(RDX, At(RDI, 8))   // W    -> dst
		a.MOVLoad(R8, At(RDI, 0))    // Out  -> pairs
		if half {
			a.MOVLoad(R11, At(RDI, 104)) // AHalfSum
		}
		a.Bind(win)
		a.MOVQ(R10, R13)
	} else {
		a.MOVLoad(R10, At(RDI, 40)) // K -> elements to scan
	}

	// ---- the window's amax, over |v| ----
	a.VPXOR(Y0, Y0, Y0)
	a.VBROADCASTSS(Y6, At(RBX, 12)) // 0x7FFFFFFF
	a.SHRimm(R10, 3)                // vectors of eight
	mx := a.Label()
	a.Bind(mx)
	a.VMOVDQULoad(Y1, At(RSI, 0))
	a.VPAND(Y1, Y1, Y6)
	a.VMAXPS(Y0, Y0, Y1)
	a.ADDimm(RSI, 32)
	a.DEC(R10)
	a.JNZ(mx)
	// VPERMQ swaps 128-bit halves and would repeat the fold VEXTRACTF128 just
	// did, leaving four of the eight lanes unread; VSHUFPS is the in-lane swap.
	a.VEXTRACTF128(Y1, Y0, 1)
	a.VMAXPS(Y0, Y0, Y1)
	a.VSHUFPS(Y1, Y0, Y0, 0x4E)
	a.VMAXPS(Y0, Y0, Y1)
	a.VSHUFPS(Y1, Y0, Y0, 0xB1)
	a.VMAXPS(Y0, Y0, Y1)
	a.VBROADCASTSSReg(Y0, Y0)

	// ---- d = amax/127 and inv = 1/d, with inv zeroed when d is zero ----
	//
	// Two divisions, not one multiply by 127/amax: d := amax/127 then 1/d is
	// what the reference computes, and a single 127/amax differs in the last
	// bit and would move a quantized value at the boundary.
	//
	// The zero case is real: MatMul zero-fills the padding rows of a ragged
	// tile, and 1/0 * 0 would be NaN.
	a.VBROADCASTSS(Y2, At(RBX, 0)) // 127.0
	a.VDIVPS(Y3, Y0, Y2)           // d
	a.VBROADCASTSS(Y4, At(RBX, 4)) // 1.0
	a.VDIVPS(Y5, Y4, Y3)           // 1/d
	a.VPXOR(Y7, Y7, Y7)
	a.VCMPPS(Y7, Y3, Y7, 4) // d != 0
	a.VPAND(Y5, Y5, Y7)     // inv

	a.VBROADCASTSS(Y12, At(RBX, 8))  // 0.5
	a.VBROADCASTSS(Y11, At(RBX, 16)) // 0x80000000

	// ---- the blocks ----
	if pairs {
		a.MOVQ(R10, R15) // blocks per window; the cursors are already live
	} else {
		a.MOVLoad(RCX, At(RDI, 112)) // Q32 -> quantize source
		a.MOVLoad(RDX, At(RDI, 8))   // W   -> dst
		a.MOVLoad(R8, At(RDI, 0))    // Out -> scale
		a.MOVLoad(R9, At(RDI, 96))   // ASum
		if half {
			a.MOVLoad(R11, At(RDI, 104)) // AHalfSum
		}
		a.MOVLoad(R10, At(RDI, 32)) // Rows -> block count
	}

	blk := a.Label()
	a.Bind(blk)
	emitQuantBlock(&a, tok, half, pairs)
	a.DEC(R10)
	a.JNZ(blk)
	if pairs {
		a.DEC(R12)
		a.JNZ(win)
	}

	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// emitQuantBlock is one 32-element block: the payload, the sums and the two or
// three stores, with d in Y3 and inv in Y5 and the cursors RCX/RDX/R8/R9/R11
// advanced on the way out. Three kernels emit it (the GEMM packer and both
// matvec forms), which differ only in how they reach a scale.
func emitQuantBlock(a *Buf, tok int, half, pairs bool) {
	a.VPXOR(Y8, Y8, Y8) // elements 0..15
	a.VPXOR(Y9, Y9, Y9) // elements 16..31
	for j := 0; j < 4; j++ {
		acc := Y8
		if j >= 2 {
			acc = Y9
		}
		a.VMOVDQULoad(Y1, At(RCX, int32(j)*32))
		a.VMULPS(Y1, Y1, Y5)
		// copysign(0.5, x), so truncation rounds half AWAY from zero.
		a.VPAND(Y2, Y1, Y11)
		a.VPOR(Y2, Y2, Y12)
		a.VADDPS(Y1, Y1, Y2)
		a.VCVTTPS2DQ(Y1, Y1)
		// The sum uses the unsaturated integers, which is exact: inv is
		// 1/(amax/127) and amax dominates its own window, so |v*inv| <= 127 and
		// the saturating pack below never fires.
		a.VPADDD(acc, acc, Y1)
		a.VPACKSSDW(Y2, Y1, Y1)
		a.VPACKSSWB(Y2, Y2, Y2)
		a.VMOVSSStore(At(RDX, int32(2*j)*int32(tok)*4), Y2)
		a.VEXTRACTI128(Y10, Y2, 1)
		a.VMOVSSStore(At(RDX, int32(2*j+1)*int32(tok)*4), Y10)
	}
	// lo16 in Y8, the whole block in Y8+Y9.
	a.VPADDD(Y10, Y8, Y9)
	hred := func(v Reg) {
		a.VEXTRACTI128(Y2, v, 1)
		a.VPADDD(v, v, Y2)
		a.VPHADDD(v, v, v)
		a.VPHADDD(v, v, v)
	}
	// The bias multiply stays in a vector lane rather than a GPR round trip.
	// bias*acc is at most 128 * 32 * 127 = 520k, nowhere near int32.
	if pairs {
		// The f32 multiply is exact: |sum| <= 32*127 = 4064 and every biasC/8
		// this tree produces is m/2^e with m 1 or 3, so sum*m < 2^24 and the
		// power-of-two scaling adds no rounding. It agrees bit for bit with the
		// reference's f64 product (TestQuantActConstantsAreExact).
		hred(Y10) // sum over the block
		a.VCVTDQ2PS(Y10, Y10)
		a.VBROADCASTSS(Y7, At(RBX, 24)) // -biasC/8
		a.VMULPS(Y10, Y10, Y7)
		a.VMOVSSStore(At(R8, 0), Y3)  // pairs[2b]   = d
		a.VMOVSSStore(At(R8, 4), Y10) // pairs[2b+1] = -sum*biasC/8
		if half {
			// Each half is reduced from its own accumulator (Y8, Y9): the
			// multiply is float here, so deriving the upper half as acc-lo after
			// it would round twice where the reference sums each sixteen alone.
			hred(Y8)                        // elements 0..15
			hred(Y9)                        // elements 16..31
			a.VBROADCASTSS(Y7, At(RBX, 28)) // -biasC/4
			a.VCVTDQ2PS(Y8, Y8)
			a.VMULPS(Y8, Y8, Y7)
			a.VMOVSSStore(At(R11, 0), Y8)
			a.VCVTDQ2PS(Y9, Y9)
			a.VMULPS(Y9, Y9, Y7)
			a.VMOVSSStore(At(R11, 4), Y9)
		}
	} else {
		a.VPBROADCASTD(Y7, At(RBX, 20)) // bias
		hred(Y10)                       // acc
		a.VPMULLD(Y10, Y10, Y7)
		a.VMOVSSStore(At(R9, 0), Y10)
		if half {
			hred(Y8) // elements 0..15
			a.VPMULLD(Y8, Y8, Y7)
			a.VMOVSSStore(At(R11, 0), Y8)
			// The upper half is acc - lo16, and acc is already multiplied.
			a.VPSUBD(Y10, Y10, Y8)
			a.VMOVSSStore(At(R11, 4*int32(tok)), Y10)
		}
		a.VMOVSSStore(At(R8, 0), Y3) // scale = d
	}

	a.ADDimm(RCX, 128)
	a.ADDimm(RDX, int32(tok)*4*8)
	if pairs {
		a.ADDimm(R8, 8)
	} else {
		a.ADDimm(R8, int32(tok)*4)
		a.ADDimm(R9, int32(tok)*4)
	}
	if half {
		a.ADDimm(R11, int32(tok)*4*2)
	}
}
