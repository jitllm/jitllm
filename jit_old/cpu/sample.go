//go:build amd64

package cpu

// The sampler's three kernels, AVX2 tier. sse_sample.go and sample_a64.go are
// the other two; sample_const.go holds the segment arithmetic and the draw
// kernel's parameter block, and MoETopKConsts() (topk_const.go) is the constant
// block all three read.

// EmitSampleSegMax generates the ordering's one primitive: the largest ELIGIBLE
// element of each of Rows consecutive segments, with the lowest id winning a
// tie.
//
//	Q32      the values                                    (f32)
//	ASum     the ids                                       (int32, idsMem only)
//	Out      the per-segment best value                    (f32, Rows, written)
//	AHalfSum the per-segment best id                       (int32, Rows, written)
//	K        the segment length in ELEMENTS
//	Rows     how many segments
//	Cols     the id of the first element                   (!idsMem only)
//	AScale   {prevVal, bitcast(prevIdx)}                   (!first only)
//	Scr      MoETopKConsts()
//
// One kernel serves the initial pass over the vocabulary (first, contiguous
// ids), the rescan of a winner's segment and the sweep of the summary (ids
// from memory). Eligibility is the MoE router's (topk.go):
// eligible(e) = v[e] < pv || (v[e] == pv && id[e] > pi), and `first` admits
// everything but a NaN. An exhausted segment reports (-Inf, INT_MAX); the
// ragged tail runs the vector body over NaN pads.
//
// The three uses and the predicate in full: docs/engineering-history/
// cpu-kernels.md, "jit_old/cpu/sample.go: EmitSampleSegMax".
func EmitSampleSegMax(first, idsMem bool) ([]byte, error) {
	return must("sample_segmax", emitSampleSegMax(first, idsMem))
}

func emitSampleSegMax(first, idsMem bool) []byte {
	var a Buf
	a.MOVLoad(RBX, At(RDI, 56))  // Scr      -> the constants
	a.MOVLoad(RSI, At(RDI, 112)) // Q32      -> the values
	a.MOVLoad(R8, At(RDI, 0))    // Out      -> the per-segment values
	a.MOVLoad(R9, At(RDI, 104))  // AHalfSum -> the per-segment ids
	a.MOVLoad(R12, At(RDI, 40))  // K        -> the segment length
	a.MOVLoad(R15, At(RDI, 32))  // Rows     -> the segment count
	a.MOVQ(R13, R12)
	a.ANDimm8(R13, 7) // the tail: 0..7 elements
	a.SHRimm(R12, 3)  // whole vectors of eight

	a.VPBROADCASTD(Y13, At(RBX, moeMaxIOff)) // INT_MAX, for the whole kernel
	a.VBROADCASTSS(Y14, At(RBX, moeNaNOff))  // the tail's filler

	if idsMem {
		a.MOVLoad(RDX, At(RDI, 96)) // ASum -> the ids
	} else {
		// The index vector starts at Cols and then flows: after a segment of K
		// elements it holds the next segment's base in lane 0.
		a.VPBROADCASTD(Y4, At(RDI, 80))
		a.VMOVDQULoad(Y2, At(RBX, moeIdxOff))
		a.VPADDD(Y2, Y2, Y4)
	}
	if !first {
		a.MOVLoad(RAX, At(RDI, 24)) // AScale -> {prevVal, prevIdx}
		a.VBROADCASTSS(Y11, At(RAX, 0))
		a.VPBROADCASTD(Y12, At(RAX, 4))
	}

	body := func() {
		if first {
			a.VCMPPS(Y5, Y4, Y4, 0) // ordered equality: every element but a NaN
		} else {
			a.VCMPPS(Y5, Y4, Y11, 1) // _CMP_LT_OS: v < prevVal
			a.VCMPPS(Y6, Y4, Y11, 0) // _CMP_EQ_OQ: v == prevVal
			a.VPCMPGTD(Y7, Y2, Y12)  // and a higher id than the tie it lost
			a.VPAND(Y6, Y6, Y7)
			a.VPOR(Y5, Y5, Y6)
		}
		a.VCMPPS(Y8, Y4, Y0, 14) // _CMP_GT_OS: strictly, so a tie keeps the
		a.VPCMPEQD(Y9, Y1, Y13)  // earlier id -- unless the lane is still empty
		a.VPOR(Y8, Y8, Y9)
		a.VPAND(Y5, Y5, Y8)
		a.VPAND(Y6, Y4, Y5) // there is no blend in this assembler, so the
		a.VPANDN(Y7, Y5, Y0)
		a.VPOR(Y0, Y6, Y7) // mask's and/andnot/or is the blend
		a.VPAND(Y6, Y2, Y5)
		a.VPANDN(Y7, Y5, Y1)
		a.VPOR(Y1, Y6, Y7)
	}

	seg := a.Label()
	a.Bind(seg)
	a.VBROADCASTSS(Y0, At(RBX, moeNegInfOff)) // the segment's running maximum
	a.VMOVAPSReg(Y1, Y13)                     // its id, INT_MAX = still empty
	a.MOVQ(R10, R12)
	a.MOVQ(R11, R13)

	vt, vd := a.Label(), a.Label()
	a.TESTQ(R10, R10)
	a.JZ(vt)
	if !idsMem {
		a.VPBROADCASTD(Y3, At(RBX, moeStep8Off))
	}
	vl := a.Label()
	a.Bind(vl)
	a.VMOVDQULoad(Y4, At(RSI, 0))
	if idsMem {
		a.VMOVDQULoad(Y2, At(RDX, 0))
		a.ADDimm(RDX, 32)
	}
	body()
	if !idsMem {
		a.VPADDD(Y2, Y2, Y3)
	}
	a.ADDimm(RSI, 32)
	a.DEC(R10)
	a.JNZ(vl)

	a.Bind(vt)
	a.TESTQ(R11, R11)
	a.JZ(vd)
	if !idsMem {
		a.VPBROADCASTD(Y3, At(RBX, moeOneIOff))
	}
	tl := a.Label()
	a.Bind(tl)
	a.VMOVSSLoad(Y4, At(RSI, 0))
	// Lane 0 is the element; the seven a scalar load zeroed become NaN, which
	// no branch of the predicate admits. The id lanes beside them are stale (or
	// zero) and nothing reads them, because their values cannot be selected.
	a.VPBLENDD(Y4, Y14, Y4, 0x01)
	if idsMem {
		a.VMOVSSLoad(Y2, At(RDX, 0))
		a.ADDimm(RDX, 4)
	}
	body()
	if !idsMem {
		a.VPADDD(Y2, Y2, Y3)
	}
	a.ADDimm(RSI, 4)
	a.DEC(R11)
	a.JNZ(tl)
	a.Bind(vd)

	// The fold. The per-lane maxima must survive it, because the winner's id is
	// found by asking which lanes equal the overall maximum.
	a.VMOVAPSReg(Y15, Y0)
	a.VEXTRACTF128(Y4, Y0, 1)
	a.VMAXPS(Y0, Y0, Y4)
	a.VSHUFPS(Y4, Y0, Y0, 0x4E)
	a.VMAXPS(Y0, Y0, Y4)
	a.VSHUFPS(Y4, Y0, Y0, 0xB1)
	a.VMAXPS(Y0, Y0, Y4)
	a.VBROADCASTSSReg(Y0, Y0)
	a.VCMPPS(Y5, Y15, Y0, 0)
	a.VPAND(Y6, Y1, Y5)
	a.VPANDN(Y7, Y5, Y13)
	a.VPOR(Y6, Y6, Y7)
	a.VEXTRACTI128(Y4, Y6, 1)
	a.VPMINSD(Y6, Y6, Y4)
	a.VSHUFPS(Y4, Y6, Y6, 0x4E)
	a.VPMINSD(Y6, Y6, Y4)
	a.VSHUFPS(Y4, Y6, Y6, 0xB1)
	a.VPMINSD(Y6, Y6, Y4)
	a.VBROADCASTSSReg(Y9, Y6) // the winner's id

	// Its value is selected by that id, not taken from the max fold, because
	// of -0.0: VMAXPS returns its second operand when both are zeros, and the
	// three tiers' outputs are compared bit for bit. Per-lane best ids are
	// unique, so exactly one lane survives the mask.
	a.VPCMPEQD(Y5, Y1, Y9)
	a.VPAND(Y6, Y15, Y5)
	a.VEXTRACTI128(Y4, Y6, 1)
	a.VPOR(Y6, Y6, Y4)
	a.VSHUFPS(Y4, Y6, Y6, 0x4E)
	a.VPOR(Y6, Y6, Y4)
	a.VSHUFPS(Y4, Y6, Y6, 0xB1)
	a.VPOR(Y6, Y6, Y4)

	a.VMOVSSStore(At(R8, 0), Y6)
	a.VMOVSSStore(At(R9, 0), Y9)
	a.ADDimm(R8, 4)
	a.ADDimm(R9, 4)
	a.DEC(R15)
	a.JNZ(seg)

	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// EmitSampleDraw generates everything the sampler does AFTER the ordering: the
// min-p cut, the top-p cumulative and its cut, the surviving mass, and the
// inverse-CDF walk.
//
//	Q32    the candidates' probabilities, DESCENDING (f32, K)
//	K      how many candidates
//	AScale the four parameters   (f32, sample_const.go)
//	ASum   the four counters     (int32, written)
//	Out    the surviving mass    (f32, 1, written)
//	Scr    MoETopKConsts()
//
// It is branchless and runs every element twice: a mask and a select is what
// all three tiers' assemblers have, and K is the survivor count (about forty
// on a normal top-k), not the vocabulary.
//
// The cuts follow llama.cpp's order: min-p is tested before the element joins
// the cumulative and top-p after, so min-p stops at the first `p < MinP*p[0]`
// having accumulated nothing of it and top-p cuts at i+1 on the first
// `acc >= TopP`. The surviving mass is the cumulative at the cut, the same
// float32 additions in the same order as summing p[:m].
//
// sumAll is the one thing a prefix cannot compute: with no top-k the
// candidates are a prefix and the divisor is the whole mass, which the caller
// measures with this kernel over the whole row. A negative value means the
// array is the whole distribution.
func EmitSampleDraw() ([]byte, error) { return must("sample_draw", emitSampleDraw()) }

func emitSampleDraw() []byte {
	var a Buf
	a.MOVLoad(RBX, At(RDI, 56))  // Scr
	a.MOVLoad(RSI, At(RDI, 112)) // Q32    -> the probabilities
	a.MOVLoad(RDX, At(RDI, 24))  // AScale -> the parameters
	a.MOVLoad(R8, At(RDI, 96))   // ASum   -> the counters
	a.MOVLoad(R9, At(RDI, 0))    // Out    -> the surviving mass
	a.MOVLoad(R12, At(RDI, 40))  // K      -> how many candidates

	a.VPXOR(Y15, Y15, Y15)                  // zero, for every compare against it
	a.VPBROADCASTD(Y7, At(RBX, moeOneIOff)) // the integer one
	a.VMOVSSLoad(Y2, At(RDX, 4*SampleParTopP))
	a.VMOVSSLoad(Y1, At(RDX, 4*SampleParMinP))
	a.VMOVSSLoad(Y4, At(RSI, 0))
	a.VMULPS(Y1, Y1, Y4) // the min-p cut is MinP * p[0]
	a.VPXOR(Y0, Y0, Y0)  // acc
	a.VPXOR(Y3, Y3, Y3)  // m
	a.VPXOR(Y10, Y10, Y10)
	a.VMOVAPSReg(Y4, Y10) // sum
	a.VPXOR(Y5, Y5, Y5)   // cut, the mask that latches
	a.VPXOR(Y6, Y6, Y6)   // i

	sel := func(dst, mask, x, y, t Reg) {
		a.VPANDN(t, mask, y)
		a.VPAND(dst, x, mask)
		a.VPOR(dst, dst, t)
	}

	a.MOVQ(R10, R12)
	p1 := a.Label()
	a.Bind(p1)
	a.VMOVSSLoad(Y13, At(RSI, 0))
	a.VCMPPS(Y8, Y13, Y1, 1) // _CMP_LT_OS: p < cut
	a.VPANDN(Y8, Y5, Y8)     // and not already cut
	sel(Y4, Y8, Y0, Y4, Y9)  // the mass is the cumulative BEFORE this element
	sel(Y3, Y8, Y6, Y3, Y9)  // and the survivor count is i
	a.VPOR(Y5, Y5, Y8)
	a.VADDPS(Y0, Y0, Y13)
	a.VCMPPS(Y8, Y0, Y2, 13) // _CMP_GE_OS: acc >= TopP
	a.VPANDN(Y8, Y5, Y8)
	sel(Y4, Y8, Y0, Y4, Y9) // here the mass INCLUDES this element
	a.VPADDD(Y11, Y6, Y7)
	sel(Y3, Y8, Y11, Y3, Y9) // and the count is i+1
	a.VPOR(Y5, Y5, Y8)
	a.VPADDD(Y6, Y6, Y7)
	a.ADDimm(RSI, 4)
	a.DEC(R10)
	a.JNZ(p1)

	// Neither cut fired: every candidate survives, and the divisor is the whole
	// distribution's mass when the caller knows it and the accumulator when it
	// does not.
	a.VMOVSSLoad(Y9, At(RDX, 4*SampleParSum))
	a.VCMPPS(Y10, Y9, Y15, 13) // _CMP_GE_OS: sumAll >= 0
	sel(Y11, Y10, Y9, Y0, Y12)
	sel(Y4, Y5, Y4, Y11, Y12)
	a.VPBROADCASTD(Y12, At(RDI, 40)) // K, the candidate count
	sel(Y3, Y5, Y3, Y12, Y11)

	a.VMOVSSStore(At(R8, 4*SampleOutM), Y3)
	a.VMOVSSStore(At(R8, 4*SampleOutCut), Y5) // a mask, read as "nonzero"
	a.VMOVSSStore(At(R9, 0), Y4)

	// The walk. r = u * mass, then subtract in order until it is spent; the
	// answer is m-1 if it never is, which is the Go loop's fallthrough.
	a.MOVLoad(RSI, At(RDI, 112))
	a.VMOVSSLoad(Y0, At(RDX, 4*SampleParU))
	a.VMULPS(Y0, Y0, Y4)
	a.VPSUBD(Y12, Y3, Y7) // res = m-1
	a.VPXOR(Y5, Y5, Y5)   // resolved
	a.VPXOR(Y6, Y6, Y6)   // i

	// RowStr != 0 asks for the mass alone (sample_const.go): the caller
	// measuring a whole distribution's total has no use for a walk over the
	// vocabulary.
	after := a.Label()
	a.MOVLoad(RAX, At(RDI, 48))
	a.TESTQ(RAX, RAX)
	a.JNZ(after)

	a.MOVQ(R10, R12)
	p2 := a.Label()
	a.Bind(p2)
	a.VMOVSSLoad(Y13, At(RSI, 0))
	a.VPCMPGTD(Y8, Y3, Y6) // i < m
	a.VPANDN(Y8, Y5, Y8)   // and not already resolved
	a.VSUBPS(Y9, Y0, Y13)
	sel(Y0, Y8, Y9, Y0, Y10)
	a.VCMPPS(Y10, Y0, Y15, 2) // _CMP_LE_OS: r <= 0
	a.VPAND(Y10, Y10, Y8)
	sel(Y12, Y10, Y6, Y12, Y11)
	a.VPOR(Y5, Y5, Y10)
	a.VPADDD(Y6, Y6, Y7)
	a.ADDimm(RSI, 4)
	a.DEC(R10)
	a.JNZ(p2)
	a.Bind(after)

	a.VMOVSSStore(At(R8, 4*SampleOutRes), Y12)
	a.VMOVSSStore(At(R8, 4*SampleOutResolved), Y5)
	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// EmitSamplePenalty generates the repeat penalty: a SCATTER over the history,
// in place.
//
//	Q32    the logits copy, read and written
//	W      the history as BYTE OFFSETS into it (int64 each)
//	K      how many
//	AScale &penalty (f32)
//
// It is O(len(history)), not O(vocabulary).
//
// The history arrives as byte offsets rather than ids: addressing with an
// int32 id needs a sign-extending 32-bit load that none of the three tiers'
// assemblers has. The caller multiplies by four once per generated token.
//
// A positive logit is divided and everything else multiplied, so `v > 0` is
// the mask; +0.0 and -0.0 stay zero, as in llama.cpp.
func EmitSamplePenalty() ([]byte, error) { return must("sample_penalty", emitSamplePenalty()) }

func emitSamplePenalty() []byte {
	var a Buf
	a.MOVLoad(RSI, At(RDI, 112)) // Q32    -> the logits
	a.MOVLoad(RDX, At(RDI, 8))   // W      -> the byte offsets
	a.MOVLoad(RAX, At(RDI, 24))  // AScale -> &penalty
	a.MOVLoad(R10, At(RDI, 40))  // K      -> how many

	a.VBROADCASTSS(Y1, At(RAX, 0))
	a.VPXOR(Y15, Y15, Y15)

	done := a.Label()
	a.TESTQ(R10, R10)
	a.JZ(done)
	lp := a.Label()
	a.Bind(lp)
	a.MOVLoad(RCX, At(RDX, 0))
	a.MOVQ(R8, RSI)
	a.ADDQ(R8, RCX)
	a.VMOVSSLoad(Y4, At(R8, 0))
	a.VDIVPS(Y5, Y4, Y1)
	a.VMULPS(Y6, Y4, Y1)
	a.VCMPPS(Y7, Y4, Y15, 14) // _CMP_GT_OS: v > 0
	a.VPAND(Y5, Y5, Y7)
	a.VPANDN(Y6, Y7, Y6)
	a.VPOR(Y5, Y5, Y6)
	a.VMOVSSStore(At(R8, 0), Y5)
	a.ADDimm(RDX, 8)
	a.DEC(R10)
	a.JNZ(lp)
	a.Bind(done)

	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}
