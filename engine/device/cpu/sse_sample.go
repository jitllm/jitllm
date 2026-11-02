package cpu

// The sampler's three kernels on the legacy-SSE tier. sample.go carries the
// contract, the ABI and the reasoning; these are the same kernels four lanes
// wide and without a VEX byte in them.
//
// They take the AVX2 twins' ABI and constant block unchanged.
//
// The lane assignment may differ from the AVX2 tier's and the gate is still
// equality, as in sse_topk.go: every ordering output is an input selected by
// comparison with ties broken by id. The draw kernel's cumulative is a strictly
// sequential chain in lane 0 on both tiers.
//
// Legacy CMPPS has no ordered greater-than, so `x > y` is spelled `y < x` and
// `x >= y` is `y <= x`; both are false for a NaN on either side, like AVX2's
// predicates 14 and 13.
//
// PMINSD (SSE4.1) decides the tier, since ids are 32-bit with a 0x7FFFFFFF
// sentinel; BLENDPS lets the ragged tail reuse the vector body.

// emitSampleSegMaxSSEBody is EmitSampleSegMax (sample.go) on legacy SSE.
//
// Registers: RSI the value cursor, RDX the id cursor, RBX Scr, R8 the output
// values, R9 the output ids, R12/R13 the segment's vector and tail counts,
// R10/R11 their working copies, R15 the segment counter; XMM0 the running
// maximum, XMM1 its id, XMM2 the element index, XMM3 the index step, XMM11
// prevVal, XMM12 prevIdx, XMM13 INT_MAX, XMM14 NaN, XMM4..XMM10 and XMM15
// temporaries.
func emitSampleSegMaxSSEBody(first, idsMem bool) []byte {
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RBX, At(RDI, 56))  // Scr
	a.MOVLoad(RSI, At(RDI, 112)) // Q32      -> the values
	a.MOVLoad(R8, At(RDI, 0))    // Out
	a.MOVLoad(R9, At(RDI, 104))  // AHalfSum -> the ids out
	a.MOVLoad(R12, At(RDI, 40))  // K
	a.MOVLoad(R15, At(RDI, 32))  // Rows
	a.MOVQ(R13, R12)
	a.ANDimm8(R13, 3)
	a.SHRimm(R12, 2)

	bcastD(&a, XMM13, At(RBX, moeMaxIOff))
	bcastSS(&a, XMM14, At(RBX, moeNaNOff))

	if idsMem {
		a.MOVLoad(RDX, At(RDI, 96))
	} else {
		bcastD(&a, XMM4, At(RDI, 80))
		a.MOVUPSLoad(XMM2, At(RBX, moeIdxOff)) // 0..3
		a.PADDD(XMM2, XMM2, XMM4)
	}
	if !first {
		a.MOVLoad(RAX, At(RDI, 24))
		bcastSS(&a, XMM11, At(RAX, 0))
		bcastD(&a, XMM12, At(RAX, 4))
	}

	body := func() {
		if first {
			a.CMPPS(XMM5, XMM4, XMM4, 0) // ordered: every element but a NaN
		} else {
			a.CMPPS(XMM5, XMM4, XMM11, 1) // v < prevVal
			a.CMPPS(XMM6, XMM4, XMM11, 0) // v == prevVal
			a.PCMPGTD(XMM7, XMM2, XMM12)  // and a higher id than the tie it lost
			a.PAND(XMM6, XMM6, XMM7)
			a.POR(XMM5, XMM5, XMM6)
		}
		a.CMPPS(XMM8, XMM0, XMM4, 1) // (max < v) ordered, i.e. `v > max`
		a.PCMPEQD(XMM9, XMM1, XMM13) // unless this lane is still empty
		a.POR(XMM8, XMM8, XMM9)
		a.PAND(XMM5, XMM5, XMM8)
		a.PAND(XMM6, XMM4, XMM5)
		a.PANDN(XMM7, XMM5, XMM0)
		a.POR(XMM0, XMM6, XMM7)
		a.PAND(XMM6, XMM2, XMM5)
		a.PANDN(XMM7, XMM5, XMM1)
		a.POR(XMM1, XMM6, XMM7)
	}

	seg := a.Label()
	a.Bind(seg)
	bcastSS(&a, XMM0, At(RBX, moeNegInfOff))
	a.MOVAPS(XMM1, XMM13)
	a.MOVQ(R10, R12)
	a.MOVQ(R11, R13)

	vt, vd := a.Label(), a.Label()
	a.TESTQ(R10, R10)
	a.JZ(vt)
	if !idsMem {
		bcastD(&a, XMM3, At(RBX, moeStep4Off))
	}
	vl := a.Label()
	a.Bind(vl)
	a.MOVUPSLoad(XMM4, At(RSI, 0))
	if idsMem {
		a.MOVDQULoad(XMM2, At(RDX, 0))
		a.ADDimm(RDX, 16)
	}
	body()
	if !idsMem {
		a.PADDD(XMM2, XMM2, XMM3)
	}
	a.ADDimm(RSI, 16)
	a.DEC(R10)
	a.JNZ(vl)

	a.Bind(vt)
	a.TESTQ(R11, R11)
	a.JZ(vd)
	if !idsMem {
		bcastD(&a, XMM3, At(RBX, moeOneIOff))
	}
	tl := a.Label()
	a.Bind(tl)
	a.MOVSSLoad(XMM10, At(RSI, 0))
	a.BLENDPS(XMM4, XMM14, XMM10, 0x01) // the element in lane 0, NaN above it
	if idsMem {
		a.MOVSSLoad(XMM2, At(RDX, 0))
		a.ADDimm(RDX, 4)
	}
	body()
	if !idsMem {
		a.PADDD(XMM2, XMM2, XMM3)
	}
	a.ADDimm(RSI, 4)
	a.DEC(R11)
	a.JNZ(tl)
	a.Bind(vd)

	a.MOVAPS(XMM15, XMM0) // the per-lane maxima must survive the fold
	hmax4SSE(&a, XMM0, XMM6)
	a.CMPPS(XMM5, XMM15, XMM0, 0)
	a.PAND(XMM6, XMM1, XMM5)
	a.PANDN(XMM7, XMM5, XMM13)
	a.POR(XMM6, XMM6, XMM7)
	hmin4SSE(&a, XMM6, XMM4)
	a.MOVAPS(XMM9, XMM6) // the winner's id, in every lane
	a.PCMPEQD(XMM5, XMM1, XMM9)
	a.PAND(XMM6, XMM15, XMM5)
	hor4SSE(&a, XMM6, XMM4) // its value's BITS, not the max fold's (sample.go)

	a.MOVSSStore(At(R8, 0), XMM6)
	a.MOVSSStore(At(R9, 0), XMM9)
	a.ADDimm(R8, 4)
	a.ADDimm(R9, 4)
	a.DEC(R15)
	a.JNZ(seg)

	a.RET() // no VZEROUPPER: it is VEX-encoded and faults on a host with no AVX
	return a.Bytes()
}

// emitSampleDrawSSEBody is EmitSampleDraw (sample.go) on legacy SSE. Every
// value lives in lane 0 and the other three are inert, so this is the AVX2
// kernel instruction for instruction with the two order-reversed compares.
func emitSampleDrawSSEBody() []byte {
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RBX, At(RDI, 56))
	a.MOVLoad(RSI, At(RDI, 112))
	a.MOVLoad(RDX, At(RDI, 24))
	a.MOVLoad(R8, At(RDI, 96))
	a.MOVLoad(R9, At(RDI, 0))
	a.MOVLoad(R12, At(RDI, 40))

	a.PXOR(XMM15, XMM15, XMM15)
	bcastD(&a, XMM7, At(RBX, moeOneIOff))
	a.MOVSSLoad(XMM2, At(RDX, 4*SampleParTopP))
	a.MOVSSLoad(XMM1, At(RDX, 4*SampleParMinP))
	a.MOVSSLoad(XMM4, At(RSI, 0))
	a.MULPS(XMM1, XMM1, XMM4)
	a.PXOR(XMM0, XMM0, XMM0)
	a.PXOR(XMM3, XMM3, XMM3)
	a.PXOR(XMM4, XMM4, XMM4)
	a.PXOR(XMM5, XMM5, XMM5)
	a.PXOR(XMM6, XMM6, XMM6)

	sel := func(dst, mask, x, y, t Reg) {
		a.PANDN(t, mask, y)
		a.PAND(dst, x, mask)
		a.POR(dst, dst, t)
	}
	// `~latched & c` cannot be written into c's own register: ANDNPS is
	// destructive, so dst == src2 is refused by sse3. XMM14 is the scratch
	// that makes it expressible.
	notCut := func(c Reg) {
		a.MOVAPS(XMM14, XMM5)
		a.PANDN(XMM14, XMM14, c)
		a.MOVAPS(c, XMM14)
	}

	a.MOVQ(R10, R12)
	p1 := a.Label()
	a.Bind(p1)
	a.MOVSSLoad(XMM13, At(RSI, 0))
	a.CMPPS(XMM8, XMM13, XMM1, 1) // p < cut
	notCut(XMM8)
	sel(XMM4, XMM8, XMM0, XMM4, XMM9)
	sel(XMM3, XMM8, XMM6, XMM3, XMM9)
	a.POR(XMM5, XMM5, XMM8)
	a.ADDPS(XMM0, XMM0, XMM13)
	a.CMPPS(XMM8, XMM2, XMM0, 2) // (TopP <= acc) ordered, i.e. `acc >= TopP`
	notCut(XMM8)
	sel(XMM4, XMM8, XMM0, XMM4, XMM9)
	a.PADDD(XMM11, XMM6, XMM7)
	sel(XMM3, XMM8, XMM11, XMM3, XMM9)
	a.POR(XMM5, XMM5, XMM8)
	a.PADDD(XMM6, XMM6, XMM7)
	a.ADDimm(RSI, 4)
	a.DEC(R10)
	a.JNZ(p1)

	a.MOVSSLoad(XMM9, At(RDX, 4*SampleParSum))
	a.CMPPS(XMM10, XMM15, XMM9, 2) // (0 <= sumAll), i.e. `sumAll >= 0`
	sel(XMM11, XMM10, XMM9, XMM0, XMM12)
	sel(XMM4, XMM5, XMM4, XMM11, XMM12)
	bcastD(&a, XMM12, At(RDI, 40))
	sel(XMM3, XMM5, XMM3, XMM12, XMM11)

	a.MOVSSStore(At(R8, 4*SampleOutM), XMM3)
	a.MOVSSStore(At(R8, 4*SampleOutCut), XMM5)
	a.MOVSSStore(At(R9, 0), XMM4)

	a.MOVLoad(RSI, At(RDI, 112))
	a.MOVSSLoad(XMM0, At(RDX, 4*SampleParU))
	a.MULPS(XMM0, XMM0, XMM4)
	a.PSUBD(XMM12, XMM3, XMM7)
	a.PXOR(XMM5, XMM5, XMM5)
	a.PXOR(XMM6, XMM6, XMM6)

	// RowStr != 0 asks for the mass alone, and skips the walk (sample.go).
	after := a.Label()
	a.MOVLoad(RAX, At(RDI, 48))
	a.TESTQ(RAX, RAX)
	a.JNZ(after)

	a.MOVQ(R10, R12)
	p2 := a.Label()
	a.Bind(p2)
	a.MOVSSLoad(XMM13, At(RSI, 0))
	a.PCMPGTD(XMM8, XMM3, XMM6)
	notCut(XMM8)
	a.SUBPS(XMM9, XMM0, XMM13)
	sel(XMM0, XMM8, XMM9, XMM0, XMM10)
	a.CMPPS(XMM10, XMM0, XMM15, 2) // r <= 0
	a.PAND(XMM10, XMM10, XMM8)
	sel(XMM12, XMM10, XMM6, XMM12, XMM11)
	a.POR(XMM5, XMM5, XMM10)
	a.PADDD(XMM6, XMM6, XMM7)
	a.ADDimm(RSI, 4)
	a.DEC(R10)
	a.JNZ(p2)
	a.Bind(after)

	a.MOVSSStore(At(R8, 4*SampleOutRes), XMM12)
	a.MOVSSStore(At(R8, 4*SampleOutResolved), XMM5)
	a.RET()
	return a.Bytes()
}

// emitSamplePenaltySSEBody is EmitSamplePenalty (sample.go) on legacy SSE.
func emitSamplePenaltySSEBody() []byte {
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RSI, At(RDI, 112))
	a.MOVLoad(RDX, At(RDI, 8))
	a.MOVLoad(RAX, At(RDI, 24))
	a.MOVLoad(R10, At(RDI, 40))

	bcastSS(&a, XMM1, At(RAX, 0))
	a.PXOR(XMM15, XMM15, XMM15)

	done := a.Label()
	a.TESTQ(R10, R10)
	a.JZ(done)
	lp := a.Label()
	a.Bind(lp)
	a.MOVLoad(RCX, At(RDX, 0))
	a.MOVQ(R8, RSI)
	a.ADDQ(R8, RCX)
	a.MOVSSLoad(XMM4, At(R8, 0))
	a.MOVAPS(XMM5, XMM4)
	a.DIVPS(XMM5, XMM5, XMM1)
	a.MOVAPS(XMM6, XMM4)
	a.MULPS(XMM6, XMM6, XMM1)
	a.CMPPS(XMM7, XMM15, XMM4, 1) // (0 < v) ordered, i.e. `v > 0`
	a.PAND(XMM5, XMM5, XMM7)
	a.PANDN(XMM8, XMM7, XMM6) // ~(v > 0) & (v * pen), into a register of its own
	a.POR(XMM5, XMM5, XMM8)
	a.MOVSSStore(At(R8, 0), XMM5)
	a.ADDimm(RDX, 8)
	a.DEC(R10)
	a.JNZ(lp)
	a.Bind(done)

	a.RET()
	return a.Bytes()
}

// EmitSampleSegMaxSSE, EmitSampleDrawSSE and EmitSamplePenaltySSE are this
// tier's entries in the Emitters table.
func EmitSampleSegMaxSSE(first, idsMem bool) ([]byte, error) {
	return must("sample_segmax_sse", emitSampleSegMaxSSEBody(first, idsMem))
}

func EmitSampleDrawSSE() ([]byte, error) {
	return must("sample_draw_sse", emitSampleDrawSSEBody())
}

func EmitSamplePenaltySSE() ([]byte, error) {
	return must("sample_penalty_sse", emitSamplePenaltySSEBody())
}
