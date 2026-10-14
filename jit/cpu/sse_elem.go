package cpu

import "fmt"

// The SSE tier's elementwise family: axpy, scale, softcap, softmax and the
// activations, in legacy SSE for a host with no AVX at all.
//
// The ABI is the AVX2 kernels' exactly: Out (0) is read and written in place,
// AScale (24) is the second vector, Scr (56) is ActConsts or &alpha, W (8) is
// softcap's {2/c, c}, and Args.K / Args.Rows are n/8 whole units and n%8 tail
// elements. An 8-float unit runs as two XMM halves (elemLoopSSE) and the tail
// is the same body on one lane.
//
// Each body is its AVX2 twin's arithmetic in the same order, except FMA.
// Scale uses none, so EmitScaleSSE is bit-identical to EmitScale; Axpy
// becomes MULPS then ADDPS, bit-identical to the unfused Go loop; everything
// built on exp is held to the AVX2 gates' bounds against the float64
// definitions.
//
// The two-operand form decides register order: dst == src2 on a
// non-commutative op panics (sse.go), so -x goes into a zeroed temporary and
// a quotient into the register holding the numerator. No arithmetic takes a
// memory operand, since a legacy m128 operand faults unless 16-byte aligned.
//
// Every kernel declares ISATierSSE and ends in a plain RET.

// EmitAxpySSE is EmitAxpy for the SSE tier: dst[i] += alpha * src[i].
//
// Registers: RCX dst, RDX src, RBX &alpha, R10/R11 counters. XMM0 holds
// alpha; XMM1 and XMM2 are the two operands of a half.
func EmitAxpySSE() ([]byte, error) {
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))  // Out -> dst, read and written in place
	a.MOVLoad(RDX, At(RDI, 24)) // AScale -> src
	a.MOVLoad(RBX, At(RDI, 56)) // Scr -> &alpha
	a.MOVLoad(R10, At(RDI, 40)) // K -> whole 8-float units
	a.MOVLoad(R11, At(RDI, 32)) // Rows -> tail elements
	bcastSS(&a, XMM0, At(RBX, 0))

	elemLoopSSE(&a, R10, R11, []Reg{RCX, RDX}, func(ld func(Reg, Reg), st func(Reg, Reg)) {
		ld(XMM1, RCX)
		ld(XMM2, RDX)
		// No FMA on this tier: the product is rounded before the add, which
		// is what `dst[i] += alpha * src[i]` computes in Go without fusion.
		a.MULPS(XMM2, XMM2, XMM0)
		a.ADDPS(XMM1, XMM1, XMM2)
		st(RCX, XMM1)
	})
	a.RET()
	return a.Bytes(), nil
}

// EmitScaleSSE is EmitScale for the SSE tier: dst[i] *= alpha. It is one
// MULPS per half, the same product in the same operand order as the AVX2
// kernel's VMULPS, so the two are bit-identical on every input.
//
// Registers: RCX dst, RBX &alpha, R10/R11 counters. XMM0 holds alpha.
func EmitScaleSSE() ([]byte, error) {
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))  // Out -> dst, read and written in place
	a.MOVLoad(RBX, At(RDI, 56)) // Scr -> &alpha
	a.MOVLoad(R10, At(RDI, 40)) // K -> whole 8-float units
	a.MOVLoad(R11, At(RDI, 32)) // Rows -> tail elements
	bcastSS(&a, XMM0, At(RBX, 0))

	elemLoopSSE(&a, R10, R11, []Reg{RCX}, func(ld func(Reg, Reg), st func(Reg, Reg)) {
		ld(XMM1, RCX)
		a.MULPS(XMM1, XMM1, XMM0)
		st(RCX, XMM1)
	})
	a.RET()
	return a.Bytes(), nil
}

// EmitClampSSE is EmitClamp for the SSE tier: x = min(max(x, lo), hi) in
// place, lo and hi the two floats Scr points at.
func EmitClampSSE() ([]byte, error) {
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))  // Out -> x, in place
	a.MOVLoad(RBX, At(RDI, 56)) // Scr -> {lo, hi}
	a.MOVLoad(R10, At(RDI, 40)) // K -> whole 8-float units
	a.MOVLoad(R11, At(RDI, 32)) // Rows -> tail elements
	bcastSS(&a, XMM0, At(RBX, 0))
	bcastSS(&a, XMM2, At(RBX, 4))

	elemLoopSSE(&a, R10, R11, []Reg{RCX}, func(ld func(Reg, Reg), st func(Reg, Reg)) {
		ld(XMM1, RCX)
		a.MAXPS(XMM1, XMM1, XMM0)
		a.MINPS(XMM1, XMM1, XMM2)
		st(RCX, XMM1)
	})
	a.RET()
	return a.Bytes(), nil
}

// EmitSoftcapSSE is EmitSoftcap for the SSE tier: x = c*tanh(x/c) in place,
// with tanh(y) = 1 - 2/(exp(2y)+1) so exp's own clamps give both saturations.
//
//	Out  x, read and written
//	W    two float32: 2/c and c
//	Scr  ActConsts
//
// Registers: RCX x, R9 W, RBX consts, R10/R11 counters. XMM7..XMM15 are exp's
// constants; XMM4 = 2, XMM5 = 2/c, XMM6 = c are hoisted out of the loop;
// XMM0..XMM3 are the body's.
func EmitSoftcapSSE() ([]byte, error) {
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))  // Out -> x, in place
	a.MOVLoad(R9, At(RDI, 8))   // W -> {2/c, c}
	a.MOVLoad(RBX, At(RDI, 56)) // Scr -> consts
	a.MOVLoad(R10, At(RDI, 40)) // K -> whole 8-float units
	a.MOVLoad(R11, At(RDI, 32)) // Rows -> tail elements
	loadExpConstsSSE(&a)
	bcastSS(&a, XMM4, At(RBX, 52)) // 2.0
	bcastSS(&a, XMM5, At(R9, 0))   // 2/c
	bcastSS(&a, XMM6, At(R9, 4))   // c

	elemLoopSSE(&a, R10, R11, []Reg{RCX}, func(ld func(Reg, Reg), st func(Reg, Reg)) {
		ld(XMM0, RCX)
		a.MULPS(XMM0, XMM0, XMM5) // 2x/c
		emitExpSSE(&a, XMM0, XMM1, XMM2)
		a.ADDPS(XMM0, XMM0, XMM8) // exp(2x/c) + 1
		a.DIVPS(XMM3, XMM4, XMM0) // 2/(that), into a third register: the
		// AVX2 form divides into its own divisor, which two operands cannot say
		a.SUBPS(XMM0, XMM8, XMM3) // tanh(x/c)
		a.MULPS(XMM0, XMM0, XMM6)
		st(RCX, XMM0)
	})
	a.RET()
	return a.Bytes(), nil
}

// EmitSoftmaxSSE is EmitSoftmax for the SSE tier: an in-place softmax over a
// row of RUNTIME length (Args.K whole units, Args.Rows tail elements), in the
// same three passes.
//
// Every reduction is the AVX2 kernel's lane for lane: each pass keeps two
// accumulators (lanes 0..3 and 4..7 of the AVX2 YMM) and folds them with
// hmax8SSE/hsum8SSE. The maximum is seeded from element 0, never a constant
// (see EmitSoftmax on qwen2's -872), and a tail element enters the maximum in
// every lane as the AVX2 tail's broadcast does, because MAXPS's operand order
// decides a NaN or a signed-zero tie.
//
// The tail's other lanes are kept out of the sum by ADDSS, which adds lane 0
// alone, instead of AVX2's VPBLENDD: the same sum, one instruction fewer.
//
// Registers: RCX row, RBX consts, R9/RDX the two counts, R10/R8 their working
// copies, R11 cursor. XMM7..XMM15 are exp's constants (XMM8 = 1.0);
// XMM3/XMM4 the maximum's halves, then XMM3 the maximum and XMM4/XMM5 the
// sum's halves, then XMM6 = 1/sum; XMM0..XMM2 are the body's.
func EmitSoftmaxSSE() ([]byte, error) { return emitSoftmaxSSE(false) }

// EmitLogSoftmaxSSE is EmitLogSoftmax for the SSE tier: pass 2 sums without
// storing, ln(sum) is emitLnSum's sequence without FMA, and pass 3 subtracts
// max + ln(sum).
func EmitLogSoftmaxSSE() ([]byte, error) { return emitSoftmaxSSE(true) }

func emitSoftmaxSSE(log bool) ([]byte, error) {
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))  // Out -> the row, read and written in place
	a.MOVLoad(RBX, At(RDI, 56)) // Scr -> expConsts
	a.MOVLoad(R9, At(RDI, 40))  // K -> whole 8-float units
	a.MOVLoad(RDX, At(RDI, 32)) // Rows -> tail elements
	loadExpConstsSSE(&a)
	pass := func(vec func(off int32), one func()) {
		a.MOVQ(R11, RCX)
		a.MOVQ(R10, R9)
		a.MOVQ(R8, RDX)
		elemLoopsSSE(&a, R10, R8, []Reg{R11}, vec, one)
	}
	half := func(off int32, lo, hi Reg) Reg {
		if off == 0 {
			return lo
		}
		return hi
	}

	// ---- pass 1: the maximum, seeded from element 0 ----
	bcastSS(&a, XMM3, At(RCX, 0))
	a.MOVAPS(XMM4, XMM3)
	pass(func(off int32) {
		a.MOVUPSLoad(XMM0, At(R11, off))
		m := half(off, XMM3, XMM4)
		a.MAXPS(m, m, XMM0)
	}, func() {
		bcastSS(&a, XMM0, At(R11, 0))
		a.MAXPS(XMM3, XMM3, XMM0)
		a.MAXPS(XMM4, XMM4, XMM0)
	})
	hmax8SSE(&a, XMM3, XMM4, XMM1) // XMM3 = the maximum, in every lane

	// ---- pass 2: exp(x - max), stored and summed ----
	a.XORPS(XMM4, XMM4, XMM4)
	a.XORPS(XMM5, XMM5, XMM5)
	pass(func(off int32) {
		a.MOVUPSLoad(XMM0, At(R11, off))
		a.SUBPS(XMM0, XMM0, XMM3)
		emitExpSSE(&a, XMM0, XMM1, XMM2)
		if !log {
			a.MOVUPSStore(At(R11, off), XMM0)
		}
		s := half(off, XMM4, XMM5)
		a.ADDPS(s, s, XMM0)
	}, func() {
		a.MOVSSLoad(XMM0, At(R11, 0))
		a.SUBPS(XMM0, XMM0, XMM3)
		emitExpSSE(&a, XMM0, XMM1, XMM2)
		if !log {
			a.MOVSSStore(At(R11, 0), XMM0)
		}
		a.ADDSS(XMM4, XMM4, XMM0) // lane 0 only: lanes 1..3 held exp(-max)
	})
	hsum8SSE(&a, XMM4, XMM5) // XMM4 = the sum, in every lane

	if log {
		emitLnSumSSE(&a)
		pass(func(off int32) {
			a.MOVUPSLoad(XMM0, At(R11, off))
			a.SUBPS(XMM0, XMM0, XMM3)
			a.MOVUPSStore(At(R11, off), XMM0)
		}, func() {
			a.MOVSSLoad(XMM0, At(R11, 0))
			a.SUBPS(XMM0, XMM0, XMM3)
			a.MOVSSStore(At(R11, 0), XMM0)
		})
		a.RET()
		return a.Bytes(), nil
	}

	// ---- pass 3: multiply by 1/sum ----
	a.DIVPS(XMM6, XMM8, XMM4) // 1/sum, XMM8 being exp's 1.0
	pass(func(off int32) {
		a.MOVUPSLoad(XMM0, At(R11, off))
		a.MULPS(XMM0, XMM0, XMM6)
		a.MOVUPSStore(At(R11, off), XMM0)
	}, func() {
		a.MOVSSLoad(XMM0, At(R11, 0))
		a.MULPS(XMM0, XMM0, XMM6)
		a.MOVSSStore(At(R11, 0), XMM0)
	})
	a.RET()
	return a.Bytes(), nil
}

// EmitActMulSSE is EmitActMul for the SSE tier: dst[i] = act(dst[i]) * up[i]
// (swiglu-oai combines with up its own way). The activation is baked.
func EmitActMulSSE(k ActKind) ([]byte, error) { return emitActKernelSSE(k, true) }

// EmitActSSE is EmitAct for the SSE tier: dst[i] = act(dst[i]), in place,
// with no `up`.
func EmitActSSE(k ActKind) ([]byte, error) { return emitActKernelSSE(k, false) }

// EmitSigmoidMulSSE is EmitSigmoidMul for the SSE tier: dst[i] =
// sigma(dst[i]) * up[i], the GATE in dst and what it gates in up.
func EmitSigmoidMulSSE() ([]byte, error) { return emitActKernelSSE(actSigmoid, true) }

// emitActKernelSSE is emitAct for the SSE tier.
//
// An unknown activation is refused by name rather than defaulted to
// GELU-tanh as the AVX2 switch does, and so is an ungated swiglu-oai, which
// would read an `up` pointer the caller never set.
//
// Registers: RCX dst, RDX up, RBX consts, R10/R11 counters. XMM7..XMM15 are
// exp's constants; XMM3 and XMM6 hold the kind's own constants, hoisted out of
// the loop (actConstsSSE); XMM0 is x and then the result, XMM1/XMM2 exp's
// temporaries, XMM4/XMM5 the body's.
func emitActKernelSSE(k ActKind, mul bool) ([]byte, error) {
	switch k {
	case ActSiLU, ActGELU, ActQuickGELU, actSigmoid, ActReLU2, ActReLU, ActSqrtSoftplus, ActGELUErf,
		ActSqrtSigmoid:
	case ActSwiGLUOAI, ActIdentity, ActSwiGLUClamp, ActSitu:
		if !mul {
			return nil, fmt.Errorf("jit: act/%s: it clamps and offsets `up`, so it exists only gated", k)
		}
	default:
		return nil, fmt.Errorf("jit: act/%s: not an activation this tier can bake", k)
	}
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))  // Out -> dst (gate), read and written in place
	a.MOVLoad(RDX, At(RDI, 24)) // AScale -> up
	a.MOVLoad(RBX, At(RDI, 56)) // Scr -> consts
	a.MOVLoad(R10, At(RDI, 40)) // K -> whole 8-float units
	a.MOVLoad(R11, At(RDI, 32)) // Rows -> tail elements
	loadExpConstsSSE(&a)
	actConstsSSE(&a, k)
	cursors := []Reg{RCX}
	if mul {
		cursors = append(cursors, RDX)
	}
	elemLoopSSE(&a, R10, R11, cursors, func(ld func(Reg, Reg), st func(Reg, Reg)) {
		emitActBodySSE(&a, k, mul, ld, st)
	})
	a.RET()
	return a.Bytes(), nil
}

// actConstsSSE hoists the constants a kind's body reads more than once into
// XMM3 and XMM6, once, before the loop. The AVX2 body broadcasts each from
// memory at its use, which is one instruction there and two (MOVSS + SHUFPS)
// here, emitted three times per unit.
func actConstsSSE(a *Buf, k ActKind) {
	switch k {
	case ActQuickGELU:
		bcastSS(a, XMM6, At(RBX, 60)) // 1.702
	case ActSwiGLUOAI:
		bcastSS(a, XMM6, At(RBX, 60)) // 1.702
		bcastSS(a, XMM3, At(RBX, 64)) // 7
	case ActSwiGLUClamp:
		bcastSS(a, XMM3, At(RBX, 72)) // 10
		bcastSS(a, XMM6, At(RBX, 76)) // -10
	case ActSqrtSoftplus:
		bcastSS(a, XMM3, At(RBX, 80))  // 2
		bcastSS(a, XMM6, At(RBX, 100)) // the |x| mask
	case ActGELU:
		bcastSS(a, XMM6, At(RBX, 52)) // 2.0
		bcastSS(a, XMM3, At(RBX, 56)) // 0.5
	}
}

// emitActBodySSE is emitActBody for one XMM (a half, or the tail's lane 0).
func emitActBodySSE(a *Buf, k ActKind, mul bool, ld func(Reg, Reg), st func(Reg, Reg)) {
	ld(XMM0, RCX)

	// negExp1 leaves 1 + exp(-v) in XMM4, for v in src (not XMM4): SiLU's,
	// sigmoid's and quick-GELU's shared denominator. -v is 0 - v, the AVX2
	// kernel's own spelling (so a +0 stays +0), built in the zeroed XMM4
	// because 0 - v into v's own register is dst == src2 on a SUBPS.
	negExp1 := func(src Reg) {
		a.XORPS(XMM4, XMM4, XMM4)
		a.SUBPS(XMM4, XMM4, src)
		emitExpSSE(a, XMM4, XMM1, XMM2)
		a.ADDPS(XMM4, XMM4, XMM8) // + 1.0
	}

	switch k {
	case ActIdentity:
		// x itself: the multiply below is the whole kernel.
	case ActSiLU:
		// x / (1 + exp(-x)).
		negExp1(XMM0)
		a.DIVPS(XMM0, XMM0, XMM4)
	case ActReLU:
		a.XORPS(XMM4, XMM4, XMM4)
		a.MAXPS(XMM0, XMM0, XMM4)
	case ActReLU2:
		// max(x, 0)^2.
		a.XORPS(XMM4, XMM4, XMM4)
		a.MAXPS(XMM0, XMM0, XMM4)
		a.MULPS(XMM0, XMM0, XMM0)
	case actSigmoid:
		// 1 / (1 + exp(-x)): SiLU with the numerator 1.
		negExp1(XMM0)
		a.DIVPS(XMM0, XMM8, XMM4)
	case ActQuickGELU:
		// x / (1 + exp(-1.702x)).
		a.MULPS(XMM5, XMM0, XMM6) // 1.702x
		negExp1(XMM5)
		a.DIVPS(XMM0, XMM0, XMM4)
	case ActSwiGLUOAI:
		// x = min(gate, 7); x*sigma(1.702x) * (clamp(up, -7, 7) + 1). MINPS
		// keeps the AVX2 operand order: a NaN gate comes out as 7 on both.
		a.MINPS(XMM0, XMM0, XMM3)
		a.MULPS(XMM5, XMM0, XMM6) // 1.702x
		negExp1(XMM5)
		a.DIVPS(XMM0, XMM0, XMM4) // x*sigma(1.702x)
		ld(XMM4, RDX)
		a.MINPS(XMM4, XMM4, XMM3)     // min(up, 7)
		bcastSS(a, XMM5, At(RBX, 68)) // -7
		a.MAXPS(XMM4, XMM4, XMM5)
		a.ADDPS(XMM4, XMM4, XMM8) // y + 1
		a.MULPS(XMM0, XMM0, XMM4)
	case ActSwiGLUClamp:
		// x = min(gate, 10); x*sigma(x) * clamp(up, -10, 10).
		a.MINPS(XMM0, XMM0, XMM3)
		negExp1(XMM0)
		a.DIVPS(XMM0, XMM0, XMM4) // x*sigma(x)
		ld(XMM4, RDX)
		a.MINPS(XMM4, XMM4, XMM3) // min(up, 10)
		a.MAXPS(XMM4, XMM4, XMM6) // max(that, -10)
		a.MULPS(XMM0, XMM0, XMM4)
	case ActSqrtSoftplus:
		// max(x,0) + log(1+exp(-|x|)), then its square root: the AVX2 body's
		// sequence (no FMA on this tier, so the series is MULPS then ADDPS).
		a.XORPS(XMM4, XMM4, XMM4)
		a.MAXPS(XMM4, XMM4, XMM0) // p = max(x, 0) (operands swapped: same for finite x)
		a.ANDPS(XMM0, XMM0, XMM6) // |x|
		a.XORPS(XMM5, XMM5, XMM5)
		a.SUBPS(XMM5, XMM5, XMM0) // -|x|
		emitExpSSE(a, XMM5, XMM1, XMM2)
		a.MOVAPS(XMM1, XMM5)
		a.ADDPS(XMM1, XMM1, XMM3) // 2 + u
		a.DIVPS(XMM5, XMM5, XMM1) // s
		a.MOVAPS(XMM0, XMM5)
		a.MULPS(XMM0, XMM0, XMM5) // s^2
		bcastSS(a, XMM1, At(RBX, 84))
		for _, off := range []int32{88, 92, 96} {
			a.MULPS(XMM1, XMM1, XMM0)
			bcastSS(a, XMM2, At(RBX, off))
			a.ADDPS(XMM1, XMM1, XMM2)
		}
		a.MULPS(XMM1, XMM1, XMM0)
		a.ADDPS(XMM1, XMM1, XMM8) // ... + 1
		a.MULPS(XMM1, XMM1, XMM5)
		a.MULPS(XMM1, XMM1, XMM3) // 2*s*h
		a.ADDPS(XMM1, XMM1, XMM4) // softplus
		a.SQRTPS(XMM0, XMM1)
	case ActSitu:
		// The AVX2 body's sequence: c*tanh(x/c) with the rational tanh at
		// both bounds (no FMA here: each Horner step a multiply and an add),
		// and sigma(g) dividing the gate's.
		bound := func(dst, x Reg, inv, c int32) {
			bcastSS(a, XMM5, At(RBX, inv))
			a.MULPS(XMM3, x, XMM5) // x/c
			bcastSS(a, XMM5, At(RBX, 120))
			a.MINPS(XMM3, XMM3, XMM5)
			bcastSS(a, XMM5, At(RBX, 124))
			a.MAXPS(XMM3, XMM3, XMM5) // clamped
			a.MULPS(XMM2, XMM3, XMM3) // x^2
			bcastSS(a, XMM1, At(RBX, 128))
			for off := int32(132); off <= 152; off += 4 {
				a.MULPS(XMM1, XMM1, XMM2)
				bcastSS(a, XMM5, At(RBX, off))
				a.ADDPS(XMM1, XMM1, XMM5)
			}
			a.MULPS(XMM1, XMM1, XMM3) // x*P(x^2)
			bcastSS(a, XMM4, At(RBX, 156))
			for off := int32(160); off <= 168; off += 4 {
				a.MULPS(XMM4, XMM4, XMM2)
				bcastSS(a, XMM5, At(RBX, off))
				a.ADDPS(XMM4, XMM4, XMM5)
			}
			a.DIVPS(dst, XMM1, XMM4) // tanh
			bcastSS(a, XMM5, At(RBX, c))
			a.MULPS(dst, dst, XMM5) // c*tanh
		}
		bound(XMM6, XMM0, 104, 108)
		negExp1(XMM0)             // 1 + exp(-g)
		a.DIVPS(XMM6, XMM6, XMM4) // 4*tanh(g/4)*sigma(g)
		ld(XMM0, RDX)
		bound(XMM0, XMM0, 112, 116)
		a.MULPS(XMM0, XMM0, XMM6)
	case ActSqrtSigmoid:
		// The AVX2 body's sequence: the root of the floored magnitude, the
		// sign bit put back, then sigmoid's.
		a.MOVAPS(XMM4, XMM0)
		bcastSS(a, XMM5, At(RBX, 100))
		a.ANDPS(XMM4, XMM4, XMM5) // |x|
		bcastSS(a, XMM5, At(RBX, 208))
		a.MAXPS(XMM4, XMM4, XMM5)
		a.SQRTPS(XMM4, XMM4)
		bcastSS(a, XMM5, At(RBX, 212))
		a.ANDPS(XMM0, XMM0, XMM5) // the sign
		a.ORPS(XMM0, XMM0, XMM4)  // the signed root
		a.XORPS(XMM4, XMM4, XMM4)
		a.SUBPS(XMM4, XMM4, XMM0) // its negation
		emitExpSSE(a, XMM4, XMM1, XMM2)
		a.ADDPS(XMM4, XMM4, XMM8)
		a.MOVAPS(XMM0, XMM8)
		a.DIVPS(XMM0, XMM0, XMM4)
	case ActGELUErf:
		// The AVX2 body's sequence with a multiply and an add for each FMA;
		// XMM3 and XMM6 carry no hoisted constant for this kind.
		bcastSS(a, XMM5, At(RBX, 100))
		a.MOVAPS(XMM3, XMM0)
		a.ANDPS(XMM3, XMM3, XMM5) // |x|
		bcastSS(a, XMM5, At(RBX, 180))
		a.MOVAPS(XMM6, XMM3)
		a.MULPS(XMM6, XMM6, XMM5) // z
		a.MOVAPS(XMM5, XMM6)
		a.MULPS(XMM5, XMM5, XMM6)
		a.XORPS(XMM4, XMM4, XMM4)
		a.SUBPS(XMM4, XMM4, XMM5) // -z^2
		emitExpSSE(a, XMM4, XMM1, XMM2)
		bcastSS(a, XMM5, At(RBX, 184))
		a.MULPS(XMM6, XMM6, XMM5)
		a.ADDPS(XMM6, XMM6, XMM8)
		a.MOVAPS(XMM5, XMM8)
		a.DIVPS(XMM5, XMM5, XMM6) // t
		bcastSS(a, XMM6, At(RBX, 204))
		for _, off := range []int32{200, 196, 192, 188} {
			a.MULPS(XMM6, XMM6, XMM5)
			bcastSS(a, XMM1, At(RBX, off))
			a.ADDPS(XMM6, XMM6, XMM1)
		}
		a.MULPS(XMM6, XMM6, XMM5) // the polynomial times t
		a.MULPS(XMM6, XMM6, XMM4) // q
		a.MULPS(XMM6, XMM6, XMM3) // |x| q
		a.ADDPS(XMM0, XMM0, XMM3) // x + |x|
		a.SUBPS(XMM0, XMM0, XMM6)
		bcastSS(a, XMM5, At(RBX, 56))
		a.MULPS(XMM0, XMM0, XMM5)
	case ActGELU:
		// GELU-tanh: z = sqrt(2/pi)*(x + 0.044715x^3), and
		// 0.5x(1 + tanh z) = 0.5x(2 - 2/(exp(2z)+1)).
		a.MULPS(XMM4, XMM0, XMM0)     // x^2
		bcastSS(a, XMM5, At(RBX, 44)) // 0.044715
		a.MULPS(XMM4, XMM4, XMM5)
		a.ADDPS(XMM4, XMM4, XMM8)     // 1 + 0.044715x^2
		a.MULPS(XMM4, XMM4, XMM0)     // x + 0.044715x^3
		bcastSS(a, XMM5, At(RBX, 48)) // sqrt(2/pi)
		a.MULPS(XMM4, XMM4, XMM5)     // z
		a.MULPS(XMM4, XMM4, XMM6)     // 2z
		emitExpSSE(a, XMM4, XMM1, XMM2)
		a.ADDPS(XMM4, XMM4, XMM8) // exp(2z) + 1
		a.DIVPS(XMM5, XMM6, XMM4) // 2/(exp(2z)+1)
		a.SUBPS(XMM4, XMM6, XMM5) // 1 + tanh(z)
		a.MULPS(XMM4, XMM4, XMM3) // * 0.5
		a.MULPS(XMM0, XMM0, XMM4) // * x
	}

	if mul && k != ActSwiGLUOAI && k != ActSwiGLUClamp && k != ActSitu {
		ld(XMM4, RDX)
		a.MULPS(XMM0, XMM0, XMM4)
	}
	st(RCX, XMM0)
}

// emitLnSumSSE is emitLnSum without FMA: XMM3 += ln(XMM4), every lane, XMM7
// the exponent bias and XMM8 1.0 (loadExpConstsSSE). Clobbers XMM0..XMM2,
// XMM5, XMM6.
func emitLnSumSSE(a *Buf) {
	a.PSRLD(XMM0, XMM4, 23)
	a.PSUBD(XMM0, XMM0, XMM7)
	a.CVTDQ2PS(XMM0, XMM0)         // e
	bcastSS(a, XMM1, At(RBX, 172)) // the mantissa mask
	a.PAND(XMM1, XMM1, XMM4)
	a.POR(XMM1, XMM1, XMM8)   // m
	a.SUBPS(XMM1, XMM1, XMM8) // u
	bcastSS(a, XMM2, At(RBX, 80))
	a.ADDPS(XMM2, XMM2, XMM1) // 2 + u
	a.MOVAPS(XMM5, XMM1)
	a.DIVPS(XMM5, XMM5, XMM2) // s
	a.MOVAPS(XMM1, XMM5)
	a.MULPS(XMM1, XMM1, XMM5) // s^2
	bcastSS(a, XMM6, At(RBX, 84))
	for _, off := range []int32{88, 92, 96} {
		a.MULPS(XMM6, XMM6, XMM1)
		bcastSS(a, XMM2, At(RBX, off))
		a.ADDPS(XMM6, XMM6, XMM2)
	}
	a.MULPS(XMM6, XMM6, XMM1)
	a.ADDPS(XMM6, XMM6, XMM8)
	a.MULPS(XMM6, XMM6, XMM5)
	bcastSS(a, XMM2, At(RBX, 80))
	a.MULPS(XMM6, XMM6, XMM2) // ln(m)
	bcastSS(a, XMM2, At(RBX, 176))
	a.MULPS(XMM0, XMM0, XMM2)
	a.ADDPS(XMM0, XMM0, XMM6) // e*ln2 + ln(m)
	a.ADDPS(XMM3, XMM3, XMM0)
}
