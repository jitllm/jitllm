package cpu

// Shared building blocks for the SSE tier's emitters.
//
// They mirror the AVX2 helpers' contracts, not their width. The elementwise
// ABI stays cpu.ElemLanes = 8 floats per unit (nn and engine/model/vision.go split
// work by it), so an SSE body runs each unit as two XMM halves (elemLoopSSE),
// and the reductions fold the halves the way the AVX2 ones fold 128-bit lanes
// (hsum8SSE, hmax8SSE), so a kernel with no FMA can be bit-identical to its
// AVX2 twin.
//
// Every SSE kernel ends in a plain RET: VZEROUPPER is VEX-encoded and faults
// without AVX, and a Buf declared without AVX2 panics on one (isa.go). RDI is
// *Args for the whole call, R14 and RBP are never written, RSP never moves
// (the 128-byte red zone below it is usable), and every XMM is scratch.

// elemLoopSSE is elemLoop for the SSE tier: body over K whole 8-float units,
// then over R single elements, where K and R are registers holding n/8 and
// n%8 (Args.K and Args.Rows, as nn computes them with cpu.ElemLanes). Either
// may be zero.
//
// body is emitted three times -- the low half of a unit, the high half, and
// the one-element tail -- and must do all its memory access through ld and st,
// which carry the half's byte offset (0 or 16) or the scalar load/store. The
// tail's ld is MOVSSLoad, which zeroes lanes 1..3, and its st is MOVSSStore,
// so nothing past the last element is read or written.
func elemLoopSSE(a *Buf, k, r Reg, cursors []Reg, body func(ld func(dst, p Reg), st func(p, src Reg))) {
	elemLoopsSSE(a, k, r, cursors, func(off int32) {
		body(func(dst, p Reg) { a.MOVUPSLoad(dst, At(p, off)) },
			func(p, src Reg) { a.MOVUPSStore(At(p, off), src) })
	}, func() {
		body(func(dst, p Reg) { a.MOVSSLoad(dst, At(p, 0)) },
			func(p, src Reg) { a.MOVSSStore(At(p, 0), src) })
	})
}

// elemLoopsSSE is elemLoopSSE with the bodies written separately, for a kernel
// whose tail is not the vector body on one lane -- a reduction, whose other
// lanes must be neutral for the operation rather than zero. vec is called
// twice per unit, with off = 0 and off = 16 (the byte offset of that half from
// each cursor); one is called once per tail element. The cursors advance by 32
// bytes per unit and 4 per tail element, exactly as elemLoops' do.
func elemLoopsSSE(a *Buf, k, r Reg, cursors []Reg, vec func(off int32), one func()) {
	tail, done := a.Label(), a.Label()
	a.TESTQ(k, k)
	a.JZ(tail)
	lp := a.Label()
	a.Bind(lp)
	vec(0)
	vec(16)
	for _, c := range cursors {
		a.ADDimm(c, 32)
	}
	a.DEC(k)
	a.JNZ(lp)

	a.Bind(tail)
	a.TESTQ(r, r)
	a.JZ(done)
	lt := a.Label()
	a.Bind(lt)
	one()
	for _, c := range cursors {
		a.ADDimm(c, 4)
	}
	a.DEC(r)
	a.JNZ(lt)
	a.Bind(done)
}

// bcastSS loads one float32 from memory and splats it across all four lanes:
// MOVSS then SHUFPS 0. It replaces VBROADCASTSS, which has no legacy form.
func bcastSS(a *Buf, x Reg, m Mem) {
	a.MOVSSLoad(x, m)
	a.SHUFPS(x, x, x, 0)
}

// bcastD loads one 32-bit integer and splats it: MOVD then PSHUFD 0. It
// replaces VPBROADCASTD.
func bcastD(a *Buf, x Reg, m Mem) {
	a.MOVDLoad(x, m)
	a.PSHUFD(x, x, 0)
}

// bcastImm32 materialises a 32-bit constant in every lane of x without a
// memory block: MOV gp, imm32; MOVD x, gp; PSHUFD x, x, 0. gp is clobbered.
// It is how a kernel gets a constant no Scr block carries, so no shared
// constant layout has to grow for the SSE tier.
func bcastImm32(a *Buf, x, gp Reg, bits uint32) {
	a.MOVimm32(gp, int32(bits))
	a.MOVDToX(x, gp)
	a.PSHUFD(x, x, 0)
}

// loadExpConstsSSE fills XMM7..XMM15 with the constants emitExpSSE needs,
// once, outside any loop. RBX must point at expConstsShared() (or any block
// that begins with it, such as ActConsts or DeltaGateConsts).
//
// The assignment is loadExpConsts' exactly, so a port keeps its register plan:
// XMM9 log2e, XMM10 -ln2hi, XMM11 -ln2lo, XMM12..XMM15 c5..c2, XMM8 1.0, XMM7
// the exponent bias 127 as an integer. XMM0..XMM6 are left to the caller.
func loadExpConstsSSE(a *Buf) {
	bcastSS(a, XMM9, At(RBX, 0))
	bcastSS(a, XMM10, At(RBX, 4))
	bcastSS(a, XMM11, At(RBX, 8))
	bcastSS(a, XMM12, At(RBX, 12))
	bcastSS(a, XMM13, At(RBX, 16))
	bcastSS(a, XMM14, At(RBX, 20))
	bcastSS(a, XMM15, At(RBX, 24))
	bcastSS(a, XMM8, At(RBX, 28))
	bcastD(a, XMM7, At(RBX, 36))
}

// emitExpSSE computes exp(dst) into dst, four lanes, clobbering t0 and t1. It
// is emitExpPS's algorithm (exp.go) with no FMA: dst, t0 and t1 are three
// distinct registers among XMM0..XMM6, and XMM7..XMM15 hold the constants
// loadExpConstsSSE put there. The clamps are re-read from RBX (offsets 32 and
// 40) through t0, exactly as the AVX2 version reads them.
//
// Two temporaries suffice without FMA because n is re-derived rather than
// kept: t1 goes to the range reduction's products, and n is converted back
// from its float form, an exact integer.
//
// Its accuracy is emitExpPS's class, not its bits: n*hi is exact (hi has 11
// significant bits and |n| <= 128), n*lo is rounded before the add, and each
// Horner step rounds twice. ssehelpers_test.go holds it to TestEmitExpPS's
// relative bound.
func emitExpSSE(a *Buf, dst, t0, t1 Reg) {
	if dst == t0 || dst == t1 || t0 == t1 || dst > XMM6 || t0 > XMM6 || t1 > XMM6 {
		panic("jit: emitExpSSE wants three distinct registers among XMM0..XMM6")
	}
	// Clamp from below so the exponent build cannot go denormal, and from
	// above so it cannot overflow the exponent field (elem_const.go).
	bcastSS(a, t0, At(RBX, 32))
	a.MAXPS(dst, dst, t0)
	bcastSS(a, t0, At(RBX, 40))
	a.MINPS(dst, dst, t0)

	// n = round(x * log2e), as an integer (t1) and as a float (t0).
	a.MULPS(t0, dst, XMM9)
	a.CVTPS2DQ(t1, t0)
	a.CVTDQ2PS(t0, t1)

	// r = x - n*ln2, in two parts, each a product then a sum.
	a.MULPS(t1, t0, XMM10)
	a.ADDPS(dst, dst, t1)
	a.MULPS(t1, t0, XMM11)
	a.ADDPS(dst, dst, t1)

	// scale = 2^n, built in the exponent field: (n + 127) << 23. n is
	// converted back from its float form, which is an exact integer.
	a.CVTPS2DQ(t1, t0)
	a.PADDD(t1, t1, XMM7)
	a.PSLLD(t1, t1, 23)

	// exp(r) by Horner: ((((c5*r + c4)*r + c3)*r + c2)*r + 1)*r + 1.
	a.MOVAPS(t0, XMM12)
	for _, c := range []Reg{XMM13, XMM14, XMM15, XMM8, XMM8} {
		a.MULPS(t0, t0, dst)
		a.ADDPS(t0, t0, c)
	}
	a.MULPS(dst, t0, t1)
}

// loadHalfMagicSSE puts 2^112 (0x77800000) in every lane of x, the multiplier
// halfToFloatSSE rebiases an f16 exponent with. gp is clobbered.
func loadHalfMagicSSE(a *Buf, x, gp Reg) { bcastImm32(a, x, gp, 0x77800000) }

// halfToFloatSSE converts four binary16 values to float32 in software: the
// replacement for VCVTPH2PS on a host with no F16C.
//
// src holds one f16 in the LOW 16 bits of each dword (the upper 16 bits are
// ignored); dst receives the four floats. t0 and t1 are clobbered; magic holds
// loadHalfMagicSSE's 2^112. src is READ AGAIN at the end for the sign, so it
// must not be dst, t0 or t1.
//
//	t    = (h & 0x7FFF) << 13          exponent and mantissa in f32 position
//	f    = float(t) * 2^112            rebias; an f16 subnormal is an f32
//	                                   denormal here and the product is exact
//	f   |= 0x7F800000 where e == 31    Inf and NaN keep their mantissa
//	f   |= (h & 0x8000) << 16          the sign
//
// It is exact for every input, subnormals included, as long as DAZ is off:
// the multiply's input is an f32 denormal for an f16 subnormal. Go leaves
// MXCSR at its default and the trampoline does not touch it; MXFP4's scales
// reach f16 subnormals, and the gate runs all 65536 halves. The one difference
// from F16C: VCVTPH2PS quiets a signalling NaN and this does not.
//
// No constant block is read: the masks are made by shifts, the Inf/NaN test
// by comparing the exponent against all-ones (PCMPEQD t1,t1), and only 2^112
// needs a register.
func halfToFloatSSE(a *Buf, dst, src, t0, t1, magic Reg) {
	if src == dst || src == t0 || src == t1 || dst == t0 || dst == t1 || t0 == t1 ||
		magic == dst || magic == src || magic == t0 || magic == t1 {
		panic("jit: halfToFloatSSE wants distinct dst, src, t0, t1 and magic")
	}
	a.PSLLD(dst, src, 17) // drop the sign and anything above bit 15
	a.PSRLD(dst, dst, 4)  // (h & 0x7FFF) << 13
	a.PSLLD(t0, dst, 4)   // exponent to the top five bits
	a.PSRAD(t0, t0, 27)   // -1 exactly when all five are set (Inf/NaN)
	a.PCMPEQD(t1, t1, t1) // all ones
	a.PCMPEQD(t0, t0, t1) // mask where e == 31
	a.PSRLD(t0, t0, 24)
	a.PSLLD(t0, t0, 23) // 0x7F800000 where e == 31
	a.MULPS(dst, dst, magic)
	a.ORPS(dst, dst, t0)
	a.PSRLD(t0, src, 15) // bit 15 to bit 0 (and anything above to bits 1+)
	a.PSLLD(t0, t0, 31)  // only bit 0 survives, at the sign position
	a.ORPS(dst, dst, t0)
}

// loadU16x4SSE loads four 16-bit values from m (8 bytes) and zero-extends
// each into its own dword -- the shape halfToFloatSSE takes. zero must hold
// zero. It is VCVTPH2PS's load half on a host without F16C; one element at the
// end of a row is PXOR then PINSRWLoad instead, which reads exactly two bytes.
func loadU16x4SSE(a *Buf, dst Reg, m Mem, zero Reg) {
	a.MOVQLoad(dst, m)
	a.PUNPCKLWD(dst, dst, zero)
}

// bf16ToFloatSSE widens the LOW four bfloat16 words of src to four float32 in
// dst: a bfloat16 is the top half of the float32 it abbreviates, so
// interleaving a zero word below each is the whole conversion, and it is exact
// by construction. bf16HiToFloatSSE does the high four words. dst must not be
// src (dst is zeroed first).
func bf16ToFloatSSE(a *Buf, dst, src Reg) {
	a.PXOR(dst, dst, dst)
	a.PUNPCKLWD(dst, dst, src)
}

func bf16HiToFloatSSE(a *Buf, dst, src Reg) {
	a.PXOR(dst, dst, dst)
	a.PUNPCKHWD(dst, dst, src)
}

// hsum4SSE leaves the sum of x's four lanes in every lane of x.
func hsum4SSE(a *Buf, x Reg) {
	a.HADDPS(x, x, x)
	a.HADDPS(x, x, x)
}

// hsum8SSE sums an 8-float unit held as two halves and leaves the total in
// every lane of lo; hi is not modified.
//
// It is the AVX2 fold, operation for operation (VEXTRACTF128, VADDPSx, then
// VHADDPSx twice), so mirrored partial sums give a bit-identical total.
func hsum8SSE(a *Buf, lo, hi Reg) {
	a.ADDPS(lo, lo, hi)
	hsum4SSE(a, lo)
}

// hmax4SSE leaves the maximum of x's four lanes in every lane of x, clobbering
// t: two in-register folds (SHUFPS 0x4E, then 0xB1) and a broadcast of lane 0.
func hmax4SSE(a *Buf, x, t Reg) {
	a.SHUFPS(t, x, x, 0x4E)
	a.MAXPS(x, x, t)
	a.SHUFPS(t, x, x, 0xB1)
	a.MAXPS(x, x, t)
	a.SHUFPS(x, x, x, 0)
}

// hmax8SSE is the softmax's maximum over two halves, into every lane of lo;
// hi is not modified and t is clobbered. It mirrors EmitSoftmax's fold --
// VMAXPS(lo, lo, hi) across the 128-bit lanes, then the two in-lane folds with
// the same operand order, then the broadcast -- because MAXPS returns its
// second operand on a NaN or a signed-zero tie, and a fold in a different
// order is a different answer there.
func hmax8SSE(a *Buf, lo, hi, t Reg) {
	a.MAXPS(lo, lo, hi)
	hmax4SSE(a, lo, t)
}
