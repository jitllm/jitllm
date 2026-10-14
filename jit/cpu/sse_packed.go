//go:build amd64

package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// The SSE tier's packed matvec: the 64-row tile, the 8-row tail and the fused
// kernel, over the jlm container's device layout, in legacy 128-bit SSE.
//
// Each kernel takes exactly the Args its AVX2 twin takes (packed.go,
// packedfused.go). The integer dot is the pre-VNNI sequence at 128 bits
// (PMADDUBSW + PMADDWD + PADDD, clobbering the unsigned operand), so the int32
// accumulators are bit-identical to the AVX2 kernel's for every format
// PreVNNIPacked admits. The f32 epilogue is the AVX2 kernel's in the same
// order, except that there is no FMA, so out += dot*scale rounds the product
// first; ssepacked_test.go holds the tiers to that.
//
// Four rows per register changes the register plans, not the loops:
//   - A 16-row fused group is four XMM accumulators. The hoisted activation
//     broadcasts of the AVX2 fused kernel do not fit beside them, so the
//     broadcast is re-issued per word inside the row loop (MOVD + PSHUFD).
//     The ones vector PMADDWD needs gets a register of its own.
//   - The 64-row tile is walked as chunks of 16 rows (8 for the tail), each
//     the AVX2 tile loop in miniature. A row's f32 accumulation order is
//     unchanged.
//   - There is no F16C. The fused kernel precomputes d (and dmin) as f32 into
//     Scratch per super-block with halfToFloatSSE, exact for every half. The
//     tile kernel has no Scratch, so it uses the 128-byte red zone below RSP
//     (the trampoline makes no call and moves no RSP while a kernel runs).
//
// Every vector load is unaligned (MOVDQU/MOVUPS/MOVQ/MOVD/MOVSS): payload spans
// start at any row and PackedScratch has no alignment promise. Its constants
// are lane-uniform, so the block is shared with the AVX2 tier unchanged.
//
// There is no DotKind here: any third value handed to an AVX2 emitter would
// silently emit VPDPBUSD, so the SSE emitters have exactly one dot (sseDot).

// SupportedPackedSSE reports whether the SSE tier has the packed kernels for
// t: every format with a device layout that PMADDUBSW cannot saturate on --
// PreVNNIPacked's bound, which is the same instruction's bound at either
// width, so it is asked rather than re-derived.
func SupportedPackedSSE(t quant.Type) bool {
	return PackedSupported(t) && PreVNNIPacked(t) == nil
}

// ssePackedFormat is one format's layout parameters, read once from kernels.
type ssePackedFormat struct {
	t                 quant.Type
	sub, bits, hi     int
	biasK             float32
	biasArray, signed bool
	perSuper, scOff   int
	codes, narrow     bool
	e8m0              bool
	// minD: the minimum is the d word's high half alone and m is one --
	// Q5_1, which has no sc plane (kernels.MinInD).
	minD            bool
	pw, hw, halfSub int
	// fields codes a payload byte, a field-run (sub/fields) apart; see
	// kernels.Fields.
	fields, run     int
	perWord, stride int
	// stream: container v27's SC stream (kernels.ScStream, scstream.go).
	stream bool
	// d32: the d word is the scale as an f32 (TQ2_F32, kernels.D32).
	d32 bool
}

func newSSEPackedFormat(t quant.Type, who string) (ssePackedFormat, error) {
	q, ok := kernels.QuantOf(t)
	if !ok {
		return ssePackedFormat{}, fmt.Errorf("jit: %s: %s has no device layout", who, t)
	}
	// The saturation refusal, before anything is emitted: PMADDUBSW saturates
	// exactly as VPMADDUBSW does, so PreVNNIPacked's answer applies.
	if err := PreVNNIPacked(t); err != nil {
		return ssePackedFormat{}, err
	}
	codes, err := hasCodes(q)
	if err != nil {
		return ssePackedFormat{}, err
	}
	f := ssePackedFormat{t: t, codes: codes, narrow: NarrowD(t), e8m0: E8M0D(t), stream: kernels.ScStream(q),
		d32: kernels.D32(q)}
	f.sub, f.bits, f.biasK, f.biasArray = kernels.Layout(q)
	f.perSuper, f.scOff = kernels.ScaleLayout(q)
	f.signed = kernels.SignedPayload(q)
	f.minD = kernels.MinInD(q)
	f.hi = kernels.HiPlane(q)
	f.pw, f.hw, f.halfSub = f.sub*f.bits/32, f.sub*f.hi/32, f.sub/2
	f.perWord, f.stride = perWordOf(f.biasArray), strideOf(f.biasArray)
	if f.bits != 2 && f.bits != 4 && f.bits != 8 {
		return ssePackedFormat{}, fmt.Errorf("jit: %s: %s has a %d-bit primary plane", who, t, f.bits)
	}
	f.fields = 8 / f.bits
	f.run = f.sub / f.fields
	return f, nil
}

// hasBias reports whether the epilogue subtracts a bias for this format.
func (f ssePackedFormat) hasBias() bool {
	return f.biasArray || f.sub == Q8Block/2 || f.biasK != 0 || f.signed
}

// sseDot is dotEmitter's DotVEX arm at 128 bits: "acc += the four-byte dot
// products of an unsigned payload against a signed activation". ones must hold
// the i16 all-ones vector PMADDWD needs.
type sseDot struct{ ones Reg }

// unsigned clobbers u, exactly as dotEmitter.unsigned's VEX sequence does.
func (d sseDot) unsigned(a *Buf, acc, u, act Reg) {
	a.PMADDUBSW(u, u, act)
	a.PMADDWD(u, u, d.ones)
	a.PADDD(acc, acc, u)
}

// signed is dotEmitter.signed's pre-VNNI arm: PSIGNB moves the weight's sign
// onto the activation, so the pair stays at 2*128*127 = 32,512 where the +128
// centring would reach 64,770 and saturate. It clobbers pay and t.
func (d sseDot) signed(a *Buf, acc, pay, act, t Reg) {
	if d.naiveSigned(a, acc, pay, act, t) {
		return
	}
	a.PSIGNB(t, act, pay) // a * sign(w)
	a.PSIGNB(pay, pay, pay)
	a.PMADDUBSW(pay, pay, t) // |w| x (a*sign(w))
	a.PMADDWD(pay, pay, d.ones)
	a.PADDD(acc, acc, pay)
}

// signedFix is dotEmitter.signedFix: it puts a signed accumulator back on
// VPDPBUSD's dot(w+128, a) scale, so the epilogue -- and its -128*sum(a)
// correction -- is the AVX2 kernel's. negSum is pairs[2b+1] = -16*sum(a),
// exact in f32. It clobbers t.
func (d sseDot) signedFix(a *Buf, acc []Reg, negSum Mem, t Reg) {
	if d.naiveActive() {
		return // the naive form already accumulated the centred dot
	}
	bcastSS(a, t, negSum)
	a.CVTPS2DQ(t, t)
	a.PSLLD(t, t, 3)
	for _, r := range acc {
		a.PSUBD(r, r, t)
	}
}

// sseOnes16 is ones16 in legacy form: PCMPEQW against itself, then a logical
// shift, and no memory.
func sseOnes16(a *Buf, dst Reg) {
	a.PCMPEQW(dst, dst, dst)
	a.PSRLW(dst, dst, 15)
}

// sseUnpack is the per-row-group payload arithmetic both packed kernels share:
// given one payload word (four rows) in pay, accumulate its dot against the
// broadcast activations into acc. It owns no loop and no address.
type sseUnpack struct {
	f                          ssePackedFormat
	dot                        sseDot
	mask, tmp, hiMask, lut, lk Reg
}

// code translates the primary codes in x through the format's table, when it
// has one, and returns the register holding what the dot takes. PSHUFB's
// legacy destination IS the table, so the table is copied into lk first.
func (u sseUnpack) code(a *Buf, x Reg) Reg {
	if !u.f.codes {
		return x
	}
	a.PSHUFB(u.lk, u.lut, x)
	return u.lk
}

// primary accumulates one primary word: for a 4-bit plane the low nibbles
// against actL and the high ones (elements half a sub-block on) against actH;
// for a 2-bit plane fields f0 and f0+1 (a field-run apart) the same way; for
// an 8-bit plane the byte against actL. pay is clobbered only on the signed
// 8-bit path.
func (u sseUnpack) primary(a *Buf, acc, pay, actL, actH Reg, f0 int) {
	switch {
	case u.f.bits < 8:
		for i, act := range []Reg{actL, actH} {
			if sh := u.f.bits * (f0 + i); sh == 0 {
				a.PAND(u.tmp, pay, u.mask)
			} else {
				a.PSRLD(u.tmp, pay, byte(sh))
				a.PAND(u.tmp, u.tmp, u.mask)
			}
			u.dot.unsigned(a, acc, u.code(a, u.tmp), act)
		}
	case u.f.signed:
		u.dot.signed(a, acc, pay, actL, u.tmp)
	default:
		u.dot.unsigned(a, acc, pay, actL)
	}
}

// secondary accumulates primary-word-slot gi's secondary codes out of the hi
// word in pay. The mask carries the <<4 already (PackedScratch), so the shift
// is relative to bit 4 and can go either way -- the AVX2 kernel's extraction.
func (u sseUnpack) secondary(a *Buf, acc, pay, act Reg, gi int) {
	lanes := 8 / u.f.hi
	switch sh := u.f.hi * (gi % lanes); {
	case sh < 4:
		a.PSLLD(u.tmp, pay, byte(4-sh))
		a.PAND(u.tmp, u.tmp, u.hiMask)
	case sh == 4:
		a.PAND(u.tmp, pay, u.hiMask)
	default:
		a.PSRLD(u.tmp, pay, byte(sh-4))
		a.PAND(u.tmp, u.tmp, u.hiMask)
	}
	u.dot.unsigned(a, acc, u.tmp, act)
}

// sseRegs hands out XMM registers in order and refuses the seventeenth.
type sseRegs struct {
	next Reg
	who  string
	t    quant.Type
}

func (r *sseRegs) take() Reg {
	if r.next > XMM15 {
		panic(fmt.Sprintf("jit: %s: %s needs more than 16 XMM registers", r.who, r.t))
	}
	x := r.next
	r.next++
	return x
}

// loadPackedConsts loads the constants a packed kernel keeps for its whole
// run from the PackedScratch block at base: the nibble mask (or, for an 8-bit
// plane, the ones vector in its place), the ones vector, -scOff, the
// pre-shifted secondary mask and the code table.
func loadPackedConsts(a *Buf, f ssePackedFormat, base Reg, u sseUnpack) {
	if f.bits < 8 {
		a.MOVDQULoad(u.mask, At(base, 0))
		if f.bits == 2 {
			// 0x03 a byte: mask & (mask >> 2), the dword shift's spill across
			// bytes cleared by the AND (fieldMask).
			a.PSRLD(u.tmp, u.mask, 2)
			a.PAND(u.mask, u.mask, u.tmp)
		}
		sseOnes16(a, u.dot.ones)
	} else {
		// No nibble mask and no 0x80 sign flip on this path (PSIGNB replaces
		// the centring), so mask IS the ones vector.
		sseOnes16(a, u.mask)
	}
	if f.hi != 0 {
		a.MOVDQULoad(u.hiMask, At(base, hiMaskSlot(f.hi)))
	}
	if f.codes {
		a.MOVDQULoad(u.lut, At(base, codesSlot))
	}
}

// halfPlane converts four rows' d (and, with dmin, their minimums) into f32 and
// stores them at dOut (and mOut). src addresses the four rows of the d plane:
// a four-byte word per row on the wide plane (d low, dmin high), a two-byte
// half per row on the narrow one. pay, x, tmp, t0, t1 are clobbered; magic
// holds 2^112 and zero holds zero (narrow only).
//
// A subnormal f16 scale is exact but costs a microcode assist on Goldmont
// (the f32 denormal multiply in halfToFloatSSE); see
// docs/engineering-history/cpu-kernels.md. An assist-free conversion is about
// twice the instructions and is only worth it if real scales turn subnormal.
func halfPlane(a *Buf, f ssePackedFormat, src, dOut, mOut Mem, pay, x, tmp, t0, t1, magic, zero Reg) {
	if f.e8m0 {
		// MXFP4's super-scale is a biased E8M0 exponent: the byte shifted
		// left 23 is the f32, with no f16 convert and so no assist.
		a.PMOVZXBDLoad(pay, src)
		a.PSLLD(tmp, pay, 23)
		a.MOVUPSStore(dOut, tmp)
		return
	}
	if f.d32 {
		// TQ2_F32: the word is the scale, and nothing converts.
		a.MOVDQULoad(pay, src)
		a.MOVUPSStore(dOut, pay)
		return
	}
	if f.narrow {
		loadU16x4SSE(a, pay, src, zero)
	} else {
		a.MOVDQULoad(pay, src)
	}
	halfToFloatSSE(a, tmp, pay, t0, t1, magic) // the high half of each dword is ignored
	a.MOVUPSStore(dOut, tmp)
	if f.biasArray {
		a.PSRLD(x, pay, 16)
		halfToFloatSSE(a, tmp, x, t0, t1, magic)
		a.MOVUPSStore(mOut, tmp)
	}
}

// stepSC reports whether the SC cursor steps a plane stride after sub-block sn.
func (f ssePackedFormat) stepSC(sn int) bool {
	if f.stream {
		return scStep(sn)
	}
	return f.perSuper > 1 && sn%f.perWord == f.perWord-1
}

// scaleF32 is scaleByte's tail for a scale already extracted into dst.
func scaleF32(a *Buf, f ssePackedFormat, dst, scOffV Reg) {
	if f.scOff != 0 {
		a.PSUBD(dst, dst, scOffV)
	}
	a.CVTDQ2PS(dst, dst)
}

// scaleByte leaves this sub-block's integer scale byte, minus scOff, in dst
// as f32 -- (word << (24-sh)) >> 24 -- from the SC word in pay.
func scaleByte(a *Buf, f ssePackedFormat, dst, pay, scOffV Reg, sh byte) {
	if 24-sh > 0 {
		a.PSLLD(dst, pay, 24-sh)
	} else {
		a.MOVAPS(dst, pay)
	}
	a.PSRLD(dst, dst, 24)
	if f.scOff != 0 {
		a.PSUBD(dst, dst, scOffV)
	}
	a.CVTDQ2PS(dst, dst)
}

// EmitPackedMatVecSSE is EmitPackedMatVec for the SSE tier: out[r] = dot(row
// r, x) over the device layout, rows = PackedRows (64) or PackedTail (8) per
// TILE, with EmitPackedMatVec's Args exactly:
//
//	Out     [Rows*rows] f32, WRITTEN (the kernel zeroes each tile first)
//	W       the QS span at this call's first row
//	A       int8 activations;  AScale {d_a, -sum*BiasC/8} per 32
//	Rows    TILES this call serves
//	K       super-blocks per row (k / PackedOuterElems)
//	RowStr  the tensor's FULL row count x4 -- the layout's stride
//	Scr     PackedScratch;  AHalf the per-16 sums (Q3_K, Q6_K)
//	PD/PSC  the d and sc spans at this call's first row;  DStr the d plane's
//	        super-block stride (DSuperBytes)
//
// GP is the AVX2 tile kernel's: RCX out, RDX payload cursor, R8 d cursor, R9
// sc cursor (the narrow d plane's chunk offset where there is no sc plane),
// R10 half-sum cursor, RSI chunk byte offset, RBX constants, RAX k counter,
// R11 activation cursor, R12 pair cursor, R13 row stride, R15 chunk counter.
// R14, RBP and RSP are never written.
func EmitPackedMatVecSSE(t quant.Type, rows int) ([]byte, error) {
	const who = "EmitPackedMatVecSSE"
	if rows < 8 || rows%8 != 0 || rows > 64 {
		return nil, fmt.Errorf("jit: %s: %d rows per tile (want 8..64, a multiple of 8)", who, rows)
	}
	f, err := newSSEPackedFormat(t, who)
	if err != nil {
		return nil, err
	}
	chunk := 16
	if rows%chunk != 0 {
		chunk = 8
	}
	nchunk, G := rows/chunk, chunk/4

	r := sseRegs{who: who, t: t}
	var u sseUnpack
	u.f = f
	u.mask = r.take()
	corr := r.take()
	if f.bits < 8 {
		u.dot.ones = r.take()
	} else {
		u.dot.ones = u.mask
	}
	iac := make([]Reg, G)
	for g := range iac {
		iac[g] = r.take()
	}
	pay, tmp, actL, actH, t0, t1 := r.take(), r.take(), r.take(), r.take(), r.take(), r.take()
	u.tmp = tmp
	scOffV := XMM0
	if f.scOff != 0 {
		scOffV = r.take()
	}
	if f.hi != 0 {
		u.hiMask = r.take()
	}
	if f.codes {
		u.lut, u.lk = r.take(), r.take()
	}

	// The chunk's d and dmin live in the red zone: sixteen f32 each, all 128
	// bytes of it. Nothing below RSP is touched otherwise.
	const dOff, mOff = -128, -64

	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(R15, At(RDI, 32)) // Rows -> tiles, then chunks
	if nchunk > 1 {
		a.MOVimm32(RAX, int32(nchunk))
		a.IMUL(R15, RAX)
	}
	a.MOVLoad(R13, At(RDI, 48)) // RowStr, the layout's row stride in bytes
	a.MOVLoad(RBX, At(RDI, 56)) // Scr
	done := a.Label()
	a.TESTQ(R15, R15)
	a.JZ(done)
	a.SUBQ(RSI, RSI) // chunk byte offset into the payload and wide spans
	if f.narrow {
		// R9 is the sc cursor for every other format and there is no sc plane
		// here, so it carries the narrow d plane's own chunk offset.
		a.SUBQ(R9, R9)
	}
	loadPackedConsts(&a, f, RBX, u)
	switch {
	case f.biasArray:
		a.MOVUPSLoad(corr, At(RBX, 128)) // -8
	case f.sub == Q8Block/2:
		a.MOVUPSLoad(corr, At(RBX, 288)) // +4
	default:
		a.MOVUPSLoad(corr, At(RBX, 160)) // +8
	}
	if f.scOff != 0 {
		a.MOVDQULoad(scOffV, At(RBX, 224+int32(scOffSlot(f.scOff))))
	}

	chunkL := a.Label()
	a.Bind(chunkL)
	a.MOVLoad(RDX, At(RDI, 8))
	a.ADDQ(RDX, RSI)
	a.MOVLoad(R8, At(RDI, 144))
	if f.narrow {
		a.ADDQ(R8, R9)
	} else {
		a.MOVLoad(R9, At(RDI, 152))
		a.ADDQ(R8, RSI)
		a.ADDQ(R9, RSI)
	}
	a.MOVLoad(R11, At(RDI, 16))
	a.MOVLoad(R12, At(RDI, 24))
	a.MOVLoad(R10, At(RDI, 64))
	a.MOVLoad(RAX, At(RDI, 40))
	// Out holds the running sums, so it starts at zero rather than being
	// written once at the end.
	a.PXOR(t0, t0, t0)
	for g := 0; g < G; g++ {
		a.MOVUPSStore(At(RCX, int32(16*g)), t0)
	}
	next := a.Label()
	a.TESTQ(RAX, RAX)
	a.JZ(next)

	outer := a.Label()
	a.Bind(outer)

	// --- this chunk's d (and dmin) as f32, once per super-block ------------
	// RBX lends itself for the constant and is given back: the chunk loop
	// reads nothing else through it until the next super-block.
	bcastImm32(&a, actL, RBX, 0x77800000) // 2^112, halfToFloatSSE's rebias
	a.MOVLoad(RBX, At(RDI, 56))
	if f.narrow {
		a.PXOR(actH, actH, actH)
	}
	for g := 0; g < G; g++ {
		src := At(R8, int32(16*g))
		if f.narrow {
			// Four rows at DRowBytes each: eight bytes of f16 halves, or four
			// of MXFP4's E8M0 exponents.
			src = At(R8, int32(4*g*DRowBytes(t)))
		}
		halfPlane(&a, f, src, At(RSP, dOff+int32(16*g)), At(RSP, mOff+int32(16*g)),
			pay, actH, tmp, t0, t1, actL, actH)
	}

	for sn := 0; sn < f.perSuper; sn++ {
		pairOff := int32(f.sub * sn / Q8Block * 8)
		for g := 0; g < G; g++ {
			a.PXOR(iac[g], iac[g], iac[g])
		}
		for w := 0; w < f.pw; w++ {
			// One prefetch per stride step, as the AVX2 tile does: what the
			// hardware prefetcher cannot see is the jump of nrows*4.
			a.PREFETCHT0(Idx(RDX, R13, 8, 0))
			// Two fields at a time, each against the activations a field-run
			// on: one pair for a 4-bit plane, two for a 2-bit one.
			for f0 := 0; f0 < f.fields; f0 += 2 {
				bcastD(&a, actL, At(R11, int32(f.sub*sn+f0*f.run+4*w)))
				if f.bits < 8 {
					bcastD(&a, actH, At(R11, int32(f.sub*sn+(f0+1)*f.run+4*w)))
				}
				for g := 0; g < G; g++ {
					a.MOVDQULoad(pay, At(RDX, int32(16*g)))
					u.primary(&a, iac[g], pay, actL, actH, f0)
				}
			}
			a.ADDQ(RDX, R13)
		}
		// The secondary plane: RDX has walked the primary words and points at
		// hi word 0.
		for hwi := 0; hwi < f.hw; hwi++ {
			lanes := 8 / f.hi
			for half := 0; half < 2; half++ {
				for w := 0; w < f.pw; w++ {
					gi := w + half*f.pw
					if gi/lanes != hwi {
						continue
					}
					bcastD(&a, actL, At(R11, int32(f.sub*sn+half*f.halfSub+4*w)))
					for g := 0; g < G; g++ {
						a.MOVDQULoad(pay, At(RDX, int32(16*g)))
						u.secondary(&a, iac[g], pay, actL, gi)
					}
				}
			}
			a.ADDQ(RDX, R13)
		}
		if f.signed {
			u.dot.signedFix(&a, iac, At(R12, pairOff+4), t1)
		}

		// --- the epilogue: the AVX2 tile's, operation for operation --------
		sh := byte((sn % f.perWord) * f.stride)
		for g := 0; g < G; g++ {
			off := int32(16 * g)
			a.MOVUPSLoad(t1, At(RSP, dOff+off)) // scale = d
			nextSC := func(r Reg) { a.MOVDQULoad(r, Idx(R9, R13, 1, off)) }
			if f.stream {
				// actH is free until Out is read below.
				a.MOVDQULoad(pay, At(R9, off))
				scField(sseSC(&a), tmp, pay, actH, sn, false, nextSC)
				scaleF32(&a, f, tmp, scOffV)
				a.MULPS(t1, t1, tmp)
			} else if f.perSuper > 1 {
				a.MOVDQULoad(pay, At(R9, off))
				scaleByte(&a, f, tmp, pay, scOffV, sh)
				a.MULPS(t1, t1, tmp)
			}
			bcastSS(&a, actL, At(R12, pairOff)) // d_a
			a.MULPS(t1, t1, actL)
			a.CVTDQ2PS(tmp, iac[g])
			a.MOVUPSLoad(actH, At(RCX, off))
			// out += dot * scale: the AVX2 kernel's VFMADD231PS, unfused.
			a.MULPS(tmp, tmp, t1)
			a.ADDPS(actH, actH, tmp)
			switch {
			case f.biasArray:
				// bias = dmin*m, and it carries d_a like the scale does.
				a.MOVUPSLoad(tmp, At(RSP, mOff+off))
				if f.stream {
					// pay is dead once read: the straddle's scratch.
					scField(sseSC(&a), t0, pay, pay, sn, true, nextSC)
					a.CVTDQ2PS(t0, t0)
					a.MULPS(tmp, tmp, t0)
				} else if !f.minD {
					a.PSLLD(t0, pay, 16-sh)
					a.PSRLD(t0, t0, 24)
					a.CVTDQ2PS(t0, t0)
					a.MULPS(tmp, tmp, t0)
				}
				a.MULPS(tmp, tmp, actL) // * d_a
				bcastSS(&a, t0, At(R12, pairOff+4))
				a.MULPS(tmp, tmp, t0)
				a.MULPS(tmp, tmp, corr)
				a.SUBPS(actH, actH, tmp)
			case f.sub == Q8Block/2:
				bcastSS(&a, tmp, At(R10, int32(4*sn)))
				a.MULPS(tmp, tmp, t1)
				a.MULPS(tmp, tmp, corr)
				a.ADDPS(actH, actH, tmp)
			case f.biasK != 0 || f.signed:
				bcastSS(&a, tmp, At(R12, pairOff+4))
				a.MULPS(tmp, tmp, t1)
				a.MULPS(tmp, tmp, corr)
				a.ADDPS(actH, actH, tmp)
			}
			a.MOVUPSStore(At(RCX, off), actH)
		}
		if f.stepSC(sn) {
			a.ADDQ(R9, R13)
		}
	}
	a.ADDimm(R11, int32(f.perSuper*f.sub))
	a.ADDimm(R12, int32(f.perSuper*f.sub/Q8Block*8))
	if f.sub == Q8Block/2 {
		a.ADDimm(R10, int32(f.perSuper*4))
	}
	if f.narrow {
		// Half the payload's stride, rounded up to a whole word; DStr carries
		// it (DSuperBytes).
		a.ADDQmem(R8, At(RDI, 160))
	} else {
		a.ADDQ(R8, R13)
	}
	a.DEC(RAX)
	a.JNZ(outer)

	a.Bind(next)
	a.ADDimm(RCX, int32(4*chunk))
	a.ADDimm(RSI, int32(4*chunk))
	if f.narrow {
		a.ADDimm(R9, int32(chunk*DRowBytes(t)))
	}
	a.DEC(R15)
	a.JNZ(chunkL)

	a.Bind(done)
	a.RET()
	return a.Bytes(), nil
}

// EmitPackedMatVecFusedSSE is EmitPackedMatVecFused for the SSE tier, with its
// Args exactly:
//
//	Out      [Rows*group] f32, ACCUMULATED into; the caller zeroes it
//	W/PD/PSC the three spans, at this call's first row
//	Rows     how many PackedFusedGroupOf(t)-row groups this call serves
//	RowStr   the layout's row stride in bytes;  DStr the d plane's
//	Scratch  [rows] f32, this call's precomputed super-block scale
//	Q32      [rows] f32, its minimum -- only for a format that has one
//
// The group is PackedFusedGroupOf(t) -- shared with the AVX2 tier, because nn
// divides by it -- which is four XMM accumulators at 16 rows and two at 8.
// GP is the AVX2 fused kernel's: R10 is AHalf or the minimum scratch (never
// both), RBX is the row counter inside rowLoop and the constant base outside
// it, RSI the payload cursor and then the scratch base.
func EmitPackedMatVecFusedSSE(t quant.Type) ([]byte, error) {
	const who = "EmitPackedMatVecFusedSSE"
	f, err := newSSEPackedFormat(t, who)
	if err != nil {
		return nil, err
	}
	group := PackedFusedGroupOf(t)
	if group%4 != 0 {
		return nil, fmt.Errorf("jit: %s: %s's group of %d rows is not whole XMM registers", who, t, group)
	}
	grp := group / 4

	r := sseRegs{who: who, t: t}
	var u sseUnpack
	u.f = f
	u.mask = r.take()
	da, bscale := r.take(), r.take()
	if f.bits < 8 {
		u.dot.ones = r.take()
	} else {
		u.dot.ones = u.mask
	}
	iac := make([]Reg, grp)
	for g := range iac {
		iac[g] = r.take()
	}
	actL, actH, pay, tmp, t2 := r.take(), r.take(), r.take(), r.take(), r.take()
	u.tmp = tmp
	scOffV := XMM0
	if f.scOff != 0 {
		scOffV = r.take()
	}
	if f.hi != 0 {
		u.hiMask = r.take()
	}
	if f.codes {
		u.lut, u.lk = r.take(), r.take()
	}

	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(R11, At(RDI, 16)) // A
	a.MOVLoad(R12, At(RDI, 24)) // AScale
	a.MOVLoad(R13, At(RDI, 48)) // RowStr
	a.MOVLoad(RBX, At(RDI, 56)) // Scr, for the constants below
	if f.biasArray {
		a.MOVLoad(R10, At(RDI, 112)) // Q32 -> the minimum scratch
	} else {
		a.MOVLoad(R10, At(RDI, 64)) // AHalf
	}
	loadPackedConsts(&a, f, RBX, u)
	if f.scOff != 0 {
		a.MOVDQULoad(scOffV, At(RBX, 224+int32(scOffSlot(f.scOff))))
	}
	a.MOVLoad(RDX, At(RDI, 8)) // QS
	if !f.narrow {
		a.MOVLoad(R9, At(RDI, 152)) // SC
	}
	a.MOVLoad(R8, At(RDI, 144)) // D, held for the whole kernel
	// No groups or no k is no work, not 2^64 iterations of a DEC/JNZ loop.
	done := a.Label()
	a.MOVLoad(RAX, At(RDI, 32))
	a.TESTQ(RAX, RAX)
	a.JZ(done)
	a.MOVLoad(RAX, At(RDI, 40)) // K, in super-blocks
	a.TESTQ(RAX, RAX)
	a.JZ(done)

	rowLoop := func(body func()) {
		a.MOVLoad(RBX, At(RDI, 32))
		a.SUBQ(R15, R15)
		l := a.Label()
		a.Bind(l)
		body()
		a.ADDimm(R15, int32(group*4))
		a.DEC(RBX)
		a.JNZ(l)
	}

	outer := a.Label()
	a.Bind(outer)

	// --- precompute this super-block's scales, once ------------------------
	// Every per-sub-block register is free here, so the software half
	// conversion costs nothing it would otherwise need: magic in da, zero (or
	// the shifted minimum) in bscale.
	bcastImm32(&a, da, RBX, 0x77800000) // 2^112; rowLoop reloads RBX
	a.MOVLoad(RSI, At(RDI, 72))         // Scratch
	if f.narrow {
		a.SUBQ(R9, R9)
		a.PXOR(bscale, bscale, bscale)
	}
	rowLoop(func() {
		for g := 0; g < grp; g++ {
			off := int32(16 * g)
			src := Idx(R8, R15, 1, off)
			if f.narrow {
				// off is four bytes a row on the wide plane, so the narrow
				// one's offset scales with DRowBytes: half for an f16 half,
				// a quarter for MXFP4's E8M0 byte.
				src = Idx(R8, R9, 1, off*int32(DRowBytes(t))/4)
			}
			halfPlane(&a, f, src, Idx(RSI, R15, 1, off), Idx(R10, R15, 1, off),
				pay, bscale, tmp, actL, actH, da, bscale)
		}
		if f.narrow {
			a.ADDimm(R9, int32(group*DRowBytes(t)))
		}
	})
	if f.narrow {
		a.ADDQmem(R8, At(RDI, 160)) // half the payload's stride; see DSuperBytes
	} else {
		a.ADDQ(R8, R13) // D advances one super-block
	}

	// --- the sub-blocks ----------------------------------------------------
	for sn := 0; sn < f.perSuper; sn++ {
		pairOff := int32(f.sub * sn / Q8Block * 8)
		sh := byte((sn % f.perWord) * f.stride)
		bcastSS(&a, da, At(R12, pairOff))
		a.MOVLoad(RBX, At(RDI, 56)) // rowLoop clobbered it
		switch {
		case f.biasArray:
			bcastSS(&a, bscale, At(R12, pairOff+4))
			a.MOVUPSLoad(tmp, At(RBX, 128)) // -8
			a.MULPS(bscale, bscale, tmp)
			a.MULPS(bscale, bscale, da)
		case f.sub == Q8Block/2:
			bcastSS(&a, bscale, At(R10, int32(4*sn)))
			a.MOVUPSLoad(tmp, At(RBX, 288)) // +4
			a.MULPS(bscale, bscale, tmp)
		case f.biasK != 0 || f.signed:
			bcastSS(&a, bscale, At(R12, pairOff+4))
			a.MOVUPSLoad(tmp, At(RBX, 160)) // +8
			a.MULPS(bscale, bscale, tmp)
		}
		rowLoop(func() {
			a.MOVQ(RSI, RDX) // the payload cursor, reset every row group
			for g := 0; g < grp; g++ {
				a.PXOR(iac[g], iac[g], iac[g])
			}
			for w := 0; w < f.pw; w++ {
				// Re-broadcast per word: four accumulators and the hoisted
				// activations of every word do not fit in sixteen XMM.
				for f0 := 0; f0 < f.fields; f0 += 2 {
					bcastD(&a, actL, At(R11, int32(f.sub*sn+f0*f.run+4*w)))
					if f.bits < 8 {
						bcastD(&a, actH, At(R11, int32(f.sub*sn+(f0+1)*f.run+4*w)))
					}
					for g := 0; g < grp; g++ {
						a.MOVDQULoad(pay, Idx(RSI, R15, 1, int32(16*g)))
						u.primary(&a, iac[g], pay, actL, actH, f0)
					}
				}
				a.ADDQ(RSI, R13)
			}
			for hwi := 0; hwi < f.hw; hwi++ {
				lanes := 8 / f.hi
				for half := 0; half < 2; half++ {
					for w := 0; w < f.pw; w++ {
						gi := w + half*f.pw
						if gi/lanes != hwi {
							continue
						}
						bcastD(&a, actL, At(R11, int32(f.sub*sn+half*f.halfSub+4*w)))
						for g := 0; g < grp; g++ {
							a.MOVDQULoad(pay, Idx(RSI, R15, 1, int32(16*g)))
							u.secondary(&a, iac[g], pay, actL, gi)
						}
					}
				}
				a.ADDQ(RSI, R13)
			}
			if f.signed {
				u.dot.signedFix(&a, iac, At(R12, pairOff+4), t2)
			}
			a.MOVLoad(RSI, At(RDI, 72)) // the scratch base, for the epilogue
			for g := 0; g < grp; g++ {
				off := int32(16 * g)
				// A straddling SC field's next word: the cursor steps a plane
				// stride and back, no register being free for the address.
				nextSC := func(r Reg) {
					a.ADDQ(R9, R13)
					a.MOVDQULoad(r, Idx(R9, R15, 1, off))
					a.SUBQ(R9, R13)
				}
				if f.stream {
					// The scale into t2 before d takes tmp, which a straddling
					// field borrows.
					a.MOVDQULoad(pay, Idx(R9, R15, 1, off))
					scField(sseSC(&a), t2, pay, tmp, sn, false, nextSC)
					scaleF32(&a, f, t2, scOffV)
					a.MOVUPSLoad(tmp, Idx(RSI, R15, 1, off)) // d, precomputed f32
					a.MULPS(tmp, tmp, t2)
				} else {
					a.MOVUPSLoad(tmp, Idx(RSI, R15, 1, off)) // d, precomputed f32
					if f.perSuper > 1 {
						a.MOVDQULoad(pay, Idx(R9, R15, 1, off)) // the scale word
						scaleByte(&a, f, t2, pay, scOffV, sh)
						a.MULPS(tmp, tmp, t2)
					}
				}
				a.MULPS(tmp, tmp, da) // scale * d_a
				a.CVTDQ2PS(iac[g], iac[g])
				// out += dot * scale: the AVX2 kernel's VFMADD231PS, unfused.
				a.MULPS(iac[g], iac[g], tmp)
				a.MOVUPSLoad(t2, Idx(RCX, R15, 1, off))
				a.ADDPS(t2, t2, iac[g])
				switch {
				case f.biasArray:
					if f.stream {
						// The minimum into pay first: a straddling one borrows tmp.
						scField(sseSC(&a), pay, pay, tmp, sn, true, nextSC)
						a.CVTDQ2PS(pay, pay)
						a.MOVUPSLoad(tmp, Idx(R10, R15, 1, off)) // dmin, precomputed
						a.MULPS(tmp, tmp, pay)
					} else {
						a.MOVUPSLoad(tmp, Idx(R10, R15, 1, off)) // dmin, precomputed
					}
					if !f.stream && !f.minD {
						a.PSLLD(pay, pay, 16-sh)
						a.PSRLD(pay, pay, 24)
						a.CVTDQ2PS(pay, pay)
						a.MULPS(tmp, tmp, pay)
					}
					a.MULPS(tmp, tmp, bscale)
					a.SUBPS(t2, t2, tmp)
				case f.hasBias():
					a.MULPS(tmp, tmp, bscale)
					a.ADDPS(t2, t2, tmp)
				}
				a.MOVUPSStore(Idx(RCX, R15, 1, off), t2)
			}
		})
		for w := 0; w < f.pw+f.hw; w++ {
			a.ADDQ(RDX, R13)
		}
		if f.stepSC(sn) {
			a.ADDQ(R9, R13)
		}
	}
	a.ADDimm(R11, int32(f.perSuper*f.sub))
	a.ADDimm(R12, int32(f.perSuper*f.sub/Q8Block*8))
	if f.sub == Q8Block/2 {
		a.ADDimm(R10, int32(f.perSuper*4))
	}
	a.DEC(RAX)
	a.JNZ(outer)

	a.Bind(done)
	a.RET()
	return a.Bytes(), nil
}
