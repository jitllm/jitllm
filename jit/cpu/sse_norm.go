package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// The SSE tier's family 2: RMSNorm, LayerNorm, the rotary rotation, the
// packed embedding row and the half-precision widen. Each takes the AVX2
// twin's ABI unchanged (norm.go, layernorm.go, rope.go, rowpacked.go), so nn
// reaches them through EmittersFor(TierSSE).
//
// Bit-identity with AVX2 is decided by FMA, not width. LayerNorm, the packed
// row and the widen use no FMA on AVX2, so mirroring each YMM as two XMM
// halves folded in the same order (hsum8SSE) reproduces them bit for bit
// (ssenorm_test.go). RMSNorm's sum of squares and the NEOX rotation are
// fused on AVX2, so they are held to the oracle instead; the NORM rotation's
// whole groups use no FMA and are bit-identical too.
//
// The one deliberate bit difference: VCVTPH2PS quiets a signalling NaN and
// halfToFloatSSE carries it through. No scale plane or embedding holds one.

// EmitRMSNormSSE is EmitRMSNorm (norm.go) on legacy SSE: y = x * w /
// sqrt(mean(x^2) + eps), n baked, Scr = [1/n, eps, 1.0].
//
// The sum of squares keeps the AVX2 kernel's four chains as eight XMM
// accumulators (chain i's lanes 0..3 in XMM(2i), 4..7 in XMM(2i+1)) and folds
// them the same way; only the rounding of each square differs (no FMA).
//
// Registers: RCX out, RSI x, RDX w, RBX Scr, R10 counter, R11/R8 cursors;
// XMM0..XMM7 the accumulators, XMM7 the scale after the fold, XMM8..XMM15
// temporaries.
func EmitRMSNormSSE(n int) ([]byte, error) {
	if n < 1 {
		return nil, fmt.Errorf("jit: EmitRMSNormSSE: width %d", n)
	}
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))  // Out -> y
	a.MOVLoad(RSI, At(RDI, 16)) // A, reinterpreted as *float32 -> x
	a.MOVLoad(RDX, At(RDI, 24)) // AScale -> w
	a.MOVLoad(RBX, At(RDI, 56)) // Scr -> [1/n, eps, 1]

	// ---- pass 1: the sum of squares ----
	for r := XMM0; r <= XMM7; r++ {
		a.XORPS(r, r, r)
	}
	sq := func(acc, t Reg, m Mem) {
		a.MOVUPSLoad(t, m)
		a.MULPS(t, t, t)
		a.ADDPS(acc, acc, t)
	}
	a.MOVQ(R11, RSI)
	// The tail in eights goes into chain 0 exactly as the AVX2 kernel does, and
	// the last n%8 one at a time: scalar ops touch lane 0 only.
	newRowWalk(&a, n, R11).run(func() {
		for h := 0; h < 8; h++ {
			sq(XMM0+Reg(h), XMM8+Reg(h), At(R11, int32(16*h)))
		}
	}, func() {
		sq(XMM0, XMM8, At(R11, 0))
		sq(XMM1, XMM9, At(R11, 16))
	}, func(i int) {
		a.MOVSSLoad(XMM8, At(R11, int32(4*i)))
		a.MULSS(XMM8, XMM8, XMM8)
		a.ADDSS(XMM0, XMM0, XMM8)
	})
	normFold8SSE(&a)

	// ---- scale = 1 / sqrt(ss/n + eps), in every lane ----
	bcastSS(&a, XMM5, At(RBX, 0)) // 1/n
	a.MULPS(XMM0, XMM0, XMM5)
	bcastSS(&a, XMM6, At(RBX, 4)) // eps
	a.ADDPS(XMM0, XMM0, XMM6)
	a.SQRTPS(XMM0, XMM0)
	bcastSS(&a, XMM7, At(RBX, 8)) // 1.0
	a.DIVPS(XMM7, XMM7, XMM0)     // the fold left the total in every lane

	// ---- pass 2: y = x * scale * w ----
	app := func(t, u Reg, d int32) {
		a.MOVUPSLoad(t, At(R11, d))
		a.MULPS(t, t, XMM7)
		a.MOVUPSLoad(u, At(RDX, d))
		a.MULPS(t, t, u)
		a.MOVUPSStore(At(R8, d), t)
	}
	a.MOVQ(R11, RSI)
	a.MOVQ(R8, RCX)
	newRowWalk(&a, n, R11, RDX, R8).run(func() {
		for h := 0; h < 8; h++ {
			app(XMM8+Reg(2*(h%4)), XMM9+Reg(2*(h%4)), int32(16*h))
		}
	}, func() {
		app(XMM8, XMM9, 0)
		app(XMM10, XMM11, 16)
	}, func(i int) {
		d := int32(4 * i)
		a.MOVSSLoad(XMM8, At(R11, d))
		a.MULSS(XMM8, XMM8, XMM7)
		a.MOVSSLoad(XMM9, At(RDX, d))
		a.MULSS(XMM8, XMM8, XMM9)
		a.MOVSSStore(At(R8, d), XMM8)
	})
	a.RET()
	return a.Bytes(), nil
}

// normFold8SSE folds eight XMM accumulators -- the AVX2 kernels' four YMM chains
// Y0..Y3, with Yi held as XMM(2i) (lanes 0..3) and XMM(2i+1) (lanes 4..7) --
// into the total, in every lane of XMM0, operation for operation:
//
//	VADDPS Y0,Y0,Y1   ->  X0 += X2, X1 += X3
//	VADDPS Y2,Y2,Y3   ->  X4 += X6, X5 += X7
//	VADDPS Y0,Y0,Y2   ->  X0 += X4, X1 += X5
//	VEXTRACTF128 + VADDPSx + VHADDPSx x2  ->  hsum8SSE(X0, X1)
//
// so bit-identical partial sums give a bit-identical total.
func normFold8SSE(a *Buf) {
	a.ADDPS(XMM0, XMM0, XMM2)
	a.ADDPS(XMM1, XMM1, XMM3)
	a.ADDPS(XMM4, XMM4, XMM6)
	a.ADDPS(XMM5, XMM5, XMM7)
	a.ADDPS(XMM0, XMM0, XMM4)
	a.ADDPS(XMM1, XMM1, XMM5)
	hsum8SSE(a, XMM0, XMM1)
}

// EmitLayerNormSSE is EmitLayerNorm (layernorm.go) on legacy SSE, and it is
// bit-identical to it: y = (x - mean) / sqrt(var + eps) * w (+ b), n and the
// bias baked, the three passes and four accumulator chains mirrored half for
// half.
//
// The variance tail needs no blend: SUBSS and MULSS keep the MOVSS load's
// zeros in lanes 1..3, and ADDSS reads lane 0 only. (AVX2 clears those lanes
// with VPBLENDD because VSUBPS computes 0 - mean there.) The gate runs against
// a packed port that drops both guards.
//
// Registers: RCX out, RSI x, RDX w, R9 b, RBX Scr, R10 counter, R11/R8
// cursors; XMM0..XMM7 the accumulators, XMM12 the mean, XMM13 the inverse
// deviation, the rest temporaries.
func EmitLayerNormSSE(n int, bias bool) ([]byte, error) { return emitRowStatSSE(n, bias, RowLayerNorm) }

// EmitGaussTopKSSE and EmitMagMatchSSE are EmitGaussTopK and EmitMagMatch
// (layernorm.go) on legacy SSE.
func EmitGaussTopKSSE(n int) ([]byte, error) { return emitRowStatSSE(n, false, RowGauss) }
func EmitMagMatchSSE(n int) ([]byte, error)  { return emitRowStatSSE(n, false, RowMag) }

// emitRowStatSSE is emitRowStat on legacy SSE. XMM15 holds the match's
// target, and the gaussian's zero in pass 3.
func emitRowStatSSE(n int, bias bool, mode RowMode) ([]byte, error) {
	if n < 1 {
		return nil, fmt.Errorf("jit: EmitLayerNormSSE: width %d", n)
	}
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))  // Out -> y
	a.MOVLoad(RSI, At(RDI, 16)) // A, reinterpreted as *float32 -> x
	a.MOVLoad(RDX, At(RDI, 24)) // AScale -> w
	a.MOVLoad(R9, At(RDI, 8))   // W, reinterpreted as *float32 -> b
	a.MOVLoad(RBX, At(RDI, 56)) // Scr -> [1/n, eps, 1]

	// sum reduces sub over x into every lane of XMM0. sub loads one XMM half
	// at a byte offset from R11 and transforms it; subOne does one element into
	// lane 0 of t with lanes 1..3 zero.
	sum := func(sub, subOne func(t Reg, off int32)) {
		for r := XMM0; r <= XMM7; r++ {
			a.XORPS(r, r, r)
		}
		a.MOVQ(R11, RSI)
		newRowWalk(&a, n, R11).run(func() {
			for h := 0; h < 8; h++ {
				t := XMM8 + Reg(h%4)
				sub(t, int32(16*h))
				a.ADDPS(XMM0+Reg(h), XMM0+Reg(h), t)
			}
		}, func() {
			sub(XMM8, 0)
			a.ADDPS(XMM0, XMM0, XMM8)
			sub(XMM9, 16)
			a.ADDPS(XMM1, XMM1, XMM9)
		}, func(i int) {
			subOne(XMM8, int32(4*i))
			a.ADDSS(XMM0, XMM0, XMM8)
		})
		normFold8SSE(&a)
	}

	// ---- pass 1: the mean, or the match's reference magnitude ----
	if mode == RowMag {
		a.MOVQ(RSI, RDX)
		sum(func(t Reg, off int32) { a.MOVUPSLoad(t, At(R11, off)); a.MULPS(t, t, t) },
			func(t Reg, off int32) { a.MOVSSLoad(t, At(R11, off)); a.MULSS(t, t, t) })
		a.MOVLoad(RSI, At(RDI, 16))
		bcastSS(&a, XMM14, At(RBX, 0)) // 1/n
		a.MULPS(XMM0, XMM0, XMM14)
		a.SQRTPS(XMM0, XMM0)
		a.SHUFPS(XMM15, XMM0, XMM0, 0)
		a.XORPS(XMM12, XMM12, XMM12)
	} else {
		sum(func(t Reg, off int32) { a.MOVUPSLoad(t, At(R11, off)) },
			func(t Reg, off int32) { a.MOVSSLoad(t, At(R11, off)) })
		bcastSS(&a, XMM14, At(RBX, 0)) // 1/n
		a.MULPS(XMM0, XMM0, XMM14)
		a.SHUFPS(XMM12, XMM0, XMM0, 0) // the mean, live from here on
	}

	// ---- pass 2: the variance ----
	sum(func(t Reg, off int32) {
		a.MOVUPSLoad(t, At(R11, off))
		a.SUBPS(t, t, XMM12)
		a.MULPS(t, t, t)
	}, func(t Reg, off int32) {
		a.MOVSSLoad(t, At(R11, off))
		a.SUBSS(t, t, XMM12)
		a.MULSS(t, t, t)
	})
	bcastSS(&a, XMM14, At(RBX, 0)) // 1/n
	a.MULPS(XMM0, XMM0, XMM14)
	switch mode {
	case RowMag:
		bcastSS(&a, XMM14, At(RBX, 4)) // eps
		a.MAXPS(XMM0, XMM0, XMM14)
		a.SQRTPS(XMM0, XMM0)
		a.DIVPS(XMM15, XMM15, XMM0) // target / magnitude
		a.SHUFPS(XMM13, XMM15, XMM15, 0)
	case RowGauss:
		bcastSS(&a, XMM15, At(RBX, 4)) // eps
		a.ADDPS(XMM0, XMM0, XMM15)
		a.SQRTPS(XMM0, XMM0)
		bcastSS(&a, XMM14, At(RBX, 8)) // c
		a.MULPS(XMM0, XMM0, XMM14)
		a.ADDPS(XMM0, XMM0, XMM12) // the cutoff, mean + c*sd
		a.SHUFPS(XMM13, XMM0, XMM0, 0)
		a.XORPS(XMM15, XMM15, XMM15)
	default:
		bcastSS(&a, XMM15, At(RBX, 4)) // eps
		a.ADDPS(XMM0, XMM0, XMM15)
		a.SQRTPS(XMM0, XMM0)
		bcastSS(&a, XMM14, At(RBX, 8)) // 1.0
		a.DIVPS(XMM14, XMM14, XMM0)
		a.SHUFPS(XMM13, XMM14, XMM14, 0)
	}

	// ---- pass 3: y = (x - mean) * inv * w + b ----
	one := func(t, u Reg, off int32) {
		a.MOVUPSLoad(t, At(R11, off))
		switch mode {
		case RowGauss:
			a.SUBPS(t, t, XMM13)
			a.MAXPS(t, t, XMM15)
		case RowMag:
			a.MULPS(t, t, XMM13)
		default:
			a.SUBPS(t, t, XMM12)
			a.MULPS(t, t, XMM13)
			a.MOVUPSLoad(u, At(RDX, off))
			a.MULPS(t, t, u)
			if bias {
				a.MOVUPSLoad(u, At(R9, off))
				a.ADDPS(t, t, u)
			}
		}
		a.MOVUPSStore(At(R8, off), t)
	}
	a.MOVQ(R11, RSI)
	a.MOVQ(R8, RCX)
	newRowWalk(&a, n, R11, RDX, R9, R8).run(func() {
		for h := 0; h < 8; h++ {
			one(XMM8+Reg(h%4), XMM4+Reg(h%4), int32(16*h))
		}
	}, func() {
		one(XMM8, XMM4, 0)
		one(XMM9, XMM5, 16)
	}, func(i int) {
		d := int32(4 * i)
		a.MOVSSLoad(XMM8, At(R11, d))
		switch mode {
		case RowGauss:
			a.SUBSS(XMM8, XMM8, XMM13)
			a.MAXSS(XMM8, XMM8, XMM15)
		case RowMag:
			a.MULSS(XMM8, XMM8, XMM13)
		default:
			a.SUBSS(XMM8, XMM8, XMM12)
			a.MULSS(XMM8, XMM8, XMM13)
			a.MOVSSLoad(XMM9, At(RDX, d))
			a.MULSS(XMM8, XMM8, XMM9)
			if bias {
				a.MOVSSLoad(XMM9, At(R9, d))
				a.ADDSS(XMM8, XMM8, XMM9)
			}
		}
		a.MOVSSStore(At(R8, d), XMM8)
	})
	a.RET()
	return a.Bytes(), nil
}

// EmitRoPESSE is EmitRoPE (rope.go) on legacy SSE: the rotary rotation of Rows
// heads of hd floats (Out, in place) by one position's {cos, sin} table
// (AScale, nrot floats), hd, nrot and the layout baked, the dimensions past
// nrot never touched.
//
//	NEOX  four pairs a vector. Two loads of {c, s} pairs deinterleave with
//	      SHUFPS 0x88 (the cosines) and 0xDD (the sines) -- at 128 bits there
//	      is no cross-lane order to undo, so the AVX2 kernel's VPERMQ has no
//	      counterpart. The rotation is two products and a sum per output
//	      (no FMA); the last nrot/2 % 4 pairs are the same arithmetic on one
//	      lane.
//	NORM  two pairs a vector: MOVSLDUP/MOVSHDUP spread cos and sin over each
//	      pair, PSHUFD 0xB1 swaps the pair and ADDSUBPS supplies the sign.
//	      An odd last pair is the same four instructions on a MOVQ -- 8 bytes,
//	      so nothing past the pair is read or written.
//
// The NORM path is the AVX2 kernel's arithmetic on every pair (VMULPS,
// VMULPS, VADDSUBPS), so it is bit-identical over that kernel's whole groups
// of four pairs and differs only on its FMA-rotated tail pairs.
func EmitRoPESSE(hd, nrot int, neox bool) ([]byte, error) {
	if nrot <= 0 || nrot%2 != 0 || nrot > hd {
		return nil, fmt.Errorf("jit: EmitRoPESSE: nrot=%d must be even and in (0, hd=%d]", nrot, hd)
	}
	half := nrot / 2
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))  // Out: head 0
	a.MOVLoad(RDX, At(RDI, 24)) // AScale: the table
	a.MOVLoad(RAX, At(RDI, 32)) // Rows: heads
	done := a.Label()
	a.TESTQ(RAX, RAX)
	a.JZ(done)

	// rot writes (x0*c - x1*s, x0*s + x1*c) into XMM6, XMM7 from x0 XMM4, x1
	// XMM5, c XMM2, s XMM3, with XMM8 as the product temporary.
	rot := func() {
		a.MULPS(XMM6, XMM4, XMM2)
		a.MULPS(XMM8, XMM5, XMM3)
		a.SUBPS(XMM6, XMM6, XMM8)
		a.MULPS(XMM7, XMM4, XMM3)
		a.MULPS(XMM8, XMM5, XMM2)
		a.ADDPS(XMM7, XMM7, XMM8)
	}
	head := a.Label()
	a.Bind(head)
	if neox {
		hb := int32(4 * half) // x[p+half] is hb bytes after x[p]
		g := half / 4
		for i := 0; i < g; i++ {
			p := int32(4 * i)
			a.MOVUPSLoad(XMM0, At(RDX, 8*p))
			a.MOVUPSLoad(XMM1, At(RDX, 8*p+16))
			a.SHUFPS(XMM2, XMM0, XMM1, 0x88) // c0 c1 c2 c3
			a.SHUFPS(XMM3, XMM0, XMM1, 0xDD) // s0 s1 s2 s3
			a.MOVUPSLoad(XMM4, At(RCX, 4*p))
			a.MOVUPSLoad(XMM5, At(RCX, 4*p+hb))
			rot()
			a.MOVUPSStore(At(RCX, 4*p), XMM6)
			a.MOVUPSStore(At(RCX, 4*p+hb), XMM7)
		}
		for p := int32(4 * g); p < int32(half); p++ {
			a.MOVSSLoad(XMM2, At(RDX, 8*p))
			a.MOVSSLoad(XMM3, At(RDX, 8*p+4))
			a.MOVSSLoad(XMM4, At(RCX, 4*p))
			a.MOVSSLoad(XMM5, At(RCX, 4*p+hb))
			rot()
			a.MOVSSStore(At(RCX, 4*p), XMM6)
			a.MOVSSStore(At(RCX, 4*p+hb), XMM7)
		}
	} else {
		// Pair p is bytes 8p of both the head and the table.
		pair := func(off int32, ld func(Reg, Mem), st func(Mem, Reg)) {
			ld(XMM0, At(RDX, off))
			a.MOVSLDUP(XMM2, XMM0) // c0 c0 c1 c1
			a.MOVSHDUP(XMM3, XMM0) // s0 s0 s1 s1
			ld(XMM4, At(RCX, off))
			a.PSHUFD(XMM5, XMM4, 0xB1) // x1 x0 x3 x2
			a.MULPS(XMM4, XMM4, XMM2)
			a.MULPS(XMM5, XMM5, XMM3)
			a.ADDSUBPS(XMM4, XMM4, XMM5)
			st(At(RCX, off), XMM4)
		}
		for i := 0; i < half/2; i++ {
			pair(int32(16*i), a.MOVUPSLoad, a.MOVUPSStore)
		}
		if half%2 == 1 {
			pair(int32(8*(half-1)), a.MOVQLoad, a.MOVQStore)
		}
	}
	a.ADDimm(RCX, int32(hd*4))
	a.DEC(RAX)
	a.JNZ(head)

	a.Bind(done)
	a.RET()
	return a.Bytes(), nil
}

// EmitPackedRowSSE is EmitPackedRow (rowpacked.go) on legacy SSE: one row of a
// packed tensor dequantized into Out, every packed format, the same Args,
// PackedRowConsts and PackedRowScratch, and bit-identical to the AVX2 kernel
// (no FMA there, and the software f16 convert is exact, subnormals included).
//
// What the AVX2-only instructions became:
//
//	VCVTPH2PSReg      halfToFloatSSE
//	VBROADCASTI128    MOVDQU (the code table is 16 bytes; an XMM holds it)
//	VPSHUFB           PSHUFB into a copy of the table -- the legacy form's
//	                  destination is the table, so the indices survive
//	VPMOVZX/SXBD      PMOVZX/SXBD from 4 bytes: two per 8-element group
//	VPSRLVD           nothing. Over one 4-element group (l/4 fixed) the
//	                  secondary plane's per-lane shift is a uniform bit shift
//	                  c = hi*((l/4) % (8/hi)) plus a byte index l%4, so PSRLD
//	                  by c then PMOVZXBD spreads the bytes. The shift table in
//	                  PackedRowConsts is unread here.
//
// Registers: RCX out, RDX the payload cursor, R8 RowStr, RBX Scr, R9 PD, R12
// DStr, R10 PSC, R11 super-blocks, RAX scratch, RSI the gather cursor;
// XMM6/XMM7 the scale and bias, XMM8 0x0F0F0F0F, XMM9 0xFF, XMM10 the code
// table, XMM11 (1<<hi)-1, XMM12 2^112, XMM13 scOff, XMM14 biasK.
func EmitPackedRowSSE(q kernels.Quant) ([]byte, error) {
	sub, bits, _, biasArray := kernels.Layout(q)
	hi := kernels.HiPlane(q)
	perSuper, _ := kernels.ScaleLayout(q)
	signed := kernels.SignedPayload(q)
	codes := kernels.Codes(q) != nil
	if sub%8 != 0 || (bits != 2 && bits != 4 && bits != 8) || perSuper < 1 || bits == 2 && (sub != 32 || hi != 0 || codes) {
		return nil, fmt.Errorf("jit: EmitPackedRowSSE: %v has no row kernel shape", q)
	}
	pw, hw := sub*bits/32, sub*hi/32
	perWord, scStride := 4, 1 // scale bytes per word, bytes between them
	if biasArray {
		perWord, scStride = 2, 2
	}
	// The scratch plan: the gathered words at [0, 4*(pw+hw)); for a 4-bit
	// payload the low nibbles at [32, 32+16*chunks) and the high at
	// [32+sub/2, 32+sub/2+16*chunks). A shape outside it is refused by name
	// rather than allowed to overlap -- it would be a new format, and the AVX2
	// kernel makes the same assumptions.
	chunks := (sub/2 + 15) / 16
	if bits == 4 && (4*(pw+hw) > 32 || 16*chunks > 32 || 32+sub/2+16*chunks > PackedRowScratch) ||
		bits == 8 && 4*(pw+hw) > PackedRowScratch {
		return nil, fmt.Errorf("jit: EmitPackedRowSSE: %v does not fit the %d-byte row scratch", q, PackedRowScratch)
	}

	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))    // Out
	a.MOVLoad(RDX, At(RDI, 8))    // W: payload word cursor, one sub-block at a time
	a.MOVLoad(R8, At(RDI, 48))    // RowStr
	a.MOVLoad(RBX, At(RDI, 56))   // Scr
	a.MOVLoad(R9, At(RDI, 144))   // PD
	a.MOVLoad(R12, At(RDI, 160))  // DStr
	a.MOVLoad(R10, At(RDI, 152))  // PSC
	a.MOVLoad(R11, At(RDI, 40))   // K: super-blocks
	a.MOVLoad(RAX, At(RDI, 72))   // Scratch
	bcastD(&a, XMM8, At(RBX, 16)) // 0x0F0F0F0F
	bcastD(&a, XMM9, At(RBX, 20)) // 0xFF
	if codes {
		a.MOVDQULoad(XMM10, At(RBX, 0))
	}
	if hi > 0 {
		bcastD(&a, XMM11, At(RBX, 32)) // (1<<hi)-1
	}
	loadHalfMagicSSE(&a, XMM12, RSI) // RSI is free until the gather
	bcastSS(&a, XMM13, At(RBX, 24))  // scOff
	bcastSS(&a, XMM14, At(RBX, 28))  // biasK

	done := a.Label()
	a.TESTQ(R11, R11)
	a.JZ(done)
	sup := a.Label()
	a.Bind(sup)
	for j := 0; j < perSuper; j++ {
		// The scale: d, times this sub-block's byte plus scOff for a k-quant.
		a.PXOR(XMM4, XMM4, XMM4)
		if E8M0Quant(q) {
			// One byte and a shift; see rowpacked.go's twin. PMOVZXBD would
			// read eight and the plane holds four rows to a word.
			a.PXOR(XMM4, XMM4, XMM4)
			a.PINSRBLoad(XMM4, XMM4, At(R9, 0), 0)
			a.PSLLD(XMM6, XMM4, 23)
		} else if kernels.D32(q) {
			a.MOVSSLoad(XMM6, At(R9, 0)) // TQ2_F32: the word is the scale
		} else {
			a.PINSRWLoad(XMM4, XMM4, At(R9, 0), 0)
			halfToFloatSSE(&a, XMM6, XMM4, XMM2, XMM3, XMM12)
		}
		if biasArray {
			a.PXOR(XMM5, XMM5, XMM5)
			a.PINSRWLoad(XMM5, XMM5, At(R9, 2), 0)
			halfToFloatSSE(&a, XMM7, XMM5, XMM2, XMM3, XMM12) // dmin
		}
		if kernels.ScStream(q) {
			// Container v27's stream (scstream.go); the next word is RowStr on.
			next := func(r Reg) { a.MOVSSLoad(r, Idx(R10, R8, 1, 0)) }
			a.MOVSSLoad(XMM1, At(R10, 0))
			scField(sseSC(&a), XMM2, XMM1, XMM3, j, false, next)
			a.CVTDQ2PS(XMM2, XMM2)
			a.ADDPS(XMM2, XMM2, XMM13)
			a.MULPS(XMM6, XMM6, XMM2)
			scField(sseSC(&a), XMM1, XMM1, XMM3, j, true, next)
			a.CVTDQ2PS(XMM1, XMM1)
			a.MULPS(XMM7, XMM7, XMM1) // bias = dmin * min
			if scStep(j) {
				a.ADDQ(R10, R8)
			}
		} else if perSuper > 1 {
			a.MOVSSLoad(XMM1, At(R10, 0))
			if sh := (j % perWord) * scStride * 8; sh > 0 {
				a.PSRLD(XMM1, XMM1, byte(sh))
			}
			a.PAND(XMM2, XMM1, XMM9)
			a.CVTDQ2PS(XMM2, XMM2)
			a.ADDPS(XMM2, XMM2, XMM13)
			a.MULPS(XMM6, XMM6, XMM2)
			if biasArray {
				a.PSRLD(XMM1, XMM1, 8)
				a.PAND(XMM1, XMM1, XMM9)
				a.CVTDQ2PS(XMM1, XMM1)
				a.MULPS(XMM7, XMM7, XMM1) // bias = dmin * min
			}
			if j%perWord == perWord-1 {
				a.ADDQ(R10, R8)
			}
		}
		if !biasArray {
			a.MULPS(XMM7, XMM6, XMM14) // bias = scale * biasK
		}
		a.SHUFPS(XMM6, XMM6, XMM6, 0)
		a.SHUFPS(XMM7, XMM7, XMM7, 0)

		// Gather the sub-block's words: pw payload, then hw secondary.
		a.MOVQ(RSI, RDX)
		for w := 0; w < pw+hw; w++ {
			a.MOVSSLoad(XMM0, At(RSI, 0))
			a.MOVSSStore(At(RAX, int32(4*w)), XMM0)
			a.ADDQ(RSI, R8)
		}
		a.MOVQ(RDX, RSI)
		base := int32(0)
		if bits == 4 {
			// Byte b of word w holds element 4w+b low and 4w+b+sub/2 high.
			// Every low store lands before any high one, so a high store may
			// overwrite a low store's bytes past sub/2 and never its data.
			for _, part := range []struct {
				dst   int32
				shift bool
			}{{32, false}, {int32(32 + sub/2), true}} {
				for c := 0; c < chunks; c++ {
					a.MOVDQULoad(XMM0, At(RAX, int32(16*c)))
					v := XMM0
					if part.shift {
						a.PSRLW(XMM0, XMM0, 4)
					}
					a.PAND(XMM0, XMM0, XMM8)
					if codes {
						a.PSHUFB(XMM1, XMM10, XMM0) // a copy of the table, indexed
						v = XMM1
					}
					a.MOVDQUStore(At(RAX, part.dst+int32(16*c)), v)
				}
			}
			base = 32
		}
		if bits == 2 {
			// TQ2: eight-element group g is field g of the sub-block's eight
			// bytes (packSub), so its two halves widen bytes 0..3 and 4..7 and
			// shift the field down; XMM5 holds 3 in every dword.
			a.PSRLD(XMM5, XMM9, 6)
		}
		for g := 0; g < sub/8; g++ {
			for h := 0; h < 2; h++ {
				gq := 2*g + h // this 4-element group's l/4
				src := At(RAX, base+int32(8*g+4*h))
				switch {
				case bits == 2:
					a.PMOVZXBDLoad(XMM1, At(RAX, int32(4*h)))
					if g > 0 {
						a.PSRLD(XMM1, XMM1, byte(2*g))
					}
					a.PAND(XMM1, XMM1, XMM5)
				case signed:
					a.PMOVSXBDLoad(XMM1, src)
				default:
					a.PMOVZXBDLoad(XMM1, src)
				}
				if hi > 0 {
					lanes := 8 / hi
					wd := pw + gq/lanes
					a.MOVDLoad(XMM2, At(RAX, int32(4*wd)))
					if c := hi * (gq % lanes); c > 0 {
						a.PSRLD(XMM2, XMM2, byte(c))
					}
					a.PMOVZXBD(XMM2, XMM2)
					a.PAND(XMM2, XMM2, XMM11)
					a.PSLLD(XMM2, XMM2, 4)
					a.POR(XMM1, XMM1, XMM2)
				}
				a.CVTDQ2PS(XMM1, XMM1)
				a.MULPS(XMM1, XMM1, XMM6)
				a.SUBPS(XMM1, XMM1, XMM7)
				a.MOVUPSStore(At(RCX, int32(32*(g+j*sub/8)+16*h)), XMM1)
			}
		}
	}
	a.ADDimm(RCX, int32(4*sub*perSuper))
	a.ADDQ(R9, R12)
	a.DEC(R11)
	a.JNZ(sup)

	a.Bind(done)
	a.RET()
	return a.Bytes(), nil
}

// EmitWidenSSE is EmitWiden (rowpacked.go) on legacy SSE: dst[i] = float32(src[i])
// for binary16 or bfloat16 src, Args.K whole units of ElemLanes (8) and
// Args.Rows single elements. Out is dst, W is src.
//
// A bfloat16 is the top half of its float32, so PUNPCKLWD/PUNPCKHWD against
// zero IS the conversion; a binary16 goes through halfToFloatSSE. Both are
// exact, so the result is the AVX2 kernel's bit for bit (bar F16C's quieting
// of a signalling NaN).
//
// Registers: RCX dst, RDX src, R10 units, R11 singles; XMM12 2^112, XMM15
// zero.
func EmitWidenSSE(bf16 bool) ([]byte, error) {
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RDX, At(RDI, 8))  // W
	a.MOVLoad(R10, At(RDI, 40)) // K
	a.MOVLoad(R11, At(RDI, 32)) // Rows
	if !bf16 {
		loadHalfMagicSSE(&a, XMM12, RAX)
		a.PXOR(XMM15, XMM15, XMM15)
	}
	vec := func() {
		a.MOVDQULoad(XMM0, At(RDX, 0)) // eight halves
		lo, hi := XMM1, XMM2
		if bf16 {
			bf16ToFloatSSE(&a, XMM1, XMM0)
			bf16HiToFloatSSE(&a, XMM2, XMM0)
		} else {
			a.PUNPCKLWD(XMM1, XMM0, XMM15) // halves 0..3, zero-extended
			a.PUNPCKHWD(XMM2, XMM0, XMM15) // halves 4..7
			halfToFloatSSE(&a, XMM3, XMM1, XMM5, XMM6, XMM12)
			halfToFloatSSE(&a, XMM4, XMM2, XMM5, XMM6, XMM12)
			lo, hi = XMM3, XMM4
		}
		a.MOVUPSStore(At(RCX, 0), lo)
		a.MOVUPSStore(At(RCX, 16), hi)
	}
	one := func() {
		a.PXOR(XMM0, XMM0, XMM0)
		a.PINSRWLoad(XMM0, XMM0, At(RDX, 0), 0)
		v := XMM0
		if bf16 {
			a.PSLLD(XMM0, XMM0, 16)
		} else {
			halfToFloatSSE(&a, XMM3, XMM0, XMM5, XMM6, XMM12)
			v = XMM3
		}
		a.MOVSSStore(At(RCX, 0), v)
	}
	for _, l := range []struct {
		n          Reg
		body       func()
		dst, srcSt int32
	}{{R10, vec, 32, 16}, {R11, one, 4, 2}} {
		skip := a.Label()
		a.TESTQ(l.n, l.n)
		a.JZ(skip)
		lp := a.Label()
		a.Bind(lp)
		l.body()
		a.ADDimm(RCX, l.dst)
		a.ADDimm(RDX, l.srcSt)
		a.DEC(l.n)
		a.JNZ(lp)
		a.Bind(skip)
	}
	a.RET()
	return a.Bytes(), nil
}
