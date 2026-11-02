package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/format/quant"
)

// The SSE tier's row-major family: the float matvec, and nothing quantized.
//
// The quantized row-major kernels read the GGUF block layout, which a
// container never decodes through, so they are declined. jlm stores float
// matrices verbatim and row-major, so the mixture routers, Qwen3-Next's BF16
// shared-expert gate and every matrix of a small HF model reach this kernel.
//
// It is emitFloatMatVec's algorithm with each YMM accumulator held as two XMM
// halves: chain i is XMM(i) (lanes 0..3) and XMM(4+i) (lanes 4..7), every
// element lands in the same lane of the same chain, and the fold is hsum8SSE.
// The only arithmetic difference from the AVX2 twin is the missing FMA.
//
// Weights arrive at any byte offset and x is a Go slice, so every load is
// unaligned (MOVUPS/MOVDQU/MOVQ/MOVSS/PINSRW) and MULPS takes registers only.

// SupportedRowMajorSSE reports whether the SSE tier has a row-major kernel for
// t: the three float formats.
func SupportedRowMajorSSE(t quant.Type) bool {
	switch t {
	case quant.F32, quant.F16, quant.BF16:
		return true
	}
	return false
}

// EmitRowMajorSSE is the SSE tier's EmitNative: the float matvec, with Emit's
// ABI for Spec{W: F32|F16|BF16, Rows: 1, Cols: 1} exactly (matvec.go,
// emitFloatMatVec):
//
//	Out     [Rows] f32, written
//	W       row-major weights, walked continuously across rows
//	A       x as float32 (reinterpreted through the *int8 field)
//	Rows    rows this call serves
//	K       ELEMENTS per row (any k; groups of 32, then 8s, then singles)
//
// Accs and ActWin are ignored, as they are by the AVX2 float kernel.
func EmitRowMajorSSE(s Spec) ([]byte, error) {
	if !SupportedRowMajorSSE(s.W) {
		return nil, fmt.Errorf("jit: the SSE tier has no row-major %s kernel: the GGUF "+
			"block-layout kernels are not ported (a container never decodes through them; "+
			"the packed family serves %s)", s.W, s.W)
	}
	if s.Cols != 1 {
		return nil, fmt.Errorf("jit: EmitRowMajorSSE: Cols=%d unsupported (decode matvec only)", s.Cols)
	}
	if s.Rows != 1 {
		return nil, fmt.Errorf("jit: EmitRowMajorSSE: %s has no interleaved kernel", s.W)
	}
	code := emitFloatMatVecSSE(s.W)
	if err := checkBudget(s, len(code)); err != nil {
		return nil, err
	}
	return code, nil
}

// emitFloatMatVecSSE generates out[r] = dot(row r, x) for row-major F32, F16
// or BF16 weights and f32 activations, at ANY k, in legacy SSE.
//
// Registers:
//
//	XMM0..XMM3   chains 0..3, lanes 0..3      XMM8/XMM9  weights, low/high half
//	XMM4..XMM7   chains 0..3, lanes 4..7      XMM10      x
//	XMM11        zero (F16/BF16)              XMM12      2^112 (F16)
//	XMM13/XMM14  halfToFloatSSE temporaries   XMM15      raw halves
//
// GP is the AVX2 kernel's: RCX out, RDX weights, RSI x base, R11 x cursor, RAX
// rows, R9/R12/R13 the three loop counts, R10 the running count.
func emitFloatMatVecSSE(t quant.Type) []byte {
	var a Buf
	a.DeclareISA(ISATierSSE)
	esz := int32(4)
	if t != quant.F32 {
		esz = 2
	}
	lo := func(i int) Reg { return Reg(i) }     // chain i, lanes 0..3
	hi := func(i int) Reg { return Reg(4 + i) } // chain i, lanes 4..7
	const (
		wlo   = XMM8
		whi   = XMM9
		xv    = XMM10
		zero  = XMM11
		magic = XMM12
		t0    = XMM13
		t1    = XMM14
		raw   = XMM15
	)

	// wide loads eight weights at off as f32, lanes 0..3 into l and 4..7 into h.
	wide := func(l, h Reg, off int32) {
		switch t {
		case quant.F32:
			a.MOVUPSLoad(l, At(RDX, off))
			a.MOVUPSLoad(h, At(RDX, off+16))
		case quant.F16:
			// VCVTPH2PS reads eight halves in one instruction; with no F16C it
			// is two four-half loads, zero-extended and widened in software
			// (exact; a subnormal weight costs a microcode assist on Goldmont,
			// see halfPlane).
			loadU16x4SSE(&a, raw, At(RDX, off), zero)
			halfToFloatSSE(&a, l, raw, t0, t1, magic)
			loadU16x4SSE(&a, raw, At(RDX, off+8), zero)
			halfToFloatSSE(&a, h, raw, t0, t1, magic)
		default:
			a.MOVDQULoad(raw, At(RDX, off))
			bf16ToFloatSSE(&a, l, raw)
			bf16HiToFloatSSE(&a, h, raw)
		}
	}
	// one loads a single weight into lane 0 of dst with lanes 1..3 zero.
	one := func(dst Reg) {
		switch t {
		case quant.F32:
			a.MOVSSLoad(dst, At(RDX, 0))
		case quant.F16:
			// PINSRW reads exactly two bytes: MOVQ would read past the row.
			a.PXOR(raw, raw, raw)
			a.PINSRWLoad(raw, raw, At(RDX, 0), 0)
			halfToFloatSSE(&a, dst, raw, t0, t1, magic)
		default:
			a.PXOR(dst, dst, dst)
			a.PINSRWLoad(dst, dst, At(RDX, 0), 0)
			a.PSLLD(dst, dst, 16)
		}
	}
	// madd is acc += w * x: the FMA, as a rounded product and a sum.
	madd := func(acc, w Reg, x Mem) {
		a.MOVUPSLoad(xv, x)
		a.MULPS(w, w, xv)
		a.ADDPS(acc, acc, w)
	}

	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RDX, At(RDI, 8))  // W, walked continuously across rows
	a.MOVLoad(RSI, At(RDI, 16)) // A, reinterpreted as float32
	a.MOVLoad(RAX, At(RDI, 32)) // Rows
	a.MOVLoad(R9, At(RDI, 40))  // K, in elements
	if t == quant.F16 {
		a.PXOR(zero, zero, zero)
		loadHalfMagicSSE(&a, magic, R10)
	}
	a.MOVQ(R12, R9)
	a.SHRimm(R12, 3)
	a.ANDimm8(R12, 3) // whole vectors of eight past the groups of 32
	a.MOVQ(R13, R9)
	a.ANDimm8(R13, 7) // single elements past those
	a.SHRimm(R9, 5)   // groups of 32

	// A zero row count is no work, not 2^64 iterations of DEC/JNZ.
	done := a.Label()
	a.TESTQ(RAX, RAX)
	a.JZ(done)

	row := a.Label()
	a.Bind(row)
	for i := 0; i < 4; i++ {
		a.PXOR(lo(i), lo(i), lo(i))
		a.PXOR(hi(i), hi(i), hi(i))
	}
	a.MOVQ(R11, RSI) // activation cursor, rewound for every row

	loop := func(n Reg, body func(), wstep, xstep int32) {
		skip := a.Label()
		a.MOVQ(R10, n)
		a.TESTQ(R10, R10)
		a.JZ(skip)
		lp := a.Label()
		a.Bind(lp)
		body()
		a.ADDimm(RDX, wstep)
		a.ADDimm(R11, xstep)
		a.DEC(R10)
		a.JNZ(lp)
		a.Bind(skip)
	}
	loop(R9, func() {
		// 32 elements: chain i takes elements 8i..8i+7, as the AVX2 kernel's
		// Y(i) does.
		for i := 0; i < 4; i++ {
			wide(wlo, whi, 8*esz*int32(i))
			madd(lo(i), wlo, At(R11, 32*int32(i)))
			madd(hi(i), whi, At(R11, 32*int32(i)+16))
		}
	}, 32*esz, 128)
	loop(R12, func() {
		wide(wlo, whi, 0)
		madd(lo(0), wlo, At(R11, 0))
		madd(hi(0), whi, At(R11, 16))
	}, 8*esz, 32)
	loop(R13, func() {
		one(wlo)
		a.MOVSSLoad(xv, At(R11, 0))
		a.MULPS(wlo, wlo, xv)
		a.ADDPS(lo(0), lo(0), wlo)
	}, esz, 4)

	// (c0 + c1) + (c2 + c3), then the 128-bit lane fold: the AVX2 order.
	a.ADDPS(lo(0), lo(0), lo(1))
	a.ADDPS(hi(0), hi(0), hi(1))
	a.ADDPS(lo(2), lo(2), lo(3))
	a.ADDPS(hi(2), hi(2), hi(3))
	a.ADDPS(lo(0), lo(0), lo(2))
	a.ADDPS(hi(0), hi(0), hi(2))
	hsum8SSE(&a, lo(0), hi(0))
	a.MOVSSStore(At(RCX, 0), lo(0))

	a.ADDimm(RCX, 4)
	a.DEC(RAX)
	a.JNZ(row)

	a.Bind(done)
	a.RET()
	return a.Bytes()
}
