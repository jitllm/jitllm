package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/format/quant"
)

// EmitFloatMatMulSSE is EmitFloatMatMul in legacy SSE (its comment is the
// contract): one accumulator of four lanes per token over four-wide groups,
// then single elements into lane 0, a multiply and an add where AVX2 fuses.
//
// Registers: XMM0..XMM7 the tokens' chains, XMM8 the weights, XMM9 the
// product, XMM11 zero, XMM12 2^112 and XMM13/XMM14 the half conversion's
// temporaries (F16), XMM15 raw halves. GP is the AVX2 kernel's.
func EmitFloatMatMulSSE(t quant.Type, nt int) ([]byte, error) {
	if t != quant.F32 && t != quant.F16 && t != quant.BF16 {
		return nil, fmt.Errorf("jit: EmitFloatMatMulSSE: %s is not a float format", t)
	}
	if nt < 1 || nt > MaxFloatTokens {
		return nil, fmt.Errorf("jit: EmitFloatMatMulSSE: a tile of %d tokens", nt)
	}
	var a Buf
	a.DeclareISA(ISATierSSE)
	esz := int32(4)
	if t != quant.F32 {
		esz = 2
	}
	const (
		w     = XMM8
		p     = XMM9
		zero  = XMM11
		magic = XMM12
		t0    = XMM13
		t1    = XMM14
		raw   = XMM15
	)
	quad := func() {
		switch t {
		case quant.F32:
			a.MOVUPSLoad(w, At(RDX, 0))
		case quant.F16:
			loadU16x4SSE(&a, raw, At(RDX, 0), zero)
			halfToFloatSSE(&a, w, raw, t0, t1, magic)
		default:
			loadU16x4SSE(&a, w, At(RDX, 0), zero)
			a.PSLLD(w, w, 16)
		}
	}
	one := func() {
		switch t {
		case quant.F32:
			a.MOVSSLoad(w, At(RDX, 0))
		case quant.F16:
			a.PXOR(raw, raw, raw)
			a.PINSRWLoad(raw, raw, At(RDX, 0), 0)
			halfToFloatSSE(&a, w, raw, t0, t1, magic)
		default:
			a.PXOR(w, w, w)
			a.PINSRWLoad(w, w, At(RDX, 0), 0)
			a.PSLLD(w, w, 16)
		}
	}
	xm := func(j int) Mem {
		base := R11
		if j >= 4 {
			base = R13
		}
		switch j % 4 {
		case 1:
			return Idx(base, RBX, 1, 0)
		case 2:
			return Idx(base, RBX, 2, 0)
		case 3:
			return Idx(base, R12, 1, 0)
		}
		return At(base, 0)
	}
	om := func(j int) Mem {
		base := RCX
		if j >= 4 {
			base = R10
		}
		switch j % 4 {
		case 1:
			return Idx(base, R8, 1, 0)
		case 2:
			return Idx(base, R8, 2, 0)
		case 3:
			return Idx(base, R15, 1, 0)
		}
		return At(base, 0)
	}

	a.MOVLoad(RCX, At(RDI, 0))
	a.MOVLoad(RDX, At(RDI, 8))
	a.MOVLoad(RSI, At(RDI, 16))
	a.MOVLoad(RAX, At(RDI, 32))
	a.MOVLoad(R9, At(RDI, 40))
	a.MOVLoad(R8, At(RDI, 88))
	if t != quant.F32 {
		a.PXOR(zero, zero, zero)
	}
	if t == quant.F16 {
		loadHalfMagicSSE(&a, magic, R10)
	}
	a.MOVQ(RBX, R9)
	a.SHLimm(RBX, 2)
	a.MOVQ(R12, RBX)
	a.SHLimm(R12, 1)
	a.ADDQ(R12, RBX)
	a.MOVQ(R15, R8)
	a.SHLimm(R15, 1)
	a.ADDQ(R15, R8)
	a.MOVQ(RDI, R9)
	a.ANDimm8(RDI, 3) // single elements
	a.SHRimm(R9, 2)   // four-wide groups

	done := a.Label()
	a.TESTQ(RAX, RAX)
	a.JZ(done)

	row := a.Label()
	a.Bind(row)
	for j := 0; j < nt; j++ {
		a.PXOR(Reg(j), Reg(j), Reg(j))
	}
	a.MOVQ(R11, RSI)
	if nt > 4 {
		a.MOVQ(R13, RBX)
		a.SHLimm(R13, 2)
		a.ADDQ(R13, RSI)
	}
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
		if nt > 4 {
			a.ADDimm(R13, xstep)
		}
		a.DEC(R10)
		a.JNZ(lp)
		a.Bind(skip)
	}
	loop(R9, func() {
		quad()
		for j := 0; j < nt; j++ {
			a.MOVUPSLoad(p, xm(j))
			a.MULPS(p, p, w)
			a.ADDPS(Reg(j), Reg(j), p)
		}
	}, 4*esz, 16)
	loop(RDI, func() {
		one()
		for j := 0; j < nt; j++ {
			a.MOVSSLoad(p, xm(j))
			a.MULPS(p, p, w)
			a.ADDPS(Reg(j), Reg(j), p)
		}
	}, esz, 4)

	if nt > 4 {
		a.MOVQ(R10, R8)
		a.SHLimm(R10, 2)
		a.ADDQ(R10, RCX)
	}
	for j := 0; j < nt; j++ {
		hsum4SSE(&a, Reg(j))
		a.MOVSSStore(om(j), Reg(j))
	}
	a.ADDimm(RCX, 4)
	a.DEC(RAX)
	a.JNZ(row)

	a.Bind(done)
	a.RET()
	return a.Bytes(), nil
}
