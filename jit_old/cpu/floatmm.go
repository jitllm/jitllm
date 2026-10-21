//go:build amd64

package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/format/quant"
)

// MaxFloatTokens is the widest token tile EmitFloatMatMul takes: one
// accumulator per token, and the weight and one temporary beside them.
const MaxFloatTokens = 8

// EmitFloatMatMul generates the float GEMM over a tile of nt tokens:
//
//	out[j*OutStr/4 + r] = dot(row r of W, x_j)   for r < Rows, j < nt
//
// for row-major F32, F16 or BF16 weights, with x_j the f32 activations at
// A + j*K*4. It is emitFloatMatVec with the weight read once per row for the
// whole tile instead of once per token: a prefill of n tokens through a float
// matrix reads it n/nt times where the matvec read it n times, which on an
// encoder of BF16 matrices is the whole cost of a forward pass.
//
// Each token is one FMA chain over eight-wide groups, then single elements
// into lane 0, then the lane reduction emitFloatMatVec does -- one chain where
// the matvec runs four, so the sum is in a different (deterministic) order.
//
// Registers: RCX the output row (token 0), R8 OutStr and R15 3*OutStr; RDX the
// weights, walked continuously across rows; RSI the activations, R11 the
// cursor of tokens 0..3 and R13 of tokens 4..7, RBX the token stride K*4 and
// R12 3*K*4; RAX rows, R9 the eight-wide groups, RDI the single elements, R10
// the running count. Y0..Y7 accumulate, Y8 holds the weights, Y9 is scratch.
func EmitFloatMatMul(t quant.Type, nt int) ([]byte, error) {
	if t != quant.F32 && t != quant.F16 && t != quant.BF16 {
		return nil, fmt.Errorf("jit: EmitFloatMatMul: %s is not a float format", t)
	}
	if nt < 1 || nt > MaxFloatTokens {
		return nil, fmt.Errorf("jit: EmitFloatMatMul: a tile of %d tokens", nt)
	}
	var a Buf
	esz := int32(4)
	if t != quant.F32 {
		esz = 2
	}
	wide := func(dst Reg) {
		switch t {
		case quant.F32:
			a.VMOVDQULoad(dst, At(RDX, 0))
		case quant.F16:
			a.VCVTPH2PS(dst, At(RDX, 0))
		default:
			a.VPMOVZXWD(dst, At(RDX, 0))
			a.VPSLLD(dst, dst, 16)
		}
	}
	one := func(dst Reg) {
		switch t {
		case quant.F32:
			a.VMOVSSLoad(dst, At(RDX, 0))
		case quant.F16:
			a.VPXOR(dst, dst, dst)
			a.VPINSRWLoad(dst, dst, At(RDX, 0), 0)
			a.VCVTPH2PSReg(dst, dst)
		default:
			a.VPXOR(dst, dst, dst)
			a.VPINSRWLoad(dst, dst, At(RDX, 0), 0)
			a.VPSLLD(dst, dst, 16)
		}
	}
	// xm is token j's activation at the cursor; om its output at the row.
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

	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RDX, At(RDI, 8))  // W
	a.MOVLoad(RSI, At(RDI, 16)) // A, as float32
	a.MOVLoad(RAX, At(RDI, 32)) // Rows
	a.MOVLoad(R9, At(RDI, 40))  // K, in elements
	a.MOVLoad(R8, At(RDI, 88))  // OutStr, in bytes
	a.MOVQ(RBX, R9)
	a.SHLimm(RBX, 2) // the token stride, K*4
	a.MOVQ(R12, RBX)
	a.SHLimm(R12, 1)
	a.ADDQ(R12, RBX) // 3*K*4
	a.MOVQ(R15, R8)
	a.SHLimm(R15, 1)
	a.ADDQ(R15, R8) // 3*OutStr
	a.MOVQ(RDI, R9)
	a.ANDimm8(RDI, 7) // single elements
	a.SHRimm(R9, 3)   // eight-wide groups

	done := a.Label()
	a.TESTQ(RAX, RAX)
	a.JZ(done)

	row := a.Label()
	a.Bind(row)
	for j := 0; j < nt; j++ {
		a.VPXOR(Reg(j), Reg(j), Reg(j))
	}
	a.MOVQ(R11, RSI) // the cursors, rewound every row
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
		wide(Y8)
		for j := 0; j < nt; j++ {
			a.VFMADD231PSMem(Reg(j), Y8, xm(j))
		}
	}, 8*esz, 32)
	loop(RDI, func() {
		one(Y8)
		for j := 0; j < nt; j++ {
			a.VMOVSSLoad(Y9, xm(j))
			a.VFMADD231PS(Reg(j), Y8, Y9)
		}
	}, esz, 4)

	if nt > 4 {
		a.MOVQ(R10, R8)
		a.SHLimm(R10, 2)
		a.ADDQ(R10, RCX) // token 4's output row
	}
	for j := 0; j < nt; j++ {
		r := Reg(j)
		a.VEXTRACTF128(Y9, r, 1)
		a.VADDPSx(r, r, Y9)
		a.VHADDPSx(r, r, r)
		a.VHADDPSx(r, r, r)
		a.VMOVSSStore(om(j), r)
	}
	a.ADDimm(RCX, 4)
	a.DEC(RAX)
	a.JNZ(row)

	a.Bind(done)
	a.VZEROUPPER()
	a.RET()
	return a.Bytes(), nil
}
