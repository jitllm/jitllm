//go:build arm64

package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/format/quant"
)

// MaxFloatTokens is the widest token tile EmitFloatMatMul takes.
const MaxFloatTokens = 8

// EmitFloatMatMul is the NEON float GEMM over a tile of nt tokens; the amd64
// emitter's comment is the contract. One four-lane chain per token over quads,
// then single elements into lane 0, reduced pairwise with FADDP.
//
// Registers: X1 the output row, X8 OutStr, X7 the store cursor; X2 the
// weights; X3 the activations, X4 the token stride K*4, X12..X17, X19, X20 the
// tokens' cursors; X5 rows, X6 quads, X10 single elements, X9 the running
// count. V0..V7 accumulate, V8 the weights, V9 an activation, V10 raw halves.
func EmitFloatMatMul(t quant.Type, nt int) ([]byte, error) {
	if t != quant.F32 && t != quant.F16 && t != quant.BF16 {
		return nil, fmt.Errorf("jit: EmitFloatMatMul: %s is not a float format", t)
	}
	if nt < 1 || nt > MaxFloatTokens {
		return nil, fmt.Errorf("jit: EmitFloatMatMul: a tile of %d tokens", nt)
	}
	var a A64
	esz := int32(4)
	if t != quant.F32 {
		esz = 2
	}
	cur := []XReg{X12, X13, X14, X15, X16, X17, X19, X20}
	quad := func() {
		switch t {
		case quant.F32:
			a.LDRq(V8, X2, 0)
		case quant.F16:
			a.LDRd(V10, X2, 0)
			a.FCVTL(V8, V10)
		default:
			a.LDRd(V10, X2, 0)
			a.SHLL(V8, V10)
		}
	}
	one := func() {
		switch t {
		case quant.F32:
			a.LDRs(V8, X2, 0)
		case quant.F16:
			a.LDRh(V8, X2, 0)
			a.FCVTL(V8, V8)
		default:
			a.LDRh(V8, X2, 0)
			a.SHLL(V8, V8)
		}
	}

	a.LDRx(X1, X0, 0)  // Out
	a.LDRx(X2, X0, 8)  // W
	a.LDRx(X3, X0, 16) // A, as float32
	a.LDRx(X5, X0, 32) // Rows
	a.LDRx(X6, X0, 40) // K, in elements
	a.LDRx(X8, X0, 88) // OutStr, in bytes
	a.ADDreg(X4, X6, X6)
	a.ADDreg(X4, X4, X4) // K*4
	a.ANDlow(X10, X6, 2) // single elements: K & 3
	a.LSRimm(X6, X6, 2)  // quads

	done := a.Label()
	a.CBZ(X5, done)
	row := a.Label()
	a.Bind(row)
	for j := 0; j < nt; j++ {
		a.MOVIzero(VReg(j))
	}
	a.MOVreg(cur[0], X3)
	for j := 1; j < nt; j++ {
		a.ADDreg(cur[j], cur[j-1], X4)
	}
	loop := func(n XReg, body func(), wstep, xstep int32) {
		skip := a.Label()
		a.MOVreg(X9, n)
		a.CBZ(X9, skip)
		lp := a.Label()
		a.Bind(lp)
		body()
		a.ADDimm(X2, X2, wstep)
		for j := 0; j < nt; j++ {
			a.ADDimm(cur[j], cur[j], xstep)
		}
		a.SUBimm(X9, X9, 1)
		a.CBNZ(X9, lp)
		a.Bind(skip)
	}
	loop(X6, func() {
		quad()
		for j := 0; j < nt; j++ {
			a.LDRq(V9, cur[j], 0)
			a.FMLA4s(VReg(j), V8, V9)
		}
	}, 4*esz, 16)
	loop(X10, func() {
		one()
		for j := 0; j < nt; j++ {
			a.LDRs(V9, cur[j], 0)
			a.FMLA4s(VReg(j), V8, V9)
		}
	}, esz, 4)

	a.MOVreg(X7, X1)
	for j := 0; j < nt; j++ {
		v := VReg(j)
		a.FADDP4s(v, v, v)
		a.FADDPs(v, v)
		a.STRs(v, X7, 0)
		a.ADDreg(X7, X7, X8)
	}
	a.ADDimm(X1, X1, 4)
	a.SUBimm(X5, X5, 1)
	a.CBNZ(X5, row)
	a.Bind(done)
	a.RET()
	return a.Bytes(), nil
}
