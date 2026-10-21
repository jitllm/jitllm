//go:build arm64

package cpu

import "fmt"

// The arm64 gated delta rule, the NEON twin of jit_old/cpu/delta.go. A NEON vector
// is four lanes, so every loop runs twice the iterations of the amd64 version.
// Every n is served: the last n%4 elements are baked scalar code, LDR s zeroing
// lanes 1..3 so the dots add them as zero.
func EmitGatedDelta(n int) ([]byte, error) { return emitGatedDelta(n, false) }

// EmitGatedDeltaChan is EmitGatedDelta with a per-channel decay in AScale2, the
// NEON twin of the amd64 emitter.
func EmitGatedDeltaChan(n int) ([]byte, error) { return emitGatedDelta(n, true) }

func emitGatedDelta(n int, perChan bool) ([]byte, error) {
	name := "EmitGatedDelta"
	if perChan {
		name = "EmitGatedDeltaChan"
	}
	if n <= 0 {
		return nil, fmt.Errorf("jit: %s: n=%d must be positive", name, n)
	}
	nv, tail := n/4, n%4
	toff := func(i int) int32 { return int32(nv*16 + 4*i) }
	var a A64
	a.LDRx(X1, X0, 0)   // Out: o
	a.LDRx(X2, X0, 8)   // W:   the state, row 0
	a.LDRx(X3, X0, 24)  // AScale: k
	a.LDRx(X4, X0, 32)  // Rows: n
	a.LDRx(X5, X0, 56)  // Scr: {decay, beta}
	a.LDRx(X6, X0, 112) // Q32: q
	a.LDRx(X7, X0, 128) // Q2:  v
	if perChan {
		a.LDRx(X8, X0, 136) // AScale2: the per-channel decay, n floats
	} else {
		a.LDRs(V16, X5, 0)
		a.DUPs4(V16, V16) // decay
	}
	a.LDRs(V17, X5, 4)
	a.DUPs4(V17, V17) // beta

	hsum := func() {
		a.FADD4s(V0, V0, V1)
		a.FADD4s(V2, V2, V3)
		a.FADD4s(V0, V0, V2)
		// Two pairwise adds reduce four lanes to one; arm64 has no HADDPS.
		a.FADDP4s(V0, V0, V0)
		a.FADDP4s(V0, V0, V0)
	}
	zero := func() {
		for i := 0; i < 4; i++ {
			a.MOVIzero(VReg(i))
		}
	}

	row := a.Label()
	a.Bind(row)
	// Pass one: decay the row in place and dot it with the key.
	zero()
	for i := 0; i < nv; i++ {
		a.LDRq(V8, X2, int32(i)*16)
		if perChan {
			a.LDRq(V16, X8, int32(i)*16)
		}
		a.FMUL4s(V8, V8, V16)
		a.STRq(V8, X2, int32(i)*16)
		a.LDRq(V9, X3, int32(i)*16)
		a.FMLA4s(VReg(i%4), V8, V9)
	}
	for i := 0; i < tail; i++ {
		a.LDRs(V8, X2, toff(i))
		if perChan {
			a.LDRs(V16, X8, toff(i))
		}
		a.FMUL4s(V8, V8, V16)
		a.STRs(V8, X2, toff(i))
		a.LDRs(V9, X3, toff(i))
		a.FMLA4s(V0, V8, V9)
	}
	hsum()
	// d = (v[j] - sk) * beta, in lane 0 and then broadcast.
	a.DUPs4(V0, V0)
	a.LDRs(V18, X7, 0)
	a.DUPs4(V18, V18)
	a.FSUB4s(V18, V18, V0)
	a.FMUL4s(V18, V18, V17)
	// Pass two: the rank-one update, and the dot with the query off the UPDATED
	// row -- which is why they share a pass and why the order matters.
	zero()
	for i := 0; i < nv; i++ {
		a.LDRq(V8, X2, int32(i)*16)
		a.LDRq(V9, X3, int32(i)*16)
		a.FMLA4s(V8, V18, V9)
		a.STRq(V8, X2, int32(i)*16)
		a.LDRq(V10, X6, int32(i)*16)
		a.FMLA4s(VReg(i%4), V8, V10)
	}
	for i := 0; i < tail; i++ {
		a.LDRs(V8, X2, toff(i))
		a.LDRs(V9, X3, toff(i))
		a.FMLA4s(V8, V18, V9)
		a.STRs(V8, X2, toff(i))
		a.LDRs(V10, X6, toff(i))
		a.FMLA4s(V0, V8, V10)
	}
	hsum()
	a.STRs(V0, X1, 0)

	a.ADDimm(X1, X1, 4)
	a.ADDimm(X7, X7, 4)
	a.addBig(X2, int32(n)*4, X9)
	a.SUBimm(X4, X4, 1)
	a.CBNZ(X4, row)

	a.RET()
	return a.Bytes(), nil
}

// EmitGatedSSD is Mamba-2's selective state update, the NEON twin of
// jit_old/cpu/delta.go's: per row j, S[j] = S[j]*decay + (x[j]*dt)*B and
// o[j] = dot(S[j], C) + D*x[j], in one pass. Scr is {decay, dt, D} and Rows
// the head's rows; n, the state size, is baked.
func EmitGatedSSD(n int) ([]byte, error) {
	if n <= 0 {
		return nil, fmt.Errorf("jit: EmitGatedSSD: n=%d must be positive", n)
	}
	nv, tail := n/4, n%4
	toff := func(i int) int32 { return int32(nv*16 + 4*i) }
	var a A64
	a.LDRx(X1, X0, 0)   // Out: o
	a.LDRx(X2, X0, 8)   // W:   the state, row 0
	a.LDRx(X3, X0, 24)  // AScale: B
	a.LDRx(X4, X0, 32)  // Rows
	a.LDRx(X5, X0, 56)  // Scr: {decay, dt, D}
	a.LDRx(X6, X0, 112) // Q32: C
	a.LDRx(X7, X0, 128) // Q2:  x
	a.LDRs(V16, X5, 0)
	a.DUPs4(V16, V16) // decay
	a.LDRs(V17, X5, 4)
	a.DUPs4(V17, V17)  // dt
	a.LDRs(V19, X5, 8) // D, lane 0

	row := a.Label()
	a.Bind(row)
	for i := 0; i < 4; i++ {
		a.MOVIzero(VReg(i))
	}
	a.LDRs(V18, X7, 0)
	a.DUPs4(V18, V18)
	a.FMUL4s(V18, V18, V17) // x[j]*dt
	for i := 0; i < nv; i++ {
		a.LDRq(V8, X2, int32(i)*16)
		a.FMUL4s(V8, V8, V16)
		a.LDRq(V9, X3, int32(i)*16)
		a.FMLA4s(V8, V18, V9)
		a.STRq(V8, X2, int32(i)*16)
		a.LDRq(V10, X6, int32(i)*16)
		a.FMLA4s(VReg(i%4), V8, V10)
	}
	for i := 0; i < tail; i++ {
		a.LDRs(V8, X2, toff(i))
		a.FMUL4s(V8, V8, V16)
		a.LDRs(V9, X3, toff(i))
		a.FMLA4s(V8, V18, V9)
		a.STRs(V8, X2, toff(i))
		a.LDRs(V10, X6, toff(i))
		a.FMLA4s(V0, V8, V10)
	}
	a.FADD4s(V0, V0, V1)
	a.FADD4s(V2, V2, V3)
	a.FADD4s(V0, V0, V2)
	a.FADDP4s(V0, V0, V0)
	a.FADDP4s(V0, V0, V0)
	// + D*x[j] in lane 0, one fused rounding.
	a.LDRs(V9, X7, 0)
	a.FMLA4s(V0, V19, V9)
	a.STRs(V0, X1, 0)

	a.ADDimm(X1, X1, 4)
	a.ADDimm(X7, X7, 4)
	a.addBig(X2, int32(n)*4, X9)
	a.SUBimm(X4, X4, 1)
	a.CBNZ(X4, row)

	a.RET()
	return a.Bytes(), nil
}

// EmitSelScan is Mamba-1's selective scan, the NEON twin of
// jit_old/cpu/delta.go's: per channel j, S[j] = S[j]*exp(dt[j]*A[j]) +
// (x[j]*dt[j])*B and o[j] = dot(S[j], C) + D[j]*x[j], in one pass. The
// argument slots are the amd64 kernel's; n, the state size, is baked.
func EmitSelScan(n int) ([]byte, error) {
	if n <= 0 {
		return nil, fmt.Errorf("jit: EmitSelScan: n=%d must be positive", n)
	}
	nv, tail := n/4, n%4
	toff := func(i int) int32 { return int32(nv*16 + 4*i) }
	var a A64
	a.LDRx(X1, X0, 0)    // Out: o
	a.LDRx(X2, X0, 8)    // W:   the state, channel 0
	a.LDRx(X3, X0, 24)   // AScale: B
	a.LDRx(X4, X0, 32)   // Rows
	a.LDRx(X5, X0, 56)   // Scr: consts
	a.LDRx(X8, X0, 64)   // AHalf: A
	a.LDRx(X6, X0, 112)  // Q32: C
	a.LDRx(X7, X0, 128)  // Q2:  x
	a.LDRx(X10, X0, 136) // AScale2: dt
	a.LDRx(X11, X0, 144) // PD: D
	loadA64ExpConsts(&a) // v16..v26

	row := a.Label()
	a.Bind(row)
	a.MOVIzero(V0)
	a.LDRs(V1, X10, 0)
	a.DUPs4(V1, V1) // dt[j]
	a.LDRs(V2, X7, 0)
	a.DUPs4(V2, V2)
	a.FMUL4s(V2, V2, V1) // x[j]*dt[j]
	step := func(ld func(d VReg, b XReg, o int32), st func(s VReg, o int32), off int32) {
		ld(V3, X8, off)
		a.FMUL4s(V3, V3, V1)
		emitA64Exp(&a, V3, V4, V5) // the decay
		ld(V6, X2, off)
		a.FMUL4s(V6, V6, V3)
		ld(V4, X3, off)
		a.FMLA4s(V6, V2, V4)
		st(V6, off)
		ld(V5, X6, off)
		a.FMLA4s(V0, V6, V5)
	}
	for i := 0; i < nv; i++ {
		step(func(d VReg, b XReg, o int32) { a.LDRq(d, b, o) },
			func(s VReg, o int32) { a.STRq(s, X2, o) }, int32(i)*16)
	}
	for i := 0; i < tail; i++ {
		// A scalar load zeroes the other lanes; they add nothing to the dot.
		step(func(d VReg, b XReg, o int32) { a.LDRs(d, b, o) },
			func(s VReg, o int32) { a.STRs(s, X2, o) }, toff(i))
	}
	a.FADDP4s(V0, V0, V0)
	a.FADDP4s(V0, V0, V0)
	// + D[j]*x[j] in lane 0, one fused rounding.
	a.LDRs(V4, X7, 0)
	a.LDRs(V5, X11, 0)
	a.FMLA4s(V0, V4, V5)
	a.STRs(V0, X1, 0)

	a.ADDimm(X1, X1, 4)
	a.ADDimm(X7, X7, 4)
	a.ADDimm(X10, X10, 4)
	a.ADDimm(X11, X11, 4)
	a.addBig(X2, int32(n)*4, X9)
	a.addBig(X8, int32(n)*4, X9)
	a.SUBimm(X4, X4, 1)
	a.CBNZ(X4, row)

	a.RET()
	return a.Bytes(), nil
}
