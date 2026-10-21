//go:build amd64

package cpu

import "fmt"

// The gated delta rule: a hybrid's answer to attention, and its hot kernel.
// It reads a fixed n x n state per head at every position, so unlike attention
// its cost does not shrink at short context.
//
// One call is one head: n rows of n, which is 3n^2 multiply-accumulates. The
// caller runs the heads across the pool, so nothing here parallelises and
// nothing here can nest inside a pool region.
//
// The recurrence per row j of the state S, with k and q the convolved key and
// query, v the value, g the decay and b the gate:
//
//	S[j] *= g
//	sk    = dot(S[j], k)
//	d     = (v[j] - sk) * b
//	S[j] += d*k
//	o[j]  = dot(S[j], q)
//
// Two passes over the row: d needs the whole of sk, so decay-and-dot must
// finish before update-and-dot starts, but the row stays in L1 between them.
//
// The state is read-modify-write. That is forbidden on the device (RULE 13)
// but fine here: a single-threaded host kernel over a private row.
//
// n is baked: it is the container's SSM.StateSize.

// EmitGatedDelta generates one head of the delta rule.
//
//	Out    o, n floats, written
//	W      the state, n*n float32, read and written
//	AScale k, n floats
//	Q32    q, n floats
//	Q2     v, n floats
//	Scr    two float32: the decay and the gate
//	Rows   n
func EmitGatedDelta(n int) ([]byte, error) { return emitGatedDelta(n, false) }

// EmitGatedDeltaChan is EmitGatedDelta with a per-channel decay, which is
// Kimi-Linear's delta rule. AScale2 carries n float32 and the row is scaled by
// decay[i] instead of by the single rate in Scr; Scr[0] is then unread and only
// the gate at Scr[4] is used. The decay varies along the key dimension, which
// in this tree's transposed state is the within-row index the loop already
// walks, so it is the same loop with one operand wider (see
// oracle.GatedDeltaChan).
func EmitGatedDeltaChan(n int) ([]byte, error) { return emitGatedDelta(n, true) }

func emitGatedDelta(n int, perChan bool) ([]byte, error) {
	name := "EmitGatedDelta"
	if perChan {
		name = "EmitGatedDeltaChan"
	}
	if n <= 0 {
		return nil, fmt.Errorf("jit: %s: n=%d must be positive", name, n)
	}
	// n is baked, so the last n%8 elements of each pass are unrolled scalar
	// code: a scalar load zeroes lanes 1..7, which the dots then add as zero.
	nv, tail := n/8, n%8
	toff := func(i int) int32 { return int32(nv*32 + 4*i) }
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out: o
	a.MOVLoad(RDX, At(RDI, 8))  // W:   the state, row 0
	a.MOVLoad(RSI, At(RDI, 24)) // AScale: k
	a.MOVLoad(RAX, At(RDI, 32)) // Rows: n
	a.MOVLoad(R10, At(RDI, 56)) // Scr: {decay, beta}
	a.MOVLoad(R8, At(RDI, 112)) // Q32: q
	a.MOVLoad(R9, At(RDI, 128)) // Q2:  v
	if perChan {
		a.MOVLoad(R11, At(RDI, 136)) // AScale2: the per-channel decay, n floats
	} else {
		a.VBROADCASTSS(Y10, At(R10, 0))
	}
	a.VBROADCASTSS(Y11, At(R10, 4))

	// Four accumulator chains, for the reason the k-quant kernels record: one
	// chain is latency-bound on the FMA rather than throughput-bound.
	hsum := func() {
		a.VADDPS(Y0, Y0, Y1)
		a.VADDPS(Y2, Y2, Y3)
		a.VADDPS(Y0, Y0, Y2)
		a.VEXTRACTF128(Y1, Y0, 1)
		a.VADDPSx(Y0, Y0, Y1)
		a.VHADDPSx(Y0, Y0, Y0)
		a.VHADDPSx(Y0, Y0, Y0)
	}
	zero := func() {
		for i := 0; i < 4; i++ {
			a.VPXOR(Reg(i), Reg(i), Reg(i))
		}
	}

	row := a.Label()
	a.Bind(row)
	// Pass one: decay the row in place and dot it with the key.
	zero()
	for i := 0; i < nv; i++ {
		a.VMOVDQULoad(Y8, At(RDX, int32(i)*32))
		if perChan {
			a.VMULPSMem(Y8, Y8, At(R11, int32(i)*32))
		} else {
			a.VMULPS(Y8, Y8, Y10)
		}
		a.VMOVDQUStore(At(RDX, int32(i)*32), Y8)
		a.VFMADD231PSMem(Reg(i%4), Y8, At(RSI, int32(i)*32))
	}
	for i := 0; i < tail; i++ {
		a.VMOVSSLoad(Y8, At(RDX, toff(i)))
		if perChan {
			a.VMOVSSLoad(Y9, At(R11, toff(i)))
			a.VMULPS(Y8, Y8, Y9)
		} else {
			a.VMULPS(Y8, Y8, Y10)
		}
		a.VMOVSSStore(At(RDX, toff(i)), Y8)
		a.VMOVSSLoad(Y9, At(RSI, toff(i)))
		a.VFMADD231PS(Y0, Y8, Y9)
	}
	hsum()
	// d = (v[j] - sk) * beta, computed in a vector lane (the assembler has no
	// scalar float ops); it is broadcast immediately afterwards anyway.
	a.VBROADCASTSSReg(Y0, Y0)
	a.VBROADCASTSS(Y13, At(R9, 0))
	a.VSUBPS(Y13, Y13, Y0)
	a.VMULPS(Y13, Y13, Y11)
	// Pass two: the rank-one update, and the dot with the query off the UPDATED
	// row -- which is why they share a pass and why the order matters.
	zero()
	for i := 0; i < nv; i++ {
		a.VMOVDQULoad(Y8, At(RDX, int32(i)*32))
		a.VMOVDQULoad(Y9, At(RSI, int32(i)*32))
		a.VFMADD231PS(Y8, Y13, Y9)
		a.VMOVDQUStore(At(RDX, int32(i)*32), Y8)
		a.VFMADD231PSMem(Reg(i%4), Y8, At(R8, int32(i)*32))
	}
	for i := 0; i < tail; i++ {
		a.VMOVSSLoad(Y8, At(RDX, toff(i)))
		a.VMOVSSLoad(Y9, At(RSI, toff(i)))
		a.VFMADD231PS(Y8, Y13, Y9)
		a.VMOVSSStore(At(RDX, toff(i)), Y8)
		a.VMOVSSLoad(Y9, At(R8, toff(i)))
		a.VFMADD231PS(Y0, Y8, Y9)
	}
	hsum()
	a.VMOVSSStore(At(RCX, 0), Y0)

	a.ADDimm(RCX, 4)
	a.ADDimm(R9, 4)
	a.ADDimm(RDX, int32(n)*4)
	a.DEC(RAX)
	a.JNZ(row)

	a.VZEROUPPER()
	a.RET()
	return a.Bytes(), nil
}

// EmitGatedSSD generates one head of Mamba-2's selective state update, the
// delta rule with its key dot removed: the state is decayed and the scaled
// value written in along the key, with no subtraction and no gate.
//
//	Out    o, Rows floats, written
//	W      the state, Rows*n float32, read and written
//	AScale B (the key), n floats
//	Q32    C (the query), n floats
//	Q2     x (the value), Rows floats
//	Scr    three float32: the decay exp(A*dt), dt, and D
//	Rows   the head's rows (its head_dim), which need not be n
//
// Per row j, in llama.cpp's ssm_scan order:
//
//	S[j] = S[j]*decay + (x[j]*dt)*B
//	o[j] = dot(S[j], C) + D*x[j]
//
// One pass: nothing reads the row before the update, so the decay, the update
// and the query dot share it. n (the state size) is baked.
func EmitGatedSSD(n int) ([]byte, error) {
	if n <= 0 {
		return nil, fmt.Errorf("jit: EmitGatedSSD: n=%d must be positive", n)
	}
	nv, tail := n/8, n%8
	toff := func(i int) int32 { return int32(nv*32 + 4*i) }
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out: o
	a.MOVLoad(RDX, At(RDI, 8))  // W:   the state, row 0
	a.MOVLoad(RSI, At(RDI, 24)) // AScale: B
	a.MOVLoad(RAX, At(RDI, 32)) // Rows
	a.MOVLoad(R10, At(RDI, 56)) // Scr: {decay, dt, D}
	a.MOVLoad(R8, At(RDI, 112)) // Q32: C
	a.MOVLoad(R9, At(RDI, 128)) // Q2:  x
	a.VBROADCASTSS(Y10, At(R10, 0))
	a.VBROADCASTSS(Y11, At(R10, 4))
	a.VBROADCASTSS(Y12, At(R10, 8))

	hsum := func() {
		a.VADDPS(Y0, Y0, Y1)
		a.VADDPS(Y2, Y2, Y3)
		a.VADDPS(Y0, Y0, Y2)
		a.VEXTRACTF128(Y1, Y0, 1)
		a.VADDPSx(Y0, Y0, Y1)
		a.VHADDPSx(Y0, Y0, Y0)
		a.VHADDPSx(Y0, Y0, Y0)
	}

	row := a.Label()
	a.Bind(row)
	for i := 0; i < 4; i++ {
		a.VPXOR(Reg(i), Reg(i), Reg(i))
	}
	// x[j]*dt, broadcast: the value written into every column of the row.
	a.VBROADCASTSS(Y13, At(R9, 0))
	a.VMULPS(Y13, Y13, Y11)
	for i := 0; i < nv; i++ {
		a.VMOVDQULoad(Y8, At(RDX, int32(i)*32))
		a.VMULPS(Y8, Y8, Y10)
		a.VMOVDQULoad(Y9, At(RSI, int32(i)*32))
		a.VFMADD231PS(Y8, Y13, Y9)
		a.VMOVDQUStore(At(RDX, int32(i)*32), Y8)
		a.VFMADD231PSMem(Reg(i%4), Y8, At(R8, int32(i)*32))
	}
	for i := 0; i < tail; i++ {
		a.VMOVSSLoad(Y8, At(RDX, toff(i)))
		a.VMULPS(Y8, Y8, Y10)
		a.VMOVSSLoad(Y9, At(RSI, toff(i)))
		a.VFMADD231PS(Y8, Y13, Y9)
		a.VMOVSSStore(At(RDX, toff(i)), Y8)
		a.VMOVSSLoad(Y9, At(R8, toff(i)))
		a.VFMADD231PS(Y0, Y8, Y9)
	}
	hsum()
	// + D*x[j], one fused rounding as the device's Fma.
	a.VMOVSSLoad(Y9, At(R9, 0))
	a.VFMADD231PS(Y0, Y12, Y9)
	a.VMOVSSStore(At(RCX, 0), Y0)

	a.ADDimm(RCX, 4)
	a.ADDimm(R9, 4)
	a.ADDimm(RDX, int32(n)*4)
	a.DEC(RAX)
	a.JNZ(row)

	a.VZEROUPPER()
	a.RET()
	return a.Bytes(), nil
}

// EmitSelScan generates Mamba-1's selective scan for Rows channels, the
// per-channel form of EmitGatedSSD: every channel is a head of one row whose
// decay is a vector over the state, exp(dt[j]*A[j]), where Mamba-2's is one
// scalar a head.
//
//	Out     o, Rows floats, written
//	W       the state, Rows*n float32, read and written
//	AScale  B, n floats, shared by every channel
//	Q32     C, n floats, shared by every channel
//	Q2      x, Rows floats
//	AScale2 dt, Rows floats, already through softplus
//	AHalf   A, Rows*n floats (already -exp(A_log))
//	PD      D, Rows floats
//	Scr     DeltaGateConsts (only its exp block is read)
//
// Per channel j, in llama.cpp's ssm_scan order:
//
//	S[j][s] = S[j][s]*exp(dt[j]*A[j][s]) + (x[j]*dt[j])*B[s]
//	o[j]    = dot(S[j], C) + D[j]*x[j]
//
// n (the state size) is baked. The decay is computed in the pass that updates
// the row, so no Rows*n plane of decays is ever written.
func EmitSelScan(n int) ([]byte, error) {
	if n <= 0 {
		return nil, fmt.Errorf("jit: EmitSelScan: n=%d must be positive", n)
	}
	nv, tail := n/8, n%8
	toff := func(i int) int32 { return int32(nv*32 + 4*i) }
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))   // Out: o
	a.MOVLoad(RDX, At(RDI, 8))   // W:   the state, channel 0
	a.MOVLoad(RSI, At(RDI, 24))  // AScale: B
	a.MOVLoad(RAX, At(RDI, 32))  // Rows
	a.MOVLoad(RBX, At(RDI, 56))  // Scr: consts
	a.MOVLoad(R10, At(RDI, 64))  // AHalf: A
	a.MOVLoad(R8, At(RDI, 112))  // Q32: C
	a.MOVLoad(R9, At(RDI, 128))  // Q2:  x
	a.MOVLoad(R11, At(RDI, 136)) // AScale2: dt
	a.MOVLoad(R12, At(RDI, 144)) // PD: D
	loadExpConsts(&a)            // Y7..Y15

	row := a.Label()
	a.Bind(row)
	a.VPXOR(Y0, Y0, Y0)
	a.VBROADCASTSS(Y1, At(R11, 0)) // dt[j]
	a.VBROADCASTSS(Y2, At(R9, 0))
	a.VMULPS(Y2, Y2, Y1) // x[j]*dt[j]
	step := func(ld func(Reg, Reg, int32), st func(Reg, int32), off int32) {
		ld(Y3, R10, off)
		a.VMULPS(Y3, Y3, Y1)
		emitExpPS(&a, Y3, Y4, Y5) // the decay
		ld(Y6, RDX, off)
		a.VMULPS(Y6, Y6, Y3)
		ld(Y4, RSI, off)
		a.VFMADD231PS(Y6, Y2, Y4)
		st(Y6, off)
		ld(Y5, R8, off)
		a.VFMADD231PS(Y0, Y6, Y5)
	}
	for i := 0; i < nv; i++ {
		step(func(d, b Reg, o int32) { a.VMOVDQULoad(d, At(b, o)) },
			func(s Reg, o int32) { a.VMOVDQUStore(At(RDX, o), s) }, int32(i)*32)
	}
	for i := 0; i < tail; i++ {
		// A scalar load zeroes the other lanes, which stay zero through the
		// update and add nothing to the dot.
		step(func(d, b Reg, o int32) { a.VMOVSSLoad(d, At(b, o)) },
			func(s Reg, o int32) { a.VMOVSSStore(At(RDX, o), s) }, toff(i))
	}
	a.VEXTRACTF128(Y1, Y0, 1)
	a.VADDPSx(Y0, Y0, Y1)
	a.VHADDPSx(Y0, Y0, Y0)
	a.VHADDPSx(Y0, Y0, Y0)
	// + D[j]*x[j], one fused rounding.
	a.VMOVSSLoad(Y4, At(R9, 0))
	a.VMOVSSLoad(Y5, At(R12, 0))
	a.VFMADD231PS(Y0, Y4, Y5)
	a.VMOVSSStore(At(RCX, 0), Y0)

	a.ADDimm(RCX, 4)
	a.ADDimm(R9, 4)
	a.ADDimm(R11, 4)
	a.ADDimm(R12, 4)
	a.ADDimm(RDX, int32(n)*4)
	a.ADDimm(R10, int32(n)*4)
	a.DEC(RAX)
	a.JNZ(row)

	a.VZEROUPPER()
	a.RET()
	return a.Bytes(), nil
}
