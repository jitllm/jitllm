//go:build amd64

package cpu

// EmitRMSNorm generates y[i] = x[i] * w[i] / sqrt(mean(x^2) + eps).
//
// One kernel, not the device's two (partial sums, then apply): one CPU worker
// owns the whole vector, so the reduction and the apply need no barrier and
// take one pool region.
//
// n is baked: the trip count and the tail are compile-time, as the matvec bakes
// K. eps and 1/n arrive through Scr because they are per-model floats, and
// baking them would key the kernel cache on a float.
//
// Registers: RCX out, RSI x, RDX w, RBX Scr, R10 counter, R11/R8 cursors.
func EmitRMSNorm(n int) []byte {
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out -> y
	a.MOVLoad(RSI, At(RDI, 16)) // A, reinterpreted as *float32 -> x
	a.MOVLoad(RDX, At(RDI, 24)) // AScale -> w
	a.MOVLoad(RBX, At(RDI, 56)) // Scr -> [invN, eps]

	// ---- pass 1: sum of squares, four independent chains ----
	//
	// Four accumulators: one chain makes this latency-bound on the FP add.
	// The tail, in eights, is baked rather than looped: n is known here, so a
	// model whose n_embd is not a multiple of 32 costs a few more instructions
	// once rather than a branch per token. The last n%8 are one at a time: a
	// scalar load zeroes lanes 1..7, so they add nothing to the sum.
	sumRowAVX2(&a, n,
		func(reg Reg, off int32) { a.VMOVDQULoad(reg, At(R11, off)) },
		func(reg Reg, off int32) { a.VMOVSSLoad(reg, At(R11, off)) },
		func(dst, t Reg) { a.VFMADD231PS(dst, t, t) })

	// ---- the scalar tail: scale = 1 / sqrt(ss/n + eps) ----
	//
	// Done in vector registers rather than moved to a GPR: there is no
	// float-to-GPR move in this assembler and no reason to want one, since the
	// result has to be broadcast back across a YMM anyway.
	a.VBROADCASTSS(Y5, At(RBX, 0)) // 1/n
	a.VBROADCASTSS(Y6, At(RBX, 4)) // eps
	a.VMULPS(Y0, Y0, Y5)
	a.VADDPS(Y0, Y0, Y6)
	a.VSQRTPS(Y0, Y0)
	a.VBROADCASTSS(Y7, At(RBX, 8)) // 1.0
	a.VDIVPS(Y0, Y7, Y0)
	a.VBROADCASTSSReg(Y0, Y0) // splat lane 0 across the whole ymm

	// ---- pass 2: y = x * scale * w ----
	a.MOVQ(R11, RSI)
	a.MOVQ(R8, RCX)
	newRowWalk(&a, n, R11, RDX, R8).run(func() {
		for i := 0; i < 4; i++ {
			d := int32(32 * i)
			a.VMOVDQULoad(Y4, At(R11, d))
			a.VMULPS(Y4, Y4, Y0)
			a.VMULPSMem(Y4, Y4, At(RDX, d))
			a.VMOVDQUStore(At(R8, d), Y4)
		}
	}, func() {
		a.VMOVDQULoad(Y4, At(R11, 0))
		a.VMULPS(Y4, Y4, Y0)
		a.VMULPSMem(Y4, Y4, At(RDX, 0))
		a.VMOVDQUStore(At(R8, 0), Y4)
	}, func(i int) {
		d := int32(4 * i)
		a.VMOVSSLoad(Y4, At(R11, d))
		a.VMULPS(Y4, Y4, Y0)
		a.VMOVSSLoad(Y5, At(RDX, d))
		a.VMULPS(Y4, Y4, Y5)
		a.VMOVSSStore(At(R8, d), Y4)
	})

	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// sumRowAVX2 reduces a transform of the n floats at RSI into lane 0 of Y0 with
// four chains Y0..Y3: it clears them, walks R11 from RSI, and folds. load puts
// one vector at an offset from R11 into reg, loadOne one element with lanes
// 1..7 zero; both may transform what they load (the variance subtracts the mean
// and squares). acc folds the loaded register t into chain dst.
func sumRowAVX2(a *Buf, n int, load, loadOne func(reg Reg, off int32), acc func(dst, t Reg)) {
	a.VPXOR(Y0, Y0, Y0)
	a.VPXOR(Y1, Y1, Y1)
	a.VPXOR(Y2, Y2, Y2)
	a.VPXOR(Y3, Y3, Y3)
	a.MOVQ(R11, RSI)
	newRowWalk(a, n, R11).run(func() {
		for i := 0; i < 4; i++ {
			load(Reg(int(Y4)+i), int32(32*i))
			acc(Reg(int(Y0)+i), Reg(int(Y4)+i))
		}
	}, func() {
		load(Y4, 0)
		acc(Y0, Y4)
	}, func(i int) {
		loadOne(Y4, int32(4*i))
		acc(Y0, Y4)
	})
	foldChainsAVX2(a)
}

// foldChainsAVX2 folds the four chains Y0..Y3 into lane 0 of Y0, in the order
// the SSE and NEON twins mirror.
func foldChainsAVX2(a *Buf) {
	a.VADDPS(Y0, Y0, Y1)
	a.VADDPS(Y2, Y2, Y3)
	a.VADDPS(Y0, Y0, Y2)
	a.VEXTRACTF128(Y1, Y0, 1)
	a.VADDPSx(Y0, Y0, Y1)
	a.VHADDPSx(Y0, Y0, Y0)
	a.VHADDPSx(Y0, Y0, Y0) // lane 0 = the total
}
