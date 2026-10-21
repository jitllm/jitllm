//go:build amd64

package cpu

// EmitLayerNorm generates y = (x - mean) / sqrt(var + eps) * w + b, the vision
// tower's norm.
//
// Three passes, not the one-pass E[x^2]-E[x]^2 identity: a ViT residual carries
// a mean far larger than its variance, so the identity cancels most of the
// mantissa in f32.
//
// bias is baked rather than tested per element, since whether the file carries
// one is a property of the model. Args.W carries the bias pointer (the one free
// pointer slot; this kernel reads no packed weights).
//
// Registers: RCX out, RSI x, RDX w, R9 b, RBX Scr, R10 counter, R11/R8 cursors.
// Y8 holds the mean, Y9 the inverse deviation and Y10 zero across the body.
func EmitLayerNorm(n int, bias bool) []byte { return emitRowStat(n, bias, RowLayerNorm) }

// EmitGaussTopK generates Gemma 3n's activation sparsity on one gate row:
// y = max(x - (mean + c*sqrt(var + eps)), 0), the population variance, with
// Scr = [1/n, eps, c]. Args as EmitLayerNorm's, with no w and no b.
func EmitGaussTopK(n int) []byte { return emitRowStat(n, false, RowGauss) }

// EmitMagMatch generates AltUp's magnitude match: y = x * sqrt(mean(r^2)) /
// sqrt(max(mean(x^2), eps)), r the reference row in AScale, with
// Scr = [1/n, eps, 1].
func EmitMagMatch(n int) []byte { return emitRowStat(n, false, RowMag) }

// EmitRowStd generates a row standardised over its unbiased deviation
// (RowStd).
func EmitRowStd(n int) []byte { return emitRowStat(n, false, RowStd) }

func emitRowStat(n int, bias bool, mode RowMode) []byte {
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out -> y
	a.MOVLoad(RSI, At(RDI, 16)) // A, reinterpreted as *float32 -> x
	a.MOVLoad(RDX, At(RDI, 24)) // AScale -> w
	a.MOVLoad(R9, At(RDI, 8))   // W, reinterpreted as *float32 -> b
	a.MOVLoad(RBX, At(RDI, 56)) // Scr -> [1/n, eps, 1]

	// ---- pass 1: the mean ----
	// The last n%8 elements are unrolled one at a time, and each pass's tail
	// must be neutral in lanes 1..7: a scalar load zeroes them, which the mean
	// can take but the variance cannot -- (0 - mean)^2 is not zero -- so the
	// variance tail blends those lanes back to zero from Y10.
	a.VPXOR(Y10, Y10, Y10)
	sum := func(sub, subOne func(reg Reg, off int32)) {
		sumRowAVX2(&a, n, sub, subOne, func(dst, t Reg) { a.VADDPS(dst, dst, t) })
	}

	if mode == RowMag {
		// The reference's magnitude, sqrt(mean(r^2)), in Y11; pass 2 then
		// squares x about a zero mean.
		a.MOVQ(RSI, RDX)
		sum(func(reg Reg, off int32) { a.VMOVDQULoad(reg, At(R11, off)); a.VMULPS(reg, reg, reg) },
			func(reg Reg, off int32) { a.VMOVSSLoad(reg, At(R11, off)); a.VMULPS(reg, reg, reg) })
		a.MOVLoad(RSI, At(RDI, 16))
		a.VBROADCASTSS(Y5, At(RBX, 0)) // 1/n
		a.VMULPS(Y0, Y0, Y5)
		a.VSQRTPS(Y0, Y0)
		a.VBROADCASTSSReg(Y11, Y0)
		a.VPXOR(Y8, Y8, Y8)
	} else {
		sum(func(reg Reg, off int32) { a.VMOVDQULoad(reg, At(R11, off)) },
			func(reg Reg, off int32) { a.VMOVSSLoad(reg, At(R11, off)) })
		a.VBROADCASTSS(Y5, At(RBX, 0)) // 1/n
		a.VMULPS(Y0, Y0, Y5)
		a.VBROADCASTSSReg(Y8, Y0) // the mean, live from here on
	}

	// ---- pass 2: the variance ----
	sum(func(reg Reg, off int32) {
		a.VMOVDQULoad(reg, At(R11, off))
		a.VSUBPS(reg, reg, Y8)
		a.VMULPS(reg, reg, reg)
	}, func(reg Reg, off int32) {
		a.VMOVSSLoad(reg, At(R11, off))
		a.VSUBPS(reg, reg, Y8)
		a.VMULPS(reg, reg, reg)
		a.VPBLENDD(reg, Y10, reg, 0x01)
	})
	if mode == RowStd {
		a.VBROADCASTSS(Y5, At(RBX, 12)) // 1/(n-1)
	} else {
		a.VBROADCASTSS(Y5, At(RBX, 0)) // 1/n
	}
	a.VBROADCASTSS(Y6, At(RBX, 4)) // eps
	a.VMULPS(Y0, Y0, Y5)
	switch mode {
	case RowMag:
		a.VMAXPS(Y0, Y0, Y6) // max(mean(x^2), eps)
	case RowStd:
	default:
		a.VADDPS(Y0, Y0, Y6)
	}
	a.VSQRTPS(Y0, Y0)
	if mode == RowStd {
		a.VADDPS(Y0, Y0, Y6) // the deviation plus eps
	}
	a.VBROADCASTSS(Y7, At(RBX, 8)) // 1.0, or the gaussian's c
	switch mode {
	case RowGauss:
		// The cutoff, mean + c*sd: what lies above it survives.
		a.VMULPS(Y0, Y0, Y7)
		a.VADDPS(Y0, Y0, Y8)
	case RowMag:
		a.VDIVPS(Y0, Y11, Y0) // target / magnitude
	default:
		a.VDIVPS(Y0, Y7, Y0)
	}
	a.VBROADCASTSSReg(Y9, Y0)

	// ---- pass 3: y = (x - mean) * inv * w + b ----
	one := func(src, dst Reg, off int32) {
		a.VMOVDQULoad(src, At(R11, off))
		switch mode {
		case RowGauss:
			a.VSUBPS(src, src, Y9)
			a.VMAXPS(src, src, Y10)
		case RowMag:
			a.VMULPS(src, src, Y9)
		case RowStd:
			a.VSUBPS(src, src, Y8)
			a.VMULPS(src, src, Y9)
		default:
			a.VSUBPS(src, src, Y8)
			a.VMULPS(src, src, Y9)
			a.VMULPSMem(src, src, At(RDX, off))
			if bias {
				a.VADDPSMem(src, src, At(R9, off))
			}
		}
		a.VMOVDQUStore(At(dst, off), src)
	}
	a.MOVQ(R11, RSI)
	a.MOVQ(R8, RCX)
	newRowWalk(&a, n, R11, RDX, R9, R8).run(func() {
		for i := 0; i < 4; i++ {
			one(Reg(int(Y4)+i), R8, int32(32*i))
		}
	}, func() {
		one(Y4, R8, 0)
	}, func(i int) {
		d := int32(4 * i)
		a.VMOVSSLoad(Y4, At(R11, d))
		switch mode {
		case RowGauss:
			a.VSUBPS(Y4, Y4, Y9)
			a.VMAXPS(Y4, Y4, Y10)
		case RowMag:
			a.VMULPS(Y4, Y4, Y9)
		case RowStd:
			a.VSUBPS(Y4, Y4, Y8)
			a.VMULPS(Y4, Y4, Y9)
		default:
			a.VSUBPS(Y4, Y4, Y8)
			a.VMULPS(Y4, Y4, Y9)
			a.VMOVSSLoad(Y5, At(RDX, d))
			a.VMULPS(Y4, Y4, Y5)
			if bias {
				a.VMOVSSLoad(Y6, At(R9, d))
				a.VADDPS(Y4, Y4, Y6)
			}
		}
		a.VMOVSSStore(At(R8, d), Y4)
	})

	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}
