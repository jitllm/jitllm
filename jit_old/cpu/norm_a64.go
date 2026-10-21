package cpu

// The NEON norm emitters. The assembler (a64.go) builds on every host, so
// these carry no build tag and the golden gate can emit them on amd64.

// EmitA64RMSNorm is the NEON twin of EmitRMSNorm: y = x/sqrt(mean(x^2)+eps) * w.
//
// x0 args, x1 out, x2 x, x3 w, x5 consts, x4 counter.
func EmitA64RMSNorm(n int) []byte {
	var a A64
	a.LDRx(X1, X0, 0)  // Out
	a.LDRx(X2, X0, 16) // A -> x
	a.LDRx(X3, X0, 24) // AScale -> w
	a.LDRx(X5, X0, 56) // Scr -> [1/n, eps, 1]

	// pass 1: sum of squares, four chains.
	// The last n%4 one at a time; LDR s zeroes lanes 1..3, which add nothing.
	sumRowA64(&a, n, X2, func(VReg) {},
		func(dst, r VReg) { a.FMLA4s(dst, r, r) }, false)

	// scale = 1/sqrt(ss/n + eps)
	a.LDRs(5, X5, 0) // 1/n
	a.LDRs(6, X5, 4) // eps
	a.FMULs(0, 0, 5)
	a.FADD4s(0, 0, 6)
	a.FSQRT4s(0, 0)
	a.LDRs(7, X5, 8) // 1.0
	a.FDIV4s(0, 7, 0)
	a.DUPs4(0, 0)

	// pass 2: y = x * scale * w
	a.MOVreg(X6, X2)
	a.MOVreg(X7, X1)
	newA64RowWalk(&a, n, X6, X3, X7).run(func() {
		for i := 0; i < 4; i++ {
			a.LDRq(VReg(4+i), X6, int32(16*i))
			a.LDRq(VReg(8+i), X3, int32(16*i))
			a.FMUL4s(VReg(4+i), VReg(4+i), 0)
			a.FMUL4s(VReg(4+i), VReg(4+i), VReg(8+i))
			a.STRq(VReg(4+i), X7, int32(16*i))
		}
	}, func() {
		a.LDRq(4, X6, 0)
		a.LDRq(8, X3, 0)
		a.FMUL4s(4, 4, 0)
		a.FMUL4s(4, 4, 8)
		a.STRq(4, X7, 0)
	}, func(i int) {
		d := int32(4 * i)
		a.LDRs(4, X6, d)
		a.LDRs(8, X3, d)
		a.FMUL4s(4, 4, 0)
		a.FMUL4s(4, 4, 8)
		a.STRs(4, X7, d)
	})
	a.RET()
	return a.Bytes()
}

// EmitA64LayerNorm is the NEON twin of EmitLayerNorm:
// y = (x - mean)/sqrt(var + eps) * w + b.
//
// Three passes for the amd64 kernel's reason: the one-pass E[x^2]-E[x]^2
// identity cancels in f32 on a residual whose mean dwarfs its variance.
//
// x1 out, x2 x, x3 w, x8 b, x5 consts, x4 counter. v30 holds the mean and v31
// the inverse deviation across the whole body.
func EmitA64LayerNorm(n int, bias bool) []byte { return emitA64RowStat(n, bias, RowLayerNorm) }

// emitA64RowStat is the layer norm, Gemma 3n's gaussian top-k or AltUp's
// magnitude match, as the amd64 emitRowStat; v29 holds the match's target.
func emitA64RowStat(n int, bias bool, mode RowMode) []byte {
	var a A64
	a.LDRx(X1, X0, 0)  // Out -> y
	a.LDRx(X2, X0, 16) // A -> x
	a.LDRx(X3, X0, 24) // AScale -> w
	a.LDRx(X8, X0, 8)  // W -> b
	a.LDRx(X5, X0, 56) // Scr -> [1/n, eps, 1]

	// sum walks x with four chains, leaving the total in lane 0 of v0. sub
	// shapes each loaded vector -- identity for the mean, (x-mean)^2 for the
	// variance.
	//
	// The last n%4 are one at a time and must be neutral in lanes 1..3: LDR s
	// zeroes them, which the mean can take and the variance cannot, so tail
	// keeps lane 0 alone (through v28) before the add.
	sum := func(base XReg, sub func(r VReg), tail bool) {
		sumRowA64(&a, n, base, sub, func(dst, r VReg) { a.FADD4s(dst, dst, r) }, tail)
	}

	// pass 1: the mean, or the match's reference magnitude (v29) with a zero
	// mean.
	if mode == RowMag {
		sum(X3, func(r VReg) { a.FMUL4s(r, r, r) }, false)
		a.LDRs(5, X5, 0) // 1/n
		a.FMULs(0, 0, 5)
		a.FSQRT4s(0, 0)
		a.DUPs4(29, 0)
		a.MOVIzero(30)
	} else {
		sum(X2, func(r VReg) {}, false)
		a.LDRs(5, X5, 0) // 1/n
		a.FMULs(0, 0, 5)
		a.DUPs4(30, 0)
	}

	// pass 2: the variance.
	sum(X2, func(r VReg) {
		a.FSUB4s(r, r, 30)
		a.FMUL4s(r, r, r)
	}, true)
	if mode == RowStd {
		a.LDRs(5, X5, 12) // 1/(n-1)
	} else {
		a.LDRs(5, X5, 0) // 1/n
	}
	a.LDRs(6, X5, 4) // eps
	a.FMULs(0, 0, 5)
	switch mode {
	case RowMag:
		a.FMAX4s(0, 0, 6)
	case RowStd:
	default:
		a.FADD4s(0, 0, 6)
	}
	a.FSQRT4s(0, 0)
	if mode == RowStd {
		a.FADD4s(0, 0, 6) // the deviation plus eps
	}
	a.LDRs(7, X5, 8) // 1.0, or the gaussian's c
	switch mode {
	case RowGauss:
		a.FMULs(0, 0, 7)
		a.FADD4s(0, 0, 30) // the cutoff, mean + c*sd
	case RowMag:
		a.FDIV4s(0, 29, 0) // target / magnitude
	default:
		a.FDIV4s(0, 7, 0)
	}
	a.DUPs4(31, 0)
	if mode == RowGauss {
		a.MOVIzero(28)
	}

	// pass 3: y = (x - mean) * inv * w + b.
	// apply is pass 3 on a loaded vector; ld reads w or b, a vector or one
	// element.
	apply := func(r VReg, off int32, ld func(VReg, XReg, int32)) {
		switch mode {
		case RowGauss:
			a.FSUB4s(r, r, 31)
			a.FMAX4s(r, r, 28)
		case RowMag:
			a.FMUL4s(r, r, 31)
		case RowStd:
			a.FSUB4s(r, r, 30)
			a.FMUL4s(r, r, 31)
		default:
			a.FSUB4s(r, r, 30)
			a.FMUL4s(r, r, 31)
			ld(12, X3, off)
			a.FMUL4s(r, r, 12)
			if bias {
				ld(13, X8, off)
				a.FADD4s(r, r, 13)
			}
		}
	}
	one := func(r VReg, off int32) {
		a.LDRq(r, X6, off)
		apply(r, off, func(v VReg, b XReg, o int32) { a.LDRq(v, b, o) })
		a.STRq(r, X7, off)
	}
	a.MOVreg(X6, X2)
	a.MOVreg(X7, X1)
	newA64RowWalk(&a, n, X6, X3, X8, X7).run(func() {
		for i := 0; i < 4; i++ {
			one(VReg(4+i), int32(16*i))
		}
	}, func() {
		one(4, 0)
	}, func(i int) {
		d := int32(4 * i)
		a.LDRs(4, X6, d)
		apply(4, d, func(v VReg, b XReg, o int32) { a.LDRs(v, b, o) })
		a.STRs(4, X7, d)
	})
	a.RET()
	return a.Bytes()
}

// sumRowA64 reduces a transform of the n floats at base into lane 0 of v0 with
// four chains v0..v3: it clears them, walks X6 from base, and folds. sub shapes
// each loaded vector (identity for the mean, (x-mean)^2 for the variance) and
// acc folds the shaped register r into chain dst.
//
// The last n%4 are one at a time and must be neutral in lanes 1..3: LDR s
// zeroes them, which the mean can take and the variance cannot, so tail keeps
// lane 0 alone (through v28) before the fold.
func sumRowA64(a *A64, n int, base XReg, sub func(r VReg), acc func(dst, r VReg), tail bool) {
	for i := 0; i < 4; i++ {
		a.MOVIzero(VReg(i))
	}
	a.MOVreg(X6, base)
	newA64RowWalk(a, n, X6).run(func() {
		for i := 0; i < 4; i++ {
			a.LDRq(VReg(4+i), X6, int32(16*i))
			sub(VReg(4 + i))
			acc(VReg(i), VReg(4+i))
		}
	}, func() {
		a.LDRq(4, X6, 0)
		sub(4)
		acc(0, 4)
	}, func(i int) {
		a.LDRs(4, X6, int32(4*i))
		sub(4)
		if tail {
			a.MOVIzero(28)
			a.INSs(28, 0, 4, 0)
			a.MOVvec(4, 28)
		}
		acc(0, 4)
	})
	foldChainsA64(a)
}

// foldChainsA64 folds the four chains v0..v3 into lane 0 of v0, in the order
// the amd64 kernels fold their four YMM chains.
func foldChainsA64(a *A64) {
	a.FADD4s(0, 0, 1)
	a.FADD4s(2, 2, 3)
	a.FADD4s(0, 0, 2)
	a.FADDP4s(0, 0, 0)
	a.FADDPs(0, 0)
}
