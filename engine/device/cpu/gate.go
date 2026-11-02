//go:build amd64

package cpu

// The linear block's two gates, fused into one kernel. They are only 64 scalars
// a layer, but nothing on a token's path runs interpreted, and transcendentals
// are expensive per element in scalar Go.
//
// decay[i] = exp(A[i] * softplus(a[i] + dt[i]))
// beta[i]  = sigmoid(b[i])
//
// softplus is computed in the |z| form. log(1+exp(z)) overflows for z above
// ~88 (llama.cpp guards it with a branch at 20); a vector kernel cannot
// branch, and the identity
//
//	softplus(z) = max(z,0) + log(1 + exp(-|z|))
//
// keeps the exp argument non-positive, so its result is in (0,1] and the
// logarithm's argument is in (1,2]. That bound is also what lets the series be
// five terms (see deltaGateConsts).

// EmitDeltaGate generates both gates for K vectors.
//
//	Out     decay, written
//	Out2    beta, written
//	W       a, the alpha half of the ba projection
//	AScale  dt, the per-head bias
//	AScale2 b, the beta half of the ba projection
//	Q32     A, the per-head decay rate (already negative)
//	Scr     DeltaGateConsts
//	K       whole vectors of ElemLanes
//	Rows    single elements after them
func EmitDeltaGate() []byte {
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))   // Out:     decay
	a.MOVLoad(RSI, At(RDI, 120)) // Out2:    beta
	a.MOVLoad(RDX, At(RDI, 8))   // W:       a
	a.MOVLoad(R8, At(RDI, 24))   // AScale:  dt
	a.MOVLoad(R11, At(RDI, 136)) // AScale2: b
	a.MOVLoad(R9, At(RDI, 112))  // Q32:     A
	a.MOVLoad(RBX, At(RDI, 56))  // Scr:     consts
	a.MOVLoad(R10, At(RDI, 40))  // K:       whole vectors
	a.MOVLoad(RAX, At(RDI, 32))  // Rows:    tail elements
	loadExpConsts(&a)

	elemLoop(&a, R10, RAX, []Reg{RCX, RSI, RDX, R8, R9, R11}, func(ld func(Reg, Reg), st func(Reg, Reg)) {
		// z = a + dt, and the two halves of softplus's |z| form.
		ld(Y0, RDX)
		ld(Y6, R8)
		a.VADDPS(Y0, Y0, Y6)
		a.VPXOR(Y4, Y4, Y4)
		a.VMAXPS(Y4, Y0, Y4) // p = max(z, 0)
		a.VBROADCASTSS(Y5, At(RBX, 64))
		a.VPAND(Y0, Y0, Y5) // |z|
		a.VPXOR(Y5, Y5, Y5)
		a.VSUBPS(Y0, Y5, Y0) // -|z|
		emitExpPS(&a, Y0, Y1, Y2)

		// log(1+u) = 2*s*(1 + s^2/3 + s^4/5 + s^6/7 + s^8/9), s = u/(2+u).
		a.VBROADCASTSS(Y5, At(RBX, 44)) // 2
		a.VADDPS(Y1, Y0, Y5)
		a.VDIVPS(Y2, Y0, Y1) // s
		a.VMULPS(Y0, Y2, Y2) // s^2
		a.VBROADCASTSS(Y1, At(RBX, 48))
		for _, off := range []int32{52, 56, 60} {
			a.VBROADCASTSS(Y5, At(RBX, off))
			a.VFMADD213PS(Y1, Y0, Y5)
		}
		a.VFMADD213PS(Y1, Y0, Y8) // ... + 1
		a.VMULPS(Y1, Y1, Y2)
		a.VBROADCASTSS(Y5, At(RBX, 44))
		a.VMULPS(Y1, Y1, Y5) // 2*s*h
		a.VADDPS(Y0, Y1, Y4) // softplus
		ld(Y6, R9)
		a.VMULPS(Y0, Y0, Y6) // * A
		emitExpPS(&a, Y0, Y1, Y2)
		st(RCX, Y0)

		// beta = sigmoid(b) = 1 / (1 + exp(-b)).
		a.VPXOR(Y4, Y4, Y4)
		ld(Y0, R11)
		a.VSUBPS(Y0, Y4, Y0)
		emitExpPS(&a, Y0, Y1, Y2)
		a.VADDPS(Y0, Y0, Y8)
		a.VDIVPS(Y0, Y8, Y0)
		st(RSI, Y0)
	})

	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// EmitDeltaDecayBound generates Kimi-K3's bounded KDA decay for K vectors:
//
//	decay[i] = exp(lb * sigma(-A[i] * (a[i] + dt[i])))
//	         = exp(lb / (1 + exp(A[i] * (a[i] + dt[i]))))
//
// fla's lower_bound gate: lb*sigmoid(exp(A_log)*(g + dt_bias)), with A the
// container's -exp(A_log). Its argument cannot overflow the exp's range in a
// way that matters: past the clamp the quotient is 0 or lb, the bound's ends.
// beta is EmitDeltaGate's, which the caller runs for it.
//
//	Out     decay, written
//	W       a, the forget gate's projection
//	AScale  dt, the per-channel bias
//	Q32     A, the per-channel decay rate (already negative)
//	AScale2 lb, one float
//	Scr     DeltaGateConsts
//	K       whole vectors of ElemLanes
//	Rows    single elements after them
func EmitDeltaDecayBound() []byte {
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))   // Out:     decay
	a.MOVLoad(RDX, At(RDI, 8))   // W:       a
	a.MOVLoad(R8, At(RDI, 24))   // AScale:  dt
	a.MOVLoad(R11, At(RDI, 136)) // AScale2: lb
	a.MOVLoad(R9, At(RDI, 112))  // Q32:     A
	a.MOVLoad(RBX, At(RDI, 56))  // Scr:     consts
	a.MOVLoad(R10, At(RDI, 40))  // K:       whole vectors
	a.MOVLoad(RAX, At(RDI, 32))  // Rows:    tail elements
	loadExpConsts(&a)
	a.VBROADCASTSS(Y6, At(R11, 0)) // lb, held

	elemLoop(&a, R10, RAX, []Reg{RCX, RDX, R8, R9}, func(ld func(Reg, Reg), st func(Reg, Reg)) {
		ld(Y0, RDX)
		ld(Y1, R8)
		a.VADDPS(Y0, Y0, Y1) // z = a + dt
		ld(Y1, R9)
		a.VMULPS(Y0, Y0, Y1) // A*z
		emitExpPS(&a, Y0, Y1, Y2)
		a.VADDPS(Y0, Y0, Y8) // 1 + exp(A*z)
		a.VDIVPS(Y0, Y6, Y0) // lb * sigma(-A*z)
		emitExpPS(&a, Y0, Y1, Y2)
		st(RCX, Y0)
	})

	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// EmitSSDGate generates Mamba-2's two per-head scalars for K vectors:
//
//	dt[i]    = softplus(a[i] + bias[i])
//	decay[i] = exp(A[i] * dt[i])
//
// It is EmitDeltaGate's decay with the softplus itself kept as the second
// output instead of a sigmoid of another stream: the selective update scales
// the value by dt, so dt is an operand and not only the decay's exponent.
//
//	Out     decay, written
//	Out2    dt, written
//	W       a, the dt projection
//	AScale  the per-head dt bias
//	Q32     A, the per-head decay rate (already negative)
//	Scr     DeltaGateConsts
//	K       whole vectors of ElemLanes
//	Rows    single elements after them
func EmitSSDGate() []byte {
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))   // Out:    decay
	a.MOVLoad(RSI, At(RDI, 120)) // Out2:   dt
	a.MOVLoad(RDX, At(RDI, 8))   // W:      a
	a.MOVLoad(R8, At(RDI, 24))   // AScale: bias
	a.MOVLoad(R9, At(RDI, 112))  // Q32:    A
	a.MOVLoad(RBX, At(RDI, 56))  // Scr:    consts
	a.MOVLoad(R10, At(RDI, 40))  // K:      whole vectors
	a.MOVLoad(RAX, At(RDI, 32))  // Rows:   tail elements
	loadExpConsts(&a)

	elemLoop(&a, R10, RAX, []Reg{RCX, RSI, RDX, R8, R9}, func(ld func(Reg, Reg), st func(Reg, Reg)) {
		ld(Y0, RDX)
		ld(Y6, R8)
		a.VADDPS(Y0, Y0, Y6)
		a.VPXOR(Y4, Y4, Y4)
		a.VMAXPS(Y4, Y0, Y4) // p = max(z, 0)
		a.VBROADCASTSS(Y5, At(RBX, 64))
		a.VPAND(Y0, Y0, Y5) // |z|
		a.VPXOR(Y5, Y5, Y5)
		a.VSUBPS(Y0, Y5, Y0) // -|z|
		emitExpPS(&a, Y0, Y1, Y2)
		a.VBROADCASTSS(Y5, At(RBX, 44)) // 2
		a.VADDPS(Y1, Y0, Y5)
		a.VDIVPS(Y2, Y0, Y1) // s
		a.VMULPS(Y0, Y2, Y2) // s^2
		a.VBROADCASTSS(Y1, At(RBX, 48))
		for _, off := range []int32{52, 56, 60} {
			a.VBROADCASTSS(Y5, At(RBX, off))
			a.VFMADD213PS(Y1, Y0, Y5)
		}
		a.VFMADD213PS(Y1, Y0, Y8) // ... + 1
		a.VMULPS(Y1, Y1, Y2)
		a.VBROADCASTSS(Y5, At(RBX, 44))
		a.VMULPS(Y1, Y1, Y5) // 2*s*h
		a.VADDPS(Y0, Y1, Y4) // softplus
		st(RSI, Y0)
		ld(Y6, R9)
		a.VMULPS(Y0, Y0, Y6) // * A
		emitExpPS(&a, Y0, Y1, Y2)
		st(RCX, Y0)
	})

	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}
