//go:build amd64

package cpu

// EmitActMul generates dst[i] = act(dst[i]) * up[i] over a runtime length.
//
// The activation is baked, not branched: it is a constant from the moment
// model.Open returns, so it is a codegen input.
//
// SiLU is x/(1+exp(-x)).
//
// GELU-tanh is 0.5x(1+tanh(k(x + 0.044715x^3))), and tanh is written as
// 1 - 2/(exp(2z)+1) rather than as its own polynomial: one exp already exists
// here and is correct over the whole range, where a tanh minimax would be a
// second approximation to get wrong. It costs one divide.
//
// Registers: RCX dst, RDX up, RBX consts, R10/R11 counters. Y7..Y15 are exp's
// constants, Y0..Y6 are free.
func EmitActMul(k ActKind) []byte { return emitAct(k, true) }

// EmitSigmoidMul generates dst[i] = sigma(dst[i]) * up[i]: the gate goes in
// dst and what it gates goes in up. qwen3next's attention out-gate and its
// linear layers' recurrent output are both sigma(gate) * value.
func EmitSigmoidMul() []byte { return emitAct(actSigmoid, true) }

// EmitAct is the ungated activation: dst[i] = act(dst[i]), in place, as a
// ViT's FFN needs (fc1, activation, fc2, no up). It takes a kind rather than a
// bool because CLIP's tower is quick-GELU.
func EmitAct(k ActKind) []byte { return emitAct(k, false) }

func emitAct(act ActKind, mul bool) []byte {
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out -> dst (gate), read and written in place
	a.MOVLoad(RDX, At(RDI, 24)) // AScale -> up
	a.MOVLoad(RBX, At(RDI, 56)) // Scr -> consts
	a.MOVLoad(R10, At(RDI, 40)) // K -> whole vectors
	a.MOVLoad(R11, At(RDI, 32)) // Rows -> tail elements
	loadExpConsts(&a)
	cursors := []Reg{RCX}
	if mul {
		cursors = append(cursors, RDX)
	}
	elemLoop(&a, R10, R11, cursors, func(ld func(Reg, Reg), st func(Reg, Reg)) {
		emitActBody(&a, act, mul, ld, st)
	})

	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

func emitActBody(a *Buf, act ActKind, mul bool, ld func(Reg, Reg), st func(Reg, Reg)) {
	ld(Y0, RCX)
	a.VMOVAPSReg(Y3, Y0) // keep x; exp clobbers its argument

	switch act {
	case ActIdentity:
		// x itself: the multiply below is the whole kernel.
	case ActSiLU:
		// SiLU: x / (1 + exp(-x)).
		a.VPXOR(Y4, Y4, Y4)
		a.VSUBPS(Y0, Y4, Y0) // -x
		emitExpPS(a, Y0, Y1, Y2)
		a.VADDPS(Y0, Y0, Y8) // + 1.0
		a.VDIVPS(Y0, Y3, Y0) // x / that
	case ActQuickGELU:
		// quick-GELU: x * sigma(1.702x) = x / (1 + exp(-1.702x)), SiLU with the
		// exponent scaled.
		a.VBROADCASTSS(Y5, At(RBX, 60)) // 1.702
		a.VMULPS(Y0, Y0, Y5)            // 1.702x
		a.VPXOR(Y4, Y4, Y4)
		a.VSUBPS(Y0, Y4, Y0) // -1.702x
		emitExpPS(a, Y0, Y1, Y2)
		a.VADDPS(Y0, Y0, Y8) // + 1.0
		a.VDIVPS(Y0, Y3, Y0) // x / that
	case ActReLU:
		a.VPXOR(Y4, Y4, Y4)
		a.VMAXPS(Y0, Y0, Y4)
	case ActReLU2:
		// Squared ReLU: max(x, 0)^2. No exp; the constants loaded above go
		// unused, which costs nothing inside the loop.
		a.VPXOR(Y4, Y4, Y4)
		a.VMAXPS(Y0, Y0, Y4)
		a.VMULPS(Y0, Y0, Y0)
	case actSigmoid:
		// sigma(x) = 1 / (1 + exp(-x)): SiLU's sequence with the numerator 1
		// instead of x.
		a.VPXOR(Y4, Y4, Y4)
		a.VSUBPS(Y0, Y4, Y0) // -x
		emitExpPS(a, Y0, Y1, Y2)
		a.VADDPS(Y0, Y0, Y8) // + 1.0
		a.VDIVPS(Y0, Y8, Y0) // 1 / that
	case ActSwiGLUOAI:
		// x = min(gate, 7); x*sigma(1.702x) is quick-GELU's sequence on the
		// clamped x. The up side is clamped both ways and offset by one, so it
		// is done here and the generic multiply below is skipped.
		a.VBROADCASTSS(Y5, At(RBX, 64)) // 7
		a.VMINPS(Y0, Y0, Y5)
		a.VMOVAPSReg(Y3, Y0)
		a.VBROADCASTSS(Y5, At(RBX, 60)) // 1.702
		a.VMULPS(Y0, Y0, Y5)
		a.VPXOR(Y4, Y4, Y4)
		a.VSUBPS(Y0, Y4, Y0) // -1.702x
		emitExpPS(a, Y0, Y1, Y2)
		a.VADDPS(Y0, Y0, Y8) // + 1.0
		a.VDIVPS(Y0, Y3, Y0) // x*sigma(1.702x)
		ld(Y4, RDX)
		a.VBROADCASTSS(Y5, At(RBX, 64)) // 7
		a.VMINPS(Y4, Y4, Y5)
		a.VBROADCASTSS(Y5, At(RBX, 68)) // -7
		a.VMAXPS(Y4, Y4, Y5)
		a.VADDPS(Y4, Y4, Y8) // y + 1
		a.VMULPS(Y0, Y0, Y4)
	case ActSwiGLUClamp:
		// x = min(gate, 10), SiLU's sequence on it; up clamped both ways
		// and multiplied here, so the generic multiply below is skipped.
		a.VBROADCASTSS(Y5, At(RBX, 72)) // 10
		a.VMINPS(Y0, Y0, Y5)
		a.VMOVAPSReg(Y3, Y0)
		a.VPXOR(Y4, Y4, Y4)
		a.VSUBPS(Y0, Y4, Y0) // -x
		emitExpPS(a, Y0, Y1, Y2)
		a.VADDPS(Y0, Y0, Y8) // + 1.0
		a.VDIVPS(Y0, Y3, Y0) // x*sigma(x)
		ld(Y4, RDX)
		a.VBROADCASTSS(Y5, At(RBX, 72)) // 10
		a.VMINPS(Y4, Y4, Y5)
		a.VBROADCASTSS(Y5, At(RBX, 76)) // -10
		a.VMAXPS(Y4, Y4, Y5)
		a.VMULPS(Y0, Y0, Y4)
	case ActSqrtSoftplus:
		// softplus(x) = max(x,0) + log(1 + exp(-|x|)); log(1+u) is the atanh
		// series of EmitDeltaGate, u in (0,1]. Then its square root.
		a.VPXOR(Y4, Y4, Y4)
		a.VMAXPS(Y4, Y0, Y4)             // p = max(x, 0)
		a.VBROADCASTSS(Y5, At(RBX, 100)) // the |x| mask
		a.VPAND(Y0, Y0, Y5)              // |x|
		a.VPXOR(Y5, Y5, Y5)
		a.VSUBPS(Y0, Y5, Y0) // -|x|
		emitExpPS(a, Y0, Y1, Y2)
		a.VBROADCASTSS(Y5, At(RBX, 80)) // 2
		a.VADDPS(Y1, Y0, Y5)
		a.VDIVPS(Y2, Y0, Y1) // s = u/(2+u)
		a.VMULPS(Y0, Y2, Y2) // s^2
		a.VBROADCASTSS(Y1, At(RBX, 84))
		for _, off := range []int32{88, 92, 96} {
			a.VBROADCASTSS(Y5, At(RBX, off))
			a.VFMADD213PS(Y1, Y0, Y5)
		}
		a.VFMADD213PS(Y1, Y0, Y8) // ... + 1
		a.VMULPS(Y1, Y1, Y2)
		a.VBROADCASTSS(Y5, At(RBX, 80))
		a.VMULPS(Y1, Y1, Y5) // 2*s*h = log(1+u)
		a.VADDPS(Y0, Y1, Y4) // softplus
		a.VSQRTPS(Y0, Y0)
	case ActSitu:
		// 4*tanh(g/4)*sigma(g) * 25*tanh(up/25), each tanh the rational
		// kernels.TanhClamp describes; up is bounded and multiplied here, so
		// the generic multiply below is skipped.
		bound := func(x Reg, inv, c int32) {
			a.VBROADCASTSS(Y5, At(RBX, inv))
			a.VMULPS(Y0, x, Y5) // x/c
			a.VBROADCASTSS(Y5, At(RBX, 120))
			a.VMINPS(Y0, Y0, Y5)
			a.VBROADCASTSS(Y5, At(RBX, 124))
			a.VMAXPS(Y0, Y0, Y5) // clamped
			a.VMULPS(Y2, Y0, Y0) // x^2
			a.VBROADCASTSS(Y1, At(RBX, 128))
			for off := int32(132); off <= 152; off += 4 {
				a.VBROADCASTSS(Y5, At(RBX, off))
				a.VFMADD213PS(Y1, Y2, Y5)
			}
			a.VMULPS(Y1, Y1, Y0) // x*P(x^2)
			a.VBROADCASTSS(Y4, At(RBX, 156))
			for off := int32(160); off <= 168; off += 4 {
				a.VBROADCASTSS(Y5, At(RBX, off))
				a.VFMADD213PS(Y4, Y2, Y5)
			}
			a.VDIVPS(Y0, Y1, Y4) // tanh
			a.VBROADCASTSS(Y5, At(RBX, c))
			a.VMULPS(Y0, Y0, Y5) // c*tanh
		}
		bound(Y3, 104, 108)
		a.VMOVAPSReg(Y6, Y0)
		a.VPXOR(Y4, Y4, Y4)
		a.VSUBPS(Y0, Y4, Y3) // -g
		emitExpPS(a, Y0, Y1, Y2)
		a.VADDPS(Y0, Y0, Y8) // 1 + exp(-g)
		a.VDIVPS(Y6, Y6, Y0) // 4*tanh(g/4)*sigma(g)
		ld(Y3, RDX)
		bound(Y3, 112, 116)
		a.VMULPS(Y0, Y0, Y6)
	case ActSqrtSigmoid:
		// sigmoid(sign(x)*sqrt(max(|x|, floor))): the magnitude's root, the
		// sign put back as a bit, then sigmoid's sequence.
		a.VBROADCASTSS(Y5, At(RBX, 100)) // the |x| mask
		a.VPAND(Y4, Y3, Y5)              // |x|
		a.VBROADCASTSS(Y5, At(RBX, 208)) // the floor
		a.VMAXPS(Y4, Y4, Y5)
		a.VSQRTPS(Y4, Y4)
		a.VBROADCASTSS(Y5, At(RBX, 212)) // the sign bit
		a.VPAND(Y5, Y3, Y5)
		a.VPOR(Y0, Y4, Y5) // the signed root
		a.VPXOR(Y4, Y4, Y4)
		a.VSUBPS(Y0, Y4, Y0)
		emitExpPS(a, Y0, Y1, Y2)
		a.VADDPS(Y0, Y0, Y8)
		a.VDIVPS(Y0, Y8, Y0)
	case ActGELUErf:
		// 0.5*((x + |x|) - |x|*q), q = 1 - erf(|x|/sqrt 2) by Abramowitz and
		// Stegun 7.1.26 (kernels.ActGELUErf).
		a.VBROADCASTSS(Y5, At(RBX, 100)) // the |x| mask
		a.VPAND(Y4, Y3, Y5)              // |x|
		a.VBROADCASTSS(Y5, At(RBX, 180)) // 1/sqrt 2
		a.VMULPS(Y6, Y4, Y5)             // z
		a.VMULPS(Y0, Y6, Y6)
		a.VPXOR(Y1, Y1, Y1)
		a.VSUBPS(Y0, Y1, Y0) // -z^2
		emitExpPS(a, Y0, Y1, Y2)
		a.VBROADCASTSS(Y5, At(RBX, 184)) // p
		a.VMULPS(Y6, Y6, Y5)
		a.VADDPS(Y6, Y6, Y8)
		a.VDIVPS(Y6, Y8, Y6)             // t = 1/(1 + p z)
		a.VBROADCASTSS(Y1, At(RBX, 204)) // a5
		for _, off := range []int32{200, 196, 192, 188} {
			a.VBROADCASTSS(Y5, At(RBX, off))
			a.VFMADD213PS(Y1, Y6, Y5)
		}
		a.VMULPS(Y1, Y1, Y6) // the polynomial times t
		a.VMULPS(Y1, Y1, Y0) // q
		a.VMULPS(Y1, Y1, Y4) // |x| q
		a.VADDPS(Y0, Y3, Y4) // x + |x|
		a.VSUBPS(Y0, Y0, Y1)
		a.VBROADCASTSS(Y5, At(RBX, 56)) // 0.5
		a.VMULPS(Y0, Y0, Y5)
	default:
		// GELU-tanh. z = k*(x + 0.044715 x^3); the cube is two multiplies and
		// an FMA rather than a pow.
		a.VMULPS(Y4, Y3, Y3)            // x^2
		a.VBROADCASTSS(Y5, At(RBX, 44)) // 0.044715
		a.VMULPS(Y4, Y4, Y5)            // 0.044715 x^2
		a.VADDPS(Y4, Y4, Y8)            // 1 + 0.044715 x^2
		a.VMULPS(Y4, Y4, Y3)            // x + 0.044715 x^3
		a.VBROADCASTSS(Y5, At(RBX, 48)) // sqrt(2/pi)
		a.VMULPS(Y0, Y4, Y5)            // z
		a.VBROADCASTSS(Y5, At(RBX, 52)) // 2.0
		a.VMULPS(Y0, Y0, Y5)            // 2z
		emitExpPS(a, Y0, Y1, Y2)        // exp(2z)
		a.VADDPS(Y0, Y0, Y8)            // exp(2z) + 1
		a.VBROADCASTSS(Y5, At(RBX, 52)) // 2.0
		a.VDIVPS(Y0, Y5, Y0)            // 2/(exp(2z)+1)
		// tanh = 1 - that, and 1 + tanh = 2 - that.
		a.VBROADCASTSS(Y5, At(RBX, 52)) // 2.0
		a.VSUBPS(Y0, Y5, Y0)            // 1 + tanh(z)
		a.VBROADCASTSS(Y5, At(RBX, 56)) // 0.5
		a.VMULPS(Y0, Y0, Y5)
		a.VMULPS(Y0, Y0, Y3) // 0.5 x (1 + tanh)
	}

	if mul && act != ActSwiGLUOAI && act != ActSwiGLUClamp && act != ActSitu {
		ld(Y4, RDX)
		a.VMULPS(Y0, Y0, Y4)
	}
	st(RCX, Y0)
}
