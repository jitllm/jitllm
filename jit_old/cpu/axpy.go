//go:build amd64

package cpu

// EmitAxpy generates dst[i] += alpha * src[i] over a runtime length.
//
// One kernel for the whole add family: residual adds, attention and projection
// biases, and the MoE's weighted accumulate.
//
// alpha comes through Scr rather than being baked: the MoE's is a routing
// probability, so baking it would key the kernel cache on a float and generate
// one kernel per token. A plain add passes 1; VFMADD231PS makes the multiply
// free on a memory-bound loop.
//
// The length is runtime: Args.K whole vectors of eight and Args.Rows single
// elements after them (see elemLoop), so every width is generated code.
//
// Registers: RCX dst, RDX src, RBX Scr, R10/R11 counters. Y0 holds alpha.
func EmitAxpy() []byte {
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out -> dst, read and written in place
	a.MOVLoad(RDX, At(RDI, 24)) // AScale -> src
	a.MOVLoad(RBX, At(RDI, 56)) // Scr -> &alpha
	a.MOVLoad(R10, At(RDI, 40)) // K -> whole vectors
	a.MOVLoad(R11, At(RDI, 32)) // Rows -> tail elements
	a.VBROADCASTSS(Y0, At(RBX, 0))

	elemLoop(&a, R10, R11, []Reg{RCX, RDX}, func(ld func(Reg, Reg), st func(Reg, Reg)) {
		ld(Y1, RCX)
		ld(Y2, RDX)
		a.VFMADD231PS(Y1, Y0, Y2)
		st(RCX, Y1)
	})

	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// EmitScale generates dst[i] *= alpha over a runtime length: gemma's embedding
// scale and the attention-score scale (text and vision).
//
// alpha comes through Scr for EmitAxpy's reason -- baking it would key the
// kernel cache on a float, and these alphas are per-model or per-head.
//
// Registers: RCX dst, RBX Scr, R10/R11 counters. Y0 holds alpha.
func EmitScale() []byte {
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out -> dst, read and written in place
	a.MOVLoad(RBX, At(RDI, 56)) // Scr -> &alpha
	a.MOVLoad(R10, At(RDI, 40)) // K -> whole vectors
	a.MOVLoad(R11, At(RDI, 32)) // Rows -> tail elements
	a.VBROADCASTSS(Y0, At(RBX, 0))

	elemLoop(&a, R10, R11, []Reg{RCX}, func(ld func(Reg, Reg), st func(Reg, Reg)) {
		ld(Y1, RCX)
		a.VMULPS(Y1, Y1, Y0)
		st(RCX, Y1)
	})

	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// EmitSoftcap generates x = c*tanh(x/c) in place over a runtime length -- the
// attention-score and final-logit cap gemma2 trains with.
//
// tanh comes from the engine's one exp: tanh(y) = 1 - 2/(exp(2y)+1), so the cap
// is c*(1 - 2/(exp(2x/c)+1)). exp's own clamps give both saturations (c and -c)
// without branches.
//
//	Out  x, read and written
//	W    two float32: 2/c and c
//	Scr  ActConsts
func EmitSoftcap() []byte {
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out -> x, in place
	a.MOVLoad(R9, At(RDI, 8))   // W -> {2/c, c}
	a.MOVLoad(RBX, At(RDI, 56)) // Scr -> consts
	a.MOVLoad(R10, At(RDI, 40)) // K -> whole vectors
	a.MOVLoad(R11, At(RDI, 32)) // Rows -> tail elements
	loadExpConsts(&a)
	a.VBROADCASTSS(Y4, At(RBX, 52)) // 2.0
	a.VBROADCASTSS(Y5, At(R9, 0))   // 2/c
	a.VBROADCASTSS(Y6, At(R9, 4))   // c

	elemLoop(&a, R10, R11, []Reg{RCX}, func(ld func(Reg, Reg), st func(Reg, Reg)) {
		ld(Y0, RCX)
		a.VMULPS(Y0, Y0, Y5)
		emitExpPS(&a, Y0, Y1, Y2)
		a.VADDPS(Y0, Y0, Y8) // exp(2x/c) + 1
		a.VDIVPS(Y0, Y4, Y0) // 2/(that)
		a.VSUBPS(Y0, Y8, Y0) // tanh(x/c)
		a.VMULPS(Y0, Y0, Y6)
		st(RCX, Y0)
	})

	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// EmitClamp generates x = min(max(x, lo), hi) in place over a runtime length:
// DBRX's clip_qkv, which clamps q, k and v to [-c, c] after the projection
// and before the rotary (llama.cpp's f_clamp_kqv, transformers' clip_qkv).
//
// lo and hi come through Scr for EmitAxpy's reason -- baking them would key
// the kernel cache on a float -- and as TWO numbers rather than one c, so the
// kernel does no negation and an asymmetric range is the same kernel.
//
//	Out  x, read and written
//	Scr  two float32: lo, hi
//
// Registers: RCX x, RBX Scr, R10/R11 counters. Y0 = lo, Y2 = hi.
func EmitClamp() []byte {
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out -> x, in place
	a.MOVLoad(RBX, At(RDI, 56)) // Scr -> {lo, hi}
	a.MOVLoad(R10, At(RDI, 40)) // K -> whole vectors
	a.MOVLoad(R11, At(RDI, 32)) // Rows -> tail elements
	a.VBROADCASTSS(Y0, At(RBX, 0))
	a.VBROADCASTSS(Y2, At(RBX, 4))

	elemLoop(&a, R10, R11, []Reg{RCX}, func(ld func(Reg, Reg), st func(Reg, Reg)) {
		ld(Y1, RCX)
		a.VMAXPS(Y1, Y1, Y0)
		a.VMINPS(Y1, Y1, Y2)
		st(RCX, Y1)
	})

	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}
