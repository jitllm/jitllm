package cpu

import "fmt"

// The SSE tier's hybrid kernels: the gated delta rule, the causal convolution
// and the two gates -- delta.go, conv.go and gate.go, legacy-encoded. The ABI
// is the AVX2 kernels' argument for argument (nn picks the emitter through
// cpu.EmittersFor); the gate's K is still a count of ElemLanes = 8 floats, run
// as two XMM halves.
//
// The recurrence's tails are scalar ops, not packed ops on a zero-padded
// register: 0*d is NaN once d is infinite, and a NaN in an upper lane reaches
// the output through the horizontal sum. The convolution's tails keep packed
// ops, since every operand there is a MOVSS load with exact zero upper lanes.
//
// There is no FMA, so none of these is bit-identical to its AVX2 twin; the
// gates hold them to the AVX2 kernels' bounds against internal/oracle.
//
// Each declares only what it uses (SSE2, plus SSE3 for the delta rule's
// HADDPS), since the declaration is what Map checks against the host; useISA
// panics if an undeclared op lands.

// EmitGatedDeltaSSE is EmitGatedDelta for the SSE tier: one head of the delta
// rule, n rows of n.
//
//	Out    o, n floats, written
//	W      the state, n*n float32, READ AND WRITTEN
//	AScale k, n floats
//	Q32    q, n floats
//	Q2     v, n floats
//	Scr    two float32: the decay and the gate
//	Rows   n
//
// Per row j, exactly as the AVX2 kernel (delta.go):
//
//	S[j] *= decay;  sk = dot(S[j], k)                    pass one
//	d = (v[j] - sk) * gate
//	S[j] += d*k;    o[j] = dot(S[j], q)                  pass two
//
// n is baked: the whole vectors of four are unrolled and the last n%4 are
// scalar code.
func EmitGatedDeltaSSE(n int) ([]byte, error) { return emitGatedDeltaSSE(n, false) }

// EmitGatedDeltaChanSSE is EmitGatedDeltaSSE with a PER-CHANNEL decay in
// AScale2. See jit_old/cpu/delta.go for why it is one operand and not one more
// kernel.
func EmitGatedDeltaChanSSE(n int) ([]byte, error) { return emitGatedDeltaSSE(n, true) }

func emitGatedDeltaSSE(n int, perChan bool) ([]byte, error) {
	name := "EmitGatedDeltaSSE"
	if perChan {
		name = "EmitGatedDeltaChanSSE"
	}
	if n <= 0 {
		return nil, fmt.Errorf("jit: %s: n=%d must be positive", name, n)
	}
	nv, tail := n/4, n%4
	voff := func(i int) int32 { return int32(i) * 16 }
	toff := func(i int) int32 { return int32(nv*16 + 4*i) }

	var a Buf
	a.DeclareISA(ISASSE2 | ISASSE3)
	a.MOVLoad(RCX, At(RDI, 0))  // Out: o
	a.MOVLoad(RDX, At(RDI, 8))  // W:   the state, row 0
	a.MOVLoad(RSI, At(RDI, 24)) // AScale: k
	a.MOVLoad(RAX, At(RDI, 32)) // Rows: n
	a.MOVLoad(R10, At(RDI, 56)) // Scr: {decay, gate}
	a.MOVLoad(R8, At(RDI, 112)) // Q32: q
	a.MOVLoad(R9, At(RDI, 128)) // Q2:  v
	if perChan {
		a.MOVLoad(R11, At(RDI, 136)) // AScale2: the per-channel decay, n floats
	} else {
		bcastSS(&a, XMM10, At(R10, 0))
	}
	bcastSS(&a, XMM11, At(R10, 4))

	// Four accumulator chains (XMM0..XMM3), for the AVX2 kernel's reason: one
	// chain is bound by the add's latency rather than its throughput.
	zero := func() {
		for i := 0; i < 4; i++ {
			a.XORPS(Reg(i), Reg(i), Reg(i))
		}
	}
	// hsum leaves the four chains' total in every lane of XMM0.
	hsum := func() {
		a.ADDPS(XMM0, XMM0, XMM1)
		a.ADDPS(XMM2, XMM2, XMM3)
		a.ADDPS(XMM0, XMM0, XMM2)
		hsum4SSE(&a, XMM0)
	}

	row := a.Label()
	a.Bind(row)

	// Pass one: decay the row in place and dot it with the key.
	zero()
	for i := 0; i < nv; i++ {
		a.MOVUPSLoad(XMM8, At(RDX, voff(i)))
		if perChan {
			a.MOVUPSLoad(XMM10, At(R11, voff(i)))
		}
		a.MULPS(XMM8, XMM8, XMM10)
		a.MOVUPSStore(At(RDX, voff(i)), XMM8)
		a.MOVUPSLoad(XMM9, At(RSI, voff(i)))
		a.MULPS(XMM9, XMM9, XMM8)
		a.ADDPS(Reg(i%4), Reg(i%4), XMM9)
	}
	for i := 0; i < tail; i++ {
		a.MOVSSLoad(XMM8, At(RDX, toff(i)))
		if perChan {
			a.MOVSSLoad(XMM10, At(R11, toff(i)))
		}
		a.MULSS(XMM8, XMM8, XMM10)
		a.MOVSSStore(At(RDX, toff(i)), XMM8)
		a.MOVSSLoad(XMM9, At(RSI, toff(i)))
		a.MULSS(XMM9, XMM9, XMM8)
		a.ADDSS(XMM0, XMM0, XMM9)
	}
	hsum()

	// d = (v[j] - sk) * gate, in every lane: sk already is.
	bcastSS(&a, XMM13, At(R9, 0))
	a.SUBPS(XMM13, XMM13, XMM0)
	a.MULPS(XMM13, XMM13, XMM11)

	// Pass two: the rank-one update, and the dot with the query off the
	// UPDATED row -- which is why they share a pass and why the order matters.
	zero()
	for i := 0; i < nv; i++ {
		a.MOVUPSLoad(XMM8, At(RDX, voff(i)))
		a.MOVUPSLoad(XMM9, At(RSI, voff(i)))
		a.MULPS(XMM9, XMM9, XMM13)
		a.ADDPS(XMM8, XMM8, XMM9)
		a.MOVUPSStore(At(RDX, voff(i)), XMM8)
		a.MOVUPSLoad(XMM9, At(R8, voff(i)))
		a.MULPS(XMM9, XMM9, XMM8)
		a.ADDPS(Reg(i%4), Reg(i%4), XMM9)
	}
	for i := 0; i < tail; i++ {
		a.MOVSSLoad(XMM8, At(RDX, toff(i)))
		a.MOVSSLoad(XMM9, At(RSI, toff(i)))
		a.MULSS(XMM9, XMM9, XMM13)
		a.ADDSS(XMM8, XMM8, XMM9)
		a.MOVSSStore(At(RDX, toff(i)), XMM8)
		a.MOVSSLoad(XMM9, At(R8, toff(i)))
		a.MULSS(XMM9, XMM9, XMM8)
		a.ADDSS(XMM0, XMM0, XMM9)
	}
	hsum()
	a.MOVSSStore(At(RCX, 0), XMM0)

	a.ADDimm(RCX, 4)
	a.ADDimm(R9, 4)
	a.ADDimm(RDX, int32(n)*4)
	a.DEC(RAX)
	a.JNZ(row)

	a.RET()
	return a.Bytes(), nil
}

// EmitConv1dSSE is EmitConv1d for the SSE tier: the causal convolution over
// plane-major state and its one-position shift, in one pass.
//
//	Out    out, chans floats; may alias Q32
//	W      the state, (taps-1)*chans float32, plane-major, READ AND WRITTEN
//	AScale the transposed weights, taps*chans float32, plane-major
//	Q32    x, the new column, chans floats
//
//	out[c] = x[c]*w[taps-1][c] + sum over t of state[t][c]*w[t][c]
//	state[t] = state[t+1], state[taps-2] = x
//
// taps and chans are baked. Groups of four channels run through a loop; the
// last chans%4 are unrolled one channel at a time.
func EmitConv1dSSE(taps, chans int) ([]byte, error) {
	if taps < 2 || taps > 8 {
		return nil, fmt.Errorf("jit: EmitConv1dSSE: taps=%d is outside [2,8]", taps)
	}
	if chans <= 0 {
		return nil, fmt.Errorf("jit: EmitConv1dSSE: chans=%d must be positive", chans)
	}
	plane := int32(chans) * 4
	var a Buf
	a.DeclareISA(ISASSE2)
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RDX, At(RDI, 8))  // W:      the state planes
	a.MOVLoad(RSI, At(RDI, 24)) // AScale: the transposed weights
	a.MOVLoad(R8, At(RDI, 112)) // Q32:    the new column

	// body is one group: four channels through ld/st at off 0 of the
	// advancing cursors, or ONE channel at off in the tail. XMM15 carries a
	// weight plane and then its product; XMM2..XMM(taps) hold every state plane
	// at once, because the shift below writes them back one plane down.
	body := func(ld func(Reg, Mem), st func(Mem, Reg), off int32) {
		// The new column and its own tap -- the LAST, because the new position
		// is the most recent entry of the window.
		ld(XMM0, At(R8, off))
		ld(XMM15, At(RSI, int32(taps-1)*plane+off))
		a.MULPS(XMM1, XMM0, XMM15)
		for t := 0; t < taps-1; t++ {
			ld(Reg(2+t), At(RDX, int32(t)*plane+off))
			ld(XMM15, At(RSI, int32(t)*plane+off))
			a.MULPS(XMM15, XMM15, Reg(2+t))
			a.ADDPS(XMM1, XMM1, XMM15)
		}
		// state[t] = state[t+1], and the newest column lands in the last plane.
		for t := 0; t < taps-2; t++ {
			st(At(RDX, int32(t)*plane+off), Reg(3+t))
		}
		st(At(RDX, int32(taps-2)*plane+off), XMM0)
		// Out may alias Q32, so it is written only after the last read of x.
		st(At(RCX, off), XMM1)
	}
	if groups := chans / 4; groups > 0 {
		a.MOVimm(RAX, int64(groups))
		lp := a.Label()
		a.Bind(lp)
		body(a.MOVUPSLoad, a.MOVUPSStore, 0)
		a.ADDimm(RCX, 16)
		a.ADDimm(RDX, 16)
		a.ADDimm(RSI, 16)
		a.ADDimm(R8, 16)
		a.DEC(RAX)
		a.JNZ(lp)
	}
	for i := 0; i < chans%4; i++ {
		body(a.MOVSSLoad, a.MOVSSStore, int32(4*i))
	}

	a.RET()
	return a.Bytes(), nil
}

// EmitDeltaGateSSE is EmitDeltaGate for the SSE tier: both of a linear
// block's gates over K units of ElemLanes and Rows single elements.
//
//	Out     decay, written
//	Out2    beta, written
//	W       a, the alpha half of the ba projection
//	AScale  dt, the per-head bias
//	AScale2 b, the beta half of the ba projection
//	Q32     A, the per-head decay rate (already negative)
//	Scr     DeltaGateConsts
//	K       whole units of ElemLanes
//	Rows    single elements after them
//
//	decay[i] = exp(A[i] * softplus(a[i] + dt[i]))
//	beta[i]  = sigmoid(b[i])
//
// softplus is the AVX2 kernel's |z| form, max(z,0) + log(1+exp(-|z|)), with
// log(1+u) as the five-term atanh series in s = u/(2+u) (gate.go and
// deltaGateConsts): branch-free, and the logarithm's argument is always in
// (1,2].
//
// Its constants stay in the Scr block and are re-read: exp's constants take
// XMM7..XMM15 and the body needs five of XMM0..XMM6 at once.
func EmitDeltaGateSSE() ([]byte, error) {
	var a Buf
	a.DeclareISA(ISASSE2)
	a.MOVLoad(RCX, At(RDI, 0))   // Out:     decay
	a.MOVLoad(RSI, At(RDI, 120)) // Out2:    beta
	a.MOVLoad(RDX, At(RDI, 8))   // W:       a
	a.MOVLoad(R8, At(RDI, 24))   // AScale:  dt
	a.MOVLoad(R11, At(RDI, 136)) // AScale2: b
	a.MOVLoad(R9, At(RDI, 112))  // Q32:     A
	a.MOVLoad(RBX, At(RDI, 56))  // Scr:     consts
	a.MOVLoad(R10, At(RDI, 40))  // K:       whole units
	a.MOVLoad(RAX, At(RDI, 32))  // Rows:    tail elements
	loadExpConstsSSE(&a)         // XMM7..XMM15; XMM8 is 1.0

	elemLoopSSE(&a, R10, RAX, []Reg{RCX, RSI, RDX, R8, R9, R11}, func(ld func(Reg, Reg), st func(Reg, Reg)) {
		// z = a + dt, and the two halves of softplus's |z| form.
		ld(XMM0, RDX)
		ld(XMM6, R8)
		a.ADDPS(XMM0, XMM0, XMM6)
		// p = max(z, 0), with z FIRST: MAXPS returns its second operand on a
		// NaN or a signed-zero tie, and the AVX2 kernel's VMAXPS(p, z, 0) is
		// that order too.
		a.XORPS(XMM5, XMM5, XMM5)
		a.MAXPS(XMM4, XMM0, XMM5)
		bcastSS(&a, XMM5, At(RBX, 64))
		a.ANDPS(XMM0, XMM0, XMM5) // |z|
		a.XORPS(XMM5, XMM5, XMM5)
		a.SUBPS(XMM5, XMM5, XMM0) // -|z|
		emitExpSSE(&a, XMM5, XMM1, XMM2)

		// log(1+u) = 2*s*(1 + s^2/3 + s^4/5 + s^6/7 + s^8/9), s = u/(2+u).
		bcastSS(&a, XMM6, At(RBX, 44)) // 2
		a.ADDPS(XMM1, XMM5, XMM6)
		a.DIVPS(XMM2, XMM5, XMM1) // s
		a.MULPS(XMM0, XMM2, XMM2) // s^2
		bcastSS(&a, XMM1, At(RBX, 48))
		for _, off := range []int32{52, 56, 60} {
			a.MULPS(XMM1, XMM1, XMM0)
			bcastSS(&a, XMM6, At(RBX, off))
			a.ADDPS(XMM1, XMM1, XMM6)
		}
		a.MULPS(XMM1, XMM1, XMM0)
		a.ADDPS(XMM1, XMM1, XMM8) // ... + 1
		a.MULPS(XMM1, XMM1, XMM2)
		bcastSS(&a, XMM6, At(RBX, 44))
		a.MULPS(XMM1, XMM1, XMM6) // 2*s*h
		a.ADDPS(XMM0, XMM1, XMM4) // softplus
		ld(XMM6, R9)
		a.MULPS(XMM0, XMM0, XMM6) // * A
		emitExpSSE(&a, XMM0, XMM1, XMM2)
		st(RCX, XMM0)

		// beta = sigmoid(b) = 1 / (1 + exp(-b)).
		a.XORPS(XMM4, XMM4, XMM4)
		ld(XMM0, R11)
		a.SUBPS(XMM4, XMM4, XMM0)
		emitExpSSE(&a, XMM4, XMM1, XMM2)
		a.ADDPS(XMM4, XMM4, XMM8)
		a.MOVAPS(XMM0, XMM8)
		a.DIVPS(XMM0, XMM0, XMM4)
		st(RSI, XMM0)
	})

	a.RET()
	return a.Bytes(), nil
}

// EmitDeltaDecayBoundSSE is EmitDeltaDecayBound for the SSE tier, its
// arguments and its sequence: exp(lb / (1 + exp(A*(a + dt)))).
func EmitDeltaDecayBoundSSE() ([]byte, error) {
	var a Buf
	a.DeclareISA(ISASSE2)
	a.MOVLoad(RCX, At(RDI, 0))    // Out:     decay
	a.MOVLoad(RDX, At(RDI, 8))    // W:       a
	a.MOVLoad(R8, At(RDI, 24))    // AScale:  dt
	a.MOVLoad(R11, At(RDI, 136))  // AScale2: lb
	a.MOVLoad(R9, At(RDI, 112))   // Q32:     A
	a.MOVLoad(RBX, At(RDI, 56))   // Scr:     consts
	a.MOVLoad(R10, At(RDI, 40))   // K:       whole units
	a.MOVLoad(RAX, At(RDI, 32))   // Rows:    tail elements
	loadExpConstsSSE(&a)          // XMM7..XMM15; XMM8 is 1.0
	bcastSS(&a, XMM6, At(R11, 0)) // lb, held

	elemLoopSSE(&a, R10, RAX, []Reg{RCX, RDX, R8, R9}, func(ld func(Reg, Reg), st func(Reg, Reg)) {
		ld(XMM0, RDX)
		ld(XMM3, R8)
		a.ADDPS(XMM0, XMM0, XMM3) // z = a + dt
		ld(XMM3, R9)
		a.MULPS(XMM0, XMM0, XMM3) // A*z
		emitExpSSE(&a, XMM0, XMM1, XMM2)
		a.ADDPS(XMM0, XMM0, XMM8) // 1 + exp(A*z)
		a.DIVPS(XMM3, XMM6, XMM0) // lb * sigma(-A*z)
		emitExpSSE(&a, XMM3, XMM1, XMM2)
		st(RCX, XMM3)
	})

	a.RET()
	return a.Bytes(), nil
}

// EmitGatedSSDSSE is EmitGatedSSD for the SSE tier: one head of Mamba-2's
// selective state update, in one pass per row (delta.go has the contract).
// With no FMA the update is a multiply and an add, and D*x is added after the
// horizontal sum.
func EmitGatedSSDSSE(n int) ([]byte, error) {
	if n <= 0 {
		return nil, fmt.Errorf("jit: EmitGatedSSDSSE: n=%d must be positive", n)
	}
	nv, tail := n/4, n%4
	voff := func(i int) int32 { return int32(i) * 16 }
	toff := func(i int) int32 { return int32(nv*16 + 4*i) }

	var a Buf
	a.DeclareISA(ISASSE2 | ISASSE3)
	a.MOVLoad(RCX, At(RDI, 0))  // Out: o
	a.MOVLoad(RDX, At(RDI, 8))  // W:   the state, row 0
	a.MOVLoad(RSI, At(RDI, 24)) // AScale: B
	a.MOVLoad(RAX, At(RDI, 32)) // Rows
	a.MOVLoad(R10, At(RDI, 56)) // Scr: {decay, dt, D}
	a.MOVLoad(R8, At(RDI, 112)) // Q32: C
	a.MOVLoad(R9, At(RDI, 128)) // Q2:  x
	bcastSS(&a, XMM10, At(R10, 0))
	bcastSS(&a, XMM11, At(R10, 4))
	bcastSS(&a, XMM12, At(R10, 8))

	row := a.Label()
	a.Bind(row)
	for i := 0; i < 4; i++ {
		a.XORPS(Reg(i), Reg(i), Reg(i))
	}
	bcastSS(&a, XMM13, At(R9, 0))
	a.MULPS(XMM13, XMM13, XMM11) // x[j]*dt
	for i := 0; i < nv; i++ {
		a.MOVUPSLoad(XMM8, At(RDX, voff(i)))
		a.MULPS(XMM8, XMM8, XMM10)
		a.MOVUPSLoad(XMM9, At(RSI, voff(i)))
		a.MULPS(XMM9, XMM9, XMM13)
		a.ADDPS(XMM8, XMM8, XMM9)
		a.MOVUPSStore(At(RDX, voff(i)), XMM8)
		a.MOVUPSLoad(XMM9, At(R8, voff(i)))
		a.MULPS(XMM9, XMM9, XMM8)
		a.ADDPS(Reg(i%4), Reg(i%4), XMM9)
	}
	for i := 0; i < tail; i++ {
		a.MOVSSLoad(XMM8, At(RDX, toff(i)))
		a.MULSS(XMM8, XMM8, XMM10)
		a.MOVSSLoad(XMM9, At(RSI, toff(i)))
		a.MULSS(XMM9, XMM9, XMM13)
		a.ADDSS(XMM8, XMM8, XMM9)
		a.MOVSSStore(At(RDX, toff(i)), XMM8)
		a.MOVSSLoad(XMM9, At(R8, toff(i)))
		a.MULSS(XMM9, XMM9, XMM8)
		a.ADDSS(XMM0, XMM0, XMM9)
	}
	a.ADDPS(XMM0, XMM0, XMM1)
	a.ADDPS(XMM2, XMM2, XMM3)
	a.ADDPS(XMM0, XMM0, XMM2)
	hsum4SSE(&a, XMM0)
	a.MOVSSLoad(XMM9, At(R9, 0))
	a.MULSS(XMM9, XMM9, XMM12)
	a.ADDSS(XMM0, XMM0, XMM9)
	a.MOVSSStore(At(RCX, 0), XMM0)

	a.ADDimm(RCX, 4)
	a.ADDimm(R9, 4)
	a.ADDimm(RDX, int32(n)*4)
	a.DEC(RAX)
	a.JNZ(row)

	a.RET()
	return a.Bytes(), nil
}

// EmitSSDGateSSE is EmitSSDGate for the SSE tier: EmitDeltaGateSSE's decay,
// with the softplus kept as the second output (gate.go has the contract).
func EmitSSDGateSSE() ([]byte, error) {
	var a Buf
	a.DeclareISA(ISASSE2)
	a.MOVLoad(RCX, At(RDI, 0))   // Out:    decay
	a.MOVLoad(RSI, At(RDI, 120)) // Out2:   dt
	a.MOVLoad(RDX, At(RDI, 8))   // W:      a
	a.MOVLoad(R8, At(RDI, 24))   // AScale: bias
	a.MOVLoad(R9, At(RDI, 112))  // Q32:    A
	a.MOVLoad(RBX, At(RDI, 56))  // Scr:    consts
	a.MOVLoad(R10, At(RDI, 40))  // K:      whole units
	a.MOVLoad(RAX, At(RDI, 32))  // Rows:   tail elements
	loadExpConstsSSE(&a)         // XMM7..XMM15; XMM8 is 1.0

	elemLoopSSE(&a, R10, RAX, []Reg{RCX, RSI, RDX, R8, R9}, func(ld func(Reg, Reg), st func(Reg, Reg)) {
		ld(XMM0, RDX)
		ld(XMM6, R8)
		a.ADDPS(XMM0, XMM0, XMM6)
		a.XORPS(XMM5, XMM5, XMM5)
		a.MAXPS(XMM4, XMM0, XMM5)
		bcastSS(&a, XMM5, At(RBX, 64))
		a.ANDPS(XMM0, XMM0, XMM5) // |z|
		a.XORPS(XMM5, XMM5, XMM5)
		a.SUBPS(XMM5, XMM5, XMM0) // -|z|
		emitExpSSE(&a, XMM5, XMM1, XMM2)
		bcastSS(&a, XMM6, At(RBX, 44)) // 2
		a.ADDPS(XMM1, XMM5, XMM6)
		a.DIVPS(XMM2, XMM5, XMM1) // s
		a.MULPS(XMM0, XMM2, XMM2) // s^2
		bcastSS(&a, XMM1, At(RBX, 48))
		for _, off := range []int32{52, 56, 60} {
			a.MULPS(XMM1, XMM1, XMM0)
			bcastSS(&a, XMM6, At(RBX, off))
			a.ADDPS(XMM1, XMM1, XMM6)
		}
		a.MULPS(XMM1, XMM1, XMM0)
		a.ADDPS(XMM1, XMM1, XMM8) // ... + 1
		a.MULPS(XMM1, XMM1, XMM2)
		bcastSS(&a, XMM6, At(RBX, 44))
		a.MULPS(XMM1, XMM1, XMM6) // 2*s*h
		a.ADDPS(XMM0, XMM1, XMM4) // softplus
		st(RSI, XMM0)
		ld(XMM6, R9)
		a.MULPS(XMM0, XMM0, XMM6) // * A
		emitExpSSE(&a, XMM0, XMM1, XMM2)
		st(RCX, XMM0)
	})

	a.RET()
	return a.Bytes(), nil
}

// EmitSelScanSSE is EmitSelScan for the SSE tier: Mamba-1's selective scan,
// one pass per channel (delta.go has the contract). With no FMA the update
// and the dot are a multiply and an add, and D*x is added after the
// horizontal sum.
func EmitSelScanSSE(n int) ([]byte, error) {
	if n <= 0 {
		return nil, fmt.Errorf("jit: EmitSelScanSSE: n=%d must be positive", n)
	}
	nv, tail := n/4, n%4
	toff := func(i int) int32 { return int32(nv*16 + 4*i) }

	var a Buf
	a.DeclareISA(ISASSE2 | ISASSE3)
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
	loadExpConstsSSE(&a)         // XMM7..XMM15

	row := a.Label()
	a.Bind(row)
	a.XORPS(XMM0, XMM0, XMM0)
	bcastSS(&a, XMM1, At(R11, 0)) // dt[j]
	bcastSS(&a, XMM2, At(R9, 0))
	a.MULPS(XMM2, XMM2, XMM1) // x[j]*dt[j]
	step := func(ld func(d, b Reg, o int32), st func(s Reg, o int32), off int32) {
		ld(XMM3, R10, off)
		a.MULPS(XMM3, XMM3, XMM1)
		emitExpSSE(&a, XMM3, XMM4, XMM5) // the decay
		ld(XMM6, RDX, off)
		a.MULPS(XMM6, XMM6, XMM3)
		ld(XMM4, RSI, off)
		a.MULPS(XMM4, XMM4, XMM2)
		a.ADDPS(XMM6, XMM6, XMM4)
		st(XMM6, off)
		ld(XMM5, R8, off)
		a.MULPS(XMM5, XMM5, XMM6)
		a.ADDPS(XMM0, XMM0, XMM5)
	}
	for i := 0; i < nv; i++ {
		step(func(d, b Reg, o int32) { a.MOVUPSLoad(d, At(b, o)) },
			func(s Reg, o int32) { a.MOVUPSStore(At(RDX, o), s) }, int32(i)*16)
	}
	for i := 0; i < tail; i++ {
		// A scalar load zeroes the other lanes; they add nothing to the dot.
		step(func(d, b Reg, o int32) { a.MOVSSLoad(d, At(b, o)) },
			func(s Reg, o int32) { a.MOVSSStore(At(RDX, o), s) }, toff(i))
	}
	hsum4SSE(&a, XMM0)
	a.MOVSSLoad(XMM4, At(R9, 0))
	a.MOVSSLoad(XMM5, At(R12, 0))
	a.MULSS(XMM4, XMM4, XMM5)
	a.ADDSS(XMM0, XMM0, XMM4)
	a.MOVSSStore(At(RCX, 0), XMM0)

	a.ADDimm(RCX, 4)
	a.ADDimm(R9, 4)
	a.ADDimm(R11, 4)
	a.ADDimm(R12, 4)
	a.ADDimm(RDX, int32(n)*4)
	a.ADDimm(R10, int32(n)*4)
	a.DEC(RAX)
	a.JNZ(row)

	a.RET()
	return a.Bytes(), nil
}
