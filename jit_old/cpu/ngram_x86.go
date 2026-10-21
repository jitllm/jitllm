package cpu

// emitNGramHashX86 is the n-gram hash on amd64 (ngram_const.go has the
// contract), n and per baked and every loop unrolled. It is scalar integer
// code with no VEX instruction, so the AVX2 and SSE tiers run the same
// instructions and it ends without VZEROUPPER; sse only declares the tier.
// reset false is a gate's violation: the window is not cut at an end of
// sequence.
//
// Registers: RSI the window, RCX the rows, R8 the next predecessors, RBX the
// constants, R9 the window's ids (Scratch); R10 an id, R11 eos, R12 the cut
// flag, R13 and R15 temporaries, RAX and RDX the product and the division.
func emitNGramHashX86(n, per int, reset, sse bool) ([]byte, error) {
	if err := ngramShape(n, per); err != nil {
		return nil, err
	}
	var a Buf
	if sse {
		a.DeclareISA(ISATierSSE)
	}
	a.MOVLoad(RCX, At(RDI, 0))  // Out: the rows
	a.MOVLoad(RSI, At(RDI, 8))  // W: the window
	a.MOVLoad(RBX, At(RDI, 56)) // Scr: the constants
	a.MOVLoad(R9, At(RDI, 72))  // Scratch: the ids
	a.MOVLoad(R8, At(RDI, 120)) // Out2: the next predecessors
	a.MOVLoad(R11, At(RBX, ngramConstEOS))

	// The token: its id, and the next position's nearest predecessor.
	a.MOVLLoad(R10, At(RSI, 0))
	a.MOVStore(At(R9, 0), R10)
	a.MOVQ(R13, R10)
	a.ADDimm(R13, 1)
	a.MOVLStore(At(R8, 0), R13)
	a.XORQ(R12, R12) // not cut
	for s := 1; s < n; s++ {
		a.MOVLLoad(R10, At(RSI, int32(4*s)))
		if s < n-1 {
			a.MOVLStore(At(R8, int32(4*s)), R10) // moves up one
		}
		// none (zero) reads as eos, anything else is its id+1.
		a.MOVQ(R13, R10)
		a.SUBimm(R10, 1)
		some := a.Label()
		a.TESTQ(R13, R13)
		a.JNZ(some)
		a.MOVQ(R10, R11)
		a.Bind(some)
		// After a cut, eos.
		live := a.Label()
		a.TESTQ(R12, R12)
		a.JZ(live)
		a.MOVQ(R10, R11)
		a.Bind(live)
		if reset {
			// eos cuts what is behind it.
			other := a.Label()
			a.MOVQ(R13, R10)
			a.XORQ(R13, R11)
			a.JNZ(other)
			a.MOVimm(R12, 1)
			a.Bind(other)
		}
		a.MOVStore(At(R9, int32(8*s)), R10)
	}
	for g := 2; g <= n; g++ {
		a.MOVLoad(R15, At(R9, 0))
		a.MOVLoad(R13, At(RBX, ngramConstMul(0)))
		a.IMUL(R15, R13)
		for j := 1; j < g; j++ {
			a.MOVLoad(R10, At(R9, int32(8*j)))
			a.MOVLoad(R13, At(RBX, ngramConstMul(j)))
			a.IMUL(R10, R13)
			a.XORQ(R15, R10)
		}
		for k := 0; k < per; k++ {
			h := (g-2)*per + k
			a.MOVQ(RAX, R15)
			a.XORQ(RDX, RDX)
			a.MOVLoad(R13, At(RBX, ngramConstVoc(n, h)))
			a.DIVQ(R13)
			a.ADDQmem(RDX, At(RBX, ngramConstOff(n, h)))
			a.MOVStore(At(RCX, int32(8*h)), RDX)
		}
	}
	a.RET()
	return a.Bytes(), nil
}

// EmitNGramHashSSE is the SSE tier's n-gram hash.
func EmitNGramHashSSE(n, per int, reset bool) ([]byte, error) {
	return emitNGramHashX86(n, per, reset, true)
}
