package cpu

// The weight prefetch a fused region's idle workers run (nn's lookahead): a
// hint per 64-byte line, nothing loaded into a register and nothing written,
// so the phase that later reads those lines computes what it would have.
//
//	Out  the first line (any address; a prefetch hint does not fault)
//	K    how many 64-byte lines
//
// The hint is T2 on x86 and PLDL2KEEP on arm64: the lines are for whichever
// worker claims them next, not for this core's L1.

// PrefetchLine is the stride a prefetch-lines kernel steps by.
const PrefetchLine = 64

// emitPrefetchLinesX86 is the x86 body; the SSE tier declares its tier and
// runs the same instructions (PREFETCHh is SSE).
//
// Registers: RCX the cursor, R10 the line count.
func emitPrefetchLinesX86(sse bool) []byte {
	var a Buf
	if sse {
		a.DeclareISA(ISATierSSE)
	}
	a.MOVLoad(RCX, At(RDI, 0))  // Out -> first line
	a.MOVLoad(R10, At(RDI, 40)) // K -> lines
	done := a.Label()
	a.TESTQ(R10, R10)
	a.JZ(done)
	lp := a.Label()
	a.Bind(lp)
	a.PREFETCHT2(At(RCX, 0))
	a.ADDimm(RCX, PrefetchLine)
	a.DEC(R10)
	a.JNZ(lp)
	a.Bind(done)
	a.RET()
	return a.Bytes()
}

// EmitPrefetchLinesSSE is the SSE tier's prefetch-lines kernel.
func EmitPrefetchLinesSSE() ([]byte, error) { return emitPrefetchLinesX86(true), nil }

// EmitA64PrefetchLines is the arm64 prefetch-lines kernel.
//
// Registers: X1 the cursor, X4 the line count.
func EmitA64PrefetchLines() []byte {
	var a A64
	a.LDRx(X1, X0, 0)  // Out -> first line
	a.LDRx(X4, X0, 40) // K -> lines
	done := a.Label()
	a.CBZ(X4, done)
	lp := a.Label()
	a.Bind(lp)
	a.PRFMop(prfmPLDL2KEEP, X1, 0)
	a.ADDimm(X1, X1, PrefetchLine)
	a.SUBimm(X4, X4, 1)
	a.CBNZ(X4, lp)
	a.Bind(done)
	a.RET()
	return a.Bytes()
}
