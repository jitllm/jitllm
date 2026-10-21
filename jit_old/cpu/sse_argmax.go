package cpu

// The SSE tier's greedy argmax: EmitArgmax (argmax.go) on legacy SSE.
//
// It takes the AVX2 twin's ABI and constant block unchanged (Args.Q32 the
// elements, Args.K the element count, Args.Out the value, Args.ASum the index,
// Args.Scr = ArgmaxConsts()). A YMM accumulator becomes two XMM halves with
// the same index step of eight, so both tiers walk the same elements in the
// same groups.
//
// It is bit-identical to the AVX2 kernel by construction: every output is an
// input selected by comparison, and the tie rule picks by index, so the gate
// is equality (argmaxtail_test.go). PMINSD (SSE4.1) is the instruction
// that decides the tier; PMINSW would not hold 32-bit indices.

// emitArgmaxSSEBody is the kernel's bytes. It is the body rather than the
// Emitters entry point because the table's EmitArgmaxSSE owns the error
// contract (an absent kernel is a work item, not a refusal -- quantact_owed.go).
//
// Registers: RSI the element cursor, RBX Scr, R10 whole vectors, R11 the tail,
// R8 ASum, R9 Out; XMM0/XMM1 the running maxima, XMM2/XMM3 the best indices,
// XMM4/XMM5 the current indices, XMM6 the +8 step, XMM7 the loaded elements,
// XMM8 the mask, XMM9/XMM10 temporaries, XMM11 the overall maximum, XMM12 and
// XMM13 the tail's maximum and index, XMM14 0x7FFFFFFF, XMM15 the +1 step.
func emitArgmaxSSEBody() []byte {
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RSI, At(RDI, 112)) // Q32 -> the elements
	a.MOVLoad(RBX, At(RDI, 56))  // Scr -> constants
	a.MOVLoad(R10, At(RDI, 40))  // K   -> ELEMENTS
	a.MOVQ(R11, R10)
	a.ANDimm8(R11, 7) // the tail: 0..7 elements
	a.SHRimm(R10, 3)  // whole vectors of eight

	bcastSS(&a, XMM0, At(RBX, 40))  // the maxima of lanes 0..3 = -Inf
	a.MOVAPS(XMM1, XMM0)            // and of lanes 4..7
	a.MOVUPSLoad(XMM2, At(RBX, 0))  // best index per lane = 0..3
	a.MOVUPSLoad(XMM3, At(RBX, 16)) //                       4..7
	a.MOVAPS(XMM4, XMM2)            // the current index vectors
	a.MOVAPS(XMM5, XMM3)
	bcastD(&a, XMM6, At(RBX, 32)) // +8 per iteration

	// The tail's accumulators, lane 0 only, for argmax.go's reason: a scalar
	// load zeroes the other lanes and zero is not neutral for a maximum. Its
	// index cursor is XMM4, the low half's, whose lane 0 holds 8k after k
	// iterations -- the index of the first tail element, and 0 when the vector
	// loop never ran.
	bcastSS(&a, XMM12, At(RBX, 40)) // the tail's maximum = -Inf
	a.PXOR(XMM13, XMM13, XMM13)     // the tail's best index
	bcastD(&a, XMM15, At(RBX, 4))   // +1 per tail element, which is lane 1 of
	//                                 the index vector and needs no constant

	// CMPPS predicate 1 spells `v > max`: legacy CMPPS has no ordered
	// greater-than, so the operands swap. (max < v) ordered is false for a NaN
	// on either side. MAXPS returns its second operand unless the first is
	// strictly greater, so passing the new element first selects the same
	// element the mask selects the index of, NaN and -0/+0 included.
	half := func(off int32, mx, best, cur Reg) {
		a.MOVUPSLoad(XMM7, At(RSI, off))
		a.CMPPS(XMM8, mx, XMM7, 1) // the lanes this element improves
		a.MAXPS(XMM9, XMM7, mx)
		a.MOVAPS(mx, XMM9)
		a.PAND(XMM9, cur, XMM8)    // the new index where it improved
		a.PANDN(XMM10, XMM8, best) // the old index where it did not
		a.POR(best, XMM9, XMM10)
		a.PADDD(cur, cur, XMM6)
	}
	elemLoopsSSE(&a, R10, R11, []Reg{RSI}, func(off int32) {
		// off is 0 for the unit's low half and 16 for its high one, which is
		// the AVX2 kernel's lanes 0..3 and 4..7.
		if off == 0 {
			half(off, XMM0, XMM2, XMM4)
		} else {
			half(off, XMM1, XMM3, XMM5)
		}
	}, func() {
		a.MOVSSLoad(XMM7, At(RSI, 0)) // one element, lanes 1..3 zeroed
		a.CMPPS(XMM8, XMM12, XMM7, 1)
		a.MAXPS(XMM9, XMM7, XMM12)
		a.MOVAPS(XMM12, XMM9)
		a.PAND(XMM9, XMM4, XMM8)
		a.PANDN(XMM10, XMM8, XMM13)
		a.POR(XMM13, XMM9, XMM10)
		a.PADDD(XMM4, XMM4, XMM15)
	})

	// The overall maximum, in every lane. The per-lane maxima must survive it:
	// the winner's index is found by asking which lanes equal the maximum.
	a.MAXPS(XMM11, XMM0, XMM1)
	hmax4SSE(&a, XMM11, XMM9)

	// Lanes that hold the maximum keep their index; the rest contribute
	// INT_MAX, so the minimum over the lanes is the lowest index that ties.
	bcastD(&a, XMM14, At(RBX, 36))
	for _, h := range [][2]Reg{{XMM0, XMM2}, {XMM1, XMM3}} {
		a.CMPPS(XMM8, h[0], XMM11, 0) // ordered equality: a NaN ties with nothing
		a.PAND(XMM9, h[1], XMM8)
		a.PANDN(XMM10, XMM8, XMM14)
		a.POR(h[1], XMM9, XMM10)
	}
	a.PMINSD(XMM2, XMM2, XMM3)
	a.PSHUFD(XMM9, XMM2, 0x4E)
	a.PMINSD(XMM2, XMM2, XMM9)
	a.PSHUFD(XMM9, XMM2, 0xB1)
	a.PMINSD(XMM2, XMM2, XMM9)

	// The tail is merged last and on a strict `>`: every tail index is higher
	// than every vector index, so an equal maximum keeps the vector's. Only
	// lane 0 of the tail's registers means anything, and only lane 0 is stored.
	a.CMPPS(XMM8, XMM11, XMM12, 1)
	a.PAND(XMM9, XMM13, XMM8)
	a.PANDN(XMM10, XMM8, XMM2)
	a.POR(XMM2, XMM9, XMM10)
	a.MAXPS(XMM9, XMM12, XMM11)

	a.MOVLoad(R8, At(RDI, 96)) // ASum -> the index
	a.MOVSSStore(At(R8, 0), XMM2)
	a.MOVLoad(R9, At(RDI, 0)) // Out -> the value
	a.MOVSSStore(At(R9, 0), XMM9)
	// No VZEROUPPER: it is VEX-encoded and faults on a host with no AVX.
	a.RET()
	return a.Bytes()
}
