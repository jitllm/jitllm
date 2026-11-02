package cpu

// EmitSparseMixerSSE is the SSE tier's EmitSparseMixer (sparsemixer.go), lane 0
// of an XMM register in place of a YMM's; sparsemixer_const.go has the
// contract.
//
// Registers: RBX Scr, R12 the logits, RSI their cursor, R8 sel, R9 ord, R10
// wt, R11 ow, RAX the loop counter. XMM7..XMM15 are exp's constants
// (loadExpConstsSSE); XMM0 the reference logit m, XMM1 the sum, XMM2 the
// index, XMM3..XMM6 temporaries.
func EmitSparseMixerSSE(n int) ([]byte, error) {
	if err := sparseMixerCheck(n); err != nil {
		return nil, err
	}
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RBX, At(RDI, 56))  // Scr
	a.MOVLoad(R12, At(RDI, 112)) // Q32      -> the logits
	a.MOVLoad(R8, At(RDI, 96))   // ASum     -> sel
	a.MOVLoad(R9, At(RDI, 104))  // AHalfSum -> ord
	a.MOVLoad(R10, At(RDI, 0))   // Out      -> wt
	a.MOVLoad(R11, At(RDI, 120)) // Out2     -> ow
	loadExpConstsSSE(&a)

	pass := func(slot int32) {
		bcastSS(&a, XMM0, At(R10, 4*slot)) // m
		a.PXOR(XMM1, XMM1, XMM1)           // the sum
		a.PXOR(XMM2, XMM2, XMM2)           // j
		a.MOVQ(RSI, R12)
		countLoopSSE(&a, RAX, n, func() {
			a.MOVSSLoad(XMM3, At(RSI, 0)) // s_j
			bcastSS(&a, XMM4, At(RBX, smAbsOff))
			a.PAND(XMM4, XMM4, XMM3) // |s_j|
			// max(|s_j|, m) with m first, so a NaN |s_j| is what comes out,
			// as on the AVX2 tier.
			a.MOVAPS(XMM5, XMM0)
			a.MAXPS(XMM5, XMM5, XMM4)
			a.MOVAPS(XMM6, XMM0)
			a.SUBPS(XMM6, XMM6, XMM3) // m - s_j
			a.DIVPS(XMM6, XMM6, XMM5) // the relative gap
			// masked: 2*eps < gap, ordered (legacy CMPPS has no ordered
			// greater-than, so the operands swap).
			bcastSS(&a, XMM4, At(RBX, smThrOff))
			a.CMPPS(XMM4, XMM4, XMM6, 1)
			if slot == 1 {
				bcastD(&a, XMM5, At(R8, 0))
				a.PCMPEQD(XMM5, XMM5, XMM2) // j == sel[0]
				a.POR(XMM4, XMM4, XMM5)
			}
			a.SUBPS(XMM3, XMM3, XMM0) // s_j - m
			emitExpSSE(&a, XMM3, XMM5, XMM6)
			a.PANDN(XMM4, XMM4, XMM3) // a masked term is zero, whatever its exp
			a.ADDPS(XMM1, XMM1, XMM4)
			bcastD(&a, XMM5, At(RBX, smOneOff))
			a.PADDD(XMM2, XMM2, XMM5)
			a.ADDimm(RSI, 4)
		})
		a.MOVAPS(XMM5, XMM8) // 1.0
		a.DIVPS(XMM5, XMM5, XMM1)
		a.MOVSSStore(At(R10, 4*slot), XMM5)
	}
	pass(0)
	pass(1)

	// ow is wt in ascending id: ord[0] is sel[0] or sel[1].
	a.MOVSSLoad(XMM0, At(R10, 0)) // w1
	a.MOVSSLoad(XMM1, At(R10, 4)) // w2
	bcastD(&a, XMM2, At(R8, 0))
	bcastD(&a, XMM3, At(R9, 0))
	a.PCMPEQD(XMM2, XMM2, XMM3) // ord[0] == sel[0]
	a.MOVAPS(XMM4, XMM0)
	a.PAND(XMM4, XMM4, XMM2)
	a.MOVAPS(XMM5, XMM2)
	a.PANDN(XMM5, XMM5, XMM1)
	a.POR(XMM4, XMM4, XMM5) // ord[0]'s weight
	a.MOVSSStore(At(R11, 0), XMM4)
	a.MOVAPS(XMM4, XMM1)
	a.PAND(XMM4, XMM4, XMM2)
	a.MOVAPS(XMM5, XMM2)
	a.PANDN(XMM5, XMM5, XMM0)
	a.POR(XMM4, XMM4, XMM5) // ord[1]'s
	a.MOVSSStore(At(R11, 4), XMM4)
	a.RET()
	return a.Bytes(), nil
}
