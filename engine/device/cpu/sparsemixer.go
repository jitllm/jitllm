//go:build amd64

package cpu

// EmitSparseMixer generates Phi-3.5-MoE's sparsemixer weights for n experts;
// sparsemixer_const.go has the contract and the constant block.
//
// It is a basic kernel: one logit an iteration in lane 0 of a YMM register,
// so no tail and no read past the logits. n is a few dozen at most (16 on
// Phi-3.5-MoE), against a token's thousands of expert rows.
//
// Registers: RBX Scr, R12 the logits, RSI their cursor, R8 sel, R9 ord, R10
// wt, R11 ow, RAX the loop counter. Y7..Y15 are exp's constants
// (loadExpConsts); Y0 the reference logit m, Y1 the sum, Y2 the index, Y3..Y6
// temporaries.
func EmitSparseMixer(n int) ([]byte, error) {
	if err := sparseMixerCheck(n); err != nil {
		return nil, err
	}
	var a Buf
	a.MOVLoad(RBX, At(RDI, 56))  // Scr
	a.MOVLoad(R12, At(RDI, 112)) // Q32      -> the logits
	a.MOVLoad(R8, At(RDI, 96))   // ASum     -> sel
	a.MOVLoad(R9, At(RDI, 104))  // AHalfSum -> ord
	a.MOVLoad(R10, At(RDI, 0))   // Out      -> wt
	a.MOVLoad(R11, At(RDI, 120)) // Out2     -> ow
	loadExpConsts(&a)

	// pass computes 1/SUM over the kept logits of exp(s_j - m), m being the
	// selected logit at wt[slot], and stores it over that logit. Slot 1 also
	// drops expert sel[0], which the reference masks to -Inf.
	pass := func(slot int32) {
		a.VBROADCASTSS(Y0, At(R10, 4*slot)) // m
		a.VPXOR(Y1, Y1, Y1)                 // the sum
		a.VPXOR(Y2, Y2, Y2)                 // j
		a.MOVQ(RSI, R12)
		countLoop(&a, RAX, n, func() {
			a.VMOVSSLoad(Y3, At(RSI, 0)) // s_j
			a.VBROADCASTSS(Y4, At(RBX, smAbsOff))
			a.VPAND(Y4, Y4, Y3)  // |s_j|
			a.VMAXPS(Y4, Y0, Y4) // max(|s_j|, m): m unless |s_j| exceeds it, and a NaN passes
			a.VSUBPS(Y5, Y0, Y3) // m - s_j
			a.VDIVPS(Y5, Y5, Y4) // the relative gap
			a.VBROADCASTSS(Y6, At(RBX, smThrOff))
			a.VCMPPS(Y5, Y5, Y6, 14) // masked: gap > 2*eps, ordered (a NaN is kept)
			if slot == 1 {
				a.VPBROADCASTD(Y6, At(R8, 0))
				a.VPCMPEQD(Y6, Y2, Y6) // j == sel[0]
				a.VPOR(Y5, Y5, Y6)
			}
			a.VSUBPS(Y3, Y3, Y0) // s_j - m
			emitExpPS(&a, Y3, Y4, Y6)
			a.VPANDN(Y3, Y5, Y3) // a masked term is zero, whatever its exp
			a.VADDPS(Y1, Y1, Y3)
			a.VPBROADCASTD(Y4, At(RBX, smOneOff))
			a.VPADDD(Y2, Y2, Y4)
			a.ADDimm(RSI, 4)
		})
		a.VDIVPS(Y1, Y8, Y1) // 1/sum: the softmax's own term is exp(0) = 1
		a.VMOVSSStore(At(R10, 4*slot), Y1)
	}
	pass(0)
	pass(1)

	// ow is wt in ascending id: ord[0] is sel[0] or sel[1].
	a.VMOVSSLoad(Y0, At(R10, 0)) // w1
	a.VMOVSSLoad(Y1, At(R10, 4)) // w2
	a.VPBROADCASTD(Y2, At(R8, 0))
	a.VPBROADCASTD(Y3, At(R9, 0))
	a.VPCMPEQD(Y2, Y2, Y3) // ord[0] == sel[0]
	a.VPAND(Y4, Y0, Y2)
	a.VPANDN(Y5, Y2, Y1)
	a.VPOR(Y4, Y4, Y5) // ord[0]'s weight
	a.VMOVSSStore(At(R11, 0), Y4)
	a.VPAND(Y4, Y1, Y2)
	a.VPANDN(Y5, Y2, Y0)
	a.VPOR(Y4, Y4, Y5) // ord[1]'s
	a.VMOVSSStore(At(R11, 4), Y4)

	a.VZEROUPPER()
	a.RET()
	return a.Bytes(), nil
}
