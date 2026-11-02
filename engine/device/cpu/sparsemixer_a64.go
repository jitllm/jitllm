//go:build arm64

package cpu

// EmitSparseMixer is the NEON twin of the AVX2 kernel in sparsemixer.go, lane
// 0 of a Q register in place of a YMM's; sparsemixer_const.go has the
// contract.
//
//	x0  the Args block      x1 the logit cursor   x3 the logits
//	x5  Scr (exp's loader reads it there)         x6 sel   x7 ord
//	x8  wt                  x9 ow                 x11 the loop counter
//	v16..v26 exp's constants (v23 is 1.0)
//	v0  the reference logit m   v1 the sum   v2 the index   v3..v6 temporaries
//
// FMAX propagates a NaN where x86's MAXPS returns an operand, which is what
// torch's clamp does to a NaN |s_j|; FCMGT is false for a NaN, so a 0/0 gap
// is kept, as on the other tiers.
func EmitSparseMixer(n int) ([]byte, error) {
	if err := sparseMixerCheck(n); err != nil {
		return nil, err
	}
	var a A64
	a.LDRx(X5, X0, 56)  // Scr
	a.LDRx(X3, X0, 112) // Q32      -> the logits
	a.LDRx(X6, X0, 96)  // ASum     -> sel
	a.LDRx(X7, X0, 104) // AHalfSum -> ord
	a.LDRx(X8, X0, 0)   // Out      -> wt
	a.LDRx(X9, X0, 120) // Out2     -> ow
	loadA64ExpConsts(&a)

	pass := func(slot int32) {
		a.LDRs(0, X8, 4*slot) // m
		a.DUPs4(0, 0)
		a.MOVIzero(1) // the sum
		a.MOVIzero(2) // j
		a.MOVreg(X1, X3)
		countLoopA64(&a, X11, n, func() {
			a.LDRs(3, X1, 0) // s_j
			a.LDRs(4, X5, smAbsOff)
			a.DUPs4(4, 4)
			a.AND16b(4, 4, 3) // |s_j|
			a.FMAX4s(5, 0, 4) // max(|s_j|, m)
			a.FSUB4s(6, 0, 3) // m - s_j
			a.FDIV4s(6, 6, 5) // the relative gap
			a.LDRs(4, X5, smThrOff)
			a.DUPs4(4, 4)
			a.FCMGT4s(4, 6, 4) // masked: gap > 2*eps
			if slot == 1 {
				a.LDRs(5, X6, 0)
				a.DUPs4(5, 5)
				a.CMEQ4s(5, 2, 5) // j == sel[0]
				a.ORR16b(4, 4, 5)
			}
			a.FSUB4s(3, 3, 0) // s_j - m
			emitA64Exp(&a, 3, 5, 6)
			a.BIC16b(3, 3, 4) // a masked term is zero, whatever its exp
			a.FADD4s(1, 1, 3)
			a.LDRs(5, X5, smOneOff)
			a.DUPs4(5, 5)
			a.ADD4s(2, 2, 5)
			a.ADDimm(X1, X1, 4)
		})
		a.FDIV4s(5, 23, 1) // 1/sum
		a.STRs(5, X8, 4*slot)
	}
	pass(0)
	pass(1)

	// ow is wt in ascending id: ord[0] is sel[0] or sel[1].
	a.LDRs(0, X8, 0) // w1
	a.LDRs(1, X8, 4) // w2
	a.LDRs(2, X6, 0)
	a.LDRs(3, X7, 0)
	a.CMEQ4s(2, 2, 3) // ord[0] == sel[0]
	a.MOVvec(4, 1)
	a.BIT16b(4, 0, 2) // ord[0]'s weight
	a.STRs(4, X9, 0)
	a.MOVvec(4, 0)
	a.BIT16b(4, 1, 2) // ord[1]'s
	a.STRs(4, X9, 4)
	a.RET()
	return a.Bytes(), nil
}
