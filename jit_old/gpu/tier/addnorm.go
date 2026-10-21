package tier

// foldTakes says block l's attention norm, in a prompt chunk on the binary16
// GEMM path, is the plain RMSNorm of bs.x that a held residual add can be
// folded into: the voltaOnly branch of the attention half, reading bs.x, with
// nothing before it in the block that reads bs.x.
func (g *devTier) foldTakes(l *layer, R int, voltaOnly bool, skip map[string]bool) bool {
	return l != nil && voltaOnly && !skip["norm"] && !l.linear && l.ds4 == nil && l.k3 == nil && l.q4e == nil &&
		l.altRouter == nil && l.nAttnB == nil && l.nAttn != nil && !l.mla
}

// addNormBroken (the wrong-answer switch of a gate, Config.hook; WithViolation) makes a folded add sum x2 with itself instead of with the
// FFN's output: the violation the fold's gate must see (BreakAddNormForTest,
// jitllmtest builds only).
