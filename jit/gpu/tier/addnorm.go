package tier

import "github.com/jitllm/jitllm/backend"

// foldTakes says block l's attention norm, in a prompt chunk on the binary16
// GEMM path, is the plain RMSNorm of bs.x that a held residual add can be
// folded into: the voltaOnly branch of the attention half, reading bs.x, with
// nothing before it in the block that reads bs.x.
func (g *devTier) foldTakes(l *layer, R int, voltaOnly bool, skip map[string]bool) bool {
	return l != nil && voltaOnly && !skip["norm"] && !l.linear && l.ds4 == nil && l.k3 == nil && l.q4e == nil &&
		l.altRouter == nil && l.nAttnB == nil && l.nAttn != nil && !l.mla
}

// foldAddNorm decides whether block l's residual add, x = x2 + fout, is held
// for block li+1's attention norm to do (kernels.RMSNormF16TRows with add,
// the kernel the FFN norm already folds its add into): one launch where a
// prompt chunk took two, with the same bits, since the sum the norm reads is
// the one it stores. It returns that norm and the conversion it stands in for,
// or nil where the add runs alone: the last block of the submission, a block
// with anything after its add, a scaled residual, or a next block whose norm
// is not the plain one (foldTakes).
func (g *devTier) foldAddNorm(bs *blockScratch, l *layer, li, hi, R int, voltaOnly, scaled, persist bool,
	fout backend.Buf, skip map[string]bool) (backend.Kernel, backend.Kernel) {
	if g.NoAddNormFold || !voltaOnly || scaled || persist || li+1 >= hi || skip["norm"] ||
		fout == bs.h || fout == bs.x || l.nResid != nil || l.altRouter != nil || l.pleGate != nil || l.outScale != nil {
		return nil, nil
	}
	nl := g.layers[li+1]
	if nl == nil || !g.voltaBlock(nl, R, bs.tok) || !g.foldTakes(nl, R, true, skip) {
		return nil, nil
	}
	return g.rmsF16For(bs, nl.mvq, R, true)
}

// addNormBroken makes a folded add sum x2 with itself instead of with the
// FFN's output: the violation the fold's gate must see (BreakAddNormForTest,
// jitllmtest builds only).
var addNormBroken bool
