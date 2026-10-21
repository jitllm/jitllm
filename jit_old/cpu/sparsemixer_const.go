package cpu

import (
	"fmt"
	"math"
)

// The constant block and the contract every sparsemixer kernel reads. One copy
// for all three tiers, in an untagged file, because the layout is the ABI
// between the emitters and nn.
//
// Sparsemixer is Phi-3.5-MoE's router (transformers' PhimoeTopKRouter, in
// eval): the top two logits m1 and m2, each weighted by a softmax over the
// logits within a relative threshold of it,
//
//	keep1(j) = !((m1 - s_j) / max(|s_j|, m1) > 2*eps)
//	w1       = 1 / SUM_{keep1(j)} exp(s_j - m1)
//	keep2(j) = !((m2 - s_j) / max(|s_j|, m2) > 2*eps)  and  j != i1
//	w2       = 1 / SUM_{keep2(j)} exp(s_j - m2)
//
// and NOT renormalised: each weight is near one, where a renormalised top-2
// sums to one. The selection itself (i1, i2, both orders) is the plain top-k's
// over the raw logits, which runs first with nothing renormalised, so on entry
// Out holds the two selected logits exactly (x/1 is x); the kernel reads m1
// and m2 there and overwrites them with w1 and w2. `keep` is the negation of a
// greater-than so a 0/0 ratio (a logit of 0 beside a maximum of 0) is kept, as
// torch's masked_fill keeps it. The ratio is the reference's f32 subtraction
// and division, so the masks are its masks bit for bit.
//
//	Q32      the logits                    (f32, n)
//	Scr      SparseMixerConsts(eps)
//	ASum     sel, the two ids in selection order   (int32, 2)
//	AHalfSum ord, the same ids ascending           (int32, 2)
//	Out      wt: the selected logits in, sel's weights out   (f32, 2)
//	Out2     ow, ord's weights                     (f32, 2, written)

// SparseMixerConsts is the block for eps: exp's (expConstsShared, offsets
// 0..40) and then
//
//	44  2*eps, the relative threshold, as the reference's f32 scalar
//	48  0x7FFFFFFF, the |x| mask
//	52  int32 1, the index step
func SparseMixerConsts(eps float32) []float32 {
	return append(expConstsShared(),
		float32(2*float64(eps)),
		math.Float32frombits(0x7FFFFFFF),
		math.Float32frombits(1))
}

const (
	smThrOff = 44
	smAbsOff = 48
	smOneOff = 52
)

// sparseMixerCheck refuses a shape the kernel does not define: sparsemixer is
// a top-2 router over at least two experts.
func sparseMixerCheck(n int) error {
	if n < 2 {
		return fmt.Errorf("jit: sparsemixer: %d experts (want at least 2)", n)
	}
	return nil
}
