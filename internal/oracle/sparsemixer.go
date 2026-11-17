package oracle

import "math"

// SparseMixer is Phi-3.5-MoE's router in eval (transformers' sparsemixer with
// training off), the
// reference cpu.EmitSparseMixer and the device's sparsemixer weights are gated
// against:
//
//	i1 = argmax(s), m1 = s[i1]
//	w1 = softmax(s masked where (m1 - s_j)/max(|s_j|, m1) > 2*eps)[i1]
//	i2 = argmax(s with i1 removed), m2 = s[i2]
//	w2 = softmax(that removal, masked where (m2 - s_j)/max(|s_j|, m2) > 2*eps)[i2]
//
// with no renormalisation. The masks are decided in f32, as torch decides them
// on an f32 router; the softmax is libm exp in f64, narrowed once, which is the
// one place a kernel is compared with a tolerance. Ties go to the lowest index,
// as torch.max's first maximum does.
//
// The result is a MoERoute with Sum 1: sel/wt in selection order, ord/ow by
// ascending id.
func SparseMixer(s []float32, eps float32, f SparseMixerFault) MoERoute {
	thr := float32(2 * float64(eps))
	i1 := moeLexTopK(s, 1, MoEFaultNone)[0]
	m1 := s[i1]
	w1 := sparseMixerWeight(s, m1, -1, thr, f)
	rest := append([]float32(nil), s...)
	rest[i1] = float32(math.Inf(-1))
	i2 := moeLexTopK(rest, 1, MoEFaultNone)[0]
	m2 := s[i2]
	drop := i1
	if f == SparseMixerFaultKeepFirst {
		drop = -1
	}
	w2 := sparseMixerWeight(s, m2, drop, thr, f)
	if f == SparseMixerFaultRenormalised {
		t := w1 + w2
		w1, w2 = w1/t, w2/t
	}
	r := MoERoute{Sel: []int32{int32(i1), int32(i2)}, Wt: []float32{w1, w2}, Sum: 1}
	if i1 < i2 {
		r.Ord, r.OWt = []int32{int32(i1), int32(i2)}, []float32{w1, w2}
	} else {
		r.Ord, r.OWt = []int32{int32(i2), int32(i1)}, []float32{w2, w1}
	}
	return r
}

// SparseMixerFault is a deliberate defect for the sparsemixer gates to fire on.
type SparseMixerFault int

const (
	SparseMixerFaultNone SparseMixerFault = iota
	// SparseMixerFaultNoMask keeps every logit in both softmaxes: a softmax
	// over all experts, which is what the plain top-k router computes.
	SparseMixerFaultNoMask
	// SparseMixerFaultKeepFirst leaves the first expert in the second
	// softmax, which the reference masks to -Inf.
	SparseMixerFaultKeepFirst
	// SparseMixerFaultRenormalised divides the two weights by their sum,
	// a renormalising top-2 router in place of the sparsemixer.
	SparseMixerFaultRenormalised
	// SparseMixerFaultAbsFactor drops the clamp: the gap divided by |s_j|
	// alone.
	SparseMixerFaultAbsFactor
)

// sparseMixerWeight is 1/SUM_{kept j} exp(s_j - m): the softmax's term at the
// maximum is exp(0) = 1.
func sparseMixerWeight(s []float32, m float32, drop int, thr float32, f SparseMixerFault) float32 {
	var sum float64
	for j, x := range s {
		if j == drop {
			continue
		}
		factor := float32(math.Abs(float64(x)))
		if f != SparseMixerFaultAbsFactor && !(factor >= m) {
			// torch's clamp(min=m): m unless |x| is not below it; a NaN |x|
			// stays NaN.
			if factor == factor {
				factor = m
			}
		}
		gap := (m - x) / factor
		if f != SparseMixerFaultNoMask && gap > thr {
			continue
		}
		sum += math.Exp(float64(x) - float64(m))
	}
	return float32(1 / sum)
}
