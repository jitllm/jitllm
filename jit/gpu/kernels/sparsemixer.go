package kernels

import "github.com/jitllm/jitllm/jit/ir"

// expertWeightsSparseMixer is Phi-3.5-MoE's router weights, the device twin of
// jit/cpu's sparsemixer kernel (sparsemixer_const.go has the arithmetic).
// ExpertRank's plain rank has already put the top two logits in pTop's slots 0
// and 1, so this needs only the logits row:
//
//	w1 = 1 / SUM_{keep1(j)} exp(s_j - m1)
//	w2 = 1 / SUM_{keep2(j), j != i1} exp(s_j - m2)
//	keep(j) = !((m - s_j) / max(|s_j|, m) > 2*eps)
//
// i1 is the lowest index holding m1, which is ExpertRank's and torch.max's
// tie rule, so a tie for the maximum keeps the twin in the second softmax.
// Params are ExpertWeights' un-renormalised ones: pTop, pW, pL. A masked term
// is selected away, not multiplied by zero: exp(s_j - m2) at the first expert
// can overflow, and 0*Inf is NaN.
func expertWeightsSparseMixer(r MoERoute) (*ir.Kernel, error) {
	n := r.NExpert
	b := ir.New("expertweightssparsemixer", [3]int{r.weightsGroup(), 1, 1})
	pTop := b.Param("pTop", ir.F32) // k+1 logits from ExpertRank
	pW := b.Param("pW", ir.F32)     // k mixture weights
	pL := b.Param("pL", ir.F32)     // nExpert router logits

	zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
	row := r.weightsRow(b)
	atTop, atW := r.rowAt(b, row, r.K+1), r.rowAt(b, row, r.K)
	atL := r.rowAt(b, row, n)
	thr := b.ConstF32(float32(2 * float64(r.SparseMixer)))
	zeroF, oneF, negOne := b.ConstF32(0), b.ConstF32(1), b.ConstF32(-1)
	m1 := b.Load(ir.F32, pTop, atTop(zero), 0)
	m2 := b.Load(ir.F32, pTop, atTop(one), 0)
	cn := b.Const(ir.U32, int64(n))

	// i1: the first index holding m1. Every Phi seed is hoisted.
	b.Loop(int64(n))
	i1 := b.Phi(ir.U32, cn)
	j := b.Phi(ir.U32, zero)
	x := b.Load(ir.F32, pL, atL(j), 0)
	lt := b.Select(ir.U32, b.Lt(ir.F32, x, m1), one, zero)
	gt := b.Select(ir.U32, b.Lt(ir.F32, m1, x), one, zero)
	eq := b.Sub(ir.U32, one, b.Add(ir.U32, lt, gt))
	cand := b.Add(ir.U32, b.Mul(ir.U32, eq, j), b.Mul(ir.U32, b.Sub(ir.U32, one, eq), cn))
	b.SetPhi(i1, b.Min(ir.U32, i1, cand))
	b.SetPhi(j, b.Add(ir.U32, j, one))
	b.EndLoop()

	// term is exp(s - m) where s is kept against m, and 0 where it is not.
	term := func(s, m ir.Value) ir.Value {
		abs := b.Max(ir.F32, s, b.Mul(ir.F32, s, negOne))
		gap := b.Div(ir.F32, b.Sub(ir.F32, m, s), b.Max(ir.F32, abs, m))
		return b.Select(ir.F32, b.Lt(ir.F32, thr, gap), zeroF, b.Exp(b.Sub(ir.F32, s, m)))
	}

	b.Loop(int64(n))
	s1 := b.Phi(ir.F32, zeroF)
	j1 := b.Phi(ir.U32, zero)
	b.SetPhi(s1, b.Add(ir.F32, s1, term(b.Load(ir.F32, pL, atL(j1), 0), m1)))
	b.SetPhi(j1, b.Add(ir.U32, j1, one))
	b.EndLoop()

	b.Loop(int64(n))
	s2 := b.Phi(ir.F32, zeroF)
	j2 := b.Phi(ir.U32, zero)
	t := term(b.Load(ir.F32, pL, atL(j2), 0), m2)
	// Expert i1 is the first softmax's: -Inf in the second. j != i1 is
	// "below or above", the IR having no Eq.
	t = b.Select(ir.F32, b.Lt(ir.U32, j2, i1), t, b.Select(ir.F32, b.Lt(ir.U32, i1, j2), t, zeroF))
	b.SetPhi(s2, b.Add(ir.F32, s2, t))
	b.SetPhi(j2, b.Add(ir.U32, j2, one))
	b.EndLoop()

	b.Store(pW, atW(zero), b.Div(ir.F32, oneF, s1), 0)
	b.Store(pW, atW(one), b.Div(ir.F32, oneF, s2), 0)
	return b.Done()
}
