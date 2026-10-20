package oldkernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// A batched mixture runs its experts over (token, slot) pairs sorted by expert,
// each expert's run padded to a whole matvec token group, so one weight read
// serves every token the router sent there (MatVec's grouped shape). These two
// kernels are the ends of that: GatherRows builds the sorted activations and
// ExpertCombineGrouped reads the expert outputs back in token order.

// GatherRows copies rows of n floats: dst[j] = src[perm[j]] for j < p, and a
// zero row wherever perm[j] >= nsrc -- which is how a padding column of an
// expert's run is spelt. Launch p*n threads.
func GatherRows(n, p, nsrc int) (*ir.Kernel, error) {
	if n < 1 || p < 1 || nsrc < 1 {
		return nil, fmt.Errorf("kernels: GatherRows: n=%d p=%d nsrc=%d", n, p, nsrc)
	}
	b := ir.New("gatherrows", [3]int{128, 1, 1})
	pSrc := b.Param("pSrc", ir.F32)
	pPerm := b.Param("pPerm", ir.U32)
	pDst := b.Param("pDst", ir.F32)
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(p*n-1)))
	cn := b.Const(ir.U32, int64(n))
	j, i := b.Div(ir.U32, flat, cn), b.Rem(ir.U32, flat, cn)
	src := b.Load(ir.U32, pPerm, j, 0)
	last := b.Const(ir.U32, int64(nsrc-1))
	v := b.Load(ir.F32, pSrc, b.Add(ir.U32, b.Mul(ir.U32, b.Min(ir.U32, src, last), cn), i), 0)
	v = b.Select(ir.F32, b.Lt(ir.U32, last, src), b.ConstF32(0), v)
	b.Store(pDst, flat, v, 0)
	return b.Done(), nil
}

// ExpertCombineGrouped is ExpertCombine for many tokens whose expert outputs sit
// in sorted order: out[r*n+i] = SUM_s w[r*k+s] * e[pos[r*k+s]*n+i], summed in
// selection order, as the one-token kernel sums. Launch rows*n threads.
func ExpertCombineGrouped(n, k, rows int) (*ir.Kernel, error) {
	if n < 1 || k < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: ExpertCombineGrouped: n=%d k=%d rows=%d", n, k, rows)
	}
	b := ir.New("expertcombinegrouped", [3]int{128, 1, 1})
	pE := b.Param("pE", ir.F32)     // sorted expert outputs, n each
	pW := b.Param("pW", ir.F32)     // rows*k weights, selection order
	pPos := b.Param("pPos", ir.U32) // rows*k sorted columns, selection order
	pOut := b.Param("pOut", ir.F32)
	zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(rows*n-1)))
	cn := b.Const(ir.U32, int64(n))
	r, i := b.Div(ir.U32, flat, cn), b.Rem(ir.U32, flat, cn)
	j0 := b.Mul(ir.U32, r, b.Const(ir.U32, int64(k)))
	zeroF := b.ConstF32(0)
	b.Loop(int64(k))
	acc := b.Phi(ir.F32, zeroF)
	s := b.Phi(ir.U32, zero)
	j := b.Add(ir.U32, j0, s)
	col := b.Load(ir.U32, pPos, j, 0)
	e := b.Load(ir.F32, pE, b.Add(ir.U32, b.Mul(ir.U32, col, cn), i), 0)
	b.SetPhi(acc, b.Fma(b.Load(ir.F32, pW, j, 0), e, acc))
	b.SetPhi(s, b.Add(ir.U32, s, one))
	b.EndLoop()
	b.Store(pOut, flat, acc, 0)
	return b.Done(), nil
}
