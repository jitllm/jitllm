package oldkernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// XIELU is Apertus's xIELU over n elements, out of place, the device twin of
// jit/cpu's EmitXIELU (xielu_const.go there has the arithmetic):
//
//	x > 0:   alpha_p*x*x + beta*x
//	x <= 0:  (g(m) + (m - x))*alpha_n + beta*x,   m = min(x, eps)
//
// where g(m) = expm1(m) - m is a degree-8 Taylor polynomial for m >= -1/2 and
// exp(m) - 1 - m below it. The layer's four numbers -- alpha_p, alpha_n, beta,
// eps, already in their effective form -- are a buffer (pP), not constants,
// so one compiled kernel serves every layer and the numbers move with the
// block when it is placed or relocated.
//
// Params: pX, pP, pOut.
func XIELU(n int) (*ir.Kernel, error) {
	if n < 1 {
		return nil, fmt.Errorf("kernels: XIELU: %d elements", n)
	}
	b := ir.New("xielu", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pP := b.Param("pP", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(n-1)))
	at := func(k int64) ir.Value { return b.Load(ir.F32, pP, b.Const(ir.U32, k), 0) }
	ap, an, beta, eps := at(0), at(1), at(2), at(3)
	x := b.Load(ir.F32, pX, i, 0)
	m := b.Min(ir.F32, x, eps)

	// The polynomial, highest coefficient first: 1/8!, 1/7!, ..., 1/2.
	coef := []float32{1.0 / 40320, 1.0 / 5040, 1.0 / 720, 1.0 / 120, 1.0 / 24, 1.0 / 6, 0.5}
	p := b.ConstF32(coef[0])
	for _, c := range coef[1:] {
		p = b.Add(ir.F32, b.Mul(ir.F32, p, m), b.ConstF32(c))
	}
	small := b.Mul(ir.F32, b.Mul(ir.F32, p, m), m)
	large := b.Sub(ir.F32, b.Sub(ir.F32, b.Exp(m), b.ConstF32(1)), m)
	g := b.Select(ir.F32, b.Lt(ir.F32, m, b.ConstF32(-0.5)), large, small)

	bx := b.Mul(ir.F32, beta, x)
	neg := b.Add(ir.F32, b.Mul(ir.F32, b.Add(ir.F32, g, b.Sub(ir.F32, m, x)), an), bx)
	pos := b.Add(ir.F32, b.Mul(ir.F32, b.Mul(ir.F32, ap, x), x), bx)
	b.Store(pOut, i, b.Select(ir.F32, b.Lt(ir.F32, b.ConstF32(0), x), pos, neg), 0)
	return b.Done(), nil
}
