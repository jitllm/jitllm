package mainkernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// A decision head's kernels (Clef's joint schema head, tier/dechead.go): a
// handful of f32 shapes over rows -- a batched matrix product either way
// round, a LayerNorm or RMSNorm per row, a row gather-add, the L2 norm, the
// scorer's features, a gathered row dot and the final combination. Each is
// one thread per output value, its loop over the shared dimension, the
// shapes baked: the head runs a few hundred rows at most, so the plain form
// is the basic kernel RULE 8 asks for, and every one of them is owed a
// faster form before the head's time shows beside the backbone's.

// DHMatABt is out[b][i][j] = scale * sum_c A[b*sa + i*lda + c] *
// B[b*sb + j*ldb + c] (+ bias[j] when bias), at out[b*so + i*ldo + j], for
// batch x m x n outputs over k: a linear layer (B the weight's rows) or a
// head's attention scores (the batch the heads). Params: pA, pB, pBias
// (unread without bias, any buffer), pOut.
func DHMatABt(batch, m, n, k, lda, ldb, ldo, sa, sb, so int, scale float32, bias bool) (*ir.Kernel, error) {
	if batch < 1 || m < 1 || n < 1 || k < 1 {
		return nil, fmt.Errorf("kernels: DHMatABt: %dx%dx%d over %d", batch, m, n, k)
	}
	b := ir.New("dh_matabt", [3]int{128, 1, 1})
	// A phi starts from a value made before its loop (ir.Validate).
	f0, z0 := b.ConstF32(0), u32(b, 0)
	pA := b.Param("pA", ir.F32)
	pB := b.Param("pB", ir.F32)
	pBias := b.Param("pBias", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	t := convItem(b, batch*m*n)
	mn := u32(b, m*n)
	bb, r := b.Div(ir.U32, t, mn), b.Rem(ir.U32, t, mn)
	i, j := b.Div(ir.U32, r, u32(b, n)), b.Rem(ir.U32, r, u32(b, n))
	a0 := b.Add(ir.U32, b.Mul(ir.U32, bb, u32(b, sa)), b.Mul(ir.U32, i, u32(b, lda)))
	b0 := b.Add(ir.U32, b.Mul(ir.U32, bb, u32(b, sb)), b.Mul(ir.U32, j, u32(b, ldb)))
	b.Loop(int64(k))
	acc := b.Phi(ir.F32, f0)
	c := b.Phi(ir.U32, z0)
	x := b.Load(ir.F32, pA, b.Add(ir.U32, a0, c), 0)
	y := b.Load(ir.F32, pB, b.Add(ir.U32, b0, c), 0)
	b.SetPhi(acc, b.Add(ir.F32, acc, b.Mul(ir.F32, x, y)))
	b.SetPhi(c, b.Add(ir.U32, c, u32(b, 1)))
	b.EndLoop()
	v := b.Mul(ir.F32, acc, b.ConstF32(scale))
	if bias {
		v = b.Add(ir.F32, v, b.Load(ir.F32, pBias, j, 0))
	}
	o := b.Add(ir.U32, b.Mul(ir.U32, bb, u32(b, so)), b.Add(ir.U32, b.Mul(ir.U32, i, u32(b, ldo)), j))
	b.Store(pOut, o, v, 0)
	return b.Done(), nil
}

// DHMatAB is out[b*so + i*ldo + j] = sum_c A[b*sa + i*lda + c] *
// B[b*sb + c*ldb + j], for batch x m x n outputs over k: a pooling matrix
// over rows (a span's mean), a head's attention over its values, a
// question's weighted options. Params: pA, pB, pOut.
func DHMatAB(batch, m, n, k, lda, ldb, ldo, sa, sb, so int) (*ir.Kernel, error) {
	if batch < 1 || m < 1 || n < 1 || k < 1 {
		return nil, fmt.Errorf("kernels: DHMatAB: %dx%dx%d over %d", batch, m, n, k)
	}
	b := ir.New("dh_matab", [3]int{128, 1, 1})
	// A phi starts from a value made before its loop (ir.Validate).
	f0, z0 := b.ConstF32(0), u32(b, 0)
	pA := b.Param("pA", ir.F32)
	pB := b.Param("pB", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	t := convItem(b, batch*m*n)
	mn := u32(b, m*n)
	bb, r := b.Div(ir.U32, t, mn), b.Rem(ir.U32, t, mn)
	i, j := b.Div(ir.U32, r, u32(b, n)), b.Rem(ir.U32, r, u32(b, n))
	a0 := b.Add(ir.U32, b.Mul(ir.U32, bb, u32(b, sa)), b.Mul(ir.U32, i, u32(b, lda)))
	b0 := b.Add(ir.U32, b.Mul(ir.U32, bb, u32(b, sb)), j)
	b.Loop(int64(k))
	acc := b.Phi(ir.F32, f0)
	c := b.Phi(ir.U32, z0)
	x := b.Load(ir.F32, pA, b.Add(ir.U32, a0, c), 0)
	y := b.Load(ir.F32, pB, b.Add(ir.U32, b0, b.Mul(ir.U32, c, u32(b, ldb))), 0)
	b.SetPhi(acc, b.Add(ir.F32, acc, b.Mul(ir.F32, x, y)))
	b.SetPhi(c, b.Add(ir.U32, c, u32(b, 1)))
	b.EndLoop()
	o := b.Add(ir.U32, b.Mul(ir.U32, bb, u32(b, so)), b.Add(ir.U32, b.Mul(ir.U32, i, u32(b, ldo)), j))
	b.Store(pOut, o, acc, 0)
	return b.Done(), nil
}

// DHNormRows is a norm of each of rows rows of width: torch's LayerNorm
// (mean and biased variance, then w and, when bias, b) or, rms set, an
// RMSNorm with its weight. One thread a row. Params: pX, pW, pB, pOut.
func DHNormRows(width, rows int, eps float32, rms, bias bool) (*ir.Kernel, error) {
	if width < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: DHNormRows: %d rows of %d", rows, width)
	}
	b := ir.New("dh_normrows", [3]int{128, 1, 1})
	// A phi starts from a value made before its loop (ir.Validate).
	f0, z0 := b.ConstF32(0), u32(b, 0)
	pX := b.Param("pX", ir.F32)
	pW := b.Param("pW", ir.F32)
	pB := b.Param("pB", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	r := convItem(b, rows)
	base := b.Mul(ir.U32, r, u32(b, width))
	inv := b.ConstF32(1 / float32(width))
	mean := b.ConstF32(0)
	if !rms {
		b.Loop(int64(width))
		s := b.Phi(ir.F32, f0)
		c := b.Phi(ir.U32, z0)
		b.SetPhi(s, b.Add(ir.F32, s, b.Load(ir.F32, pX, b.Add(ir.U32, base, c), 0)))
		b.SetPhi(c, b.Add(ir.U32, c, u32(b, 1)))
		b.EndLoop()
		mean = b.Mul(ir.F32, s, inv)
	}
	b.Loop(int64(width))
	q := b.Phi(ir.F32, f0)
	c2 := b.Phi(ir.U32, z0)
	d := b.Sub(ir.F32, b.Load(ir.F32, pX, b.Add(ir.U32, base, c2), 0), mean)
	b.SetPhi(q, b.Add(ir.F32, q, b.Mul(ir.F32, d, d)))
	b.SetPhi(c2, b.Add(ir.U32, c2, u32(b, 1)))
	b.EndLoop()
	rstd := b.Div(ir.F32, b.ConstF32(1), b.Sqrt(b.Add(ir.F32, b.Mul(ir.F32, q, inv), b.ConstF32(eps))))
	b.Loop(int64(width))
	c3 := b.Phi(ir.U32, z0)
	at := b.Add(ir.U32, base, c3)
	v := b.Mul(ir.F32, b.Mul(ir.F32, b.Sub(ir.F32, b.Load(ir.F32, pX, at, 0), mean), rstd), b.Load(ir.F32, pW, c3, 0))
	if bias {
		v = b.Add(ir.F32, v, b.Load(ir.F32, pB, c3, 0))
	}
	b.Store(pOut, at, v, 0)
	b.SetPhi(c3, b.Add(ir.U32, c3, u32(b, 1)))
	b.EndLoop()
	return b.Done(), nil
}

// DHGatherAdd is out[r][c] = in[r][c] + T[idx[r]][c] for rows rows of
// width: a broadcast row (idx all zero), a type embedding, an option's
// question. Params: pIn, pIdx (u32), pT, pOut.
func DHGatherAdd(width, rows int) (*ir.Kernel, error) {
	if width < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: DHGatherAdd: %d rows of %d", rows, width)
	}
	b := ir.New("dh_gatheradd", [3]int{128, 1, 1})
	pIn := b.Param("pIn", ir.F32)
	pIdx := b.Param("pIdx", ir.U32)
	pT := b.Param("pT", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := convItem(b, width*rows)
	w := u32(b, width)
	r, c := b.Div(ir.U32, i, w), b.Rem(ir.U32, i, w)
	tr := b.Load(ir.U32, pIdx, r, 0)
	v := b.Add(ir.F32, b.Load(ir.F32, pIn, i, 0), b.Load(ir.F32, pT, b.Add(ir.U32, b.Mul(ir.U32, tr, w), c), 0))
	b.Store(pOut, i, v, 0)
	return b.Done(), nil
}

// DHL2Rows is y = x / max(||x||, 1e-12) for each of rows rows of width
// (torch's F.normalize). One thread a row. Params: pX, pOut.
func DHL2Rows(width, rows int) (*ir.Kernel, error) {
	if width < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: DHL2Rows: %d rows of %d", rows, width)
	}
	b := ir.New("dh_l2rows", [3]int{128, 1, 1})
	// A phi starts from a value made before its loop (ir.Validate).
	f0, z0 := b.ConstF32(0), u32(b, 0)
	pX := b.Param("pX", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	r := convItem(b, rows)
	base := b.Mul(ir.U32, r, u32(b, width))
	b.Loop(int64(width))
	q := b.Phi(ir.F32, f0)
	c := b.Phi(ir.U32, z0)
	x := b.Load(ir.F32, pX, b.Add(ir.U32, base, c), 0)
	b.SetPhi(q, b.Add(ir.F32, q, b.Mul(ir.F32, x, x)))
	b.SetPhi(c, b.Add(ir.U32, c, u32(b, 1)))
	b.EndLoop()
	inv := b.Div(ir.F32, b.ConstF32(1), b.Max(ir.F32, b.Sqrt(q), b.ConstF32(1e-12)))
	b.Loop(int64(width))
	c2 := b.Phi(ir.U32, z0)
	at := b.Add(ir.U32, base, c2)
	b.Store(pOut, at, b.Mul(ir.F32, b.Load(ir.F32, pX, at, 0), inv), 0)
	b.SetPhi(c2, b.Add(ir.U32, c2, u32(b, 1)))
	b.EndLoop()
	return b.Done(), nil
}

// DHRowDot is out[r] = sum_c A[r][c] * B[idx[r]][c] for rows rows of width.
// Params: pA, pIdx (u32), pB, pOut.
func DHRowDot(width, rows int) (*ir.Kernel, error) {
	if width < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: DHRowDot: %d rows of %d", rows, width)
	}
	b := ir.New("dh_rowdot", [3]int{128, 1, 1})
	// A phi starts from a value made before its loop (ir.Validate).
	f0, z0 := b.ConstF32(0), u32(b, 0)
	pA := b.Param("pA", ir.F32)
	pIdx := b.Param("pIdx", ir.U32)
	pB := b.Param("pB", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	r := convItem(b, rows)
	w := u32(b, width)
	a0 := b.Mul(ir.U32, r, w)
	b0 := b.Mul(ir.U32, b.Load(ir.U32, pIdx, r, 0), w)
	b.Loop(int64(width))
	acc := b.Phi(ir.F32, f0)
	c := b.Phi(ir.U32, z0)
	x := b.Load(ir.F32, pA, b.Add(ir.U32, a0, c), 0)
	y := b.Load(ir.F32, pB, b.Add(ir.U32, b0, c), 0)
	b.SetPhi(acc, b.Add(ir.F32, acc, b.Mul(ir.F32, x, y)))
	b.SetPhi(c, b.Add(ir.U32, c, u32(b, 1)))
	b.EndLoop()
	b.Store(pOut, r, acc, 0)
	return b.Done(), nil
}

// DHFeatures is the joint scorer's input for each of rows options: the
// field of its question f = F[idx[r]], the option o = O[r], and
// [f, o, f*o, |f-o|], rows of 4*width. Params: pF, pIdx (u32), pO, pOut.
func DHFeatures(width, rows int) (*ir.Kernel, error) {
	if width < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: DHFeatures: %d rows of %d", rows, width)
	}
	b := ir.New("dh_features", [3]int{128, 1, 1})
	pF := b.Param("pF", ir.F32)
	pIdx := b.Param("pIdx", ir.U32)
	pO := b.Param("pO", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := convItem(b, width*rows)
	w := u32(b, width)
	r, c := b.Div(ir.U32, i, w), b.Rem(ir.U32, i, w)
	f := b.Load(ir.F32, pF, b.Add(ir.U32, b.Mul(ir.U32, b.Load(ir.U32, pIdx, r, 0), w), c), 0)
	o := b.Load(ir.F32, pO, i, 0)
	d := b.Sub(ir.F32, f, o)
	at := b.Add(ir.U32, b.Mul(ir.U32, r, u32(b, 4*width)), c)
	b.Store(pOut, at, f, 0)
	b.Store(pOut, b.Add(ir.U32, at, w), o, 0)
	b.Store(pOut, b.Add(ir.U32, at, u32(b, 2*width)), b.Mul(ir.F32, f, o), 0)
	b.Store(pOut, b.Add(ir.U32, at, u32(b, 3*width)), b.Max(ir.F32, d, b.Sub(ir.F32, b.ConstF32(0), d)), 0)
	return b.Done(), nil
}

// DHCombine is out = s0*prior + s2*(s1*cos + res) over n options: Clef's
// logit from its prior, its cosine and its scorer's residual. Params:
// pPrior, pCos, pRes, pOut.
func DHCombine(n int, s0, s1, s2 float32) (*ir.Kernel, error) {
	if n < 1 {
		return nil, fmt.Errorf("kernels: DHCombine: %d options", n)
	}
	b := ir.New("dh_combine", [3]int{128, 1, 1})
	pP := b.Param("pPrior", ir.F32)
	pC := b.Param("pCos", ir.F32)
	pR := b.Param("pRes", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := convItem(b, n)
	joint := b.Add(ir.F32, b.Mul(ir.F32, b.ConstF32(s1), b.Load(ir.F32, pC, i, 0)), b.Load(ir.F32, pR, i, 0))
	v := b.Add(ir.F32, b.Mul(ir.F32, b.ConstF32(s0), b.Load(ir.F32, pP, i, 0)), b.Mul(ir.F32, b.ConstF32(s2), joint))
	b.Store(pOut, i, v, 0)
	return b.Done(), nil
}
