package mainkernels

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// Gemma 3n's AltUp and LAuReL on a device (engine/model/altup.go has the
// graph). The residual is S streams of d for each of `rows` rows, stream-major:
// stream k of row r at (k*rows + r)*d, so stream 0 is the ordinary residual
// every other block kernel reads. Each kernel is basic: one thread per output
// element, S unrolled where it appears.

// flatOf is a launch's clamped flat index over n items.
func flatOf(b *ir.Builder, n int) ir.Value {
	return b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()), b.Const(ir.U32, int64(n-1)))
}

// tanhOf is 1 - 2/(exp(2x) + 1), Softcap's spelling.
func tanhOf(b *ir.Builder, x ir.Value) ir.Value {
	one := b.ConstF32(1)
	e := b.Exp(b.Mul(ir.F32, b.ConstF32(2), x))
	return b.Sub(ir.F32, one, b.Div(ir.F32, b.ConstF32(2), b.Add(ir.F32, e, one)))
}

// AltUpRoute is AltUp's router: m[r][j] = tanh((h_r . W_j) / d), h the
// router-normed active stream, W the S x d router as plain floats.
// Parameters: pH (rows x d), pW (S x d), pM (rows x S).
func AltUpRoute(rows, d, s int) (*ir.Kernel, error) {
	if rows < 1 || d < 1 || s < 2 {
		return nil, fmt.Errorf("kernels: AltUpRoute: rows=%d d=%d streams=%d", rows, d, s)
	}
	b := ir.New("altuproute", [3]int{128, 1, 1})
	pH := b.Param("pH", ir.F32)
	pW := b.Param("pW", ir.F32)
	pM := b.Param("pM", ir.F32)
	flat := flatOf(b, rows*s)
	r := b.Div(ir.U32, flat, b.Const(ir.U32, int64(s)))
	j := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(s)))
	hb := b.Mul(ir.U32, r, b.Const(ir.U32, int64(d)))
	wb := b.Mul(ir.U32, j, b.Const(ir.U32, int64(d)))
	zf, zu, one := b.ConstF32(0), b.Const(ir.U32, 0), b.Const(ir.U32, 1)
	b.Loop(int64(d))
	i := b.Phi(ir.U32, zu)
	acc := b.Phi(ir.F32, zf)
	b.SetPhi(acc, b.Fma(b.Load(ir.F32, pH, b.Add(ir.U32, hb, i), 0), b.Load(ir.F32, pW, b.Add(ir.U32, wb, i), 0), acc))
	b.SetPhi(i, b.Add(ir.U32, i, one))
	b.EndLoop()
	b.Store(pM, flat, tanhOf(b, b.Mul(ir.F32, acc, b.ConstF32(1/float32(d)))), 0)
	return b.Done(), nil
}

// AltUpPredict is AltUp's prediction: out_k = x_k + sum_j c_kj x_j with
// c = sum_i m_i T_i, T the prediction coefficients transposed (S rows of
// S*S). Parameters: pX (S x rows x d), pM (rows x S), pT (S x S*S), pOut.
func AltUpPredict(rows, d, s int) (*ir.Kernel, error) {
	if rows < 1 || d < 1 || s < 2 {
		return nil, fmt.Errorf("kernels: AltUpPredict: rows=%d d=%d streams=%d", rows, d, s)
	}
	b := ir.New("altuppredict", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pM := b.Param("pM", ir.F32)
	pT := b.Param("pT", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	plane := rows * d
	flat := flatOf(b, s*plane)
	k := b.Div(ir.U32, flat, b.Const(ir.U32, int64(plane)))
	rem := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(plane)))
	r := b.Div(ir.U32, rem, b.Const(ir.U32, int64(d)))
	mb := b.Mul(ir.U32, r, b.Const(ir.U32, int64(s)))
	kb := b.Mul(ir.U32, k, b.Const(ir.U32, int64(s)))
	acc := b.Load(ir.F32, pX, flat, 0)
	for j := 0; j < s; j++ {
		var c ir.Value = b.ConstF32(0)
		for i := 0; i < s; i++ {
			// T[i][k*S + j]
			t := b.Load(ir.F32, pT, b.Add(ir.U32, kb, b.Const(ir.U32, int64(i*s*s+j))), 0)
			c = b.Fma(b.Load(ir.F32, pM, mb, int64(i)), t, c)
		}
		acc = b.Fma(c, b.Load(ir.F32, pX, rem, int64(j*plane)), acc)
	}
	b.Store(pOut, flat, acc, 0)
	return b.Done(), nil
}

// AltUpCorrect is AltUp's correction: out_k = pred_k + (1 + sum_i m_i T_ik) *
// (o - pred_0), o the block's output (stream 0's plane) and T the correction
// coefficients transposed (S rows of S). Parameters: pPred (S x rows x d), pO
// (rows x d), pM (rows x S), pT (S x S), pOut (S x rows x d).
func AltUpCorrect(rows, d, s int) (*ir.Kernel, error) {
	if rows < 1 || d < 1 || s < 2 {
		return nil, fmt.Errorf("kernels: AltUpCorrect: rows=%d d=%d streams=%d", rows, d, s)
	}
	b := ir.New("altupcorrect", [3]int{128, 1, 1})
	pPred := b.Param("pPred", ir.F32)
	pO := b.Param("pO", ir.F32)
	pM := b.Param("pM", ir.F32)
	pT := b.Param("pT", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	plane := rows * d
	flat := flatOf(b, s*plane)
	k := b.Div(ir.U32, flat, b.Const(ir.U32, int64(plane)))
	rem := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(plane)))
	r := b.Div(ir.U32, rem, b.Const(ir.U32, int64(d)))
	mb := b.Mul(ir.U32, r, b.Const(ir.U32, int64(s)))
	var c ir.Value = b.ConstF32(0)
	for i := 0; i < s; i++ {
		t := b.Load(ir.F32, pT, k, int64(i*s))
		c = b.Fma(b.Load(ir.F32, pM, mb, int64(i)), t, c)
	}
	c = b.Add(ir.F32, c, b.ConstF32(1))
	inn := b.Sub(ir.F32, b.Load(ir.F32, pO, rem, 0), b.Load(ir.F32, pPred, rem, 0))
	b.Store(pOut, flat, b.Fma(c, inn, b.Load(ir.F32, pPred, flat, 0)), 0)
	return b.Done(), nil
}

// AltUpFinish adds the per-layer input's projection p to every stream but the
// first: out = src + (k > 0 ? p : 0). Parameters: pSrc (S x rows x d), pP
// (rows x d), pOut (S x rows x d).
func AltUpFinish(rows, d, s int) (*ir.Kernel, error) {
	if rows < 1 || d < 1 || s < 2 {
		return nil, fmt.Errorf("kernels: AltUpFinish: rows=%d d=%d streams=%d", rows, d, s)
	}
	b := ir.New("altupfinish", [3]int{128, 1, 1})
	pSrc := b.Param("pSrc", ir.F32)
	pP := b.Param("pP", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	plane := rows * d
	flat := flatOf(b, s*plane)
	rem := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(plane)))
	rest := b.Lt(ir.U32, b.Const(ir.U32, int64(plane-1)), flat)
	p := b.Select(ir.F32, rest, b.Load(ir.F32, pP, rem, 0), b.ConstF32(0))
	b.Store(pOut, flat, b.Add(ir.F32, b.Load(ir.F32, pSrc, flat, 0), p), 0)
	return b.Done(), nil
}

// MulRows is out[r][i] = x[r][i] * w[i] over rows rows of d: the corrected
// active stream times AltUp's scale before the per-layer input's gate.
// Parameters: pX, pW, pOut.
func MulRows(rows, d int) (*ir.Kernel, error) {
	if rows < 1 || d < 1 {
		return nil, fmt.Errorf("kernels: MulRows: rows=%d d=%d", rows, d)
	}
	b := ir.New("mulrows", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pW := b.Param("pW", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	flat := flatOf(b, rows*d)
	i := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(d)))
	b.Store(pOut, flat, b.Mul(ir.F32, b.Load(ir.F32, pX, flat, 0), b.Load(ir.F32, pW, i, 0)), 0)
	return b.Done(), nil
}

// LaurelJoin is the attention's residual add with LAuReL's branch beside it:
// out = (x + y + z) / sqrt(2). Parameters: pX, pY, pZ, pOut.
func LaurelJoin(n int) (*ir.Kernel, error) {
	if n < 1 {
		return nil, fmt.Errorf("kernels: LaurelJoin: n=%d", n)
	}
	b := ir.New("laureljoin", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pY := b.Param("pY", ir.F32)
	pZ := b.Param("pZ", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := flatOf(b, n)
	v := b.Add(ir.F32, b.Add(ir.F32, b.Load(ir.F32, pX, i, 0), b.Load(ir.F32, pY, i, 0)), b.Load(ir.F32, pZ, i, 0))
	b.Store(pOut, i, b.Mul(ir.F32, v, b.ConstF32(float32(1/math.Sqrt2))), 0)
	return b.Done(), nil
}

// GaussTopKApplyRows is Gemma 3n's activation sparsity over rows rows of k,
// after LayerNormPartRows and LayerNormVarRows: out = max(x - (mean +
// c*sqrt(var)), 0), the population variance. Parameters: pX, pSum, pVar,
// pOut.
func GaussTopKApplyRows(k, parts int, c float32, rows int) (*ir.Kernel, error) {
	if rows < 1 || k < 1 || parts < 1 || k%parts != 0 {
		return nil, fmt.Errorf("kernels: GaussTopKApplyRows: k=%d parts=%d rows=%d", k, parts, rows)
	}
	b := ir.New("gausstopk", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pSum := b.Param("pSum", ir.F32)
	pVar := b.Param("pVar", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := flatOf(b, k*rows)
	base := b.Mul(ir.U32, b.Div(ir.U32, i, b.Const(ir.U32, int64(k))), b.Const(ir.U32, int64(parts)))
	mean := layerNormMeanAt(b, pSum, base, parts, k)
	vsum := b.ConstF32(0)
	for p := 0; p < parts; p++ {
		vsum = b.Add(ir.F32, vsum, b.Load(ir.F32, pVar, base, int64(p)))
	}
	sd := b.Sqrt(b.Mul(ir.F32, vsum, b.ConstF32(1/float32(k))))
	cut := b.Fma(sd, b.ConstF32(c), mean)
	b.Store(pOut, i, b.Max(ir.F32, b.Sub(ir.F32, b.Load(ir.F32, pX, i, 0), cut), b.ConstF32(0)), 0)
	return b.Done(), nil
}
