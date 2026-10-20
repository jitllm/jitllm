package oldkernels

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// A FastConformer block's kernels (d1-omni's audio tower; engine/model's
// audio.go is the host's transcription). Each runs over a scratch of R rows
// of which the first t are the clip's: t is read from a one-word buffer pN,
// so one compiled program serves every clip up to R rows. A key, a value or a
// neighbour past t is never read into a row: it is SELECTED away, not
// multiplied by zero, since the rows past t hold whatever the scratch held
// (0 * NaN is NaN; RULE 13).

// RelPosScores is relative-position attention's scores (Transformer-XL with
// per-head u and v biases): for row i, head h and key j < t,
//
//	out[(i*heads+h)*R+j] = ((q_i+u) . k_j + (q_i+v) . P[R-1-i+j]) / sqrt(kd)
//
// over the head's kd columns, and -3e38 for j >= t. P is the projected
// relative-position table, 2R-1 rows from position R-1 down to -(R-1), so
// row R-1-i+j is position i-j. Params: pQ, pK, pP (rows of heads*kd), pU, pV
// (heads*kd), pN (u32 t), pOut. Launch R*heads*R threads.
func RelPosScores(r, heads, kd int) (*ir.Kernel, error) {
	if r < 1 || heads < 1 || kd < 1 {
		return nil, fmt.Errorf("kernels: RelPosScores: %d rows of %d heads of %d", r, heads, kd)
	}
	b := ir.New("relposscores", [3]int{128, 1, 1})
	pQ := b.Param("pQ", ir.F32)
	pK := b.Param("pK", ir.F32)
	pP := b.Param("pP", ir.F32)
	pU := b.Param("pU", ir.F32)
	pV := b.Param("pV", ir.F32)
	pN := b.Param("pN", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	d := heads * kd
	i := convItem(b, r*heads*r)
	j := b.Rem(ir.U32, i, u32(b, r))
	ih := b.Div(ir.U32, i, u32(b, r))
	h := b.Rem(ir.U32, ih, u32(b, heads))
	row := b.Div(ir.U32, ih, u32(b, heads))
	t := b.Load(ir.U32, pN, u32(b, 0), 0)
	hc := b.Mul(ir.U32, h, u32(b, kd))
	qb := b.Add(ir.U32, b.Mul(ir.U32, row, u32(b, d)), hc)
	kb := b.Add(ir.U32, b.Mul(ir.U32, j, u32(b, d)), hc)
	// P's row R-1-i+j: in range for every i, j < R.
	pr := b.Sub(ir.U32, b.Add(ir.U32, u32(b, r-1), j), row)
	pb := b.Add(ir.U32, b.Mul(ir.U32, pr, u32(b, d)), hc)
	f0, z := b.ConstF32(0), u32(b, 0)
	b.Loop(int64(kd))
	acc := b.Phi(ir.F32, f0)
	e := b.Phi(ir.U32, z)
	q := b.Load(ir.F32, pQ, b.Add(ir.U32, qb, e), 0)
	ue := b.Load(ir.F32, pU, b.Add(ir.U32, hc, e), 0)
	ve := b.Load(ir.F32, pV, b.Add(ir.U32, hc, e), 0)
	kv := b.Load(ir.F32, pK, b.Add(ir.U32, kb, e), 0)
	pv := b.Load(ir.F32, pP, b.Add(ir.U32, pb, e), 0)
	a2 := b.Fma(b.Add(ir.F32, q, ue), kv, acc)
	b.SetPhi(acc, b.Fma(b.Add(ir.F32, q, ve), pv, a2))
	b.SetPhi(e, b.Add(ir.U32, e, u32(b, 1)))
	b.EndLoop()
	s := b.Mul(ir.F32, acc, b.ConstF32(float32(1/math.Sqrt(float64(kd)))))
	b.Store(pOut, i, b.Select(ir.F32, b.Lt(ir.U32, j, t), s, b.ConstF32(-3e38)), 0)
	return b.Done(), nil
}

// AttnAccRows is multi-head attention's weighted sum over the first t keys:
// out[i*heads*kd + h*kd + e] = sum over j < t of p[(i*heads+h)*R+j] *
// v[j*heads*kd + h*kd + e]. Params: pP, pV, pN (u32 t), pOut. Launch
// R*heads*kd threads.
func AttnAccRows(r, heads, kd int) (*ir.Kernel, error) {
	if r < 1 || heads < 1 || kd < 1 {
		return nil, fmt.Errorf("kernels: AttnAccRows: %d rows of %d heads of %d", r, heads, kd)
	}
	b := ir.New("attnaccrows", [3]int{128, 1, 1})
	pP := b.Param("pP", ir.F32)
	pV := b.Param("pV", ir.F32)
	pN := b.Param("pN", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	d := heads * kd
	i := convItem(b, r*d)
	col := b.Rem(ir.U32, i, u32(b, d))
	row := b.Div(ir.U32, i, u32(b, d))
	h := b.Div(ir.U32, col, u32(b, kd))
	t := b.Load(ir.U32, pN, u32(b, 0), 0)
	pb := b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, row, u32(b, heads)), h), u32(b, r))
	f0, z := b.ConstF32(0), u32(b, 0)
	b.Loop(int64(r))
	acc := b.Phi(ir.F32, f0)
	j := b.Phi(ir.U32, z)
	pv := b.Load(ir.F32, pP, b.Add(ir.U32, pb, j), 0)
	vv := b.Load(ir.F32, pV, b.Add(ir.U32, b.Mul(ir.U32, j, u32(b, d)), col), 0)
	b.SetPhi(acc, b.Select(ir.F32, b.Lt(ir.U32, j, t), b.Fma(pv, vv, acc), acc))
	b.SetPhi(j, b.Add(ir.U32, j, u32(b, 1)))
	b.EndLoop()
	b.Store(pOut, i, acc, 0)
	return b.Done(), nil
}

// DWConv1dRows is a depthwise convolution of k taps (k odd) centred on each
// row over the first t rows of c channels, zero past either end of them:
// out[i][ch] = sum over u of w[u][ch] * x[i+u-k/2][ch], the term taken only
// where 0 <= i+u-k/2 < t. w is tap-major. Params: pX, pW, pN (u32 t), pOut.
// Launch R*c threads.
func DWConv1dRows(k, c, r int) (*ir.Kernel, error) {
	if k < 1 || k%2 != 1 || c < 1 || r < 1 {
		return nil, fmt.Errorf("kernels: DWConv1dRows: %d taps over %d channels of %d rows", k, c, r)
	}
	b := ir.New("dwconv1drows", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pW := b.Param("pW", ir.F32)
	pN := b.Param("pN", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	last := u32(b, r*c-1)
	i := convItem(b, r*c)
	ch := b.Rem(ir.U32, i, u32(b, c))
	row := b.Div(ir.U32, i, u32(b, c))
	t := b.Load(ir.U32, pN, u32(b, 0), 0)
	acc := b.ConstF32(0)
	half := k / 2
	for u := 0; u < k; u++ {
		// src = row + u - half, in u32: a row before the first wraps past
		// every t and is selected away.
		src := b.Sub(ir.U32, b.Add(ir.U32, row, u32(b, u)), u32(b, half))
		ok := b.Lt(ir.U32, src, t)
		xv := b.Load(ir.F32, pX, b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, src, u32(b, c)), ch), last), 0)
		wv := b.Load(ir.F32, pW, b.Add(ir.U32, u32(b, u*c), ch), 0)
		acc = b.Select(ir.F32, ok, b.Fma(wv, xv, acc), acc)
	}
	b.Store(pOut, i, acc, 0)
	return b.Done(), nil
}

// AffineRows is out = x*s + h, s and h one value a column shared by every
// row: a BatchNorm folded into a scale and a shift. Params: pX, pS, pH,
// pOut. Launch width*rows threads.
func AffineRows(width, rows int) (*ir.Kernel, error) {
	if width < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: AffineRows width=%d rows=%d", width, rows)
	}
	b := ir.New("affinerows", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pS := b.Param("pS", ir.F32)
	pH := b.Param("pH", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := convItem(b, width*rows)
	c := b.Rem(ir.U32, i, u32(b, width))
	b.Store(pOut, i, b.Fma(b.Load(ir.F32, pX, i, 0), b.Load(ir.F32, pS, c, 0), b.Load(ir.F32, pH, c, 0)), 0)
	return b.Done(), nil
}

// AddHalf is out = a + b/2, one rounding (ir.Fma, as AddScaled): a
// FastConformer block's half feed-forward residual, its one half baked.
// Params: pA, pB, pOut. Launch n threads.
func AddHalf(n int) (*ir.Kernel, error) {
	if n < 1 {
		return nil, fmt.Errorf("kernels: AddHalf n=%d", n)
	}
	b := ir.New("add_half", [3]int{128, 1, 1})
	pA := b.Param("pA", ir.F32)
	pB := b.Param("pB", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := convItem(b, n)
	b.Store(pOut, i, b.Fma(b.ConstF32(0.5), b.Load(ir.F32, pB, i, 0), b.Load(ir.F32, pA, i, 0)), 0)
	return b.Done(), nil
}
