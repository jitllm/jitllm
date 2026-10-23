package kernels

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/jit/ir"
)

// A convolutional tower's own kernels (Gemma 3n's MobileNet-V5), over
// channel-last rows: a position is a row of c floats. Every shape is baked,
// one thread an output element, the surplus clamped to the last (RULE 13: no
// kernel reads what it writes). The matrices are the matvec every block runs,
// the norms and activations the elementwise set's.

// convItem is this thread's item of n, clamped to the last.
func convItem(b *ir.Builder, n int) ir.Value {
	return b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(n-1)))
}

func u32(b *ir.Builder, v int) ir.Value { return b.Const(ir.U32, int64(v)) }

// PadRows copies an h x w grid of c channels into an hp x wp one, zero
// outside, the source's first sample at (top, left): TF SAME's padding.
// Params: pX, pOut. Launch hp*wp*c threads.
func PadRows(h, w, c, hp, wp, top, left int) (*ir.Kernel, error) {
	if h < 1 || w < 1 || c < 1 || top < 0 || left < 0 || hp < 1 || wp < 1 {
		return nil, fmt.Errorf("kernels: PadRows: %dx%dx%d into %dx%d at (%d, %d)", h, w, c, hp, wp, top, left)
	}
	b := ir.New("padrows", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := convItem(b, hp*wp*c)
	cc := u32(b, c)
	ch := b.Rem(ir.U32, i, cc)
	pix := b.Div(ir.U32, i, cc)
	x := b.Rem(ir.U32, pix, u32(b, wp))
	y := b.Div(ir.U32, pix, u32(b, wp))
	// Unsigned: a sample before the source wraps far past it.
	sy := b.Sub(ir.U32, y, u32(b, top))
	sx := b.Sub(ir.U32, x, u32(b, left))
	cy := b.Min(ir.U32, sy, u32(b, h-1))
	cx := b.Min(ir.U32, sx, u32(b, w-1))
	v := b.Load(ir.F32, pX, b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, cy, u32(b, w)), cx), cc), ch), 0)
	zero := b.ConstF32(0)
	v = b.Select(ir.F32, b.Lt(ir.U32, sx, u32(b, w)), v, zero)
	v = b.Select(ir.F32, b.Lt(ir.U32, sy, u32(b, h)), v, zero)
	b.Store(pOut, i, v, 0)
	return b.Done()
}

// DWConv is a depthwise k x k convolution at stride s over a padded grid wp
// samples wide, writing ho x wo positions of c channels: out[y][x][ch] = sum
// over taps of pad[y*s+ky][x*s+kx][ch] * w[ky*k+kx][ch]. Params: pPad, pW,
// pOut. Launch ho*wo*c threads.
func DWConv(k, s, wp, c, ho, wo int) (*ir.Kernel, error) {
	if k < 1 || s < 1 || c < 1 || ho < 1 || wo < 1 || (wo-1)*s+k > wp {
		return nil, fmt.Errorf("kernels: DWConv: k=%d s=%d over %d samples to %dx%dx%d", k, s, wp, ho, wo, c)
	}
	b := ir.New("dwconv", [3]int{128, 1, 1})
	pPad := b.Param("pPad", ir.F32)
	pW := b.Param("pW", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := convItem(b, ho*wo*c)
	cc := u32(b, c)
	ch := b.Rem(ir.U32, i, cc)
	pix := b.Div(ir.U32, i, cc)
	x := b.Rem(ir.U32, pix, u32(b, wo))
	y := b.Div(ir.U32, pix, u32(b, wo))
	base := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.Mul(ir.U32, y, u32(b, s)), u32(b, wp)),
		b.Mul(ir.U32, x, u32(b, s))), cc), ch)
	f0, z := b.ConstF32(0), u32(b, 0)
	b.Loop(int64(k * k))
	acc := b.Phi(ir.F32, f0)
	t := b.Phi(ir.U32, z)
	ky := b.Div(ir.U32, t, u32(b, k))
	kx := b.Rem(ir.U32, t, u32(b, k))
	off := b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, ky, u32(b, wp)), kx), cc)
	xv := b.Load(ir.F32, pPad, b.Add(ir.U32, base, off), 0)
	wv := b.Load(ir.F32, pW, b.Add(ir.U32, b.Mul(ir.U32, t, cc), ch), 0)
	b.SetPhi(acc, b.Fma(xv, wv, acc))
	b.SetPhi(t, b.Add(ir.U32, t, u32(b, 1)))
	b.EndLoop()
	b.Store(pOut, i, acc, 0)
	return b.Done()
}

// Im2col unfolds a padded grid wp samples wide for a full k x k convolution at
// stride s: row p (output position y*wo+x) of kpad floats is the window's
// taps, tap-major (ky, kx, c), and zero past k*k*c. Params: pPad, pOut. n is
// the output positions, padLen the padded grid's floats. Launch n*kpad
// threads.
func Im2col(k, s, wp, c, wo, kpad, n, padLen int) (*ir.Kernel, error) {
	if k < 1 || s < 1 || c < 1 || wo < 1 || n < 1 || kpad < k*k*c || padLen < 1 {
		return nil, fmt.Errorf("kernels: Im2col: k=%d s=%d c=%d into %d of %d", k, s, c, n, kpad)
	}
	b := ir.New("im2col", [3]int{128, 1, 1})
	pPad := b.Param("pPad", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := convItem(b, n*kpad)
	j := b.Rem(ir.U32, i, u32(b, kpad))
	p := b.Div(ir.U32, i, u32(b, kpad))
	x := b.Rem(ir.U32, p, u32(b, wo))
	y := b.Div(ir.U32, p, u32(b, wo))
	cc := u32(b, c)
	tap := b.Div(ir.U32, j, cc)
	ch := b.Rem(ir.U32, j, cc)
	ky := b.Div(ir.U32, tap, u32(b, k))
	kx := b.Rem(ir.U32, tap, u32(b, k))
	row := b.Add(ir.U32, b.Mul(ir.U32, y, u32(b, s)), ky)
	col := b.Add(ir.U32, b.Mul(ir.U32, x, u32(b, s)), kx)
	idx := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, row, u32(b, wp)), col), cc), ch)
	v := b.Load(ir.F32, pPad, b.Min(ir.U32, idx, u32(b, padLen-1)), 0)
	v = b.Select(ir.F32, b.Lt(ir.U32, j, u32(b, k*k*c)), v, b.ConstF32(0))
	b.Store(pOut, i, v, 0)
	return b.Done()
}

// MQAScores is multi-query attention's scores: n rows of heads queries of kd
// against m keys of kd, scaled by 1/sqrt(kd); out[(r*heads+h)*m+j]. Params:
// pQ, pK, pOut. Launch n*heads*m threads.
func MQAScores(n, m, heads, kd int) (*ir.Kernel, error) {
	if n < 1 || m < 1 || heads < 1 || kd < 1 {
		return nil, fmt.Errorf("kernels: MQAScores: %d rows of %d heads of %d over %d keys", n, heads, kd, m)
	}
	b := ir.New("mqascores", [3]int{128, 1, 1})
	pQ := b.Param("pQ", ir.F32)
	pK := b.Param("pK", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := convItem(b, n*heads*m)
	j := b.Rem(ir.U32, i, u32(b, m))
	rh := b.Div(ir.U32, i, u32(b, m))
	qb := b.Mul(ir.U32, rh, u32(b, kd))
	kb := b.Mul(ir.U32, j, u32(b, kd))
	f0, z := b.ConstF32(0), u32(b, 0)
	b.Loop(int64(kd))
	acc := b.Phi(ir.F32, f0)
	d := b.Phi(ir.U32, z)
	qv := b.Load(ir.F32, pQ, b.Add(ir.U32, qb, d), 0)
	kv := b.Load(ir.F32, pK, b.Add(ir.U32, kb, d), 0)
	b.SetPhi(acc, b.Fma(qv, kv, acc))
	b.SetPhi(d, b.Add(ir.U32, d, u32(b, 1)))
	b.EndLoop()
	b.Store(pOut, i, b.Mul(ir.F32, acc, b.ConstF32(float32(1/math.Sqrt(float64(kd))))), 0)
	return b.Done()
}

// RowSoftmax is a softmax over each of rows rows of m, one thread a row.
// Params: pX, pOut. Launch rows threads.
func RowSoftmax(rows, m int) (*ir.Kernel, error) {
	if rows < 1 || m < 1 {
		return nil, fmt.Errorf("kernels: RowSoftmax: %d rows of %d", rows, m)
	}
	b := ir.New("rowsoftmax", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	r := convItem(b, rows)
	base := b.Mul(ir.U32, r, u32(b, m))
	lowest, f0, z := b.ConstF32(-math.MaxFloat32), b.ConstF32(0), u32(b, 0)
	b.Loop(int64(m))
	mx := b.Phi(ir.F32, lowest)
	j := b.Phi(ir.U32, z)
	b.SetPhi(mx, b.Max(ir.F32, mx, b.Load(ir.F32, pX, b.Add(ir.U32, base, j), 0)))
	b.SetPhi(j, b.Add(ir.U32, j, u32(b, 1)))
	b.EndLoop()
	b.Loop(int64(m))
	sum := b.Phi(ir.F32, f0)
	j2 := b.Phi(ir.U32, z)
	b.SetPhi(sum, b.Add(ir.F32, sum, b.Exp(b.Sub(ir.F32, b.Load(ir.F32, pX, b.Add(ir.U32, base, j2), 0), mx))))
	b.SetPhi(j2, b.Add(ir.U32, j2, u32(b, 1)))
	b.EndLoop()
	inv := b.Div(ir.F32, b.ConstF32(1), sum)
	b.Loop(int64(m))
	j3 := b.Phi(ir.U32, z)
	at := b.Add(ir.U32, base, j3)
	b.Store(pOut, at, b.Mul(ir.F32, b.Exp(b.Sub(ir.F32, b.Load(ir.F32, pX, at, 0), mx)), inv), 0)
	b.SetPhi(j3, b.Add(ir.U32, j3, u32(b, 1)))
	b.EndLoop()
	return b.Done()
}

// MQAAcc is multi-query attention's weighted sum: out[(r*heads+h)*kd+d] = sum
// over j < m of p[(r*heads+h)*m+j] * v[j*kd+d]. Params: pP, pV, pOut. Launch
// n*heads*kd threads.
func MQAAcc(n, m, heads, kd int) (*ir.Kernel, error) {
	if n < 1 || m < 1 || heads < 1 || kd < 1 {
		return nil, fmt.Errorf("kernels: MQAAcc: %d rows of %d heads of %d over %d keys", n, heads, kd, m)
	}
	b := ir.New("mqaacc", [3]int{128, 1, 1})
	pP := b.Param("pP", ir.F32)
	pV := b.Param("pV", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := convItem(b, n*heads*kd)
	d := b.Rem(ir.U32, i, u32(b, kd))
	rh := b.Div(ir.U32, i, u32(b, kd))
	pb := b.Mul(ir.U32, rh, u32(b, m))
	f0, z := b.ConstF32(0), u32(b, 0)
	b.Loop(int64(m))
	acc := b.Phi(ir.F32, f0)
	j := b.Phi(ir.U32, z)
	pv := b.Load(ir.F32, pP, b.Add(ir.U32, pb, j), 0)
	vv := b.Load(ir.F32, pV, b.Add(ir.U32, b.Mul(ir.U32, j, u32(b, kd)), d), 0)
	b.SetPhi(acc, b.Fma(pv, vv, acc))
	b.SetPhi(j, b.Add(ir.U32, j, u32(b, 1)))
	b.EndLoop()
	b.Store(pOut, i, acc, 0)
	return b.Done()
}
