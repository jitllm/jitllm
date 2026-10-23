package kernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/ir"
)

// Conv3Centred is LFM2's encoder convolution (d1-omni's trunk) over a whole
// non-causal call: a depthwise convolution of three taps centred on the row,
//
//	out[t][c] = W[0][c] * X[t-1][c] + W[1][c] * X[t][c] + W[2][c] * X[t+1][c]
//
// a neighbour taken only where row t's window, [lo, hi) in pWin (the row
// windows a non-causal call's attention reads, two words a row), holds it --
// so the rows past either end of the call read zero, and a media prefix's
// last row, whose window is the prefix, does not read the text's first. W is
// plane-major, a tap's channels in a row, as the short convolution's state
// is. No state: the call is the whole sequence. Launch rows*chans threads.
func Conv3Centred(chans, rows int) (*ir.Kernel, error) {
	if chans <= 0 || rows <= 0 {
		return nil, fmt.Errorf("kernels: Conv3Centred chans=%d rows=%d", chans, rows)
	}
	b := ir.New("conv3centred", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pW := b.Param("pW", ir.F32)
	pWin := b.Param("pWin", ir.U32)
	pOut := b.Param("pOut", ir.F32)

	last := b.Const(ir.U32, int64(rows*chans-1))
	idx := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()), last)
	cw := b.Const(ir.U32, int64(chans))
	t := b.Div(ir.U32, idx, cw)
	c := b.Sub(ir.U32, idx, b.Mul(ir.U32, t, cw))
	one := b.Const(ir.U32, 1)
	tt := b.Add(ir.U32, t, t)
	lo := b.Load(ir.U32, pWin, tt, 0)
	hi := b.Load(ir.U32, pWin, tt, 1)
	// lo <= t-1 and t+1 < hi. Both neighbours are loaded, clamped into the
	// buffer (t-1 wraps at row 0 and is clamped), and selected away.
	okL := b.Lt(ir.U32, lo, t)
	okR := b.Lt(ir.U32, b.Add(ir.U32, t, one), hi)
	xl := b.Load(ir.F32, pX, b.Min(ir.U32, b.Sub(ir.U32, idx, cw), last), 0)
	xr := b.Load(ir.F32, pX, b.Min(ir.U32, b.Add(ir.U32, idx, cw), last), 0)
	xm := b.Load(ir.F32, pX, idx, 0)
	w0 := b.Load(ir.F32, pW, c, 0)
	w1 := b.Load(ir.F32, pW, b.Add(ir.U32, c, cw), 0)
	w2 := b.Load(ir.F32, pW, b.Add(ir.U32, c, b.Add(ir.U32, cw, cw)), 0)
	zero := b.ConstF32(0)
	acc := b.Mul(ir.F32, w1, xm)
	acc = b.Fma(w0, b.Select(ir.F32, okL, xl, zero), acc)
	acc = b.Fma(w2, b.Select(ir.F32, okR, xr, zero), acc)
	b.Store(pOut, idx, acc, 0)
	return b.Done()
}
