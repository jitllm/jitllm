package oldkernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// Kimi-K3's residual attention on the device (nn.LayerPlan.K3;
// engine/model/k3.go is the graph and jit/gpu/tier/k3.go the wiring).
//
// The residual is streams of d, stream-major for `rows` rows (stream k of row
// r at (k*rows + r)*d): stream 0 the running residual, stream 1+j checkpoint
// j. A sublayer's input is the softmax mix of the first nb checkpoints and the
// running residual:
//
//	K3ResScores  each row's nb+1 scores, sum(w * v_j / rms(v_j)), the
//	             checkpoints first and the running residual last
//	K3ResMix     each row's sum_j p_j v_j, p the scores' softmax
//	K3Post       every stream after a sublayer: the running residual plus its
//	             output (or, restarting, the output alone), and the block's raw
//	             input banked at the checkpoint stream when it has one

// k3Stream is the stream a score or mix term j of nb reads: checkpoint j at
// stream j+1, and the running residual (j == nb) at stream 0.
func k3Stream(j, nb int) int {
	if j == nb {
		return 0
	}
	return j + 1
}

// K3ResScores writes each row's nb+1 scores over x: sum_i w[i] *
// v[i] * rsqrt(mean(v^2) + eps), or with raw the plain dot (a gate's
// violation). One thread a (row, term), the width walked in one loop.
// Params pX, pW, pS; pS is [rows][nb+1].
func K3ResScores(d, rows, nb int, eps float32, raw bool) (*ir.Kernel, error) {
	if d < 1 || rows < 1 || nb < 1 {
		return nil, fmt.Errorf("kernels: K3ResScores: d=%d rows=%d bank=%d", d, rows, nb)
	}
	ns := nb + 1
	b := ir.New("k3resscores", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pW := b.Param("pW", ir.F32)
	pS := b.Param("pS", ir.F32)
	t := flatID(b, rows*ns)
	r := b.Div(ir.U32, t, b.Const(ir.U32, int64(ns)))
	j := b.Sub(ir.U32, t, b.Mul(ir.U32, r, b.Const(ir.U32, int64(ns))))
	// stream = j+1, or 0 at j == nb: (j+1) mod ns.
	k := b.Rem(ir.U32, b.Add(ir.U32, j, b.Const(ir.U32, 1)), b.Const(ir.U32, int64(ns)))
	base := b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, k, b.Const(ir.U32, int64(rows))), r),
		b.Const(ir.U32, int64(d)))
	zeroF, zeroU := b.ConstF32(0), b.Const(ir.U32, 0)
	b.Loop(int64(d))
	ss := b.Phi(ir.F32, zeroF)
	dot := b.Phi(ir.F32, zeroF)
	i := b.Phi(ir.U32, zeroU)
	v := b.Load(ir.F32, pX, b.Add(ir.U32, base, i), 0)
	b.SetPhi(ss, b.Fma(v, v, ss))
	b.SetPhi(dot, b.Fma(v, b.Load(ir.F32, pW, i, 0), dot))
	b.SetPhi(i, b.Add(ir.U32, i, b.Const(ir.U32, 1)))
	b.EndLoop()
	s := dot
	if !raw {
		ms := b.Add(ir.F32, b.Div(ir.F32, ss, b.ConstF32(float32(d))), b.ConstF32(eps))
		s = b.Div(ir.F32, dot, b.Sqrt(ms))
	}
	b.Store(pS, t, s, 0)
	return b.Done(), nil
}

// K3ResMix writes each row's mix over x: sum_j p_j v_j[i], p the softmax of
// the row's nb+1 scores (K3ResScores' order). One thread an element.
// Params pX, pS, pOut; pOut is [rows][d].
func K3ResMix(d, rows, nb int) (*ir.Kernel, error) {
	if d < 1 || rows < 1 || nb < 1 {
		return nil, fmt.Errorf("kernels: K3ResMix: d=%d rows=%d bank=%d", d, rows, nb)
	}
	ns := nb + 1
	b := ir.New("k3resmix", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pS := b.Param("pS", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	t := flatID(b, rows*d)
	r := b.Div(ir.U32, t, b.Const(ir.U32, int64(d)))
	i := b.Sub(ir.U32, t, b.Mul(ir.U32, r, b.Const(ir.U32, int64(d))))
	sb := b.Mul(ir.U32, r, b.Const(ir.U32, int64(ns)))
	sc := make([]ir.Value, ns)
	var mx ir.Value
	for j := range sc {
		sc[j] = b.Load(ir.F32, pS, sb, int64(j))
		if j == 0 {
			mx = sc[0]
		} else {
			mx = b.Max(ir.F32, mx, sc[j])
		}
	}
	sum := b.ConstF32(0)
	for j := range sc {
		sc[j] = b.Exp(b.Sub(ir.F32, sc[j], mx))
		sum = b.Add(ir.F32, sum, sc[j])
	}
	acc := b.ConstF32(0)
	for j := range sc {
		at := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, b.Const(ir.U32, int64(k3Stream(j, nb)*rows)), r),
			b.Const(ir.U32, int64(d))), i)
		acc = b.Fma(b.Div(ir.F32, sc[j], sum), b.Load(ir.F32, pX, at, 0), acc)
	}
	b.Store(pOut, t, acc, 0)
	return b.Done(), nil
}

// K3Post writes every stream of out from x and a sublayer's output y
// ([rows][d]): stream 0 is x's plus y, or y alone where restart; stream push
// (when non-zero) is x's stream 0, the block's raw input banked; every other
// stream is x's. One thread an element of every stream. Params pX, pY, pOut.
func K3Post(d, rows, streams, push int, restart bool) (*ir.Kernel, error) {
	if d < 1 || rows < 1 || streams < 2 || push < 0 || push >= streams {
		return nil, fmt.Errorf("kernels: K3Post: d=%d rows=%d streams=%d push=%d", d, rows, streams, push)
	}
	b := ir.New("k3post", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pY := b.Param("pY", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	n := rows * d
	t := flatID(b, streams*n)
	nn := b.Const(ir.U32, int64(n))
	k := b.Div(ir.U32, t, nn)
	e := b.Sub(ir.U32, t, b.Mul(ir.U32, k, nn)) // the element within a stream
	x0 := b.Load(ir.F32, pX, e, 0)
	y := b.Load(ir.F32, pY, e, 0)
	s0 := y
	if !restart {
		s0 = b.Add(ir.F32, x0, y)
	}
	v := b.Load(ir.F32, pX, t, 0)
	if push > 0 {
		// k == push: not below it, and below push+1 (the IR compares by Lt).
		at := b.Select(ir.F32, b.Lt(ir.U32, k, b.Const(ir.U32, int64(push+1))), x0, v)
		v = b.Select(ir.F32, b.Lt(ir.U32, k, b.Const(ir.U32, int64(push))), v, at)
	}
	b.Store(pOut, t, b.Select(ir.F32, b.Lt(ir.U32, k, b.Const(ir.U32, 1)), s0, v), 0)
	return b.Done(), nil
}
