package kernels

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// Qwen3.8-Flash-Next on the device (nn.LayerPlan.Q4E; engine/model/q4e.go is
// the graph and jit/gpu/tier/q4e.go the wiring).
//
// The residual is h streams, stream-major as DeepSeek V4's: stream k
// of row r at (k*rows + r)*d. A sublayer's mix:
//
//	Q4EGroupNorm  each stream RMSNormed against its own row of the weight,
//	              row-major out (row r's stream k at (r*h + k)*d)
//	Q4EScaleSiLU  the gate's low rank: silu(x/h)
//	Q4EGateMix    the mean over the streams of sigmoid(gate) * normed
//	Q4EInject     the placement weights, 2*sigmoid(inject/h)
//	Q4EPlace      the sublayer's output added into every stream at its weight
//
// and the n-gram block's: Q4EStreamDot (each stream's gate, the signed
// square root's sigmoid of its key and residual's scaled dot), Q4EGatedValue
// (the gate times the shared value, per stream), Q4ENGramConv (the dilated
// convolution over the slot's history, through silu, and the history moved
// on) and Q4ENGramAdd (both into the streams).

// Q4E's placement and violation switches, the host's (nn.Q4EFault).
const (
	Q4EPlaceGated = iota
	Q4EPlacePlain
	Q4EPlaceHalf
)

// Q4EGroupNorm is each of rows*h stream rows RMSNormed against its own
// d-wide row of the weight: in[(k*rows + r)*d + i] when streamMajor, else
// in[(r*h + k)*d + i], out[(r*h + k)*d + i] = in * rsqrt(mean(in^2) + eps) *
// w[k*d + i]. Params pX, pW [h*d], pOut; rows*h workgroups of RMSNormGroup.
func Q4EGroupNorm(d, h, rows int, eps float32, streamMajor bool) (*ir.Kernel, error) {
	if d < 1 || h < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: Q4EGroupNorm: d=%d h=%d rows=%d", d, h, rows)
	}
	const g = RMSNormGroup
	name := "q4egroupnorm"
	if streamMajor {
		name = "q4egroupnorms"
	}
	b := ir.New(name, [3]int{g, 1, 1})
	pX := b.Param("pX", ir.F32)
	pW := b.Param("pW", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	part := b.Shared("ss", ir.F32, g)
	row := b.Min(ir.U32, b.CTAID(), b.Const(ir.U32, int64(rows*h-1)))
	r := b.Div(ir.U32, row, b.Const(ir.U32, int64(h)))
	k := b.Rem(ir.U32, row, b.Const(ir.U32, int64(h)))
	dd := b.Const(ir.U32, int64(d))
	inBase := b.Mul(ir.U32, row, dd)
	if streamMajor {
		inBase = b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, k, b.Const(ir.U32, int64(rows))), r), dd)
	}
	outBase := b.Mul(ir.U32, row, dd)
	wBase := b.Mul(ir.U32, k, dd)
	last := b.Const(ir.U32, int64(d-1))
	step := b.Const(ir.U32, g)
	zero := b.ConstF32(0)
	tid := b.TID()
	iters := int64((d + g - 1) / g)
	b.Loop(iters)
	acc := b.Phi(ir.F32, zero)
	i := b.Phi(ir.U32, tid)
	v := b.Load(ir.F32, pX, b.Add(ir.U32, inBase, b.Min(ir.U32, i, last)), 0)
	b.SetPhi(acc, b.Add(ir.F32, acc, b.Select(ir.F32, b.Lt(ir.U32, i, dd), b.Mul(ir.F32, v, v), zero)))
	b.SetPhi(i, b.Add(ir.U32, i, step))
	b.EndLoop()
	ss := treeSum(b, part, tid, acc, g)
	mean := b.Add(ir.F32, b.Mul(ir.F32, ss, b.ConstF32(1/float32(d))), b.ConstF32(eps))
	scale := b.Div(ir.F32, b.ConstF32(1), b.Sqrt(mean))
	b.Loop(iters)
	i2 := b.Phi(ir.U32, tid)
	idx := b.Min(ir.U32, i2, last)
	x := b.Load(ir.F32, pX, b.Add(ir.U32, inBase, idx), 0)
	w := b.Load(ir.F32, pW, b.Add(ir.U32, wBase, idx), 0)
	b.Store(pOut, b.Add(ir.U32, outBase, idx), b.Mul(ir.F32, b.Mul(ir.F32, x, scale), w), 0)
	b.SetPhi(i2, b.Add(ir.U32, i2, step))
	b.EndLoop()
	return b.Done(), nil
}

// treeSum is a workgroup's sum of every thread's v, through shared part.
func treeSum(b *ir.Builder, part, tid, v ir.Value, g int) ir.Value {
	b.Store(part, tid, v, 0)
	b.Barrier()
	top := b.Const(ir.U32, int64(g-1))
	for s := g / 2; s >= 1; s /= 2 {
		mine := b.Load(ir.F32, part, tid, 0)
		other := b.Load(ir.F32, part, b.Min(ir.U32, b.Add(ir.U32, tid, b.Const(ir.U32, int64(s))), top), 0)
		sum := b.Add(ir.F32, mine, other)
		b.Barrier()
		b.Store(part, tid, sum, 0)
		b.Barrier()
	}
	return b.Load(ir.F32, part, b.Const(ir.U32, 0), 0)
}

// Q4EScaleSiLU is the gate's low rank, out = silu(x/h), or silu(x) where
// unscaled (a gate's violation). Params pX, pOut; n threads.
func Q4EScaleSiLU(n, h int, unscaled bool) (*ir.Kernel, error) {
	if n < 1 || h < 1 {
		return nil, fmt.Errorf("kernels: Q4EScaleSiLU: n=%d h=%d", n, h)
	}
	b := ir.New("q4escalesilu", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := flatID(b, n)
	x := b.Load(ir.F32, pX, i, 0)
	if !unscaled {
		x = b.Mul(ir.F32, x, b.ConstF32(1/float32(h)))
	}
	b.Store(pOut, i, act(b, x, ActSiLU), 0)
	return b.Done(), nil
}

// Q4EGateMix is each row's sublayer input, the mean over its h streams of
// sigmoid(gate) * normed (normed alone where plain, a gate's violation).
// Params pXN and pGate [rows][h][d], pOut [rows][d]; rows*d threads.
func Q4EGateMix(d, h, rows int, plain bool) (*ir.Kernel, error) {
	if d < 1 || h < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: Q4EGateMix: d=%d h=%d rows=%d", d, h, rows)
	}
	b := ir.New("q4egatemix", [3]int{128, 1, 1})
	pXN := b.Param("pXN", ir.F32)
	pGate := b.Param("pGate", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	flat := flatID(b, rows*d)
	r := b.Div(ir.U32, flat, b.Const(ir.U32, int64(d)))
	i := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(d)))
	base := b.Add(ir.U32, b.Mul(ir.U32, r, b.Const(ir.U32, int64(h*d))), i)
	one := b.ConstF32(1)
	acc := b.ConstF32(0)
	for k := 0; k < h; k++ {
		x := b.Load(ir.F32, pXN, base, int64(k*d))
		if !plain {
			gt := b.Load(ir.F32, pGate, base, int64(k*d))
			x = b.Mul(ir.F32, b.Div(ir.F32, one, b.Add(ir.F32, one, b.Exp(b.Sub(ir.F32, b.ConstF32(0), gt)))), x)
		}
		acc = b.Add(ir.F32, acc, x)
	}
	b.Store(pOut, flat, b.Mul(ir.F32, acc, b.ConstF32(1/float32(h))), 0)
	return b.Done(), nil
}

// Q4EInject is each row's placement weights from its h logits: 2*sigmoid(
// logit/h), one where place is Q4EPlacePlain, half where Q4EPlaceHalf.
// Params pInj, pOut [rows][h]; n = rows*h threads.
func Q4EInject(n, h, place int) (*ir.Kernel, error) {
	if n < 1 || h < 1 {
		return nil, fmt.Errorf("kernels: Q4EInject: n=%d h=%d", n, h)
	}
	b := ir.New("q4einject", [3]int{128, 1, 1})
	pInj := b.Param("pInj", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := flatID(b, n)
	one := b.ConstF32(1)
	x := b.Mul(ir.F32, b.Load(ir.F32, pInj, i, 0), b.ConstF32(1/float32(h)))
	w := b.Div(ir.F32, one, b.Add(ir.F32, one, b.Exp(b.Sub(ir.F32, b.ConstF32(0), x))))
	switch place {
	case Q4EPlacePlain:
		w = one
	case Q4EPlaceGated:
		w = b.Mul(ir.F32, b.ConstF32(2), w)
	}
	b.Store(pOut, i, w, 0)
	return b.Done(), nil
}

// Q4EPlace adds each row's sublayer output into every stream at its weight:
// out[(k*rows + r)*d + i] = x[same] + w[r*h + k] * y[r*d + i]. Params pX,
// pY, pW, pOut (the streams, out of place); h*rows*d threads.
func Q4EPlace(d, h, rows int) (*ir.Kernel, error) {
	if d < 1 || h < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: Q4EPlace: d=%d h=%d rows=%d", d, h, rows)
	}
	b := ir.New("q4eplace", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pY := b.Param("pY", ir.F32)
	pW := b.Param("pW", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	flat := flatID(b, h*rows*d)
	rd := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(rows*d)))
	k := b.Div(ir.U32, flat, b.Const(ir.U32, int64(rows*d)))
	r := b.Div(ir.U32, rd, b.Const(ir.U32, int64(d)))
	w := b.Load(ir.F32, pW, b.Add(ir.U32, b.Mul(ir.U32, r, b.Const(ir.U32, int64(h))), k), 0)
	b.Store(pOut, flat, b.Fma(w, b.Load(ir.F32, pY, rd, 0), b.Load(ir.F32, pX, flat, 0)), 0)
	return b.Done(), nil
}

// Q4EStreamDot is the n-gram block's gate per (row, stream): the signed
// square root's sigmoid of the dot of its normed key and normed residual
// over sqrt(d) (ActSqrtSigmoid). Params pK and pQ [rows][h][d], pOut
// [rows][h]; rows*h workgroups of RMSNormGroup.
func Q4EStreamDot(d, h, rows int) (*ir.Kernel, error) {
	if d < 1 || h < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: Q4EStreamDot: d=%d h=%d rows=%d", d, h, rows)
	}
	const g = RMSNormGroup
	b := ir.New("q4estreamdot", [3]int{g, 1, 1})
	pK := b.Param("pK", ir.F32)
	pQ := b.Param("pQ", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	part := b.Shared("dot", ir.F32, g)
	row := b.Min(ir.U32, b.CTAID(), b.Const(ir.U32, int64(rows*h-1)))
	dd := b.Const(ir.U32, int64(d))
	base := b.Mul(ir.U32, row, dd)
	last := b.Const(ir.U32, int64(d-1))
	zero := b.ConstF32(0)
	tid := b.TID()
	b.Loop(int64((d + g - 1) / g))
	acc := b.Phi(ir.F32, zero)
	i := b.Phi(ir.U32, tid)
	at := b.Add(ir.U32, base, b.Min(ir.U32, i, last))
	p := b.Mul(ir.F32, b.Load(ir.F32, pK, at, 0), b.Load(ir.F32, pQ, at, 0))
	b.SetPhi(acc, b.Add(ir.F32, acc, b.Select(ir.F32, b.Lt(ir.U32, i, dd), p, zero)))
	b.SetPhi(i, b.Add(ir.U32, i, b.Const(ir.U32, g)))
	b.EndLoop()
	s := b.Mul(ir.F32, treeSum(b, part, tid, acc, g), b.ConstF32(float32(1/math.Sqrt(float64(d)))))
	b.Store(pOut, row, act(b, s, ActSqrtSigmoid), 0)
	return b.Done(), nil
}

// Q4EGatedValue is each (row, stream)'s gated value, gate[r*h + k] times the
// row's value. Params pGate [rows][h], pVal [rows][d], pOut [rows][h][d];
// rows*h*d threads.
func Q4EGatedValue(d, h, rows int) (*ir.Kernel, error) {
	if d < 1 || h < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: Q4EGatedValue: d=%d h=%d rows=%d", d, h, rows)
	}
	b := ir.New("q4egatedvalue", [3]int{128, 1, 1})
	pGate := b.Param("pGate", ir.F32)
	pVal := b.Param("pVal", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	flat := flatID(b, rows*h*d)
	rk := b.Div(ir.U32, flat, b.Const(ir.U32, int64(d)))
	i := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(d)))
	r := b.Div(ir.U32, rk, b.Const(ir.U32, int64(h)))
	v := b.Load(ir.F32, pVal, b.Add(ir.U32, b.Mul(ir.U32, r, b.Const(ir.U32, int64(d))), i), 0)
	b.Store(pOut, flat, b.Mul(ir.F32, b.Load(ir.F32, pGate, rk, 0), v), 0)
	return b.Done(), nil
}

// Q4ENGramConv is the n-gram block's convolution over a call's rows, one
// thread a channel of the hd = h*d: each row, in its sequence's position
// order, reads its normed gated value cur and, tap t of taps reading
// (taps-1-t)*dil positions back, its sequence's history -- the in slot's, or
// the previous row's when that was the same sequence's, carried in
// registers -- and writes silu(sum_t w[t][c] * value) to out; the history
// moved on is written at the sequence's out slot. The history is the last
// (taps-1)*dil positions' values, oldest first, hd wide each, at off floats
// into a slot of stride floats of the pool (the recurrent tail).
//
// pN is the recurrent descriptor (RecDescWords; with runs, the run
// descriptor, RunDescWords): its row count bounds the rows, so a padded
// row moves no history. Params pState (the in half), pCur [rows][hd], pW
// [taps][hd], pN, pOut [rows][hd], pSOut (the out half); hd threads. With
// noConv the convolution is zeros (a gate's violation) and the history still
// moves on.
func Q4ENGramConv(hd, taps, dil, rows, stride, off int, runs, noConv bool) (*ir.Kernel, error) {
	hist := (taps - 1) * dil
	if hd < 1 || taps < 2 || dil < 1 || rows < 1 || stride < off+hist*hd {
		return nil, fmt.Errorf("kernels: Q4ENGramConv: hd=%d taps=%d dil=%d rows=%d stride=%d off=%d",
			hd, taps, dil, rows, stride, off)
	}
	name := "q4engramconv"
	if runs {
		name = "q4engramconvruns"
	}
	b := ir.New(name, [3]int{128, 1, 1})
	pState := b.Param("pState", ir.F32)
	pCur := b.Param("pCur", ir.F32)
	pW := b.Param("pW", ir.F32)
	pN := b.Param("pN", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	pSOut := b.Param("pSOut", ir.F32)
	c := flatID(b, hd)
	hdC := b.Const(ir.U32, int64(hd))
	strideC := b.Const(ir.U32, int64(stride))
	offC := b.Const(ir.U32, int64(off))
	w := make([]ir.Value, taps)
	for t := range w {
		w[t] = b.Load(ir.F32, pW, c, int64(t*hd))
	}
	count := b.Load(ir.U32, pN, b.Const(ir.U32, 0), 0)
	if runs {
		count = b.Const(ir.U32, int64(rows))
	}
	zero := b.ConstF32(0)
	one := b.ConstF32(1)
	e0, none := b.Const(ir.U32, 0), b.Const(ir.U32, 0xFFFFFFFF)
	b.Loop(int64(rows))
	e := b.Phi(ir.U32, e0)
	prevJ := b.Phi(ir.U32, none)
	h := make([]ir.Value, hist)
	for p := range h {
		h[p] = b.Phi(ir.F32, zero)
	}
	row, j := e, b.Const(ir.U32, 0)
	if runs {
		row = orderRow(b, pN, e, rows)
		j, _ = runOf(b, pN, row, rows)
	}
	in, outS := slotIn(b, pN, j), slotOut(b, pN, j)
	valid := b.Lt(ir.U32, e, count)
	same := b.Lt(ir.U32, b.Xor(ir.U32, j, prevJ), b.Const(ir.U32, 1))
	base := b.Add(ir.U32, b.Mul(ir.U32, in, strideC), offC)
	cur0 := make([]ir.Value, hist)
	for p := 0; p < hist; p++ {
		ld := b.Load(ir.F32, pState, b.Add(ir.U32, base, b.Add(ir.U32, b.Mul(ir.U32, b.Const(ir.U32, int64(p)), hdC), c)), 0)
		cur0[p] = b.Select(ir.F32, same, h[p], ld)
	}
	rowAt := b.Add(ir.U32, b.Mul(ir.U32, b.Min(ir.U32, row, b.Const(ir.U32, int64(rows-1))), hdC), c)
	cur := b.Load(ir.F32, pCur, rowAt, 0)
	acc := zero
	for t := 0; t < taps; t++ {
		back := (taps - 1 - t) * dil
		v := cur
		if back > 0 {
			v = cur0[hist-back]
		}
		acc = b.Fma(w[t], v, acc)
	}
	y := b.Div(ir.F32, acc, b.Add(ir.F32, one, b.Exp(b.Sub(ir.F32, zero, acc))))
	if noConv {
		y = zero
	}
	b.Store(pOut, rowAt, y, 0)
	// The history moves on, the oldest out and cur in; a padded row (past the
	// count) keeps it and writes it back unchanged.
	next := make([]ir.Value, hist)
	for p := 0; p < hist; p++ {
		nv := cur
		if p < hist-1 {
			nv = cur0[p+1]
		}
		next[p] = b.Select(ir.F32, valid, nv, cur0[p])
	}
	ob := b.Add(ir.U32, b.Mul(ir.U32, outS, strideC), offC)
	for p := 0; p < hist; p++ {
		b.Store(pSOut, b.Add(ir.U32, ob, b.Add(ir.U32, b.Mul(ir.U32, b.Const(ir.U32, int64(p)), hdC), c)), next[p], 0)
	}
	for p := range h {
		b.SetPhi(h[p], next[p])
	}
	b.SetPhi(prevJ, j)
	b.SetPhi(e, b.Add(ir.U32, e, b.Const(ir.U32, 1)))
	b.EndLoop()
	return b.Done(), nil
}

// Q4ENGramAdd adds the n-gram block's gated value and its convolution into
// every stream: x[(k*rows + r)*d + i] += gv[(r*h + k)*d + i] + conv[r][k*d +
// i]. Params pX, pGV, pConv, pOut (the streams, out of place); h*rows*d
// threads.
func Q4ENGramAdd(d, h, rows int) (*ir.Kernel, error) {
	if d < 1 || h < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: Q4ENGramAdd: d=%d h=%d rows=%d", d, h, rows)
	}
	b := ir.New("q4engramadd", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pGV := b.Param("pGV", ir.F32)
	pConv := b.Param("pConv", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	flat := flatID(b, h*rows*d)
	rd := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(rows*d)))
	k := b.Div(ir.U32, flat, b.Const(ir.U32, int64(rows*d)))
	r := b.Div(ir.U32, rd, b.Const(ir.U32, int64(d)))
	i := b.Rem(ir.U32, rd, b.Const(ir.U32, int64(d)))
	rk := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, r, b.Const(ir.U32, int64(h))), k),
		b.Const(ir.U32, int64(d))), i)
	s := b.Add(ir.F32, b.Load(ir.F32, pGV, rk, 0), b.Load(ir.F32, pConv, rk, 0))
	b.Store(pOut, flat, b.Add(ir.F32, b.Load(ir.F32, pX, flat, 0), s), 0)
	return b.Done(), nil
}
