package kernels

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/jit/ir"
)

// FlashAttentionMergeRun folds one pass of partials into a running partial:
// the attention over a history longer than one launch's plane goes in passes
// over key ranges (each pass's descriptors clipped to its range), and each
// pass's s.Splits partials are combined with the run so far into a new run --
// still unnormalised, one split's record in FlashAttentionMerge's layout, so
// the next pass folds into it and FlashAttentionMerge at one split finishes
// it (the sink and the division, once). Out of place: pOut is not pRun.
//
// first drops pRun, for the first pass. A thread per record element (the
// padded value width, at least 32): element d < Dim of the numerator, and
// replica d < 32 of the maximum and the sum. Parameters: pPart, pRun (unless
// first), pOut. Launch ceil(Rows*Heads*padded/128) groups of 128.
func FlashAttentionMergeRun(s FlashShape, first bool) (*ir.Kernel, error) {
	if s.Splits < 1 || s.Heads < 1 || s.Rows < 1 || s.Dim < 1 {
		return nil, fmt.Errorf("kernels: FlashAttentionMergeRun: invalid shape %+v", s)
	}
	padded := (s.Dim + 31) / 32 * 32
	stride := int64(padded + 64)
	b := ir.New("flashmergerun", [3]int{128, 1, 1})
	pPart := b.Param("pPart", ir.F32)
	var pRun ir.Value
	if !first {
		pRun = b.Param("pRun", ir.F32)
	}
	pOut := b.Param("pOut", ir.F32)
	u := func(x int) ir.Value { return b.Const(ir.U32, int64(x)) }
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()), u(s.Rows*s.Heads*padded-1))
	item := udiv(b, flat, padded)
	d := urem(b, flat, padded)
	dv := b.Min(ir.U32, d, u(s.Dim-1)) // a padding element reads the last and writes 0
	rep := b.Min(ir.U32, d, u(31))
	base := b.Mul(ir.U32, item, u(s.Splits*(padded+64)))
	mx := b.ConstF32(-math.MaxFloat32)
	var ms, ls, xs []ir.Value
	for j := 0; j < s.Splits; j++ {
		ms = append(ms, b.Load(ir.F32, pPart, base, int64(j)*stride+int64(padded)))
		ls = append(ls, b.Load(ir.F32, pPart, base, int64(j)*stride+int64(padded+32)))
		xs = append(xs, b.Load(ir.F32, pPart, b.Add(ir.U32, base, dv), int64(j)*stride))
	}
	rb := b.Mul(ir.U32, item, u(padded+64))
	if !first {
		ms = append(ms, b.Load(ir.F32, pRun, rb, int64(padded)))
		ls = append(ls, b.Load(ir.F32, pRun, rb, int64(padded+32)))
		xs = append(xs, b.Load(ir.F32, pRun, b.Add(ir.U32, rb, dv), 0))
	}
	for _, m := range ms {
		mx = b.Max(ir.F32, mx, m)
	}
	sum, acc := b.ConstF32(0), b.ConstF32(0)
	for j := range ms {
		w := b.Exp(b.Sub(ir.F32, ms[j], mx))
		sum = b.Fma(ls[j], w, sum)
		acc = b.Fma(xs[j], w, acc)
	}
	acc = b.Select(ir.F32, b.Lt(ir.U32, d, u(s.Dim)), acc, b.ConstF32(0))
	b.Store(pOut, b.Add(ir.U32, rb, d), acc, 0)
	b.Store(pOut, b.Add(ir.U32, rb, rep), mx, int64(padded))
	b.Store(pOut, b.Add(ir.U32, rb, rep), sum, int64(padded+32))
	return b.Done()
}
