package mainkernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

// PLESlice gathers one block's slice of Gemma 4's per-layer embedding inputs:
// out[r*p+j] = in[r*width + off + j] for rows rows, where in holds each row's
// width (NLayer*p) inputs and off, the block's li*p, is the one u32 in pOff.
// The offset is a buffer rather than a constant so every block shares one
// kernel. Out of place for RULE 13.
func PLESlice(rows, width, p int) (*ir.Kernel, error) {
	if rows < 1 || p < 1 || width < p {
		return nil, fmt.Errorf("kernels: PLESlice: rows=%d width=%d p=%d", rows, width, p)
	}
	b := ir.New("pleslice", [3]int{128, 1, 1})
	pIn := b.Param("pIn", ir.F32)
	pOff := b.Param("pOff", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(rows*p-1)))
	np := b.Const(ir.U32, int64(p))
	r, j := b.Div(ir.U32, i, np), b.Rem(ir.U32, i, np)
	off := b.Load(ir.U32, pOff, b.Const(ir.U32, 0), 0)
	src := b.Add(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, r, b.Const(ir.U32, int64(width))), off), j)
	b.Store(pOut, i, b.Load(ir.F32, pIn, src, 0), 0)
	return b.Done(), nil
}
