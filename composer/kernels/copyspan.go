package kernels

import (
	"fmt"

	"github.com/jitllm/jitllm/composer/ir"
)

// CopySpan copies n floats from src at a runtime offset: dst[i] = src[off[0]+i].
// It moves one row of a chunk's residual into a one-row scratch -- the last
// row, which is the only one a prefill's output projection reads -- without a
// kernel per row index. Launch n threads in groups of 128; the grid is clamped.
func CopySpan(n int) (*ir.Kernel, error) {
	if n <= 0 {
		return nil, fmt.Errorf("kernels: CopySpan: n=%d", n)
	}
	b := ir.New("copyspan", [3]int{128, 1, 1})
	pSrc := b.Param("pSrc", ir.F32)
	pDst := b.Param("pDst", ir.F32)
	pOff := b.Param("pOff", ir.U32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()), b.Const(ir.U32, int64(n-1)))
	off := b.Load(ir.U32, pOff, b.Const(ir.U32, 0), 0)
	b.Store(pDst, i, b.Load(ir.F32, pSrc, b.Add(ir.U32, off, i), 0), 0)
	return b.Done()
}
