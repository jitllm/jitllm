package mainkernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

// PagedGatherPages copies pages of a paged pool into one contiguous buffer:
// dst[j*words + i] = src[pTab[j]*words + i] for the n pages the table names.
// It is how a sequence's history leaves the device (a block moving home, a
// seam shrinking): Buf reads a whole buffer, and a pool is every sequence's.
// Parameters: pSrc (the pool), pDst, pTab.
func PagedGatherPages(words, n int) (*ir.Kernel, error) {
	if words < 1 || n < 1 {
		return nil, fmt.Errorf("kernels: PagedGatherPages: words=%d n=%d", words, n)
	}
	b := ir.New("pagedgather", [3]int{128, 1, 1})
	src := b.Param("pSrc", ir.F32)
	dst := b.Param("pDst", ir.F32)
	tab := b.Param("pTab", ir.U32)
	u := func(x int) ir.Value { return b.Const(ir.U32, int64(x)) }
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()), u(words*n-1))
	j := b.Div(ir.U32, flat, u(words))
	i := b.Rem(ir.U32, flat, u(words))
	pid := b.Load(ir.U32, tab, j, 0)
	b.Store(dst, flat, b.Load(ir.F32, src, b.Add(ir.U32, b.Mul(ir.U32, pid, u(words)), i), 0), 0)
	return b.Done(), nil
}
