package oldkernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// RoPEQKVPaged is a decode block's three position writers as one launch: q
// rotated into its row-major buffer (RoPERows), k rotated into the paged,
// transposed K pool (PagedRoPERowsT) and v copied into the paged V pool
// (PagedCopyRows, f16 packing it when f16). Each part is the same emitted
// body as its own kernel, so the bits are the same; what goes is two launches
// a block -- on a decode token, two dependent kernels of a few microseconds
// each whose cost is the launch and the drain between them, not the work.
//
// A thread does one item of each part, its index clamped into each part's
// range separately: the IR has no conditional, so the surplus threads of the
// shorter parts recompute and store their part's last item, the clamping
// every kernel here already relies on (RULE 13: no part reads what another
// writes). Launch RoPEQKVPagedThreads threads, 128 wide.
//
// Parameters: pQ, pCS, pQOff, pQDst, pK, pKPool, pV, pVPool, pTab, pRow.
func RoPEQKVPaged(nHead, nKVHead, headDim, nRot int, neox bool, rows, page int, f16 bool) (*ir.Kernel, error) {
	if err := checkPage("RoPEQKVPaged", page); err != nil {
		return nil, err
	}
	if nRot%2 != 0 || nRot <= 0 || nRot > headDim || nHead < 1 || nKVHead < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: RoPEQKVPaged: nRot=%d headDim=%d heads=%d/%d rows=%d",
			nRot, headDim, nHead, nKVHead, rows)
	}
	vw := nKVHead * headDim
	if f16 && vw%2 != 0 {
		return nil, fmt.Errorf("kernels: RoPEQKVPaged: an f16 V row of %d elements", vw)
	}
	nq, nk, nv := ropeQKVParts(nHead, nKVHead, headDim, nRot, rows, f16)
	b := ir.New("ropeqkvpaged", [3]int{128, 1, 1})
	pQ := b.Param("pQ", ir.F32)
	pCS := b.Param("pCS", ir.F32)
	pQOff := b.Param("pQOff", ir.U32)
	pQDst := b.Param("pQDst", ir.F32)
	pK := b.Param("pK", ir.F32)
	pKPool := b.Param("pKPool", ir.F32)
	pV := b.Param("pV", ir.F32)
	pVPool := b.Param("pVPool", ir.F32)
	pTab := b.Param("pTab", ir.U32)
	pRow := b.Param("pRow", ir.U32)
	gid := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	clamp := func(n int) ir.Value { return b.Min(ir.U32, gid, b.Const(ir.U32, int64(n-1))) }
	ropeBody(b, nHead, headDim, nRot, neox, false, rows, 0, pQ, pCS, pQOff, pQDst, clamp(nq))
	pagedRopeBody(b, nKVHead, headDim, nKVHead*headDim, nRot, neox, page, pK, pCS, pKPool, pTab, pRow, clamp(nk))
	pagedCopyBody(b, vw, page, vw, 0, f16, pV, pVPool, pTab, pRow, clamp(nv))
	k := b.Done()
	return k, k.Validate()
}

// RoPEQKVPagedThreads is how many threads RoPEQKVPaged launches: its longest
// part's items.
func RoPEQKVPagedThreads(nHead, nKVHead, headDim, nRot, rows int, f16 bool) int {
	nq, nk, nv := ropeQKVParts(nHead, nKVHead, headDim, nRot, rows, f16)
	return max(nq, nk, nv)
}

func ropeQKVParts(nHead, nKVHead, headDim, nRot, rows int, f16 bool) (nq, nk, nv int) {
	pairs := nRot / 2
	nv = nKVHead * headDim * rows
	if f16 {
		nv /= 2
	}
	return nHead * pairs * rows, nKVHead * pairs * rows, nv
}
