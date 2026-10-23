package kernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/ir"
)

// ArgmaxGroup is Argmax's one workgroup.
const ArgmaxGroup = 1024

// Argmax writes the index of the largest of n floats to pOut[0], the lowest
// index winning a tie -- the host's rule (jit/cpu/argmax.go), so a greedy
// decode takes the same token on either side of the bus.
//
// It keeps greedy decode from bringing the logits home: the next token is one
// integer, so this reads four bytes instead of the whole vocabulary. One
// workgroup: each thread keeps the best of a strided run, seeded with its
// first element so no -Inf constant is needed, and a shared tree reduces the
// (value, index) pairs.
func Argmax(n int) (*ir.Kernel, error) { return argmaxRows(n, 1) }

// ArgmaxRows is Argmax over rows independent vectors of n, one workgroup per
// row: row r's index lands at pOut[r]. A batch of sequences takes one token
// each from it.
func ArgmaxRows(n, rows int) (*ir.Kernel, error) { return argmaxRows(n, rows) }

func argmaxRows(n, rows int) (*ir.Kernel, error) {
	if n < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: Argmax: n=%d rows=%d", n, rows)
	}
	const g = ArgmaxGroup
	name := "argmax"
	if rows > 1 {
		name = "argmaxrows"
	}
	b := ir.New(name, [3]int{g, 1, 1})
	pIn := b.Param("pIn", ir.F32)
	pOut := b.Param("pOut", ir.U32)
	sv := b.Shared("sv", ir.F32, g)
	si := b.Shared("si", ir.U32, g)

	tid := b.TID()
	last := b.Const(ir.U32, int64(n-1))
	nn := b.Const(ir.U32, int64(n))
	step := b.Const(ir.U32, g)
	top := b.Const(ir.U32, g-1)
	first := b.Min(ir.U32, tid, last)
	var row, rowBase ir.Value
	if rows > 1 {
		row = b.Min(ir.U32, b.CTAID(), b.Const(ir.U32, int64(rows-1)))
		rowBase = b.Mul(ir.U32, row, nn)
	}
	load := func(idx ir.Value) ir.Value {
		if rows > 1 {
			idx = b.Add(ir.U32, rowBase, idx)
		}
		return b.Load(ir.F32, pIn, idx, 0)
	}
	v0 := load(first)
	iters := int64((n + g - 1) / g)

	b.Loop(iters)
	bv := b.Phi(ir.F32, v0)
	bi := b.Phi(ir.U32, first)
	i := b.Phi(ir.U32, tid)
	idx := b.Min(ir.U32, i, last)
	v := load(idx)
	// Strictly greater and in range: an index only rises inside a thread, so
	// a tie keeps the earlier one.
	take := b.Lt(ir.F32, bv, v)
	nv := b.Select(ir.F32, take, v, bv)
	ni := b.Select(ir.U32, take, idx, bi)
	in := b.Lt(ir.U32, i, nn)
	b.SetPhi(bv, b.Select(ir.F32, in, nv, bv))
	b.SetPhi(bi, b.Select(ir.U32, in, ni, bi))
	b.SetPhi(i, b.Add(ir.U32, i, step))
	b.EndLoop()

	b.Store(sv, tid, bv, 0)
	b.Store(si, tid, bi, 0)
	b.Barrier()
	for s := g / 2; s >= 1; s /= 2 {
		mv := b.Load(ir.F32, sv, tid, 0)
		mi := b.Load(ir.U32, si, tid, 0)
		o := b.Min(ir.U32, b.Add(ir.U32, tid, b.Const(ir.U32, int64(s))), top)
		ov := b.Load(ir.F32, sv, o, 0)
		oi := b.Load(ir.U32, si, o, 0)
		// The other side wins if it is larger, or equal with a lower index.
		gt, lt := b.Lt(ir.F32, mv, ov), b.Lt(ir.F32, ov, mv)
		ri := b.Select(ir.U32, gt, oi, b.Select(ir.U32, lt, mi, b.Select(ir.U32, b.Lt(ir.U32, oi, mi), oi, mi)))
		rv := b.Select(ir.F32, gt, ov, mv)
		b.Barrier()
		b.Store(sv, tid, rv, 0)
		b.Store(si, tid, ri, 0)
		b.Barrier()
	}
	at := b.Const(ir.U32, 0)
	if rows > 1 {
		at = row
	}
	b.Store(pOut, at, b.Load(ir.U32, si, b.Const(ir.U32, 0), 0), 0)
	return b.Done()
}
