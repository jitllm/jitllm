package compose

import "github.com/jitllm/jitllm/jit/gpu/ir"

// groupReduce is a workgroup's sum (or maximum) of one value a thread, by
// its definition: every thread stores its value to sh, and after a barrier
// every thread folds slots 0 to g-1 in order, so each holds the same result.
// A barrier after the fold frees sh for what follows. Linear in g; a pass
// that schedules it as a tree is an optimization over this form.
func groupReduce(b *ir.Builder, sh, tid, x ir.Value, g int, max bool) ir.Value {
	b.Store(sh, tid, x, 0)
	b.Barrier()
	first := b.Load(ir.F32, sh, b.Const(ir.U32, 0), 0)
	one := b.Const(ir.U32, 1)
	b.Loop(int64(g - 1))
	acc := b.Phi(ir.F32, first)
	i := b.Phi(ir.U32, one)
	y := b.Load(ir.F32, sh, i, 0)
	if max {
		b.SetPhi(acc, b.Max(ir.F32, acc, y))
	} else {
		b.SetPhi(acc, b.Add(ir.F32, acc, y))
	}
	b.SetPhi(i, b.Add(ir.U32, i, one))
	b.EndLoop()
	b.Barrier()
	return acc
}
