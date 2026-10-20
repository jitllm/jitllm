package compose

import "github.com/jitllm/jitllm/jit/gpu/ir"

// groupReduce is a workgroup's sum (or maximum) of one value a thread, every
// thread holding the result, by the schedule's algorithm:
//
//   - a butterfly (warp): each 32-lane subgroup combines by shuffles at
//     halving distances, each subgroup's result to its slot of sh, and the
//     slots combined by a second butterfly -- what a target guaranteeing 32
//     lanes runs, the group a multiple of 32 subgroups at most;
//   - a tree: each thread's value to sh, then halving strides, each thread
//     combining its slot with the one s further on (clamped to the last, so
//     surplus threads re-read rather than leave the array: RULE 13), behind a
//     barrier each side of the store.
func groupReduce(b *ir.Builder, sh, tid, x ir.Value, g int, max, warp bool) ir.Value {
	comb := func(p, q ir.Value) ir.Value {
		if max {
			return b.Max(ir.F32, p, q)
		}
		return b.Add(ir.F32, p, q)
	}
	if warp {
		const lanes = ir.SubgroupLanes
		v := x
		for m := lanes / 2; m >= 1; m /= 2 {
			v = comb(v, b.ShuffleXor(ir.F32, v, int64(m)))
		}
		b.Store(sh, b.Shr(ir.U32, tid, b.Const(ir.U32, 5)), v, 0)
		b.Barrier()
		t := b.Load(ir.F32, sh, b.And(ir.U32, tid, b.Const(ir.U32, lanes-1)), 0)
		for m := lanes / 2; m >= 1; m /= 2 {
			t = comb(t, b.ShuffleXor(ir.F32, t, int64(m)))
		}
		return t
	}
	b.Store(sh, tid, x, 0)
	b.Barrier()
	top := b.Const(ir.U32, int64(g-1))
	for s := g / 2; s >= 1; s /= 2 {
		mine := b.Load(ir.F32, sh, tid, 0)
		other := b.Load(ir.F32, sh, b.Min(ir.U32, b.Add(ir.U32, tid, b.Const(ir.U32, int64(s))), top), 0)
		v := comb(mine, other)
		b.Barrier()
		b.Store(sh, tid, v, 0)
		b.Barrier()
	}
	return b.Load(ir.F32, sh, b.Const(ir.U32, 0), 0)
}
