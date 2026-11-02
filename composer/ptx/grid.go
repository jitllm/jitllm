package ptx

// gridBarrier lowers ir.OpGridBarrier: the block's threads meet at bar.sync,
// thread 0 arrives on the sync word and waits for the generation to move, and
// the block meets again. It is the shape cooperative_groups' grid.sync()
// compiles to, written out against the PTX memory model (PTX ISA, "Memory
// Consistency Model"):
//
//   - bar.sync makes every write the block made before it visible to thread 0
//     at CTA scope, and fences are cumulative, so thread 0's device-scope
//     release on arrival publishes the whole block's writes;
//   - every arrival is an acq_rel RMW on one location, so the arrivals form one
//     release sequence and the last arriver's acquire sees them all;
//   - a waiter's acquire load that reads the moved generation reads from that
//     sequence and so synchronises with every arrival, and the second
//     bar.sync carries it to the block.
//
// The sync buffer's two words are one 64-bit word: the arrival count in the
// low half and the generation in the high. An arrival adds 1 and learns both
// halves from the one RMW; the last arriver adds 2^32 - n, which in one
// atomic step empties the count and moves the generation. No other arrival
// can fall between the two, since every other block is waiting. Every barrier
// leaves the count at zero, so launches of different grids can share the
// buffer in turn. One round trip to L2 fewer than a separate generation read:
// a V100 at 80 workgroups measured 1.46 us a barrier with the read and 0.83
// without.
//
// Below sm_70 there are no scoped acquire/release forms; membar.gl and a
// volatile load give the same order there, which is how the pre-Volta grid
// sync was built.
func gridBarrier(b *Buf, sync Reg, sm int) {
	tid, n32 := b.Reg(B32), b.Reg(B32)
	n, old, cnt, gen, cur, curGen, step := b.Reg(B64), b.Reg(B64), b.Reg(B64), b.Reg(B64), b.Reg(B64), b.Reg(B64), b.Reg(B64)
	notT0, notLast, same := b.Reg(Pred), b.Reg(Pred), b.Reg(Pred)
	done, spin := b.Label(), b.Label()
	b.Op("bar.sync 0;")
	b.Op("mov.u32 %v,%%tid.x;", tid)
	b.Op("setp.ne.u32 %v,%v,0;", notT0, tid)
	b.Op("@%v bra %s;", notT0, done)
	b.Op("mov.u32 %v,%%nctaid.x;", n32)
	b.Op("cvt.u64.u32 %v,%v;", n, n32)
	if sm >= 70 {
		b.Op("atom.acq_rel.gpu.global.add.u64 %v,[%v],1;", old, sync)
	} else {
		b.Op("membar.gl;")
		b.Op("atom.global.add.u64 %v,[%v],1;", old, sync)
	}
	b.Op("and.b64 %v,%v,4294967295;", cnt, old)
	b.Op("shr.u64 %v,%v,32;", gen, old)
	b.Op("add.u64 %v,%v,1;", cnt, cnt)
	b.Op("setp.ne.u64 %v,%v,%v;", notLast, cnt, n)
	if gridBroken {
		b.Op("@%v bra %s;", notLast, done)
	} else {
		b.Op("@%v bra %s;", notLast, spin)
	}
	// The last arriver: count back to zero and the generation on, one add.
	b.Op("mov.u64 %v,4294967296;", step)
	b.Op("sub.u64 %v,%v,%v;", step, step, n)
	if sm >= 70 {
		b.Op("red.release.gpu.global.add.u64 [%v],%v;", sync, step)
	} else {
		b.Op("membar.gl;")
		b.Op("red.global.add.u64 [%v],%v;", sync, step)
	}
	b.Op("bra.uni %s;", done)
	b.Bind(spin)
	if sm >= 70 {
		b.Op("ld.acquire.gpu.global.u64 %v,[%v];", cur, sync)
	} else {
		b.Op("ld.volatile.global.u64 %v,[%v];", cur, sync)
	}
	b.Op("shr.u64 %v,%v,32;", curGen, cur)
	b.Op("setp.eq.u64 %v,%v,%v;", same, curGen, gen)
	b.Op("@%v bra %s;", same, spin)
	if sm < 70 {
		b.Op("membar.gl;")
	}
	b.Bind(done)
	b.Op("bar.sync 0;")
}

// gridBroken makes every waiter leave without waiting: the barrier still
// counts, so a launch ends, and orders nothing. It is the violation the grid
// barrier's gate must see (BreakGridBarrierForTest, jitllmtest builds only).
var gridBroken bool
