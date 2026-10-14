package ptx

// The dependency counters (ir/depsync.go) against the PTX memory model (PTX
// ISA, "Memory Consistency Model"), the same argument as the grid barrier's
// (grid.go):
//
//   - a signal is bar.sync, then thread 0's red.release.gpu: the barrier
//     orders the block's writes before thread 0 at CTA scope, and a release
//     is cumulative, so it publishes them;
//   - every signal on a word is an RMW, so the signals form one release
//     sequence, and a wait's ld.acquire.gpu that reads a value at or past its
//     target synchronises with every signal it counted; the bar.sync after
//     the spin carries that to the block;
//   - the spin backs off with nanosleep, as Mirage's and the Hazy
//     megakernel's waits do, so a waiting SM leaves the memory system to the
//     producers it waits for.
//
// Targets are compared modulo 2^32 -- (s32)(cur - target) >= 0 -- because
// the counters only grow (the epoch scales every target).
//
// Below sm_70 there is no scoped acquire/release and no nanosleep; membar.gl
// and a volatile load give the same order, and the spin does not sleep.

func depWait(b *Buf, buf, word, target Reg, sm int) {
	tid, cur, d := b.Reg(B32), b.Reg(B32), b.Reg(B32)
	ad := b.Reg(B64)
	notT0, wait := b.Reg(Pred), b.Reg(Pred)
	done, spin := b.Label(), b.Label()
	b.Op("mov.u32 %v,%%tid.x;", tid)
	b.Op("setp.ne.u32 %v,%v,0;", notT0, tid)
	b.Op("@%v bra %s;", notT0, done)
	b.Op("mad.wide.u32 %v,%v,4,%v;", ad, word, buf)
	b.Bind(spin)
	if sm >= 70 {
		b.Op("ld.acquire.gpu.global.u32 %v,[%v];", cur, ad)
	} else {
		b.Op("ld.volatile.global.u32 %v,[%v];", cur, ad)
	}
	b.Op("sub.u32 %v,%v,%v;", d, cur, target)
	b.Op("setp.lt.s32 %v,%v,0;", wait, d)
	if depBroken {
		b.Op("bra.uni %s;", done)
	}
	if sm >= 70 {
		b.Op("@%v nanosleep.u32 32;", wait)
	}
	b.Op("@%v bra %s;", wait, spin)
	if sm < 70 {
		b.Op("membar.gl;")
	}
	b.Bind(done)
	b.Op("bar.sync 0;")
}

func depSignal(b *Buf, buf, word Reg, sm int) {
	tid := b.Reg(B32)
	ad := b.Reg(B64)
	notT0 := b.Reg(Pred)
	done := b.Label()
	b.Op("bar.sync 0;")
	b.Op("mov.u32 %v,%%tid.x;", tid)
	b.Op("setp.ne.u32 %v,%v,0;", notT0, tid)
	b.Op("@%v bra %s;", notT0, done)
	b.Op("mad.wide.u32 %v,%v,4,%v;", ad, word, buf)
	if sm >= 70 {
		b.Op("red.release.gpu.global.add.u32 [%v],1;", ad)
	} else {
		b.Op("membar.gl;")
		b.Op("red.global.add.u32 [%v],1;", ad)
	}
	b.Bind(done)
}

// depEpoch loads the region's epoch. Every thread loads it; the launch
// boundary orders the previous launch's advance before it.
func depEpoch(b *Buf, buf Reg, word int64, sm int) Reg {
	r := b.Reg(B32)
	if sm >= 70 {
		b.Op("ld.relaxed.gpu.global.u32 %v,[%v+%d];", r, buf, 4*word)
	} else {
		b.Op("ld.volatile.global.u32 %v,[%v+%d];", r, buf, 4*word)
	}
	return r
}

// depExit arrives on the exit count at base+1; the last arrival zeroes it
// and advances the epoch at base. Every workgroup read the epoch before it
// arrived, so no workgroup of this launch sees the advance; the next launch
// is ordered after this one by the stream.
func depExit(b *Buf, buf Reg, base int64, sm int) {
	tid, n, old, ep := b.Reg(B32), b.Reg(B32), b.Reg(B32), b.Reg(B32)
	notT0, notLast := b.Reg(Pred), b.Reg(Pred)
	done := b.Label()
	b.Op("bar.sync 0;")
	b.Op("mov.u32 %v,%%tid.x;", tid)
	b.Op("setp.ne.u32 %v,%v,0;", notT0, tid)
	b.Op("@%v bra %s;", notT0, done)
	b.Op("mov.u32 %v,%%nctaid.x;", n)
	b.Op("sub.u32 %v,%v,1;", n, n)
	if sm >= 70 {
		b.Op("atom.acq_rel.gpu.global.add.u32 %v,[%v+%d],1;", old, buf, 4*(base+1))
	} else {
		b.Op("membar.gl;")
		b.Op("atom.global.add.u32 %v,[%v+%d],1;", old, buf, 4*(base+1))
	}
	b.Op("setp.ne.u32 %v,%v,%v;", notLast, old, n)
	b.Op("@%v bra %s;", notLast, done)
	b.Op("st.global.u32 [%v+%d],0;", buf, 4*(base+1))
	b.Op("ld.global.u32 %v,[%v+%d];", ep, buf, 4*base)
	b.Op("add.u32 %v,%v,1;", ep, ep)
	b.Op("st.global.u32 [%v+%d],%v;", buf, 4*base, ep)
	b.Bind(done)
}

// depBroken makes every wait return without waiting: the counters still
// count and the launch still ends, and nothing is ordered. It is the
// violation the dep block's gate must see (BreakDepWaitForTest, jitllmtest
// builds only).
var depBroken bool
