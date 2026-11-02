package spirv

// The grid barrier (ir.OpGridBarrier) in SPIR-V: the same arrive-and-wait
// counter as ptx/grid.go, in structured control flow.
//
//	ControlBarrier(Workgroup, Device, AcqRel|Uniform|Workgroup)
//	if tid == 0 {
//	    gen := AtomicLoad(sync[1], Device, Relaxed)
//	    old := AtomicIAdd(sync[0], Device, AcqRel|Uniform, 1)
//	    if old == NumWorkgroups.x-1 {
//	        AtomicStore(sync[0], Device, Relaxed, 0)
//	        AtomicIAdd(sync[1], Device, Release|Uniform, 1)
//	    } else {
//	        loop { if AtomicLoad(sync[1], Device, Acquire|Uniform) != gen break }
//	    }
//	}
//	ControlBarrier(Workgroup, Device, AcqRel|Uniform|Workgroup)
//
// A kernel with one also decorates every storage buffer Coherent (see build):
// without the Vulkan memory model, that is what makes a plain load after the
// barrier see another workgroup's plain store before it, rather than a line a
// non-coherent cache kept from an earlier phase. The Vulkan specification
// gives no forward-progress guarantee between workgroups; the launch is bounded
// by the device's co-residency limit (vulkan.GridLimit).
const (
	opAtomicLoad      = 227
	opAtomicStore     = 228
	opAtomicIAdd      = 234
	opSelectionMerge  = 247
	builtInNumGroups  = 24
	decoCoherent      = 23
	scopeDevice       = 1
	scopeWorkgroup    = 2
	semRelaxed        = 0
	semAcquire        = 0x2
	semRelease        = 0x4
	semAcqRel         = 0x8
	semUniform        = 0x40
	semWorkgroupMem   = 0x100
	selectControlNone = 0
)

func (l *lowerer) gridBarrier(buf uint32) {
	m := &l.m
	f := &m.funcs
	bar := func() {
		inst(f, opControlBarrier, l.u32c(scopeWorkgroup), l.u32c(scopeDevice),
			l.u32c(semAcqRel|semUniform|semWorkgroupMem))
	}
	word := func(i uint32) uint32 {
		p := m.id()
		inst(f, opAccessChain, l.ptrElem(l.tu32), p, buf, l.u32c(0), l.u32c(i))
		return p
	}
	dev := l.u32c(scopeDevice)
	bar()
	isT0 := m.id()
	inst(f, opIEqual, l.tbool, isT0, l.tid, l.u32c(0))
	then, merge := m.id(), m.id()
	inst(f, opSelectionMerge, merge, selectControlNone)
	inst(f, opBranchConditional, isT0, then, merge)

	inst(f, opLabel, then)
	cnt, gw := word(0), word(1)
	gen := m.id()
	inst(f, opAtomicLoad, l.tu32, gen, gw, dev, l.u32c(semRelaxed))
	old := m.id()
	inst(f, opAtomicIAdd, l.tu32, old, cnt, dev, l.u32c(semAcqRel|semUniform), l.u32c(1))
	last := m.id()
	inst(f, opISub, l.tu32, last, l.nGroups, l.u32c(1))
	isLast := m.id()
	inst(f, opIEqual, l.tbool, isLast, old, last)
	lastB, waitB, inner := m.id(), m.id(), m.id()
	inst(f, opSelectionMerge, inner, selectControlNone)
	inst(f, opBranchConditional, isLast, lastB, waitB)

	inst(f, opLabel, lastB)
	inst(f, opAtomicStore, cnt, dev, l.u32c(semRelaxed), l.u32c(0))
	bump := m.id()
	inst(f, opAtomicIAdd, l.tu32, bump, gw, dev, l.u32c(semRelease|semUniform), l.u32c(1))
	inst(f, opBranch, inner)

	inst(f, opLabel, waitB)
	if gridBroken {
		inst(f, opBranch, inner)
		inst(f, opLabel, inner)
		inst(f, opBranch, merge)
		inst(f, opLabel, merge)
		l.cur = merge
		bar()
		return
	}
	hdr, body, cont, exit := m.id(), m.id(), m.id(), m.id()
	inst(f, opBranch, hdr)
	inst(f, opLabel, hdr)
	inst(f, opLoopMerge, exit, cont, loopControlNone)
	inst(f, opBranch, body)
	inst(f, opLabel, body)
	cur := m.id()
	inst(f, opAtomicLoad, l.tu32, cur, gw, dev, l.u32c(semAcquire|semUniform))
	same := m.id()
	inst(f, opIEqual, l.tbool, same, cur, gen)
	inst(f, opBranchConditional, same, cont, exit)
	inst(f, opLabel, cont)
	inst(f, opBranch, hdr)
	inst(f, opLabel, exit)
	inst(f, opBranch, inner)

	inst(f, opLabel, inner)
	inst(f, opBranch, merge)
	inst(f, opLabel, merge)
	l.cur = merge
	bar()
}

// gridBroken makes every waiter leave without waiting (BreakGridBarrierForTest,
// jitllmtest builds only): the violation the grid barrier's gate must see.
var gridBroken bool
