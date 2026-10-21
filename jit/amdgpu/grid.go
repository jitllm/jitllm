package amdgpu

import "fmt"

// gridBarrier lowers ir.OpGridBarrier: the arrive-and-wait counter of
// ptx/grid.go under the AMDGPU memory model (LLVM AMDGPUUsage, "Memory Model"):
// every thread releases at agent scope and meets at s_barrier; thread 0
// arrives with an acq_rel atomicrmw at agent scope, the last arriver resets the
// count and advances the generation with a release, the others spin on an
// acquire load of it; everyone meets at s_barrier again and acquires at agent
// scope, which invalidates the vector L1 a later plain load would otherwise
// hit. The workgroup count is the dispatch packet's grid_size_x (byte 12 of
// hsa_kernel_dispatch_packet_t) over the declared width, which the launch
// matches (backend.checkWidth).
func (l *lowerer) gridBarrier(id int, sync string) {
	l.declare("declare void @llvm.amdgcn.s.barrier()")
	l.declare("declare i32 @llvm.amdgcn.workitem.id.x()")
	l.declare("declare ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()")
	t0, last, wait, spin, join := fmt.Sprintf("G%dt", id), fmt.Sprintf("G%dl", id), fmt.Sprintf("G%dw", id),
		fmt.Sprintf("G%ds", id), fmt.Sprintf("G%dj", id)
	l.emit("fence syncscope(\"agent\") release")
	l.emit("call void @llvm.amdgcn.s.barrier()")
	tid, isT0 := l.fresh(), l.fresh()
	l.emit("%s = call i32 @llvm.amdgcn.workitem.id.x()", tid)
	l.emit("%s = icmp eq i32 %s, 0", isT0, tid)
	l.emit("br i1 %s, label %%%s, label %%%s", isT0, t0, join)

	l.label(t0)
	gp, gen, old := l.fresh(), l.fresh(), l.fresh()
	l.emit("%s = getelementptr i32, ptr addrspace(1) %s, i64 1", gp, sync)
	l.emit("%s = load atomic i32, ptr addrspace(1) %s syncscope(\"agent\") monotonic, align 4", gen, gp)
	l.emit("%s = atomicrmw add ptr addrspace(1) %s, i32 1 syncscope(\"agent\") acq_rel, align 4", old, sync)
	dp, gsp, gs, n, nm1, isLast := l.fresh(), l.fresh(), l.fresh(), l.fresh(), l.fresh(), l.fresh()
	l.emit("%s = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()", dp)
	l.emit("%s = getelementptr i8, ptr addrspace(4) %s, i64 12", gsp, dp)
	l.emit("%s = load i32, ptr addrspace(4) %s, align 4", gs, gsp)
	l.emit("%s = udiv i32 %s, %d", n, gs, l.k.Group[0])
	l.emit("%s = sub i32 %s, 1", nm1, n)
	l.emit("%s = icmp eq i32 %s, %s", isLast, old, nm1)
	l.emit("br i1 %s, label %%%s, label %%%s", isLast, last, wait)

	l.label(last)
	bump := l.fresh()
	l.emit("store atomic i32 0, ptr addrspace(1) %s syncscope(\"agent\") monotonic, align 4", sync)
	l.emit("%s = atomicrmw add ptr addrspace(1) %s, i32 1 syncscope(\"agent\") release, align 4", bump, gp)
	l.emit("br label %%%s", join)

	l.label(wait)
	l.emit("br label %%%s", spin)
	l.label(spin)
	cur, same := l.fresh(), l.fresh()
	l.emit("%s = load atomic i32, ptr addrspace(1) %s syncscope(\"agent\") acquire, align 4", cur, gp)
	l.emit("%s = icmp eq i32 %s, %s", same, cur, gen)
	l.emit("br i1 %s, label %%%s, label %%%s", same, spin, join)

	l.label(join)
	l.emit("call void @llvm.amdgcn.s.barrier()")
	l.emit("fence syncscope(\"agent\") acquire")
}
