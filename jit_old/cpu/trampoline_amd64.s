//go:build amd64

#include "textflag.h"
#include "funcdata.h"

// func callKernel(code *byte, args *Args)
//
// Loads args into RDI (the System V first argument register, which is what the
// generated code expects) and CALLs the kernel.
//
// RBX, R12, R13 AND R15 ARE SAVED HERE, NOT IN THE KERNEL.
//
// R12/R13/R15 were added for Q6_K, which needs more live pointers than the
// caller-saved set has: the weights, the activations, two scale streams and the
// output, plus two counters.
//
// R14 is still NOT offered on any terms. Go's arm64 and amd64 register ABIs both
// pin a register to the current goroutine -- R14 here -- and handing the runtime
// a corrupted g is not a bug that reproduces near its cause.
// RBX is callee-saved under the contract
// this trampoline owes Go. Saving it here rather than in each generated prologue
// means kernels perform no pushes at all, so RSP is identical on every path
// through a kernel body — which matters the moment a kernel has more than one
// exit, and costs one store either way.
//
// R14 and R15 are deliberately NOT offered the same way. Go's amd64 register ABI
// pins R14 to the current goroutine; a kernel that clobbered it would corrupt g.
//
// NOSPLIT because there is no stack growth check to perform and no Go call to
// make from here. NO_LOCAL_POINTERS declares the saved-RBX slot pointer-free so
// the GC does not try to scan it.
//
// Note what is NOT here: no stack switch and no call into the runtime. That is
// what keeps entry cheap — and it is also why a kernel cannot be async-preempted
// while it runs, which is what forces the per-call execution budget in budget.go.
TEXT ·callKernel(SB), NOSPLIT, $32-16
	NO_LOCAL_POINTERS
	MOVQ BX, 0(SP)
	MOVQ R12, 8(SP)
	MOVQ R13, 16(SP)
	MOVQ R15, 24(SP)
	MOVQ code+0(FP), AX
	MOVQ args+8(FP), DI
	CALL AX
	MOVQ 0(SP), BX
	MOVQ 8(SP), R12
	MOVQ 16(SP), R13
	MOVQ 24(SP), R15
	RET
