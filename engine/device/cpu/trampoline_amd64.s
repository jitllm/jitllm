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
// R14 is a kernel's to use and is not saved: Go's amd64 ABIInternal pins g to
// R14, but this function is ABI0, where R14 and X15 are scratch, and every
// ABIInternal call site of an ABI0 function reloads g from TLS into R14 and
// zeroes X15 after the CALL (the disassembly shows MOVQ FS:-8, R14 after
// each CALL callKernel.abi0). A signal taken inside a kernel reloads g from
// TLS too (runtime·sigtramp), and a kernel's PC is no Go safe point, so
// nothing reads R14 as g while a kernel runs. X15 is a vector register to
// kernels on the same terms.
// RBX is callee-saved under the contract
// this trampoline owes Go. Saving it here rather than in each generated prologue
// means kernels perform no pushes at all, so RSP is identical on every path
// through a kernel body — which matters the moment a kernel has more than one
// exit, and costs one store either way.
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
