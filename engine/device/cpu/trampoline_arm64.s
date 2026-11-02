//go:build arm64

#include "textflag.h"
#include "funcdata.h"

// func callKernel(code *byte, args *Args)
//
// Loads args into R0 (AAPCS64's first integer argument register, which is what
// the generated code expects) and calls the kernel.
//
// NOTHING IS SAVED HERE, AND NOTHING NEEDS TO BE.
//
// The amd64 sibling saves RBX/R12/R13/R15 on the kernel's behalf, because those
// are callee-saved under System V and saving them once here keeps kernels free
// of pushes. Go's arm64 ABIInternal has NO callee-saved registers at all --
// "a call may overwrite any register that doesn't have a fixed meaning,
// including argument registers" (cmd/compile/abi-internal.md:221) -- so a
// kernel may freely use R0-R17, R19-R26 and every V register with no prologue
// whatsoever. It is the one place where arm64 is simpler than x86 for a reason
// that has nothing to do with the instruction encoding.
//
// The registers with a fixed meaning, which a kernel must NEVER write:
//
//	R28  the current goroutine, g (REGG, cmd/internal/obj/arm64/a.out.go:203).
//	     R14 IS THE amd64 ANSWER AND IS WRONG HERE. Corrupting g does not fail
//	     near its cause.
//	R18  the platform register, reserved by macOS. Go does not use it either.
//	R27  REGTMP, the assembler's and linker's scratch.
//	R29  the frame pointer, walked by tracebacks and profilers.
//	R30  the link register.
//	RSP  must stay 16-byte aligned.
//
// ★ THE FRAME SIZE IS $16 AND MUST NOT BE $0.
//
// This is not a leaf function -- it performs a nested BL -- so it needs the
// compiler-inserted prologue that spills the caller's LR. With a zero frame the
// prologue does not spill it, so the kernel's RET overwrites nothing but this
// function's own RET returns INTO the kernel: an infinite loop between the
// instruction after BL and RET, with no diagnostic. NOFRAME would suppress the
// same save and produce the same loop. $16 was verified executing generated
// code on an M4.
//
// NOSPLIT because there is no stack growth check to perform and no Go call to
// make from here. NO_LOCAL_POINTERS declares the frame pointer-free so the GC
// does not try to scan it.
//
// Note what is NOT here: no stack switch and no call into the runtime. That is
// what keeps entry cheap -- and it is also why a kernel cannot be
// async-preempted while it runs, which is what forces the per-call execution
// budget in budget.go.
TEXT ·callKernel(SB), NOSPLIT, $16-16
	NO_LOCAL_POINTERS
	MOVD code+0(FP), R1
	MOVD args+8(FP), R0
	BL   (R1)
	RET
