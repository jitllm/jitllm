package cpu

import "github.com/jitllm/jitllm/composer/x86"

// A kernel's ISA, declared in its own bytes.
//
// The declaration is code that executes as nothing. An SSE-tier kernel's first
// seven bytes are
//
//	0F 1F 80 4A 49 lo hi      nop DWORD PTR [rax+disp32]
//
// with disp32 = 'J', 'I', then the 16-bit ISA mask. 0F 1F /0 is the multi-byte
// NOP every x86-64 part executes, so the entry point stays code[0] and no Map
// or emitter signature changes.
//
// A byte scan for VEX prefixes cannot do this job: C5 is a legal ModRM byte in
// legacy code (PXOR xmm0,xmm5 is 66 0F EF C5) and C4 a legal opcode (PINSRW),
// so a scan would decline correct SSE kernels, and an SSE host has no other
// kernel to fall back to.
//
// The composer's assembler writes it (composer/x86's DeclareISA), refusing a
// VEX instruction in a buffer declared without AVX2 and a legacy one whose
// extension is outside the declared set; the objdump gate (internal/ssegate)
// checks that bookkeeping independently. The runtime reads it here when it
// maps code.

// KernelISA reads a kernel's declaration, as the assembler that wrote it
// reads it (x86.KernelISA): the extensions it needs and whether it carried a
// declaration. An undeclared kernel is an AVX2-tier kernel and needs
// ISATierAVX2 -- plus AVX-VNNI if it contains VPDPBUSD, which runnable()
// checks separately because that half predates the declaration.
func KernelISA(code []byte) (need ISA, declared bool) {
	n, declared := x86.KernelISA(code)
	return ISA(n), declared
}

// KernelTier is the tier a kernel was generated for: TierSSE for a declared
// kernel that needs no AVX2, TierAVX2 for an undeclared one on amd64.
func KernelTier(code []byte) Tier {
	need, declared := KernelISA(code)
	if declared && need&ISAAVX2 == 0 {
		return TierSSE
	}
	return TierAVX2
}
