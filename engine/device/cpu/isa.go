package cpu

import "fmt"

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
// The declaration is true by construction: DeclareISA arms the Buf so vex()
// and VZEROUPPER panic in a buffer declared without AVX2, and every legacy
// method panics when its extension is outside the declared set. The objdump
// gate (ssegate_test.go) independently checks the encoder's bookkeeping.

// isaMarker is the fixed prefix of the declaration; the mask follows it.
var isaMarker = [5]byte{0x0F, 0x1F, 0x80, 'J', 'I'}

// ISAMarkerLen is how many bytes the declaration occupies at the head of a
// kernel.
const ISAMarkerLen = len(isaMarker) + 2

// DeclareISA marks this buffer as a kernel needing exactly the extensions in
// mask, and arms emit-time enforcement of it. It must be the kernel's first
// emission.
//
// SSE-tier emitters call it with ISATierSSE (or a narrower set); AVX2-tier
// emitters never call it, so "undeclared" means "AVX2|FMA|F16C".
func (a *Buf) DeclareISA(mask ISA) {
	if a.declared {
		panic("jit: DeclareISA called twice")
	}
	if a.Len() != 0 {
		panic(fmt.Sprintf("jit: DeclareISA after %d bytes; the declaration must "+
			"be the kernel's first instruction so Map finds it at code[0]", a.Len()))
	}
	if mask&ISASSE2 == 0 {
		panic(fmt.Sprintf("jit: DeclareISA(%s) without sse2; SSE2 is the x86-64 "+
			"floor every legacy vector instruction here is checked against", mask))
	}
	a.declared, a.isa = true, mask
	for _, b := range isaMarker {
		a.u8(b)
	}
	a.u8(byte(mask))
	a.u8(byte(mask >> 8))
}

// KernelISA reads a kernel's declaration: the extensions it needs and whether
// it carried a declaration. An undeclared kernel is an AVX2-tier kernel and
// needs ISATierAVX2 -- plus AVX-VNNI if it contains VPDPBUSD, which runnable()
// checks separately because that half predates the declaration.
func KernelISA(code []byte) (need ISA, declared bool) {
	if len(code) >= ISAMarkerLen && [5]byte(code[:5]) == isaMarker {
		return ISA(code[5]) | ISA(code[6])<<8, true
	}
	return ISATierAVX2, false
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

// useISA is the emit-time half of the declaration: a legacy instruction from
// extension bit may only land in a buffer that declared it. An undeclared
// buffer is an AVX2-tier kernel (or an encoder test), where every legacy form
// executes, and is not checked.
func (a *Buf) useISA(bit ISA, mnemonic string) {
	if a.declared && a.isa&bit == 0 {
		panic(fmt.Sprintf("jit: %s needs %s, and this kernel declared only %s; "+
			"declare it (DeclareISA) or do without it", mnemonic, bit, a.isa))
	}
}

// noVEXInSSE panics when a VEX instruction is emitted into a kernel declared
// without AVX2. It is called from vex() and VZEROUPPER, the only two paths
// that produce a VEX prefix, and it only ever panics: it emits nothing, so the
// AVX2 tier's bytes cannot move because of it.
func (a *Buf) noVEXInSSE(what string) {
	if a.declared && a.isa&ISAAVX2 == 0 {
		panic(fmt.Sprintf("jit: %s in a kernel declared %s: a VEX prefix faults "+
			"on a host without AVX (a Goldmont Atom SIGILLs on it whatever the "+
			"vector length)", what, a.isa))
	}
}
