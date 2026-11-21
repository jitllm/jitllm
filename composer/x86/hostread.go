package x86

// CPUIDRead is code that reads CPUID at leaf and sub. Called with the
// argument block's address in RDI, it stores EAX, EBX, ECX and EDX, a 64-bit
// word each, at the address the block's first word holds. RBX is
// callee-saved and CPUID writes it, so it waits in R11 across the
// instruction: EBX is stored before RBX comes back, which is why this is not
// a push and pop around the whole.
func CPUIDRead(leaf, sub int32) ([]byte, error) {
	var a Buf
	a.MOVQ(R11, RBX)
	a.MOVLoad(R10, At(RDI, 0))
	a.MOVimm32(RAX, leaf)
	a.MOVimm32(RCX, sub)
	a.CPUID()
	a.MOVStore(At(R10, 0), RAX)
	a.MOVStore(At(R10, 8), RBX)
	a.MOVStore(At(R10, 16), RCX)
	a.MOVStore(At(R10, 24), RDX)
	a.MOVQ(RBX, R11)
	a.RET()
	return a.Bytes()
}

// XCR0Read is code that reads XCR0 (XGETBV with ECX 0) as CPUIDRead reads
// CPUID: EAX and EDX, a 64-bit word each, at the address in the block's
// first word.
func XCR0Read() ([]byte, error) {
	var a Buf
	a.MOVLoad(R10, At(RDI, 0))
	a.MOVimm32(RCX, 0)
	a.XGETBV()
	a.MOVStore(At(R10, 0), RAX)
	a.MOVStore(At(R10, 8), RDX)
	a.RET()
	return a.Bytes()
}
