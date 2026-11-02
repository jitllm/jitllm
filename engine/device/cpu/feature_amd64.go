package cpu

// The CPUID probe the capability set is built from (features_amd64.go).
//
// The probe is itself JIT-compiled, which reads CPUID without a dependency
// (golang.org/x/sys/cpu would end go.mod's single require; internal/cpu cannot
// be imported). It is the one probe: SupportedNative, HostDotKind and
// SupportedPackedNative all ask CPU(), so forcing a feature off reaches all of
// them.

// cpuidProbe emits: save RBX, CPUID(leaf, subleaf), store the four result
// registers through Args.Out, restore RBX.
//
// RBX is callee-saved under System V and CPUID clobbers it, so it is parked in
// R11 across the instruction; EBX has to be read out before the restore, which
// is why this cannot simply push and pop around the whole thing.
func cpuidProbe(leaf, sub int32) []byte {
	var a Buf
	a.MOVQ(R11, RBX)           // save RBX
	a.MOVLoad(R10, At(RDI, 0)) // Args.Out -> where to write
	a.MOVimm32(RAX, leaf)
	a.MOVimm32(RCX, sub)
	a.CPUID()
	a.MOVStore(At(R10, 0), RAX)
	a.MOVStore(At(R10, 8), RBX)
	a.MOVStore(At(R10, 16), RCX)
	a.MOVStore(At(R10, 24), RDX)
	a.MOVQ(RBX, R11) // restore
	a.RET()
	return a.Bytes()
}
