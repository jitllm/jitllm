package cpu

import "github.com/jitllm/jitllm/composer/x86"

// The CPUID probe the capability set is built from (features_amd64.go).
//
// The probe is itself JIT-compiled, which reads CPUID without a dependency
// (golang.org/x/sys/cpu would end go.mod's single require; internal/cpu cannot
// be imported). It is the one probe: Probe and every Target are built from
// it.

// cpuidProbe emits: save RBX, CPUID(leaf, subleaf), store the four result
// registers through Args.Out, restore RBX.
//
// RBX is callee-saved under System V and CPUID clobbers it, so it is parked in
// R11 across the instruction; EBX has to be read out before the restore, which
// is why this cannot simply push and pop around the whole thing.
func cpuidProbe(leaf, sub int32) ([]byte, error) {
	var a x86.Buf
	a.MOVQ(x86.R11, x86.RBX)               // save RBX
	a.MOVLoad(x86.R10, x86.At(x86.RDI, 0)) // Args.Out -> where to write
	a.MOVimm32(x86.RAX, leaf)
	a.MOVimm32(x86.RCX, sub)
	a.CPUID()
	a.MOVStore(x86.At(x86.R10, 0), x86.RAX)
	a.MOVStore(x86.At(x86.R10, 8), x86.RBX)
	a.MOVStore(x86.At(x86.R10, 16), x86.RCX)
	a.MOVStore(x86.At(x86.R10, 24), x86.RDX)
	a.MOVQ(x86.RBX, x86.R11) // restore
	a.RET()
	return a.Bytes()
}
