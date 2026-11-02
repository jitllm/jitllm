// Package amdgpu lowers jit_old/gpu/ir kernels to LLVM IR text for AMD GPUs.
//
// It is the fourth lowering beside ptx, spirv and msl. The text is handed to
// comgr (jit_old/gpu/hip.Comgr.Compile), whose AMDGPU code generator owns
// instruction selection, register allocation and scheduling, as ptxas does for
// PTX. See docs/design/amd-hip.md for why the target is LLVM IR and not GCN
// assembly.
package amdgpu

import (
	"fmt"
	"strconv"
	"strings"
)

// Target is the GPU a kernel is generated for.
type Target struct {
	// Arch is HIP's gcnArchName, target features included
	// ("gfx90a:sramecc+:xnack-"); comgr is told exactly this.
	Arch string
	// Wave is the wavefront width the kernel is generated at, stated in the
	// function's target features rather than left to the code generator's
	// default: 64 on gfx9 (GCN, CDNA), 32 on gfx10 and later (RDNA).
	Wave int
	// Dot4 is how this target computes a four-way int8 dot product.
	Dot4 Dot4Kind
}

// Dot4Kind is the instruction OpDot4 lowers to.
type Dot4Kind uint8

const (
	// Dot4Bytes is four sign-extended multiplies and adds: correct on every
	// target, the form for one whose dot instruction is not listed below.
	Dot4Bytes Dot4Kind = iota
	// Dot4Signed is v_dot4_i32_i8 (llvm.amdgcn.sdot4), the dot1-insts
	// instruction of gfx908, gfx90a and gfx94x.
	Dot4Signed
	// Dot4Mixed is v_dot4_i32_iu8 (llvm.amdgcn.sudot4) with both operands
	// signed: gfx11 and gfx12 dropped the first and kept this one.
	Dot4Mixed
)

func (d Dot4Kind) String() string {
	return [...]string{"bytes", "sdot4", "sudot4"}[d]
}

// TargetFor is the target for a gcnArchName. An arch this package cannot place
// in a generation is refused: its wave size would be a guess.
func TargetFor(arch string) (Target, error) {
	base := arch
	if i := strings.IndexByte(base, ':'); i >= 0 {
		base = base[:i]
	}
	if !strings.HasPrefix(base, "gfx") || len(base) < 6 {
		return Target{}, fmt.Errorf("amdgpu: %q is not a gfx target", arch)
	}
	num := base[3:]
	// The generation is every digit but the last two, which are the minor and
	// stepping (hex): gfx90a is 9, gfx1100 is 11, gfx1201 is 12.
	major, err := strconv.Atoi(num[:len(num)-2])
	if err != nil {
		return Target{}, fmt.Errorf("amdgpu: %q is not a gfx target", arch)
	}
	t := Target{Arch: arch, Wave: 32}
	switch {
	case major == 9:
		t.Wave = 64
		switch base {
		case "gfx908", "gfx90a", "gfx940", "gfx941", "gfx942", "gfx950":
			t.Dot4 = Dot4Signed
		}
	case major == 10:
		switch base {
		case "gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036":
			t.Dot4 = Dot4Signed
		}
	case major == 11 || major == 12:
		t.Dot4 = Dot4Mixed
	case major < 9:
		return Target{}, fmt.Errorf("amdgpu: %s predates the generations this lowering targets (gfx9 and later)", arch)
	}
	return t, nil
}

// Base is the arch without its target features.
func (t Target) Base() string {
	if i := strings.IndexByte(t.Arch, ':'); i >= 0 {
		return t.Arch[:i]
	}
	return t.Arch
}
