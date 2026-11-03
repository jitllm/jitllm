package composer

import (
	"github.com/jitllm/jitllm/composer/x86"
	"github.com/jitllm/jitllm/engine/device/cpu"
)

// CPUForm describes how the composer forms CPU device d's kernels: on arm64
// NEON, its int8 dots by SDOT where d states FEAT_DotProd and widened where
// not; on x86 its wide passes (AVX2, FMA and F16C) or its narrow ones,
// and its int8 dots by VPDPBUSD where d states AVX-VNNI.
func CPUForm(d Device, arm64 bool) string {
	if arm64 {
		if d.ISA&uint64(cpu.ISADotProd) != 0 {
			return "arm64 NEON, int8 dots by SDOT"
		}
		return "arm64 NEON, int8 dots widened (no FEAT_DotProd)"
	}
	s := "x86, wide passes (AVX2, FMA, F16C)"
	if NarrowsCPU(d) {
		s = "x86, narrow passes (SSE4.1: four floats a vector, binary16 in software)"
	}
	if d.ISA&uint64(x86.ISAVNNI) != 0 {
		return s + ", int8 dots by VPDPBUSD"
	}
	return s + ", int8 dots by byte multiplies and pairwise adds"
}
