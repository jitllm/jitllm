package composer

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/composer/x86"
	"github.com/jitllm/jitllm/engine/device/cpu"
)

// TestCPUFormFollowsTheISA: the composer's description of a CPU device is
// the passes and instructions its stated ISA selects -- the wide passes on
// AVX2, FMA and F16C, the narrow ones on SSE4.1 alone (NarrowsCPU, which the
// KV width default reads), VPDPBUSD where AVX-VNNI is stated, and on arm64
// SDOT or its widening by FEAT_DotProd.
func TestCPUFormFollowsTheISA(t *testing.T) {
	wide := uint64(x86.ISATierSSE | x86.ISATierAVX2)
	for _, c := range []struct {
		isa    uint64
		arm64  bool
		want   []string
		narrow bool
	}{
		{wide, false, []string{"wide passes", "byte multiplies"}, false},
		{wide | uint64(x86.ISAVNNI), false, []string{"wide passes", "VPDPBUSD"}, false},
		{uint64(x86.ISATierSSE), false, []string{"narrow passes", "byte multiplies"}, true},
		{uint64(x86.ISATierSSE | x86.ISAAVX2), false, []string{"narrow passes"}, true},
		{uint64(cpu.ISADotProd), true, []string{"SDOT"}, false},
		{0, true, []string{"widened"}, false},
	} {
		d := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: c.isa}
		got := CPUForm(d, c.arm64)
		for _, w := range c.want {
			if !strings.Contains(got, w) {
				t.Errorf("ISA %#x (arm64 %v): %q lacks %q", c.isa, c.arm64, got, w)
			}
		}
		if !c.arm64 && NarrowsCPU(d) != c.narrow {
			t.Errorf("ISA %#x: NarrowsCPU %v, want %v", c.isa, !c.narrow, c.narrow)
		}
	}
}
