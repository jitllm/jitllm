//go:build amd64

// amd64 only: every assertion here is about CPUID.

package cpu

import (
	"runtime"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
)

// TestFeatureProbeGatesTheQuantizedKernels is the production gate: a CPU
// without the instruction (VPDPBUSD, SDOT) must decline the kernel, not emit
// it and SIGILL on the first token.
func TestFeatureProbeGatesTheQuantizedKernels(t *testing.T) {
	// The emitters stay host-pure: they must still produce code for a target
	// this machine cannot run, or cross-arch disassembly tests break.
	if !Supported(quant.Q4_0) || !SupportedA64(quant.Q4_0) {
		t.Error("the emitters must not consult the host CPU; that is SupportedNative's job")
	}

	// F32 needs no feature on either arch, so it is always available -- which
	// is what keeps nn.NewJIT returning a usable tier (with its pool) on a
	// machine that has nothing else.
	if !SupportedNative(quant.F32) {
		t.Error("F32 uses only baseline AVX2 / ARMv8-A and must never be gated")
	}

	// The probe must actually be consulted: forcing the feature off (which
	// re-probes) has to change the answer.
	if !SupportedNative(quant.Q4_0) {
		t.Skip("this host declines Q4_0 already, so forcing cannot change the answer")
	}
	forceNoFeatures(t)
	if SupportedNative(quant.Q4_0) {
		t.Error("the feature probe was forced to report absent and SupportedNative " +
			"still admits a quantized kernel: a SIGILL on the first token")
	}
	// The packed family has a pre-VNNI form on amd64, so it stays available
	// (the point of SupportedPackedNative).
	if runtime.GOARCH == "amd64" && !SupportedPackedNative(quant.Q4_0) {
		t.Error("a pre-VNNI amd64 host must still build the PACKED kernel: it has " +
			"an AVX2 sequence (jit_old/cpu/prevnni.go), and declining sends the " +
			"container decode path to the float64 oracle")
	}
}

// forceNoFeatures makes the CPU probes answer "absent" for the rest of this
// test. The switches are package variables rather than environment variables,
// so a shipped binary cannot pretend the CPU lacks an instruction it has.
func forceNoFeatures(t *testing.T) {
	t.Helper()
	vnni, dot := forceNoVNNI, forceNoDotProd
	t.Cleanup(func() {
		forceNoVNNI, forceNoDotProd = vnni, dot
		reprobeHostFeatures()
	})
	forceNoVNNI, forceNoDotProd = true, true
	// Re-run the one capability probe (features_amd64.go): without it the
	// Once has already fired and the flag reaches nothing, and without the
	// Cleanup reset it would reach the rest of the binary.
	reprobeHostFeatures()
}
