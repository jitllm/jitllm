//go:build amd64 || arm64

package nn

import (
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// requireRowMajorHost skips a gate over the GGUF row-major family when this CPU
// cannot execute it, naming the missing capability. That family has no
// pre-VNNI form by design (a container never decodes through it). It skips on
// the capability (cpu.SupportedNative, the predicate nn.MatVec and nn.MatMul
// consult), not the architecture, so the gate runs exactly when the path it
// covers is reachable.
func requireRowMajorHost(t testing.TB, qt quant.Type) {
	t.Helper()
	if cpu.SupportedNative(qt) {
		return
	}
	t.Skipf("ROW-MAJOR %v IS NOT RUNNABLE ON THIS HOST (%s, no AVX-VNNI) -- this "+
		"gate covered nothing here. The row-major family has no pre-VNNI form by "+
		"decision (native_amd64.go); the PACKED family is what a container uses "+
		"and jit_old/cpu's prevnni.go is where its AVX2 sequence lives.",
		qt, cpu.NativeArch())
}
