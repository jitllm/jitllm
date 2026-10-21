//go:build amd64 && jitllmtest

package nn

import (
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestPackedGEMMPreVNNIMatchesTheRowLoopExactly runs the GEMM gate with VNNI
// forced absent for the whole test, reference included: the GEMM is emitted
// lazily per shape and reads HostDotKind at emission time, so forcing only
// around NewJIT (as forcedPreVNNI does) would hand it the VNNI sequence.
func TestPackedGEMMPreVNNIMatchesTheRowLoopExactly(t *testing.T) {
	needPreVNNI(t)
	old := cpu.ForceNoVNNIForTest(true)
	defer cpu.ForceNoVNNIForTest(old)
	if cpu.HostDotKind() != cpu.DotVEX {
		t.Fatal("ForceNoVNNIForTest did not make HostDotKind report DotVEX")
	}
	mk := func(maxK, maxRows int, ts []quant.Type, opts ...Option) *JIT {
		if cpu.HostDotKind() != cpu.DotVEX {
			t.Fatal("the forcing lapsed mid-test")
		}
		return NewJIT(maxK, maxRows, ts, opts...)
	}
	n := runGEMMGate(t, mk)
	if n == 0 {
		t.Fatal("no (format, shape) pair ran -- this gate proved nothing")
	}
	t.Logf("%d (format, shape) pair(s) bit-identical (or, integer-accumulated, within NMSE 1e-10) under the forced VEX sequence", n)
}
