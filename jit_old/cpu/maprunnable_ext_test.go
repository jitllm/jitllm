package cpu_test

import (
	"errors"
	"testing"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// hybMapRunnable is mustMap (hostexec_test.go) for the external test
// package: it maps code, and skips (naming the refusal) where this host cannot
// run it. On an SSE host an AVX2-tier gate can only skip; each op has an
// SSE-tier gate beside it that runs there.
func hybMapRunnable(t testing.TB, code []byte) *cpu.Code {
	t.Helper()
	c, err := cpu.Map(code)
	if err == nil {
		return c
	}
	if errors.Is(err, cpu.ErrNoVNNI) || errors.Is(err, cpu.ErrISA) {
		t.Skipf("NOT RUNNABLE ON THIS HOST (tier %v): %v -- the SSE-tier twin's gate covers "+
			"this op here", cpu.HostTier(), err)
	}
	t.Fatalf("Map: %v", err)
	return nil
}
