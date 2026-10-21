//go:build amd64 || arm64

package cpu_test

import (
	"testing"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// hostTable is the external package's view of the emitter table of the tier
// this host runs (hosttier_test.go has the reason): an AVX2 host's gate runs
// the AVX2 kernel and an SSE-only host's the SSE one, against the same oracle.
func hostTable() *cpu.Emitters { return cpu.EmittersFor(cpu.HostTier()) }

// onHost maps a kernel emitted by hostTable, failing the gate on the emitter's
// error or on Map's: a host-tier kernel the host refuses is a table and a host
// that disagree about the tier, never a reason to skip.
func onHost(t testing.TB) func([]byte, error) *cpu.Code {
	return func(b []byte, err error) *cpu.Code {
		t.Helper()
		if err != nil {
			t.Fatalf("the %v tier's emitter: %v", cpu.HostTier(), err)
		}
		c, merr := cpu.Map(b)
		if merr != nil {
			t.Fatalf("the %v tier's kernel does not map on this host: %v", cpu.HostTier(), merr)
		}
		return c
	}
}
