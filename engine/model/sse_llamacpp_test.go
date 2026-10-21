//go:build amd64 && jitllmtest

package model_test

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestGreedyMatchesLlamaCppSSE is the llama.cpp golden gate over every model
// with the SSE tier forced -- the same gate, the same tie rule (a divergence
// on a knife-edge margin is reduction order, not a bug), and no AVX2 kernel
// mapped while it runs. It is the external-package twin of TestSSEModelGates,
// which cannot call this package's TestGreedyMatchesLlamaCpp.
//
// Like the other model-level SSE gates it needs every SSE family, and until
// then it fails naming the ops still pending rather than skip.
func TestGreedyMatchesLlamaCppSSE(t *testing.T) {
	old := cpu.ForceTierForTest(cpu.TierSSE)
	defer cpu.ForceTierForTest(old)
	if cpu.HostTier() != cpu.TierSSE {
		t.Fatalf("forced SSE and the probe reports %v -- the force reached nothing", cpu.HostTier())
	}
	if p := cpu.SSEPending(); len(p) > 0 {
		t.Fatalf("the SSE tier is incomplete: %d ops have no kernel (%s); red by design until every family has landed",
			len(p), strings.Join(p, ", "))
	}
	before := cpu.MappedByTier()
	TestGreedyMatchesLlamaCpp(t)
	if d := cpu.MappedByTier()[cpu.TierAVX2] - before[cpu.TierAVX2]; d != 0 {
		t.Errorf("%d AVX2 kernels were mapped while the SSE tier was forced", d)
	}
}
