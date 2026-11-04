//go:build amd64 && jitllmtest

package model_test

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
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
	// Every model it opens maps onto a Host at the SSE tier, whose gate
	// refuses an AVX2 kernel by name: none can be mapped silently.
	TestGreedyMatchesLlamaCpp(t)
}
