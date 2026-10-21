//go:build (amd64 || arm64) && jitllmtest

package nn

import (
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestMoERouteJITOnTheForcedSSETier runs the whole DeepSeek-V3 router -- the
// generated sigmoid AND the generated selection -- with the probe answering for
// an Atom.
//
// This entry point has two caches, moeTopKs and moeSigmoids. Both are warmed
// on the host's own tier first, so a cache that ignored the tier would answer
// the forced run with an AVX2 kernel; MappedByTier says it did not.
func TestMoERouteJITOnTheForcedSSETier(t *testing.T) {
	skipOnAnSSEHost(t, "that the router's two caches are keyed by tier; the SSE router "+
		"itself is held to the oracle here by TestMoERouteJITMatchesTheOracle and "+
		"jit_old/cpu's TestMoERouteMatchesTheOracle")
	// Warm both caches on the host tier first.
	for _, cfg := range moeV3NNCfgs() {
		r := NewMoERouteFor(cfg.n, cfg.k, cfg.g)
		if !MoERouteJIT(moeV3Logits(cfg.n)[0].x, cfg.g, r) {
			t.Fatalf("%s: no kernel on the host tier", cfg.name)
		}
	}

	old := cpu.ForceTierForTest(cpu.TierSSE)
	defer cpu.ForceTierForTest(old)
	if cpu.HostTier() != cpu.TierSSE {
		t.Skipf("the tier could not be forced to SSE on this build (host tier %v)", cpu.HostTier())
	}
	before := cpu.MappedByTier()

	moeV3Sweep(t, oracle.MoEFaultNone, nil)

	after := cpu.MappedByTier()
	if d := after[cpu.TierAVX2] - before[cpu.TierAVX2]; d != 0 {
		t.Errorf("%d AVX2 kernel(s) were mapped under the forced SSE tier", d)
	}
	if after[cpu.TierSSE] == before[cpu.TierSSE] {
		t.Error("no SSE-tier kernel was mapped -- this arm did not run the tier it names")
	}
}
