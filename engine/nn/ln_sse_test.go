//go:build amd64 && jitllmtest

package nn

import (
	"testing"

	"github.com/jitllm/jitllm/jit/cpu"
)

// TestLnMatchesTheDefinitionOnTheSSETier is the same gate under the forced SSE
// tier, whose body has no FMA and is a transcription of its own.
func TestLnMatchesTheDefinitionOnTheSSETier(t *testing.T) {
	old := cpu.ForceTierForTest(cpu.TierSSE)
	defer cpu.ForceTierForTest(old)
	if cpu.HostTier() != cpu.TierSSE {
		t.Fatalf("forced the SSE tier and the probe reports %v", cpu.HostTier())
	}
	checkLn(t)
}
