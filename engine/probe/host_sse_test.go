//go:build jitllmtest && amd64

package probe

import (
	"testing"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestHostMeasuresSSE runs the host probe's SSE kernels on any x86 host,
// the tier forced, and holds them to the orders TestHostMeasures holds.
func TestHostMeasuresSSE(t *testing.T) {
	defer cpu.ForceTierForTest(cpu.ForceTierForTest(cpu.TierSSE))
	if hostLanes() != 4 {
		t.Fatalf("the SSE tier has %d lanes", hostLanes())
	}
	TestHostMeasures(t)
}
