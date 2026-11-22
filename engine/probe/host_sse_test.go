//go:build jitllmtest && amd64

package probe

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
)

// TestHostMeasuresSSE runs the host probe's SSE kernels on any x86 host,
// the tier forced, and holds them to the orders TestHostMeasures holds.
func TestHostMeasuresSSE(t *testing.T) {
	h := cpu.New(cpu.AtTier(cpu.TierSSE))
	if h.Tier() != cpu.TierSSE {
		t.Fatalf("a Host at the SSE tier reports %v", h.Tier())
	}
	if n := (emitter{sse: true}).hostLanes(); n != 4 {
		t.Fatalf("the SSE tier has %d lanes", n)
	}
	hostMeasures(t, h)
}
