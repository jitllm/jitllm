//go:build amd64 && jitllmtest

package hardware

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestISAFollowsTheTier: the machine page's isa field is the tier report off
// the AVX2 tier, not the AVX2 tier's VNNI line on a host that cannot run it.
func TestISAFollowsTheTier(t *testing.T) {
	if cpu.HostTier() == cpu.TierAVX2 && !strings.Contains(isa(), "VPDPBUSD") {
		t.Errorf("the AVX2 host's isa changed: %q", isa())
	}
	old := cpu.ForceTierForTest(cpu.TierSSE)
	defer cpu.ForceTierForTest(old)
	if cpu.HostTier() != cpu.TierSSE {
		t.Fatalf("forced SSE and the probe reports %v", cpu.HostTier())
	}
	if s := isa(); !strings.HasPrefix(s, "sse ") || strings.Contains(s, "VPDPBUSD") {
		t.Errorf("an SSE host's isa is %q, want the tier report", s)
	}
}
