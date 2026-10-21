//go:build amd64 && jitllmtest

package main

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestISALineFollowsTheTier: `jitllm hardware`'s isa line describes the tier
// the host runs. On an SSE host it must not print the AVX2 tier's VNNI line --
// "AVX2 + AVX-VNNI ... (probed: ABSENT)" -- which is true and says nothing
// about what a run does there.
func TestISALineFollowsTheTier(t *testing.T) {
	if cpu.HostTier() == cpu.TierAVX2 && !strings.Contains(isa(), "VPDPBUSD") {
		t.Errorf("the AVX2 host's isa line changed: %q", isa())
	}
	old := cpu.ForceTierForTest(cpu.TierSSE)
	defer cpu.ForceTierForTest(old)
	if cpu.HostTier() != cpu.TierSSE {
		t.Fatalf("forced SSE and the probe reports %v", cpu.HostTier())
	}
	if s := isa(); strings.Contains(s, "VPDPBUSD") || !strings.Contains(s, "legacy SSE") {
		t.Errorf("an SSE host's isa line is %q", s)
	}
}
