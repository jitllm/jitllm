//go:build (amd64 || arm64) && jitllmtest

package nn

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestRopeTableIsKeyedByTier checks a forced tier actually selects its own
// rotary-table kernel rather than a cached one (the hazard
// engine/nn/ssemoetopk_test.go gates for the router). It only runs where the tier
// can be forced.
func TestRopeTableIsKeyedByTier(t *testing.T) {
	skipOnAnSSEHost(t, "that a forced SSE call maps its own rotary-table kernel rather than "+
		"the warmed AVX2 one; the SSE kernel itself is held to the oracle here by the "+
		"host-tier rotary-table gates (ropetab_test.go)")
	r := Rope{NRot: 128, Base: 1000000}
	cs := make([]float32, r.NRot)
	r.Table(cs, 4097)

	old := cpu.ForceTierForTest(cpu.TierSSE)
	defer cpu.ForceTierForTest(old)
	if cpu.HostTier() != cpu.TierSSE {
		t.Skipf("the tier could not be forced to SSE on this build (host tier %v)", cpu.HostTier())
	}
	before := cpu.MappedByTier()
	sse := make([]float32, r.NRot)
	r.Table(sse, 4097)
	after := cpu.MappedByTier()
	if d := after[cpu.TierAVX2] - before[cpu.TierAVX2]; d != 0 {
		t.Errorf("%d AVX2 kernel(s) were mapped while the SSE tier was forced", d)
	}
	if after[cpu.TierSSE] == before[cpu.TierSSE] {
		t.Fatal("no SSE-tier kernel was mapped -- the cache is not keyed by tier and " +
			"this call reused the AVX2 kernel warmed above")
	}
	// The two tiers are held to the oracle, not to each other: the SSE tier
	// has no FMA, so its roundings differ.
	want := ropeOracle(r, 4097, r.NRot/2)
	var cross float64
	for i := range want {
		if d := math.Abs(float64(sse[i]) - float64(cs[i])); d > cross {
			cross = d
		}
	}

	// The whole sweep on this tier, not one shape: the ragged widths and the
	// large positions are where a four-lane kernel would part from an
	// eight-lane one, and those are precisely the cases one config cannot
	// reach.
	const bound = 3e-7
	var worst float64
	var worstAt string
	ran := 0
	for _, c := range ropeConfigs() {
		npairs := c.r.NRot / 2
		out := make([]float32, c.r.NRot)
		for _, pos := range ropePositions() {
			if pos >= cpu.RopeTabMaxPos {
				continue
			}
			c.r.Table(out, pos)
			ran++
			ref := ropeOracle(c.r, pos, npairs)
			for i := range ref {
				d := math.Abs(float64(out[i]) - ref[i])
				if d > worst {
					worst, worstAt = d, c.name
				}
				if d > bound {
					t.Errorf("SSE %s pos %d entry %d: |d| %.3e > %.1e", c.name, pos, i, d, bound)
				}
			}
		}
	}
	// And the tail guard on this tier as well: the SSE store narrows to four
	// pairs where the AVX2 one narrows to eight, so the two have different
	// ragged counts and a gate run on one says nothing about the other.
	const guard = float32(-12345.5)
	for _, c := range ropeConfigs() {
		npairs := c.r.NRot / 2
		buf := make([]float32, 2*npairs+32)
		for i := range buf {
			buf[i] = guard
		}
		for _, pos := range []int{0, 1, 129, 4096, 131071} {
			c.r.Table(buf[:2*npairs], pos)
			for i := 2 * npairs; i < len(buf); i++ {
				if buf[i] != guard {
					t.Fatalf("SSE %s (%d pairs) at pos %d: word %d past the table was written",
						c.name, npairs, pos, i-2*npairs)
				}
			}
		}
	}
	t.Logf("SSE tier: %d tables, worst |d| %.3e (%s); the two tiers differ by %.3e at one shape",
		ran, worst, worstAt, cross)
}
