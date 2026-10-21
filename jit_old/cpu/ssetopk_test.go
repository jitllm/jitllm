//go:build amd64 && jitllmtest

package cpu

import (
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"
)

// The MoE top-k on the forced SSE tier. TestMoETopKMatchesTheGoLoops already
// calls the SSE kernel; this arm runs it through the table under the force and
// asserts the SSE tier mapped a kernel and no AVX2 kernel was mapped, so the
// tier under test is the one selected.
func TestMoETopKOnTheForcedSSETier(t *testing.T) {
	old := ForceTierForTest(TierSSE)
	defer ForceTierForTest(old)
	if HostTier() != TierSSE {
		t.Fatalf("the force did not reach the probe: host tier %v", HostTier())
	}
	before := MappedByTier()

	for _, s := range moeShapes {
		n, k := s[0], s[1]
		for _, norm := range []bool{true, false} {
			b, err := EmittersFor(HostTier()).MoETopK(n, k, norm)
			if err != nil {
				t.Fatalf("n=%d k=%d: %v", n, k, err)
			}
			if got := KernelTier(b); got != TierSSE {
				t.Fatalf("n=%d k=%d: the forced tier handed a kernel declaring %v", n, k, got)
			}
			requireSSEKernel(t, "sse_moe_topk", b)
			c, err := MapNamed(b, "moe_topk_forced_sse")
			if err != nil {
				t.Fatalf("n=%d k=%d: mapping: %v", n, k, err)
			}
			for _, in := range moeInputs(n) {
				want := moeTopKGo(in.p, k, norm, moeFaultNone)
				if d := moeDiff(want, moeCall(c, in.p, k)); d != "" {
					t.Errorf("forced SSE n=%d k=%d norm=%v %s: %s", n, k, norm, in.name, d)
				}
			}
			c.Close()
		}
	}

	after := MappedByTier()
	if d := after[TierAVX2] - before[TierAVX2]; d != 0 {
		t.Errorf("%d AVX2 kernel(s) were mapped under the forced SSE tier", d)
	}
	if after[TierSSE] == before[TierSSE] {
		t.Error("no SSE-tier kernel was mapped -- this arm did not run the tier it names")
	}
}

// TestMoERouteOnTheForcedSSETier is the same arm for the DeepSeek-V3 router:
// the bias plane, the group phase, the unbiased gather and the scaling factor
// on legacy SSE under the force, asserting the tier as well as the answer.
func TestMoERouteOnTheForcedSSETier(t *testing.T) {
	old := ForceTierForTest(TierSSE)
	defer ForceTierForTest(old)
	if HostTier() != TierSSE {
		t.Fatalf("the force did not reach the probe: host tier %v", HostTier())
	}
	before := MappedByTier()

	moeV3RunAll(t, func(t *testing.T, cfg moeV3Cfg) ([]moeV3Kernel, func()) {
		t.Helper()
		b, err := EmittersFor(HostTier()).MoERoute(cfg.n, cfg.k, cfg.g)
		if err != nil {
			t.Fatalf("%s: %v", cfg.name, err)
		}
		if got := KernelTier(b); got != TierSSE {
			t.Fatalf("%s: the forced tier handed a kernel declaring %v", cfg.name, got)
		}
		requireSSEKernel(t, "sse_moe_route", b)
		c, err := MapNamed(b, "moe_route_forced_sse")
		if err != nil {
			t.Fatalf("%s: mapping: %v", cfg.name, err)
		}
		return []moeV3Kernel{{"forced-sse/" + cfg.name, c, MoETopKConstsFor(cfg.scale)}},
			func() { c.Close() }
	}, oracle.MoEFaultNone, nil)

	after := MappedByTier()
	if d := after[TierAVX2] - before[TierAVX2]; d != 0 {
		t.Errorf("%d AVX2 kernel(s) were mapped under the forced SSE tier", d)
	}
	if after[TierSSE] == before[TierSSE] {
		t.Error("no SSE-tier kernel was mapped -- this arm did not run the tier it names")
	}
}
