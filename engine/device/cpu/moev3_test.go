//go:build amd64

package cpu

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"
)

// The DeepSeek-V3 router gate for both x86 tiers. moev3ref_test.go has the
// fixtures and the oracle adapter; the bar is bit equality, as for the pre-V3
// gate.

func moeV3Kernels(t *testing.T, cfg moeV3Cfg) ([]moeV3Kernel, func()) {
	t.Helper()
	avx, err := EmitMoERoute(cfg.n, cfg.k, cfg.g)
	if err != nil {
		t.Fatalf("EmitMoERoute(%s): %v", cfg.name, err)
	}
	sse, err := EmitMoERouteSSE(cfg.n, cfg.k, cfg.g)
	if err != nil {
		t.Fatalf("EmitMoERouteSSE(%s): %v", cfg.name, err)
	}
	if got := KernelTier(sse); got != TierSSE {
		t.Fatalf("%s: the SSE router was generated for tier %v -- a VEX instruction "+
			"would fault on the host this kernel exists for", cfg.name, got)
	}
	requireSSEKernel(t, "sse_moe_route", sse)

	kons := MoETopKConstsFor(cfg.scale)
	var out []moeV3Kernel
	var codes []*Code
	for _, e := range []struct {
		name string
		body []byte
	}{{"avx2", avx}, {"sse", sse}} {
		if !tierArmRuns(t, e.name, e.body) {
			continue
		}
		c, err := MapNamed(e.body, "moe_route_"+e.name)
		if err != nil {
			t.Fatalf("mapping the %s router: %v", e.name, err)
		}
		codes = append(codes, c)
		out = append(out, moeV3Kernel{e.name + "/" + cfg.name, c, kons})
	}
	return out, func() {
		for _, c := range codes {
			c.Close()
		}
	}
}

// TestMoERouteMatchesTheOracle is the gate.
func TestMoERouteMatchesTheOracle(t *testing.T) {
	moeV3RunAll(t, moeV3Kernels, oracle.MoEFaultNone, nil)
}

// TestMoERouteViolations injects each plausible defect into the oracle (not
// the emitter, which would put a test hook in shipping code) and requires the
// comparison to go red. The V3-specific ones: the bias reaching the weights, a
// group scored by its best instead of its best two, and the mask arriving
// after the selection.
func TestMoERouteViolations(t *testing.T) {
	for _, f := range oracle.MoEFaults {
		t.Run(strings.ReplaceAll(f.String(), " ", "_"), func(t *testing.T) {
			var first string
			hits := moeV3RunAll(t, moeV3Kernels, f, func(s string) { first = s })
			if hits == 0 {
				t.Fatalf("%v changes nothing the gate compares -- it does not "+
					"discriminate on that axis", f)
			}
			t.Logf("%v: %d comparison(s) disagree; the first is %s", f, hits, first)
		})
	}
}

// TestMoERouteTableHandsTheTiersKernel checks the emitter table returns this
// tier's kernel, so a right kernel behind a miswired table fails.
func TestMoERouteTableHandsTheTiersKernel(t *testing.T) {
	g := MoEGate{Norm: true, Bias: true, NGroup: 8, NGroupUsed: 4, Scale: true}
	for _, e := range []struct {
		tier Tier
		want func() ([]byte, error)
	}{
		{TierAVX2, func() ([]byte, error) { return EmitMoERoute(256, 8, g) }},
		{TierSSE, func() ([]byte, error) { return EmitMoERouteSSE(256, 8, g) }},
	} {
		want, err := e.want()
		if err != nil {
			t.Fatal(err)
		}
		got, err := EmittersFor(e.tier).MoERoute(256, 8, g)
		if err != nil {
			t.Fatalf("%v: %v", e.tier, err)
		}
		if string(got) != string(want) {
			t.Errorf("%v: the table's MoERoute is not that tier's emitter", e.tier)
		}
	}
}
