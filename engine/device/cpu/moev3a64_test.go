//go:build arm64

package cpu

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"
)

// The NEON DeepSeek-V3 router gate. It is the amd64 file's arms against the A64
// kernel, over the same fixtures and the same oracle (moev3ref_test.go).
//
// Its tag is arm64 alone, not `arm64 && darwin`, so it also runs under
// qemu-aarch64 on an amd64 box (evidence about the encoding and arithmetic,
// not about the hardware).

func moeV3KernelA64(t *testing.T, cfg moeV3Cfg) ([]moeV3Kernel, func()) {
	t.Helper()
	b, err := EmitMoERoute(cfg.n, cfg.k, cfg.g)
	if err != nil {
		t.Fatalf("EmitMoERoute(%s): %v", cfg.name, err)
	}
	c, err := MapNamed(b, "moe_route_neon")
	if err != nil {
		t.Fatalf("mapping the NEON router: %v", err)
	}
	return []moeV3Kernel{{"neon/" + cfg.name, c, MoETopKConstsFor(cfg.scale)}},
		func() { c.Close() }
}

func TestMoERouteA64MatchesTheOracle(t *testing.T) {
	moeV3RunAll(t, moeV3KernelA64, oracle.MoEFaultNone, nil)
}

// TestMoERouteA64Violations: the same defects, injected into the oracle,
// must make the comparison go red here too.
func TestMoERouteA64Violations(t *testing.T) {
	for _, f := range oracle.MoEFaults {
		t.Run(strings.ReplaceAll(f.String(), " ", "_"), func(t *testing.T) {
			var first string
			hits := moeV3RunAll(t, moeV3KernelA64, f, func(s string) { first = s })
			if hits == 0 {
				t.Fatalf("%v changes nothing the gate compares -- it does not "+
					"discriminate on that axis", f)
			}
			t.Logf("%v: %d comparison(s) disagree; the first is %s", f, hits, first)
		})
	}
}

// TestMoERouteA64TableHandsTheTiersKernel checks the emitter table returns this
// tier's kernel.
func TestMoERouteA64TableHandsTheTiersKernel(t *testing.T) {
	g := MoEGate{Norm: true, Bias: true, NGroup: 8, NGroupUsed: 4, Scale: true}
	want, err := EmitMoERoute(256, 8, g)
	if err != nil {
		t.Fatal(err)
	}
	got, err := EmittersFor(TierNEON).MoERoute(256, 8, g)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Error("the table's MoERoute is not the NEON emitter")
	}
}
