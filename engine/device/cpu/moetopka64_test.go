//go:build arm64

package cpu

import (
	"fmt"
	"strings"
	"testing"
)

// The NEON MoE router top-k gate. It is the amd64 file's arms against the A64
// kernel, over the same oracle and the same fixtures (moetopkref_test.go).
//
// Its tag is arm64 alone, not `arm64 && darwin`, so it also runs under
// qemu-aarch64 on an amd64 box.

func moeKernelA64(t *testing.T, n, k int, norm bool) (*Code, func()) {
	t.Helper()
	b, err := EmitMoETopK(n, k, norm)
	if err != nil {
		t.Fatalf("EmitMoETopK(%d, %d, %v): %v", n, k, norm, err)
	}
	c, err := MapNamed(b, "moe_topk_gate_neon")
	if err != nil {
		t.Fatalf("mapping the NEON top-k: %v", err)
	}
	return c, func() { c.Close() }
}

func TestMoETopKA64MatchesTheGoLoops(t *testing.T) {
	for _, s := range moeShapes {
		n, k := s[0], s[1]
		for _, norm := range []bool{true, false} {
			c, done := moeKernelA64(t, n, k, norm)
			for _, in := range moeInputs(n) {
				want := moeTopKGo(in.p, k, norm, moeFaultNone)
				got := moeCall(c, in.p, k)
				if d := moeDiff(want, got); d != "" {
					t.Errorf("neon n=%d k=%d norm=%v %s: %s", n, k, norm, in.name, d)
				}
				if in.name == "allEqual" {
					moeCheckAllEqual(t, "neon", got)
				}
			}
			done()
		}
	}
}

// TestMoETopKA64Violations: the same three defects, injected into the oracle,
// must make the comparison go red here too.
func TestMoETopKA64Violations(t *testing.T) {
	for _, f := range []moeFault{moeFaultLastOfTie, moeFaultAscendingSum, moeFaultOrdInSelectionOrder} {
		t.Run(strings.ReplaceAll(f.String(), " ", "_"), func(t *testing.T) {
			var first string
			hits := 0
			for _, s := range moeShapes {
				n, k := s[0], s[1]
				if k < 2 {
					continue
				}
				c, done := moeKernelA64(t, n, k, true)
				for _, in := range moeInputs(n) {
					bad := moeTopKGo(in.p, k, true, f)
					if d := moeDiff(bad, moeCall(c, in.p, k)); d != "" {
						hits++
						if first == "" {
							first = fmt.Sprintf("n=%d k=%d %s: %s", n, k, in.name, d)
						}
					}
				}
				done()
			}
			if hits == 0 {
				t.Fatalf("%v changes nothing the gate compares -- it does not "+
					"discriminate on that axis", f)
			}
			t.Logf("%v: %d of the fixtures disagree; the first is %s", f, hits, first)
		})
	}
}

func TestMoETopKA64RefusesAShapeItCannotServe(t *testing.T) {
	for _, s := range [][2]int{{4, 8}, {0, 1}, {8, 0}, {8, -1}} {
		if _, err := EmitMoETopK(s[0], s[1], true); err == nil {
			t.Errorf("EmitMoETopK(%d, %d) produced code", s[0], s[1])
		}
	}
}

// TestMoETopKA64TableHandsTheTiersKernel checks the emitter table returns this
// tier's kernel.
func TestMoETopKA64TableHandsTheTiersKernel(t *testing.T) {
	want, err := EmitMoETopK(128, 8, true)
	if err != nil {
		t.Fatal(err)
	}
	got, err := EmittersFor(TierNEON).MoETopK(128, 8, true)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Error("the table's MoETopK is not the NEON emitter")
	}
}
