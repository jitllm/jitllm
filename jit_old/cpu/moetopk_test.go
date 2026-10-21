//go:build amd64

package cpu

import (
	"fmt"
	"strings"
	"testing"
)

// The MoE router top-k gate, for both x86 tiers.
//
// The bar is bit equality with the Go loops: sel and ord are expert ids, and
// wt, ow and sum must match bit for bit because Forward, Prefill and
// ForwardBatch are gated bit-identical on a mixture. Both tiers run against the
// same oracle; their lane grouping differs, which is fine since the selection
// does no arithmetic and the divisor is a k-term chain in selection order.

// moeKernels maps both tiers' kernels for one shape, and checks the SSE bytes
// declare the SSE tier and carry no VEX prefix before either executes.
func moeKernels(t *testing.T, n, k int, norm bool) ([]struct {
	name string
	c    *Code
}, func()) {
	t.Helper()
	avx, err := EmitMoETopK(n, k, norm)
	if err != nil {
		t.Fatalf("EmitMoETopK(%d, %d, %v): %v", n, k, norm, err)
	}
	sse, err := EmitMoETopKSSE(n, k, norm)
	if err != nil {
		t.Fatalf("EmitMoETopKSSE(%d, %d, %v): %v", n, k, norm, err)
	}
	if got := KernelTier(sse); got != TierSSE {
		t.Fatalf("the SSE top-k was generated for tier %v -- a VEX instruction would "+
			"fault on the host this kernel exists for", got)
	}
	requireSSEKernel(t, "sse_moe_topk", sse)

	var out []struct {
		name string
		c    *Code
	}
	var codes []*Code
	for _, e := range []struct {
		name string
		body []byte
	}{{"avx2", avx}, {"sse", sse}} {
		if !tierArmRuns(t, e.name, e.body) {
			continue
		}
		c, err := MapNamed(e.body, "moe_topk_gate_"+e.name)
		if err != nil {
			t.Fatalf("mapping the %s top-k: %v", e.name, err)
		}
		codes = append(codes, c)
		out = append(out, struct {
			name string
			c    *Code
		}{e.name, c})
	}
	return out, func() {
		for _, c := range codes {
			c.Close()
		}
	}
}

// TestMoETopKMatchesTheGoLoops is the gate.
func TestMoETopKMatchesTheGoLoops(t *testing.T) {
	for _, s := range moeShapes {
		n, k := s[0], s[1]
		for _, norm := range []bool{true, false} {
			ks, done := moeKernels(t, n, k, norm)
			for _, in := range moeInputs(n) {
				want := moeTopKGo(in.p, k, norm, moeFaultNone)
				for _, kern := range ks {
					got := moeCall(kern.c, in.p, k)
					if d := moeDiff(want, got); d != "" {
						t.Errorf("%s n=%d k=%d norm=%v %s: %s", kern.name, n, k, norm, in.name, d)
					}
					if in.name == "allEqual" {
						moeCheckAllEqual(t, kern.name, got)
					}
				}
			}
			done()
		}
	}
}

// TestMoETopKViolations injects each of the three plausible-looking defects
// into the oracle (not the emitter, which would put a test hook in shipping
// code) and requires the comparison to go red, proving it discriminates on
// each axis.
func TestMoETopKViolations(t *testing.T) {
	for _, f := range []moeFault{moeFaultLastOfTie, moeFaultAscendingSum, moeFaultOrdInSelectionOrder} {
		t.Run(strings.ReplaceAll(f.String(), " ", "_"), func(t *testing.T) {
			var first string
			hits := 0
			for _, s := range moeShapes {
				n, k := s[0], s[1]
				if k < 2 {
					continue // every one of these needs at least two selections
				}
				ks, done := moeKernels(t, n, k, true)
				for _, in := range moeInputs(n) {
					bad := moeTopKGo(in.p, k, true, f)
					got := moeCall(ks[0].c, in.p, k)
					if d := moeDiff(bad, got); d != "" {
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

// TestMoETopKRefusesAShapeItCannotServe: k > n is not a configuration, so every
// tier refuses it by name.
func TestMoETopKRefusesAShapeItCannotServe(t *testing.T) {
	for _, s := range [][2]int{{4, 8}, {0, 1}, {8, 0}, {8, -1}} {
		if _, err := EmitMoETopK(s[0], s[1], true); err == nil {
			t.Errorf("EmitMoETopK(%d, %d) produced code", s[0], s[1])
		}
		if _, err := EmitMoETopKSSE(s[0], s[1], true); err == nil {
			t.Errorf("EmitMoETopKSSE(%d, %d) produced code", s[0], s[1])
		}
	}
}

// TestMoETopKTableHandsTheTiersKernel checks the emitter table returns this
// tier's kernel, so a right kernel behind a miswired table fails.
func TestMoETopKTableHandsTheTiersKernel(t *testing.T) {
	for _, e := range []struct {
		tier Tier
		want func() ([]byte, error)
	}{
		{TierAVX2, func() ([]byte, error) { return EmitMoETopK(128, 8, true) }},
		{TierSSE, func() ([]byte, error) { return EmitMoETopKSSE(128, 8, true) }},
	} {
		want, err := e.want()
		if err != nil {
			t.Fatal(err)
		}
		got, err := EmittersFor(e.tier).MoETopK(128, 8, true)
		if err != nil {
			t.Fatalf("%v: %v", e.tier, err)
		}
		if string(got) != string(want) {
			t.Errorf("%v: the table's MoETopK is not that tier's emitter", e.tier)
		}
	}
}
