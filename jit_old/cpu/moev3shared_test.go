//go:build amd64 || arm64

package cpu

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"
)

// The DeepSeek-V3 router's architecture-independent gates: the two byte
// identities that say the pre-V3 path did not move, and the shape refusals.

// TestMoETopKIsMoERouteAtItsDefault is the regression gate for olmoe,
// Qwen3-MoE, Qwen3-Next and gpt-oss: the emitter that grew a bias plane, a
// group phase, a gather and a scaling factor must produce the same bytes for a
// configuration that asks for none of them. Bytes, because the pre-V3
// contract includes a sum's float32 parenthesisation and the bits of a -0.0
// weight, which many non-identical rewrites would preserve.
func TestMoETopKIsMoERouteAtItsDefault(t *testing.T) {
	for _, s := range moeShapes {
		for _, norm := range []bool{true, false} {
			want, err := EmitMoETopK(s[0], s[1], norm)
			if err != nil {
				t.Fatalf("EmitMoETopK(%d, %d, %v): %v", s[0], s[1], norm, err)
			}
			got, err := EmitMoERoute(s[0], s[1], MoEGate{Norm: norm})
			if err != nil {
				t.Fatalf("EmitMoERoute(%d, %d, {Norm:%v}): %v", s[0], s[1], norm, err)
			}
			if string(got) != string(want) {
				t.Fatalf("n=%d k=%d norm=%v: MoERoute at its default is %d bytes "+
					"against MoETopK's %d -- the pre-V3 router moved",
					s[0], s[1], norm, len(got), len(want))
			}
		}
	}
}

// TestMoERouteDegenerateGroupingIsTheUngroupedKernel is Kimi-K2's
// configuration: n_group=1, topk_group=1 keeps every expert, so the emitter
// must emit the ungrouped path with no group phase. It is checked as bytes
// because a group phase that kept everything would give the same answers at
// extra cost.
func TestMoERouteDegenerateGroupingIsTheUngroupedKernel(t *testing.T) {
	for _, bias := range []bool{false, true} {
		for _, scale := range []bool{false, true} {
			base := MoEGate{Norm: true, Bias: bias, Scale: scale}
			want, err := EmitMoERoute(128, 8, base)
			if err != nil {
				t.Fatal(err)
			}
			for _, ng := range []int{1} {
				g := base
				g.NGroup, g.NGroupUsed = ng, 1
				got, err := EmitMoERoute(128, 8, g)
				if err != nil {
					t.Fatalf("bias=%v scale=%v nGroup=%d: %v", bias, scale, ng, err)
				}
				if string(got) != string(want) {
					t.Errorf("bias=%v scale=%v: nGroup=%d emitted %d bytes against the "+
						"ungrouped kernel's %d -- a group that keeps everything is "+
						"not a phase worth running", bias, scale, ng, len(got), len(want))
				}
			}
			// And the scratch must not grow for a phase that does not exist.
			if a, b := MoERouteScratch(128, 8, base), MoERouteScratch(128, 8,
				MoEGate{Norm: true, Bias: bias, Scale: scale, NGroup: 1, NGroupUsed: 1}); a != b {
				t.Errorf("bias=%v scale=%v: the degenerate grouping asks for %d scratch "+
					"words against the ungrouped %d", bias, scale, b, a)
			}
		}
	}
}

// TestMoERouteScratchIsBackwardCompatible: a route allocated for the plain
// router is exactly what the plain router always asked for, and the extra
// planes only ever append. moePlanes puts the id and weight pads first for
// this reason, so nn's existing MoERoute allocation is still right.
func TestMoERouteScratchIsBackwardCompatible(t *testing.T) {
	for _, k := range []int{1, 2, 4, 8, 10} {
		if a, b := MoERouteScratch(128, k, MoEGate{Norm: true}), MoETopKScratch(k); a != b {
			t.Errorf("k=%d: the default route asks for %d scratch words, the plain one %d", k, a, b)
		}
		pl := moePlanes(128, k, MoEGate{Norm: true, Bias: true, NGroup: 8, NGroupUsed: 4})
		if pl.id != 0 || pl.wt != MoETopKPad(k) {
			t.Errorf("k=%d: the V3 layout moved the id/weight pads to %d/%d", k, pl.id, pl.wt)
		}
	}
	// The arm64 kernel has a separate path for a plane offset past an ADD
	// immediate's twelve bits; the matrix must contain a shape that reaches it.
	far := false
	for _, cfg := range moeV3Cfgs {
		if pl := moePlanes(cfg.n, cfg.k, cfg.g); cfg.g.Grouped() && 4*pl.gs > 4095 {
			far = true
		}
	}
	if !far {
		t.Error("no configuration puts a group plane past 4095 bytes -- the NEON " +
			"emitter's register-offset path is not covered")
	}
}

// TestMoERouteRefusesShapesItCannotServe. Every one of these is a container
// that does not describe a router, so it is refused by name at emission rather
// than producing code that reads outside a group.
func TestMoERouteRefusesShapesItCannotServe(t *testing.T) {
	for _, r := range moeV3Refusals {
		if b, err := EmitMoERoute(r.n, r.k, r.g); err == nil {
			t.Errorf("%s: EmitMoERoute(%d, %d, %+v) produced %d bytes",
				r.name, r.n, r.k, r.g, len(b))
		}
	}
}

// TestMoERouteDivergencesAreRecorded: the places llama.cpp and
// modeling_deepseek.py disagree about this router are a reviewed list, and an
// empty list would mean nobody looked.
func TestMoERouteDivergencesAreRecorded(t *testing.T) {
	if len(oracle.MoEDivergences) < 3 {
		t.Fatalf("only %d divergence(s) recorded between the two authorities",
			len(oracle.MoEDivergences))
	}
	for _, d := range oracle.MoEDivergences {
		for i, s := range d {
			if s == "" {
				t.Errorf("divergence %q has an empty field %d", d[0], i)
			}
		}
		t.Logf("%s\n    llama.cpp: %s\n    HF:        %s\n    jitllm:    %s", d[0], d[1], d[2], d[3])
	}
}

// TestMoERouteReadsNothingPastTheExperts: the scores and the bias are n long
// and the kernel must not touch element n. Every phase walks whole vectors
// plus a scalar remainder, and rounding the vector count up would read what
// follows the slice (which, on a selection, can win) without faulting, so the
// gate changes what follows and requires the answer not to.
func TestMoERouteReadsNothingPastTheExperts(t *testing.T) {
	for _, cfg := range moeV3Cfgs {
		n, k := cfg.n, cfg.k
		b, err := EmittersFor(HostTier()).MoERoute(n, k, cfg.g) // this host's tier, SSE on an Atom
		if err != nil {
			t.Fatalf("%s: %v", cfg.name, err)
		}
		c, err := MapNamed(b, "moe_route_overread")
		if err != nil {
			t.Fatalf("%s: mapping: %v", cfg.name, err)
		}
		kons := MoETopKConstsFor(cfg.scale)
		in := moeV3Inputs(cfg)[0]
		var first moeRoute
		for pass, pad := range []float32{
			float32(math.Inf(1)),  // would win every selection it reached
			float32(math.Inf(-1)), // and would lose every one
		} {
			sc := append(append([]float32(nil), in.sc...), pad, pad, pad, pad, pad, pad, pad, pad)
			bias := append(append([]float32(nil), in.bias...), pad, pad, pad, pad, pad, pad, pad, pad)
			got := moeV3Call(c, sc[:n:n+8], bias[:n:n+8], k, cfg.g, kons)
			if pass == 0 {
				first = got
				continue
			}
			if d := moeDiff(first, got); d != "" {
				t.Errorf("%s: the answer moved when the bytes AFTER the %d expert(s) "+
					"changed: %s", cfg.name, n, d)
			}
		}
		c.Close()
	}
}
