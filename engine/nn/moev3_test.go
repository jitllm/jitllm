//go:build amd64 || arm64

package nn

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit/cpu"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// MoERouteJIT end to end: the generated gating function AND the generated
// router, against internal/oracle.
//
// The comparison is split at the one approximate step: the sigmoid goes through
// a minimax exp, so the scores are checked against libm with a relative bound,
// and then the kernel's own scores are fed to the oracle and every downstream
// output must match bit for bit. One combined tolerance would hide a wrong
// selection behind the exp's error bar.

// moeV3NNCfg is a configuration and the fixtures that exercise it.
type moeV3NNCfg struct {
	name string
	n, k int
	g    MoEGate
}

func moeV3NNCfgs() []moeV3NNCfg {
	bias := func(n int, f func(i int) float32) []float32 {
		b := make([]float32, n)
		for i := range b {
			b[i] = f(i)
		}
		return b
	}
	return []moeV3NNCfg{
		// DeepSeek-V3 and R1: 256 experts, 8 groups, 4 used, 8 routed,
		// norm_topk_prob, routed_scaling_factor 2.5, noaux_tc bias.
		{"deepseek-v3", 256, 8, MoEGate{
			Sigmoid: true, Norm: true, Scale: 2.5, NGroup: 8, NGroupUsed: 4,
			Bias: bias(256, func(i int) float32 { return float32(i%13)*0.05 - 0.3 }),
		}},
		// Kimi-K2: n_group = 1, topk_group = 1 -- degenerate, and it must be
		// the ungrouped path.
		{"kimi-k2", 128, 8, MoEGate{
			Sigmoid: true, Norm: true, Scale: 2.827, NGroup: 1, NGroupUsed: 1,
			Bias: bias(128, func(i int) float32 { return float32(i%7)*0.1 - 0.35 }),
		}},
		// GLM-4-MoE: sigmoid and a bias, no grouping.
		{"glm4-moe", 128, 8, MoEGate{
			Sigmoid: true, Norm: true, Scale: 1.0,
			Bias: bias(128, func(i int) float32 { return float32(i%5)*0.08 - 0.16 }),
		}},
		// A ragged one, so the tails of every phase run.
		{"ragged", 63, 5, MoEGate{
			Sigmoid: true, Norm: true, Scale: 2.5, NGroup: 7, NGroupUsed: 3,
			Bias: bias(63, func(i int) float32 { return float32(i%11)*0.06 - 0.3 }),
		}},
		// The pre-V3 router reached through the same entry point: softmax
		// gating, no bias, no groups, no scale.
		{"softmax/olmoe-shaped", 64, 8, MoEGate{Norm: true}},
		{"softmax/no-norm", 64, 8, MoEGate{}},
		// Gemma 4: softmax, renormalised, each weight times its expert's own
		// scale; ragged so the gather's tail runs.
		{"gemma4", 128, 8, MoEGate{Norm: true,
			ExpScale: bias(128, func(i int) float32 { return 0.5 + float32(i%13)*0.125 })}},
		{"gemma4/ragged", 63, 5, MoEGate{Norm: true,
			ExpScale: bias(63, func(i int) float32 { return 0.25 + float32(i%7)*0.25 })}},
	}
}

// moeV3Logits is what the router matvec hands the gating kernel. The range is
// deliberately wide: sigma saturates outside about +-90, so a large-magnitude
// fixture makes whole runs of scores tie, which is what a lane-parallel
// selection can get wrong.
func moeV3Logits(n int) []struct {
	name string
	x    []float32
} {
	rng := rand.New(rand.NewSource(int64(n)*104729 + 7))
	mk := func(f func(i int) float32) []float32 {
		p := make([]float32, n)
		for i := range p {
			p[i] = f(i)
		}
		return p
	}
	var out []struct {
		name string
		x    []float32
	}
	add := func(name string, x []float32) {
		out = append(out, struct {
			name string
			x    []float32
		}{name, x})
	}
	add("random", mk(func(int) float32 { return rng.Float32()*12 - 6 }))
	add("narrow", mk(func(int) float32 { return rng.Float32()*0.2 - 0.1 }))
	add("allEqual", mk(func(int) float32 { return 0.75 }))
	add("saturating", mk(func(i int) float32 {
		if i%3 == 0 {
			return 200
		}
		return -200
	}))
	add("ascending", mk(func(i int) float32 { return float32(i)*0.01 - 2 }))
	add("oneHot", mk(func(i int) float32 {
		if i == n/2 {
			return 9
		}
		return -9
	}))
	return out
}

// moeToOracle turns a filled route into the reference's shape.
func moeToOracle(r *MoERoute) oracle.MoERoute {
	return oracle.MoERoute{Sel: r.Sel, Ord: r.Ord, Wt: r.Wt, OWt: r.OWt, Sum: r.Sum[0]}
}

func (c moeV3NNCfg) oracleGate() oracle.MoEGate {
	return oracle.MoEGate{
		Bias: c.g.Bias, NGroup: c.g.NGroup, NGroupUsed: c.g.NGroupUsed,
		Norm: c.g.Norm, Scale: c.g.scale(), ExpScale: c.g.ExpScale,
	}
}

// moeRouteDiff reports the first place two routes disagree. The weights and the
// divisor are compared bitwise: there is no tolerance to spend downstream of
// the gating function.
func moeRouteDiff(want, got oracle.MoERoute) string {
	for i := range want.Sel {
		if want.Sel[i] != got.Sel[i] {
			return fmt.Sprintf("sel[%d] = %d, want %d (sel %v, want %v)",
				i, got.Sel[i], want.Sel[i], got.Sel, want.Sel)
		}
	}
	for i := range want.Ord {
		if want.Ord[i] != got.Ord[i] {
			return fmt.Sprintf("ord[%d] = %d, want %d (ord %v, want %v)",
				i, got.Ord[i], want.Ord[i], got.Ord, want.Ord)
		}
	}
	if a, b := math.Float32bits(want.Sum), math.Float32bits(got.Sum); a != b {
		return fmt.Sprintf("sum = %v (%#08x), want %v (%#08x)", got.Sum, b, want.Sum, a)
	}
	for i := range want.Wt {
		if a, b := math.Float32bits(want.Wt[i]), math.Float32bits(got.Wt[i]); a != b {
			return fmt.Sprintf("wt[%d] = %v (%#08x), want %v (%#08x)", i, got.Wt[i], b, want.Wt[i], a)
		}
	}
	for i := range want.OWt {
		if a, b := math.Float32bits(want.OWt[i]), math.Float32bits(got.OWt[i]); a != b {
			return fmt.Sprintf("ow[%d] = %v (%#08x), want %v (%#08x)", i, got.OWt[i], b, want.OWt[i], a)
		}
	}
	return ""
}

// moeV3Sweep runs every configuration against the reference under fault f and
// returns how many comparisons disagreed.
func moeV3Sweep(t *testing.T, f oracle.MoEFault, report func(string)) int {
	t.Helper()
	hits := 0
	for _, cfg := range moeV3NNCfgs() {
		r := NewMoERouteFor(cfg.n, cfg.k, cfg.g)
		for _, in := range moeV3Logits(cfg.n) {
			if !MoERouteJIT(in.x, cfg.g, r) {
				t.Fatalf("%s: this tier has no kernel for the router", cfg.name)
			}
			// The gating step, against libm, with a tolerance. Only the
			// sigmoid is checked here: the softmax has its own gate in this
			// package and its reference is not an elementwise function.
			if f == oracle.MoEFaultNone && cfg.g.Sigmoid {
				for i, x := range in.x {
					want := oracle.MoESigmoid(x)
					if d := abs32(r.Sc[i] - want); d > 2e-6+2e-6*abs32(want) {
						t.Fatalf("%s %s: the generated sigmoid of logit %d is %v against libm's %v",
							cfg.name, in.name, i, r.Sc[i], want)
					}
				}
			}
			// Everything downstream, against the kernel's own scores, bitwise.
			want := oracle.MoERouter(r.Sc, cfg.k, cfg.oracleGate(), f)
			d := moeRouteDiff(want, moeToOracle(r))
			if f == oracle.MoEFaultNone {
				if d != "" {
					t.Errorf("%s %s: %s", cfg.name, in.name, d)
				}
				continue
			}
			if d != "" {
				hits++
				if report != nil {
					report(cfg.name + " " + in.name + ": " + d)
					report = nil
				}
			}
		}
	}
	return hits
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// TestMoERouteJITMatchesTheOracle is the gate.
func TestMoERouteJITMatchesTheOracle(t *testing.T) {
	moeV3Sweep(t, oracle.MoEFaultNone, nil)
}

// TestMoERouteJITViolations: every defect in oracle.MoEFaults, injected into
// the reference, must make the end-to-end comparison go red.
func TestMoERouteJITViolations(t *testing.T) {
	for _, f := range oracle.MoEFaults {
		t.Run(strings.ReplaceAll(f.String(), " ", "_"), func(t *testing.T) {
			var first string
			hits := moeV3Sweep(t, f, func(s string) { first = s })
			if hits == 0 {
				t.Fatalf("%v changes nothing the gate compares -- it does not "+
					"discriminate on that axis", f)
			}
			t.Logf("%v: %d comparison(s) disagree; the first is %s", f, hits, first)
		})
	}
}

// TestMoERouteJITRunsTheGatingItWasAskedFor checks the gating function
// actually ran and was the one requested: the scores are not the logits,
// sigmoid and softmax disagree, and a sigmoid score is in [0,1] while the
// softmax sums to one.
func TestMoERouteJITRunsTheGatingItWasAskedFor(t *testing.T) {
	const n, k = 64, 4
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(i)*0.13 - 4
	}
	sig := MoEGate{Sigmoid: true, Norm: true}
	sm := MoEGate{Norm: true}

	rs := NewMoERouteFor(n, k, sig)
	if !MoERouteJIT(x, sig, rs) {
		t.Fatal("no kernel for the sigmoid-gated router")
	}
	rm := NewMoERouteFor(n, k, sm)
	if !MoERouteJIT(x, sm, rm) {
		t.Fatal("no kernel for the softmax-gated router")
	}

	same := true
	for i := range x {
		if rs.Sc[i] != x[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("the scores are the logits -- no gating kernel ran")
	}
	var smSum float32
	for i := range x {
		if rs.Sc[i] < 0 || rs.Sc[i] > 1 {
			t.Fatalf("sigmoid score %d is %v, outside [0,1]", i, rs.Sc[i])
		}
		smSum += rm.Sc[i]
	}
	if d := abs32(smSum - 1); d > 1e-5 {
		t.Errorf("the softmax-gated scores sum to %v, not 1", smSum)
	}
	differs := false
	for i := range x {
		if rs.Sc[i] != rm.Sc[i] {
			differs = true
			break
		}
	}
	if !differs {
		t.Error("sigmoid and softmax produced the same scores -- the flag does not reach the kernel")
	}
}

// TestMoERouteJITIsTheOldRouterAtItsDefault: a softmax gate with no bias, no
// groups and no scale must give MoETopK32JIT's answer bit for bit on the same
// probabilities. It is the behavioural half of jit/cpu's byte-identity gate,
// taken through the entry point engine/model/moe.go will keep using.
func TestMoERouteJITIsTheOldRouterAtItsDefault(t *testing.T) {
	const n, k = 64, 6
	for _, norm := range []bool{true, false} {
		x := make([]float32, n)
		rng := rand.New(rand.NewSource(int64(n) + 1))
		for i := range x {
			x[i] = rng.Float32()*8 - 4
		}
		g := MoEGate{Norm: norm}
		got := NewMoERouteFor(n, k, g)
		if !MoERouteJIT(x, g, got) {
			t.Fatal("no kernel")
		}
		// The old path: softmax the caller's own buffer, then MoETopK32JIT.
		p := append([]float32(nil), x...)
		Softmax32JIT(p, n)
		want := NewMoERoute(k)
		if !MoETopK32JIT(p, norm, want) {
			t.Fatal("no kernel for the pre-V3 router")
		}
		if d := moeRouteDiff(moeToOracle(want), moeToOracle(got)); d != "" {
			t.Errorf("norm=%v: the V3 entry point at its default differs from MoETopK32JIT: %s", norm, d)
		}
	}
}

// TestMoERouteJITRefusesARouteBuiltForAnotherGate: a route is sized for one
// configuration, and handing it to another is a programming error that must
// stop rather than read a plane nobody allocated.
func TestMoERouteJITRefusesARouteBuiltForAnotherGate(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("a route built by NewMoERoute reached MoERouteJIT without a panic")
		}
	}()
	g := MoEGate{Sigmoid: true, Norm: true, NGroup: 8, NGroupUsed: 4, Scale: 2.5}
	MoERouteJIT(make([]float32, 64), g, NewMoERoute(4))
}

// TestMoERouterFloorsTheDivisor drives the router with logits so negative that
// every selected expert's sigmoid underflows to exactly zero, and requires
// finite, zero weights. The divisor guard is DeepSeek's sum + 1e-20 rather than
// llama.cpp's clamp (see AGENTS.md, Open questions). The second half checks an
// ordinary selection still renormalises to 1, so a floor that is too large
// would fail.
func TestMoERouterFloorsTheDivisor(t *testing.T) {
	const nExp, k = 8, 4
	g := oracle.MoEGate{NGroup: 1, NGroupUsed: 1, Norm: true, Scale: 1}

	// sigmoid(-800) is exactly 0 in float32 -- and in float64 -- so every
	// selected weight is zero and the divisor is the sum of zeros.
	dead := make([]float32, nExp)
	for i := range dead {
		dead[i] = float32(1 / (1 + math.Exp(800)))
	}
	for i, v := range dead {
		if v != 0 {
			t.Fatalf("expert %d scored %v, not 0 -- the underflow this gate needs did not happen", i, v)
		}
	}
	r := oracle.MoERouter(dead, k, g, oracle.MoEFaultNone)
	if len(r.Wt) != k {
		t.Fatalf("router returned %d weights, want %d", len(r.Wt), k)
	}
	for i, w := range r.Wt {
		if math.IsNaN(float64(w)) || math.IsInf(float64(w), 0) {
			t.Fatalf("weight %d is %v: the divisor has no floor, so an "+
				"all-underflowed selection is 0/0", i, w)
		}
		if w != 0 {
			t.Errorf("weight %d is %v, want exactly 0: both authorities return "+
				"zero weights for an all-zero selection", i, w)
		}
	}

	// An ordinary selection is untouched: the floor is 1e-20 against sums of
	// order 1.
	rng := rand.New(rand.NewSource(7))
	ord := make([]float32, nExp)
	for i := range ord {
		ord[i] = float32(1 / (1 + math.Exp(-rng.NormFloat64())))
	}
	got := oracle.MoERouter(ord, k, g, oracle.MoEFaultNone)
	var sum float32
	for _, w := range got.Wt {
		sum += w
	}
	// The k weights renormalise to 1 at Scale 1; a floor big enough to matter
	// would show here as a sum below 1.
	if d := math.Abs(float64(sum) - 1); d > 1e-6 {
		t.Errorf("an ordinary selection renormalises to %v, not 1 (off by %.3e) "+
			"-- the floor is participating where it must not", sum, d)
	}
	t.Logf("all-underflowed selection -> weights %v; ordinary selection sums to %v",
		r.Wt, sum)
}

// TestRouterFloorMatchesTheKernel: internal/oracle, the device IR and the host
// kernels' constant block must all guard the divide with the same number. The
// host's is checked through the block word the kernel reads, not only through
// the Go constant.
func TestRouterFloorMatchesTheKernel(t *testing.T) {
	host := cpu.MoETopKConsts()
	const wordsBefore = 15 // moeSumFloor is byte 60
	if len(host) <= wordsBefore {
		t.Fatalf("the host constant block is %d words: it carries no floor", len(host))
	}
	got := []struct {
		name string
		v    float32
	}{
		{"internal/oracle", oracle.MoESumFloorForTest},
		{"jit/gpu/kernels", kernelMoESumFloorForTest},
		{"jit/cpu constant", cpu.MoESumFloor()},
		{"jit/cpu block word", host[wordsBefore]},
	}
	for _, g := range got[1:] {
		if g.v != got[0].v {
			t.Errorf("%s floors the divisor at %v, %s at %v",
				got[0].name, got[0].v, g.name, g.v)
		}
	}
	if got[0].v <= 0 {
		t.Fatalf("the floor is %v: a non-positive floor does not guard 0/0", got[0].v)
	}
	t.Logf("all four agree at %v", got[0].v)
}

// kernelMoESumFloorForTest is jit/gpu/kernels' constant, named here so the
// comparison above reads as one line rather than an import in the middle of it.
const kernelMoESumFloorForTest = oldkernels.MoESumFloor
