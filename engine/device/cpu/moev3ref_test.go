//go:build amd64 || arm64

package cpu

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/internal/oracle"
)

// The DeepSeek-V3 router gate's fixtures and its oracle adapter, shared by the
// amd64 and arm64 arms so the fixture table is written once.
//
// The oracle is internal/oracle (nn's end-to-end gate uses the same one). Its
// derivation deliberately differs from the kernel's: oracle.MoERouter builds
// the kept-group set and masks from it, while the kernel finds a threshold
// pair and re-derives the set with two compares.

// moeV3Cfg is one baked router configuration, with the runtime value behind
// MoEGate.Scale beside it.
type moeV3Cfg struct {
	name  string
	n, k  int
	g     MoEGate
	scale float32
}

// moeV3Cfgs is the matrix. The first four are real: DeepSeek-V3 and R1 ship
// 256 experts in 8 groups with 4 used, 8 routed, a correction bias and
// routed_scaling_factor 2.5; Kimi-K2 ships n_group=1/topk_group=1, which is the
// degenerate grouping and must come out as the ungrouped path; GLM-4-MoE ships
// the bias and scale with no grouping at all.
//
// The rest cover ragged tails: 30 experts in 5 groups of 6 is ragged in the
// group span, 63 in 7 groups of 9 in the span and the total, and 12 groups in
// the group vector itself.
var moeV3Cfgs = []moeV3Cfg{
	{"deepseek-v3", 256, 8, MoEGate{Norm: true, Bias: true, NGroup: 8, NGroupUsed: 4, Scale: true}, 2.5},
	{"deepseek-v3/nonorm", 256, 8, MoEGate{Bias: true, NGroup: 8, NGroupUsed: 4, Scale: true}, 2.5},
	{"kimi-k2/degenerate", 128, 8, MoEGate{Norm: true, Bias: true, NGroup: 1, NGroupUsed: 1, Scale: true}, 2.827},
	{"glm4-moe/ungrouped", 128, 8, MoEGate{Norm: true, Bias: true, Scale: true}, 1.5},

	{"bias-only", 64, 4, MoEGate{Norm: true, Bias: true}, 1},
	{"scale-only", 64, 4, MoEGate{Norm: true, Scale: true}, 2.5},
	{"group-only", 64, 4, MoEGate{Norm: true, NGroup: 4, NGroupUsed: 2}, 1},
	{"allgroups", 64, 6, MoEGate{Norm: true, Bias: true, NGroup: 4, NGroupUsed: 4, Scale: true}, 0.5},
	{"onegroupused", 64, 4, MoEGate{Norm: true, Bias: true, NGroup: 8, NGroupUsed: 1, Scale: true}, 2},
	{"pairgroups", 32, 4, MoEGate{Norm: true, Bias: true, NGroup: 16, NGroupUsed: 3, Scale: true}, 1.25},

	{"ragged/span", 30, 4, MoEGate{Norm: true, Bias: true, NGroup: 5, NGroupUsed: 2, Scale: true}, 2.5},
	{"ragged/both", 63, 5, MoEGate{Norm: true, Bias: true, NGroup: 7, NGroupUsed: 3, Scale: true}, 2.5},
	{"ragged/groups", 60, 4, MoEGate{Norm: true, Bias: true, NGroup: 12, NGroupUsed: 5, Scale: true}, 2.5},
	{"ragged/total", 65, 3, MoEGate{Norm: true, Bias: true, Scale: true}, 2.5},
	// Ragged and plain: the only shape where the selection walks the caller's
	// own buffer with a tail (a bias or group selects over a scratch plane),
	// so these are what let TestMoERouteReadsNothingPastTheExperts fire.
	{"ragged/plain", 63, 5, MoEGate{Norm: true, Scale: true}, 2.5},

	// Wide enough that the group planes sit past arm64's 12-bit add
	// immediate, so the NEON kernel's register-offset branch is exercised
	// (no shipped expert count reaches it yet).
	{"wide", 1024, 8, MoEGate{Norm: true, Bias: true, NGroup: 8, NGroupUsed: 4, Scale: true}, 2.5},
	{"ragged/plain-nonorm", 45, 3, MoEGate{}, 1},
	{"ragged/k1", 33, 1, MoEGate{Norm: true, Bias: true, NGroup: 3, NGroupUsed: 2, Scale: true}, 2.5},
	{"groupsize1", 8, 2, MoEGate{Norm: true, Bias: true, NGroup: 8, NGroupUsed: 3, Scale: true}, 2.5},

	// Gemma 4's router: softmax, renormalised, then each weight times its
	// expert's own scale, gathered by id. Ragged so the gather's tail runs, and
	// combined with everything else so the multiply's place is pinned.
	{"gemma4", 128, 8, MoEGate{Norm: true, ExpScale: true}, 1},
	{"expscale/ragged", 63, 5, MoEGate{Norm: true, ExpScale: true}, 1},
	{"expscale/all", 65, 3, MoEGate{Norm: true, Bias: true, NGroup: 5, NGroupUsed: 2, Scale: true, ExpScale: true}, 2.5},
	{"expscale/k1", 33, 1, MoEGate{ExpScale: true}, 1},
}

// moeExpScale is the per-expert scale every ExpScale fixture runs with:
// distinct per expert, so a gather that takes a neighbour's shows.
func moeExpScale(n int) []float32 {
	s := make([]float32, n)
	for e := range s {
		s[e] = 0.5 + float32((e*37)%13)/8
	}
	return s
}

// oracleGate is the same configuration in the reference's vocabulary.
func (c moeV3Cfg) oracleGate(bias []float32) oracle.MoEGate {
	og := oracle.MoEGate{
		NGroup: c.g.NGroup, NGroupUsed: c.g.NGroupUsed, Norm: c.g.Norm, Scale: 1,
	}
	if c.g.Bias {
		og.Bias = bias
	}
	if c.g.Scale {
		og.Scale = c.scale
	}
	if c.g.ExpScale {
		og.ExpScale = moeExpScale(c.n)
	}
	return og
}

type moeV3Case struct {
	name     string
	sc, bias []float32
}

// moeV3Inputs is the score and bias vectors for one configuration.
//
// The scores are sigmoid outputs: finite, non-negative and often exactly equal
// at 0 or 1, hence the tie fixtures and no infinity fixture. The adversarial
// ones target the phases: topTwo separates best-one from best-two group
// scoring, biasFlips lets the bias reorder the selection (so weights taken from
// the biased plane show), and groupHot puts every large score in groups the
// masking must drop.
func moeV3Inputs(c moeV3Cfg) []moeV3Case {
	n, ng := c.n, c.g.NGroup
	if ng < 1 {
		ng = 1
	}
	sz := n / ng
	rng := rand.New(rand.NewSource(int64(n)*7919 + int64(c.k)*31 + int64(ng)))
	mk := func(f func(i int) float32) []float32 {
		p := make([]float32, n)
		for i := range p {
			p[i] = f(i)
		}
		return p
	}
	sig := func(f func(i int) float32) []float32 {
		return mk(func(i int) float32 { return oracle.MoESigmoid(f(i)) })
	}
	zero := mk(func(int) float32 { return 0 })

	var out []moeV3Case
	add := func(name string, sc, bias []float32) {
		out = append(out, moeV3Case{name, sc, bias})
	}

	randBias := mk(func(int) float32 { return rng.Float32()*0.4 - 0.2 })
	add("random", sig(func(int) float32 { return rng.Float32()*8 - 4 }), randBias)
	add("random/nobias", sig(func(int) float32 { return rng.Float32()*8 - 4 }), zero)
	// Every score equal: the selection is decided entirely by index, every
	// group score is equal, and the first NGroupUsed groups must be kept.
	add("allEqual", mk(func(int) float32 { return 0.5 }), zero)
	add("allEqual/bias", mk(func(int) float32 { return 0.5 }), randBias)
	// Saturation: sigma of a large magnitude is exactly 1 or exactly 0, so
	// whole runs tie.
	add("saturated", sig(func(i int) float32 {
		if i%3 == 0 {
			return 120
		}
		return -120
	}), zero)
	add("allZero", zero, zero)

	// One group holds everything; the masking has to keep it and drop the rest.
	add("groupHot", mk(func(i int) float32 {
		if i/sz == ng/2 {
			return 0.9
		}
		return 0.01
	}), randBias)

	// Separates top-1 from top-2 group scoring: group A has the highest single
	// score and nothing else, group B has two mediums.
	add("topTwo", mk(func(i int) float32 {
		g, j := i/sz, i%sz
		switch {
		case g == 0 && j == 0:
			return 0.90
		case g == 1 && (j == 0 || j == 1):
			return 0.60
		}
		return 0.01 + float32(i)*1e-4
	}), zero)

	// A bias large enough to reorder the selection: the unbiased scores
	// ascend, the bias descends harder.
	add("biasFlips",
		mk(func(i int) float32 { return 0.1 + 0.8*float32(i)/float32(n) }),
		mk(func(i int) float32 { return 0.9 - 1.8*float32(i)/float32(n) }))

	// Ties placed where a fold crosses a lane, a vector, a group and the end.
	pairs := mk(func(int) float32 { return rng.Float32() * 0.1 })
	for _, at := range []int{0, 3, 7, 8, sz - 1, sz, 2*sz - 1, n - 2} {
		if at >= 0 && at+1 < n {
			pairs[at+1] = pairs[at]
		}
	}
	add("tiesAtBoundaries", pairs, randBias)

	// Two values only: every element ties with half the model.
	add("twoValues", mk(func(i int) float32 {
		if i%2 == 0 {
			return 0.5
		}
		return 0.25
	}), zero)
	return out
}

// moeV3Call runs a mapped kernel in the ABI nn uses.
func moeV3Call(c *Code, sc, bias []float32, k int, g MoEGate, consts []float32) moeRoute {
	r := moeRoute{
		sel: make([]int32, k), ord: make([]int32, k),
		wt: make([]float32, k), ow: make([]float32, k),
	}
	sum := make([]float32, 1)
	scr := make([]float32, MoERouteScratch(len(sc), k, g))
	// Poisoned, not fresh: every plane the kernel reads must be one it wrote
	// this call, since the engine reuses scratch across tokens.
	for i := range scr {
		scr[i] = math.Float32frombits(0xDEADBEEF)
	}
	args := Args{
		Q32:      &sc[0],
		Q2:       &bias[0],
		Scr:      (*byte)(unsafe.Pointer(&consts[0])),
		Scratch:  (*byte)(unsafe.Pointer(&scr[0])),
		ASum:     &r.sel[0],
		AHalfSum: &r.ord[0],
		Out:      &r.wt[0],
		Out2:     &r.ow[0],
		AScale:   &sum[0],
	}
	if g.ExpScale {
		// Exactly n floats: the gather must read nothing past the experts.
		es := moeExpScale(len(sc))
		args.AScale2 = &es[0]
	}
	c.Call(&args)
	r.sum = sum[0]
	return r
}

// moeFromOracle is the reference's route in the shape moeDiff compares.
func moeFromOracle(o oracle.MoERoute) moeRoute {
	return moeRoute{sel: o.Sel, ord: o.Ord, wt: o.Wt, ow: o.OWt, sum: o.Sum}
}

// moeV3Kernel maps one tier's kernel for a configuration.
type moeV3Kernel struct {
	name string
	c    *Code
	kons []float32
}

// moeV3RunAll is the body of every tier's agreement arm: for every
// configuration and every fixture, the kernel must equal the reference bit for
// bit. fault selects the reference's defect; with MoEFaultNone it is the gate,
// and with a defect it counts the fixtures on which the comparison goes red.
func moeV3RunAll(t *testing.T, mk func(*testing.T, moeV3Cfg) ([]moeV3Kernel, func()),
	fault oracle.MoEFault, report func(string)) int {
	t.Helper()
	hits := 0
	for _, cfg := range moeV3Cfgs {
		ks, done := mk(t, cfg)
		for _, in := range moeV3Inputs(cfg) {
			want := moeFromOracle(oracle.MoERouter(in.sc, cfg.k, cfg.oracleGate(in.bias), fault))
			for _, kern := range ks {
				got := moeV3Call(kern.c, in.sc, in.bias, cfg.k, cfg.g, kern.kons)
				d := moeDiff(want, got)
				if fault == oracle.MoEFaultNone {
					if d != "" {
						t.Errorf("%s %s %s: %s", kern.name, cfg.name, in.name, d)
					}
					continue
				}
				if d != "" {
					hits++
					if report != nil {
						report(fmt.Sprintf("%s %s %s: %s", kern.name, cfg.name, in.name, d))
					}
					report = nil
				}
			}
		}
		done()
	}
	return hits
}

// moeV3Refusals is every shape the emitters must refuse by name. Each is a
// config error, which nn reports naming the router.
var moeV3Refusals = []struct {
	name string
	n, k int
	g    MoEGate
}{
	{"more groups used than exist", 64, 4, MoEGate{NGroup: 4, NGroupUsed: 5}},
	{"no groups used", 64, 4, MoEGate{NGroup: 4}},
	{"a negative group count used", 64, 4, MoEGate{NGroup: 4, NGroupUsed: -1}},
	{"experts that do not divide into groups", 65, 4, MoEGate{NGroup: 8, NGroupUsed: 2}},
	{"k above n, grouped", 4, 8, MoEGate{NGroup: 2, NGroupUsed: 1}},
}
