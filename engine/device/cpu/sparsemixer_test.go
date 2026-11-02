//go:build amd64 || arm64

package cpu

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/internal/oracle"
)

// The sparsemixer weights kernel on every tier this host executes -- AVX2 and
// SSE on amd64, NEON on arm64 -- held to internal/oracle.SparseMixer behind
// the plain top-k kernel at Norm off, which is how nn runs it.

// smShapes is the expert counts: 16 (Phi-3.5-MoE), the smallest legal one,
// either side of every vector width, and a few dozen.
var smShapes = []int{2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64}

// smEps is Phi-3.5-MoE's router_jitter_noise.
const smEps = 0.01

// smInputs are the logit vectors for one expert count. Random logits leave
// every other expert masked, so the cases that keep several are built: a
// cluster within the threshold of the maximum, a cluster within it of the
// second, a tie for the maximum, a negative maximum, and zeros (a 0/0 gap,
// which the reference keeps).
func smInputs(n int) []struct {
	name string
	s    []float32
} {
	rng := rand.New(rand.NewSource(int64(n)*104729 + 7))
	mk := func(f func(i int) float32) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = f(i)
		}
		return s
	}
	rnd := func(int) float32 { return float32(rng.NormFloat64()) }
	var out []struct {
		name string
		s    []float32
	}
	add := func(name string, s []float32) {
		out = append(out, struct {
			name string
			s    []float32
		}{name, s})
	}
	add("random", mk(rnd))
	add("wide", mk(func(int) float32 { return float32(rng.NormFloat64() * 8) }))
	near := mk(rnd)
	top := 5 + float32(rng.Float64())
	for i := 0; i < n; i += 3 {
		near[i] = top * (1 - float32(rng.Float64())*0.03) // within 2*eps of the top, or just past it
	}
	add("cluster", near)
	second := mk(func(int) float32 { return float32(rng.NormFloat64()) - 4 })
	second[n-1] = 3
	for i := 0; i < n-1; i += 2 {
		second[i] = 1.5 * (1 - float32(rng.Float64())*0.025)
	}
	add("clusterBehind", second)
	tie := mk(rnd)
	tie[0], tie[n-1] = 4, 4
	add("tie", tie)
	add("negative", mk(func(int) float32 { return -1 - float32(rng.Float64()) }))
	zeros := mk(func(i int) float32 { return -float32(i % 3) })
	add("zeros", zeros)
	return out
}

// smKernel is one tier's pair: the plain top-k at Norm off and the
// sparsemixer behind it.
type smKernel struct {
	name       string
	route, mix *Code
}

func smKernels(t *testing.T, n int) ([]smKernel, func()) {
	t.Helper()
	tiers := []Tier{HostTier()}
	if runtime.GOARCH == "amd64" && HostTier() != TierSSE {
		tiers = append(tiers, TierSSE)
	}
	var out []smKernel
	var codes []*Code
	for _, tr := range tiers {
		e := EmittersFor(tr)
		rb, err := e.MoERoute(n, 2, MoEGate{})
		if err != nil {
			t.Fatalf("%v MoERoute(%d, 2): %v", tr, n, err)
		}
		mb, err := e.SparseMixer(n)
		if err != nil {
			t.Fatalf("%v SparseMixer(%d): %v", tr, n, err)
		}
		if tr == TierSSE && KernelTier(mb) != TierSSE {
			t.Fatalf("the SSE sparsemixer declares tier %v", KernelTier(mb))
		}
		rc, err := MapNamed(rb, "sm_route")
		if err != nil {
			t.Fatal(err)
		}
		mc, err := MapNamed(mb, "sm_mix")
		if err != nil {
			t.Fatal(err)
		}
		codes = append(codes, rc, mc)
		out = append(out, smKernel{fmt.Sprint(tr), rc, mc})
	}
	return out, func() {
		for _, c := range codes {
			c.Close()
		}
	}
}

// smCall runs the pair on s, as nn.MoERouteJIT does for a sparsemixer gate.
func smCall(k smKernel, s []float32) oracle.MoERoute {
	r := oracle.MoERoute{Sel: make([]int32, 2), Ord: make([]int32, 2),
		Wt: make([]float32, 2), OWt: make([]float32, 2)}
	sum := make([]float32, 1)
	scr := make([]float32, MoERouteScratch(len(s), 2, MoEGate{}))
	for i := range scr {
		scr[i] = math.Float32frombits(0xDEADBEEF)
	}
	kons := MoETopKConstsFor(1)
	k.route.Call(&Args{
		Q32: &s[0], Scr: (*byte)(unsafe.Pointer(&kons[0])), Scratch: (*byte)(unsafe.Pointer(&scr[0])),
		ASum: &r.Sel[0], AHalfSum: &r.Ord[0], Out: &r.Wt[0], Out2: &r.OWt[0], AScale: &sum[0],
	})
	mix := SparseMixerConsts(smEps)
	k.mix.Call(&Args{
		Q32: &s[0], Scr: (*byte)(unsafe.Pointer(&mix[0])),
		ASum: &r.Sel[0], AHalfSum: &r.Ord[0], Out: &r.Wt[0], Out2: &r.OWt[0],
	})
	r.Sum = sum[0]
	return r
}

// smDiff compares a route with the reference: ids exactly, weights to the
// relative error of the kernels' exp: a degree-5 polynomial on |r| <= ln2/2,
// whose truncation is r^6/720 ~ 2.4e-6 (exp.go).
func smDiff(want, got oracle.MoERoute) string {
	for i := 0; i < 2; i++ {
		if want.Sel[i] != got.Sel[i] || want.Ord[i] != got.Ord[i] {
			return fmt.Sprintf("ids sel %v ord %v, want sel %v ord %v", got.Sel, got.Ord, want.Sel, want.Ord)
		}
		for _, p := range [][2]float32{{want.Wt[i], got.Wt[i]}, {want.OWt[i], got.OWt[i]}} {
			if d := math.Abs(float64(p[1]-p[0])) / math.Abs(float64(p[0])); !(d <= 1e-5) {
				return fmt.Sprintf("weights wt %v ow %v, want wt %v ow %v", got.Wt, got.OWt, want.Wt, want.OWt)
			}
		}
	}
	return ""
}

func TestSparseMixerMatchesTheOracle(t *testing.T) {
	kept := 0
	for _, n := range smShapes {
		ks, done := smKernels(t, n)
		for _, in := range smInputs(n) {
			want := oracle.SparseMixer(in.s, smEps, oracle.SparseMixerFaultNone)
			if want.Wt[0] < 0.99 || want.Wt[1] < 0.99 {
				kept++ // a softmax over more than the maximum: the mask mattered
			}
			for _, k := range ks {
				if d := smDiff(want, smCall(k, in.s)); d != "" {
					t.Errorf("%s n=%d %s: %s", k.name, n, in.name, d)
				}
			}
		}
		done()
	}
	// Random logits mask every other expert, where every weight is 1 and the
	// mask is invisible; the built cases must reach the softmax.
	if kept < len(smShapes) {
		t.Fatalf("only %d inputs kept more than the maximum -- the masks were barely exercised", kept)
	}
}

// TestSparseMixerViolations runs each plausible defect in the reference and
// requires the comparison to fail on some input: a fault the gate cannot see
// is one it certifies nothing about.
func TestSparseMixerViolations(t *testing.T) {
	for _, f := range []oracle.SparseMixerFault{oracle.SparseMixerFaultNoMask,
		oracle.SparseMixerFaultKeepFirst, oracle.SparseMixerFaultRenormalised,
		oracle.SparseMixerFaultAbsFactor} {
		t.Run(fmt.Sprint(f), func(t *testing.T) {
			hits := 0
			for _, n := range smShapes {
				ks, done := smKernels(t, n)
				for _, in := range smInputs(n) {
					bad := oracle.SparseMixer(in.s, smEps, f)
					if smDiff(bad, smCall(ks[0], in.s)) != "" {
						hits++
					}
				}
				done()
			}
			if hits == 0 {
				t.Fatalf("fault %d: the gate cannot see it", f)
			}
			t.Logf("fault %d caught on %d inputs", f, hits)
		})
	}
}
