package composer

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// TestComposedCPUSampleTypical holds the draws with typical-p, XTC,
// top-n-sigma and a dynamic temperature -- a top-k
// draw's walk, the cut with no top-k and its top-k variant past the walk's
// bound (graph.SampleCut's Typical forms) -- on the CPU device to their
// definition (internal/oracle): llama.cpp's locally typical cut, the
// candidates whose surprise is nearest the entropy until the mass before one
// passes typical-p, then top-p and min-p over the survivors renormalized, then
// XTC's cut of all but the least of those past its threshold; top-n-sigma,
// the cut's forms', masks the row before them all. Every decisive draw is the
// definition's, and typical-p moves the token on a share of them, XTC,
// top-n-sigma and the dynamic temperature each on a share of theirs: the gate
// sees every cut.
func TestComposedCPUSampleTypical(t *testing.T) {
	const n, K = 300, 16
	type arm struct {
		name       string
		g          *graph.Graph
		walk, topK bool
	}
	for _, a := range []arm{
		{"walk", (&graph.SampleWalk{SampleTopK: graph.SampleTopK{NVocab: n, K: K}}).Build(), true, true},
		{"cut", (&graph.SampleCut{NVocab: n, Typical: true}).Build(), false, false},
		{"cutk", (&graph.SampleCut{NVocab: n, TopK: true, Typical: true}).Build(), false, true},
	} {
		t.Run(a.name, func(t *testing.T) {
			g := a.g
			if err := graph.Verify(g, graph.Facts{}); err != nil {
				t.Fatal(err)
			}
			in := map[string]graph.ValueID{}
			for _, v := range g.In {
				in[g.Values[v].Name] = v
			}
			tokV := g.Out[0]
			_, codes, err := Default().Form(Request{Graph: g, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: 1, Page: 64, Keys: 1}, Device: cpuDevice(hostArm())})
			if err != nil {
				t.Fatal(err)
			}
			ids := make([]uint32, n)
			for i := range ids {
				ids[i] = uint32(i)
			}
			f := func(x float64) uint32 { return math.Float32bits(float32(x)) }
			inf := math.Inf(1)
			rng := rand.New(rand.NewSource(43))
			decisive, cases, moved, xtcs, xmoved, sigs, smoved, dyns, dmoved := 0, 0, 0, 0, 0, 0, 0, 0, 0
			for c := range 400 {
				l := make([]float32, n)
				for i := range l {
					l[i] = float32(rng.NormFloat64() * 2)
				}
				k := n
				switch {
				case a.walk:
					k = 2 + rng.Intn(K-1)
				case a.topK:
					k = 2 + rng.Intn(n-2)
				}
				invT := float64(float32(1 / (0.4 + 1.4*rng.Float64())))
				topp, minp := inf, 0.0
				switch c % 4 {
				case 0:
					topp = float64(float32(0.5 + 0.49*rng.Float64()))
				case 1:
					minp = float64(float32(0.05 * rng.Float64()))
				}
				typ := float64(float32(0.2 + 0.75*rng.Float64()))
				xtc := inf
				if c%3 == 0 {
					xtc = float64(float32(0.02 + 0.2*rng.Float64()))
				}
				dyn := [2]float64{}
				if c%5 == 2 {
					dyn = [2]float64{float64(float32(0.2 + 0.8*rng.Float64())), float64(float32(0.5 + 1.5*rng.Float64()))}
				}
				// Top-n-sigma is the cut's forms' alone; the walk never reads it.
				ns, lo, sm := 0.0, l, inf
				if !a.walk && c%4 == 1 {
					ns = float64(float32(0.5 + 1.5*rng.Float64()))
					lo, sm = oracle.TopNSigma(l, ns)
				}
				u := float64(float32(rng.Float64()))
				lw := make([]uint32, n)
				for i, v := range l {
					lw[i] = math.Float32bits(v)
				}
				vals := map[graph.ValueID][]uint32{in["logits"]: lw, in["sample.window"]: make([]uint32, n),
					in["sample.penalty"]: {f(1), 0, 0}, tokV: {0}}
				var want, plain, noX, noS, noD uint32
				var margin float64
				switch {
				case a.walk:
					want, margin = oracle.SampleWalk(l, K, k, invT, topp, minp, typ, xtc, dyn, u)
					plain, _ = oracle.SampleWalk(l, K, k, invT, topp, minp, inf, xtc, dyn, u)
					noX, _ = oracle.SampleWalk(l, K, k, invT, topp, minp, typ, inf, dyn, u)
					noD, _ = oracle.SampleWalk(l, K, k, invT, topp, minp, typ, xtc, [2]float64{}, u)
					vals[in["sample.walk"]] = []uint32{f(float64(k)), f(invT), f(topp), f(minp), f(u), f(typ), f(xtc), 0, f(dyn[0]), f(dyn[1])}
				case a.topK:
					want, margin = oracle.SampleWalk(lo, n, k, invT, topp, minp, typ, xtc, dyn, u)
					plain, _ = oracle.SampleWalk(lo, n, k, invT, topp, minp, inf, xtc, dyn, u)
					noX, _ = oracle.SampleWalk(lo, n, k, invT, topp, minp, typ, inf, dyn, u)
					noS, _ = oracle.SampleWalk(l, n, k, invT, topp, minp, typ, xtc, dyn, u)
					noD, _ = oracle.SampleWalk(lo, n, k, invT, topp, minp, typ, xtc, [2]float64{}, u)
					vals[in["sample.ids"]] = append([]uint32(nil), ids...)
					vals[in["sample.cut"]] = []uint32{f(invT), f(topp), f(minp), f(u), f(float64(k)), f(ns), f(typ), f(xtc), 0, f(dyn[0]), f(dyn[1])}
				default:
					want, margin = oracle.SampleCut(lo, nil, 1, float32(invT), topp, minp, typ, xtc, dyn, u)
					plain, _ = oracle.SampleCut(lo, nil, 1, float32(invT), topp, minp, inf, xtc, dyn, u)
					noX, _ = oracle.SampleCut(lo, nil, 1, float32(invT), topp, minp, typ, inf, dyn, u)
					noS, _ = oracle.SampleCut(l, nil, 1, float32(invT), topp, minp, typ, xtc, dyn, u)
					noD, _ = oracle.SampleCut(lo, nil, 1, float32(invT), topp, minp, typ, xtc, [2]float64{}, u)
					vals[in["sample.ids"]] = append([]uint32(nil), ids...)
					vals[in["sample.cut"]] = []uint32{f(invT), f(topp), f(minp), f(u), f(ns), f(typ), f(xtc), 0, f(dyn[0]), f(dyn[1])}
				}
				if err := runCPUValues(hostArm(), codes, g, vals, [3][]uint32{}); err != nil {
					t.Fatal(err)
				}
				margin = min(margin, sm)
				cases++
				if margin < 1e-4 {
					continue
				}
				decisive++
				if want != plain {
					moved++
				}
				if dyn[0] > 0 {
					dyns++
					if want != noD {
						dmoved++
					}
				}
				if ns > 0 {
					sigs++
					if want != noS {
						smoved++
					}
				}
				if !math.IsInf(xtc, 1) {
					xtcs++
					if want != noX {
						xmoved++
					}
				}
				if got := vals[tokV][0]; got != want {
					t.Fatalf("case %d (k %d, 1/T %.3f, top-p %.4f, min-p %.4f, typical-p %.4f, u %.6f): drew %d, want %d (margin %.3g)",
						c, k, invT, topp, minp, typ, u, got, want, margin)
				}
			}
			if decisive < cases*7/10 {
				t.Fatalf("%d of %d draws decisive: the comparison sees too little", decisive, cases)
			}
			if moved < decisive/10 || xmoved < xtcs/10 || smoved < sigs/10 || dmoved < dyns/10 {
				t.Fatalf("typical-p moved %d of %d decisive draws, XTC %d of %d, top-n-sigma %d of %d, the dynamic temperature %d of %d: "+
					"the gate cannot see a cut", moved, decisive, xmoved, xtcs, smoved, sigs, dmoved, dyns)
			}
			t.Logf("%d of %d draws decisive and right, %d of them moved by typical-p, %d of %d by XTC", decisive, cases, moved, xmoved, xtcs)
		})
	}
}
