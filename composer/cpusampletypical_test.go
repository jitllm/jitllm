package composer

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// TestComposedCPUSampleTypical holds the draws with typical-p -- a top-k
// draw's walk, the cut with no top-k and its top-k variant past the walk's
// bound (graph.SampleCut's Typical forms) -- on the CPU device to their
// definition (internal/oracle): llama.cpp's locally typical cut, the
// candidates whose surprise is nearest the entropy until the mass before one
// passes typical-p, then top-p and min-p over the survivors renormalized.
// Every decisive draw is the definition's, and typical-p moves the token on a
// share of them: the gate sees the cut.
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
			decisive, cases, moved := 0, 0, 0
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
				u := float64(float32(rng.Float64()))
				lw := make([]uint32, n)
				for i, v := range l {
					lw[i] = math.Float32bits(v)
				}
				vals := map[graph.ValueID][]uint32{in["logits"]: lw, in["sample.window"]: make([]uint32, n),
					in["sample.penalty"]: {f(1), 0, 0}, tokV: {0}}
				var want, plain uint32
				var margin float64
				switch {
				case a.walk:
					want, margin = oracle.SampleWalk(l, K, k, invT, topp, minp, typ, u)
					plain, _ = oracle.SampleWalk(l, K, k, invT, topp, minp, inf, u)
					vals[in["sample.walk"]] = []uint32{f(float64(k)), f(invT), f(topp), f(minp), f(u), f(typ)}
				case a.topK:
					want, margin = oracle.SampleWalk(l, n, k, invT, topp, minp, typ, u)
					plain, _ = oracle.SampleWalk(l, n, k, invT, topp, minp, inf, u)
					vals[in["sample.ids"]] = append([]uint32(nil), ids...)
					vals[in["sample.cut"]] = []uint32{f(invT), f(topp), f(minp), f(u), f(float64(k)), f(typ)}
				default:
					want, margin = oracle.SampleCut(l, nil, 1, float32(invT), topp, minp, typ, u)
					plain, _ = oracle.SampleCut(l, nil, 1, float32(invT), topp, minp, inf, u)
					vals[in["sample.ids"]] = append([]uint32(nil), ids...)
					vals[in["sample.cut"]] = []uint32{f(invT), f(topp), f(minp), f(u), f(typ)}
				}
				if err := runCPUValues(hostArm(), codes, g, vals, [3][]uint32{}); err != nil {
					t.Fatal(err)
				}
				cases++
				if margin < 1e-4 {
					continue
				}
				decisive++
				if want != plain {
					moved++
				}
				if got := vals[tokV][0]; got != want {
					t.Fatalf("case %d (k %d, 1/T %.3f, top-p %.4f, min-p %.4f, typical-p %.4f, u %.6f): drew %d, want %d (margin %.3g)",
						c, k, invT, topp, minp, typ, u, got, want, margin)
				}
			}
			if decisive < cases*7/10 {
				t.Fatalf("%d of %d draws decisive: the comparison sees too little", decisive, cases)
			}
			if moved < decisive/10 {
				t.Fatalf("typical-p moved %d of %d decisive draws: the gate cannot see the cut", moved, decisive)
			}
			t.Logf("%d of %d draws decisive and right, %d of them moved by typical-p", decisive, cases, moved)
		})
	}
}
