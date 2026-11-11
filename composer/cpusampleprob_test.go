package composer

import (
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// TestComposedCPUSampleProb holds graph.SampleWalk and graph.SampleCut with
// Prob on the CPU device to the distribution their draw draws from, read at
// a token (the walk's at the candidate whose id it is, 0 at none): over
// random logits, penalties, temperatures, cuts (off among them) and tokens
// (the best, survivors and cut ones), within float32's rounding of the
// definition wherever no cut is within rounding of flipping.
func TestComposedCPUSampleProb(t *testing.T) {
	const n, K, H = 300, 16, 8
	type arm struct {
		name   string
		g      *graph.Graph
		k, typ bool
	}
	for _, a := range []arm{
		{"walk", (&graph.SampleWalk{SampleTopK: graph.SampleTopK{NVocab: n, K: K}, Prob: true}).Build(), true, false},
		{"cut", (&graph.SampleCut{NVocab: n, Prob: true}).Build(), false, false},
		{"walk-typical", (&graph.SampleWalk{SampleTopK: graph.SampleTopK{NVocab: n, K: K}, Prob: true}).Build(), true, true},
		{"cut-typical", (&graph.SampleCut{NVocab: n, Prob: true, Typical: true}).Build(), false, true},
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
			out := g.Out
			// Apart, as the engine forms it: the dense distribution would take the
			// window's storage otherwise.
			_, codes, err := Default().Form(Request{Graph: g, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: 1, Page: 64, Keys: 1}, Device: cpuDevice(hostArm()), Apart: true})
			if err != nil {
				t.Fatal(err)
			}
			ids := make([]uint32, n)
			for i := range ids {
				ids[i] = uint32(i)
			}
			f := math.Float32bits
			typ, xtc, mix := float32(math.Inf(1)), float32(math.Inf(1)), float32(0)
			prob := func(l []float32, hist []int, pen, invT float32, k int, topp, minp float32, x int) float64 {
				lw := make([]uint32, n)
				for i, v := range l {
					lw[i] = f(v)
				}
				win := windowWords(hist, n)
				vals := map[graph.ValueID][]uint32{
					in["logits"]: lw, in["sample.window"]: win, in["sample.penalty"]: {f(pen), 0, 0},
				}
				for _, o := range out {
					vals[o] = make([]uint32, g.Values[o].Width)
				}
				if a.k {
					vals[in["sample.walk"]] = []uint32{f(float32(k)), f(invT), f(topp), f(minp), 0, f(typ), f(xtc), f(mix)}
				} else {
					vals[in["sample.ids"]] = append([]uint32(nil), ids...)
					vals[in["sample.cut"]] = []uint32{f(invT), f(topp), f(minp), 0}
					if a.typ {
						vals[in["sample.cut"]] = append(vals[in["sample.cut"]], 0, f(typ), f(xtc), f(mix))
					}
				}
				if err := runCPUValues(hostArm(), codes, g, vals, [3][]uint32{}); err != nil {
					t.Fatal(err)
				}
				p := vals[out[0]]
				if !a.k {
					return float64(math.Float32frombits(p[x]))
				}
				for i, id := range vals[out[1]] {
					if int(id) == x {
						return float64(math.Float32frombits(p[i]))
					}
				}
				return 0
			}
			rng := rand.New(rand.NewSource(41))
			decisive, nonzero := 0, 0
			for c := range 300 {
				l := make([]float32, n)
				for i := range l {
					l[i] = float32(rng.NormFloat64() * 3)
				}
				var hist []int
				pen := float32(1)
				if c%2 == 0 {
					pen = 1.1 + 0.5*rng.Float32()
					for range rng.Intn(H + 1) {
						hist = append(hist, rng.Intn(n/10))
					}
				}
				invT := float32(1 / (0.4 + 1.4*rng.Float64()))
				k := 0
				if a.k {
					k = 1 + rng.Intn(K)
				}
				topp, minp := math.Inf(1), 0.0
				switch c % 4 {
				case 0:
					topp = float64(float32(0.3 + 0.69*rng.Float64()))
				case 1:
					minp = float64(float32(0.2 * rng.Float64()))
				case 2:
					topp, minp = float64(float32(0.5+0.49*rng.Float64())), float64(float32(0.1*rng.Float64()))
				}
				// A token among the best mostly, any token else.
				x := rng.Intn(n)
				if c%3 != 0 {
					byLogit := make([]int, n)
					for i := range byLogit {
						byLogit[i] = i
					}
					sort.SliceStable(byLogit, func(i, j int) bool { return l[byLogit[i]] > l[byLogit[j]] })
					x = byLogit[rng.Intn(K)]
				}
				if a.typ {
					typ = float32(0.2 + 0.75*rng.Float64())
					xtc, mix = float32(math.Inf(1)), 0
					if c%2 == 1 {
						xtc, mix = float32(0.02+0.2*rng.Float64()), float32(rng.Float64())
					}
				}
				want, margin := oracle.SampleProb(l, hist, pen, invT, k, topp, minp, float64(typ), float64(xtc), float64(mix), x)
				if margin < 1e-4 {
					continue
				}
				got := prob(l, hist, pen, invT, k, float32(topp), float32(minp), x)
				decisive++
				if want > 0 {
					nonzero++
				}
				if math.Abs(got-want) > 1e-5*want+1e-9 {
					t.Fatalf("case %d (k %d, history %v at %.3f, 1/T %.3f, top-p %.4f, min-p %.4f): p(%d) %.9g, want %.9g",
						c, k, hist, pen, invT, topp, minp, x, got, want)
				}
			}
			// Typical-p cuts the best tokens often, so fewer of the drawn ones
			// survive it.
			if decisive < 200 || nonzero < 40 && !a.typ || nonzero < 20 {
				t.Fatalf("%d decisive cases, %d of a survivor: the comparison sees too little", decisive, nonzero)
			}
			t.Logf("%d probabilities right, %d of them a survivor's", decisive, nonzero)
		})
	}
}
