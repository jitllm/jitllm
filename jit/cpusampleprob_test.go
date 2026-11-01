package jit

import (
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// sampleProbRef is the probability the host sampler's draw gives token x, in
// float64: of the candidates (the k best penalized logits, or the whole row
// where k is 0) in their order, min-p and top-p over their softmax at
// temperature 1 (the best kept), then x's share of the survivors' softmax at
// 1/T; with no top-k and no cut firing, x's share of the row's softmax at 1/T.
// It returns the probability and how far the cuts are from flipping.
func sampleProbRef(logits []float32, hist []int, pen, invT float32, k int, topp, minp float64, x int) (p, margin float64) {
	v := append([]float32(nil), logits...)
	for _, h := range hist {
		if v[h] > 0 {
			v[h] /= pen
		} else {
			v[h] *= pen
		}
	}
	n := len(v)
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return v[order[a]] > v[order[b]] })
	cand := order
	if k > 0 {
		cand = order[:k]
	}
	v0 := float64(v[cand[0]])
	q1 := make([]float64, len(cand))
	sum := 0.0
	for j, i := range cand {
		q1[j] = math.Exp(float64(v[i]) - v0)
		sum += q1[j]
	}
	keep, fired, acc := len(cand), false, 0.0
	margin = math.Inf(1)
	for j := range cand {
		q1[j] /= sum
		if minp > 0 {
			margin = min(margin, math.Abs(q1[j]-minp*q1[0])/q1[0])
		}
		if q1[j] < minp*q1[0] {
			keep, fired = j, true
			break
		}
		acc += q1[j]
		if !math.IsInf(topp, 1) {
			margin = min(margin, math.Abs(acc-topp))
		}
		if acc >= topp {
			keep, fired = j+1, true
			break
		}
	}
	surv := cand[:max(keep, 1)]
	if k == 0 && !fired {
		surv = order
	}
	top := float64(v[surv[0]] * invT)
	mass, px := 0.0, 0.0
	for _, i := range surv {
		e := math.Exp(float64(v[i]*invT) - top)
		mass += e
		if i == x {
			px = e
		}
	}
	return px / mass, margin
}

// TestComposedCPUSampleProb holds graph.SampleWalk and graph.SampleCut with
// Prob on the CPU device to the distribution their draw draws from, read at
// a token (the walk's at the candidate whose id it is, 0 at none): over
// random logits, penalties, temperatures, cuts (off among them) and tokens
// (the best, survivors and cut ones), within float32's rounding of the
// definition wherever no cut is within rounding of flipping.
func TestComposedCPUSampleProb(t *testing.T) {
	const n, K, H = 300, 16, 8
	type arm struct {
		name string
		g    *graph.Graph
		k    bool
	}
	for _, a := range []arm{
		{"walk", (&graph.SampleWalk{SampleTopK: graph.SampleTopK{NVocab: n, K: K, Hist: H}, Prob: true}).Build(), true},
		{"cut", (&graph.SampleCut{NVocab: n, Hist: H, Prob: true}).Build(), false},
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
			_, codes, err := Default().Form(Request{Graph: g, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: 1, Page: 64, Keys: 1}, Device: cpuDevice()})
			if err != nil {
				t.Fatal(err)
			}
			ids := make([]uint32, n)
			for i := range ids {
				ids[i] = uint32(i)
			}
			f := math.Float32bits
			prob := func(l []float32, hist []int, pen, invT float32, k int, topp, minp float32, x int) float64 {
				lw := make([]uint32, n)
				for i, v := range l {
					lw[i] = f(v)
				}
				args := make([]uint32, 2+H)
				args[0], args[1] = uint32(len(hist)), f(pen)
				for i := 2; i < len(args); i++ {
					args[i] = math.MaxUint32
				}
				for i, h := range hist {
					args[2+i] = uint32(h)
				}
				vals := map[graph.ValueID][]uint32{
					in["logits"]: lw, in["sample.args"]: args, in["sample.penalty"]: {f(pen)},
				}
				for _, o := range out {
					vals[o] = make([]uint32, g.Values[o].Width)
				}
				if a.k {
					vals[in["sample.walk"]] = []uint32{f(float32(k)), f(invT), f(topp), f(minp), 0}
				} else {
					vals[in["sample.ids"]] = append([]uint32(nil), ids...)
					vals[in["sample.cut"]] = []uint32{f(invT), f(topp), f(minp), 0}
				}
				if err := runCPUValues(codes, g, vals, [3][]uint32{}); err != nil {
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
				want, margin := sampleProbRef(l, hist, pen, invT, k, topp, minp, x)
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
			if decisive < 200 || nonzero < 40 {
				t.Fatalf("%d decisive cases, %d of a survivor: the comparison sees too little", decisive, nonzero)
			}
			t.Logf("%d probabilities right, %d of them a survivor's", decisive, nonzero)
		})
	}
}
