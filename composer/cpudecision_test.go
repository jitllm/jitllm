package composer

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUDecisionScores executes a label readout's graph
// (graph.DecisionScores) on the CPU against its definition: each option's
// group max, the softmax at the temperature over the question's options,
// the mean with a prior at its weight, and the statistics and confidences
// the answers read (one option's confidence is 1).
// A question of fewer options than the form holds leaves the rest at zero,
// and a group's repeated slot does not move its score. Dropping the mask
// (every option kept) must fail it.
func TestComposedCPUDecisionScores(t *testing.T) {
	const N, G = 8, 3
	d := graph.DecisionScores{N: N, G: G}
	g := d.Build()
	rng := rand.New(rand.NewSource(7))
	words := func(x []float32) []uint32 {
		out := make([]uint32, len(x))
		for i, v := range x {
			out[i] = math.Float32bits(v)
		}
		return out
	}
	for _, c := range []struct {
		k    int
		invT float32
		w    float32
	}{{5, 1, 0}, {5, 0.5, 1}, {8, 2, 0}, {2, 1, 1}, {1, 1, 0}} {
		vals := make([]float32, N*G)
		for i := range vals {
			vals[i] = float32(rng.NormFloat64() * 3)
		}
		// Option 1's group holds one id, its other slots repeating it.
		vals[G+1], vals[G+2] = vals[G], vals[G]
		keep, params, prior := make([]float32, N), make([]float32, 2+N), make([]float32, N)
		params[0], params[1] = c.invT, c.w
		for i := range N {
			params[2+i] = float32(i)
			if i < c.k {
				keep[i] = 1
				prior[i] = rng.Float32()
			}
		}
		run := func(keep []float32) ([]float32, []float32) {
			io := map[graph.ValueID][]uint32{g.In[0]: words(vals), g.In[1]: words(keep), g.In[2]: words(params), g.In[3]: words(prior)}
			formOnCPU(t, hostArm(), g, io)
			return wordsF32(io[g.Out[0]]), wordsF32(io[g.Out[1]])
		}
		// The definition, in float64.
		want := make([]float64, N)
		top := math.Inf(-1)
		score := make([]float64, c.k)
		for i := range c.k {
			s := math.Inf(-1)
			for j := range G {
				s = math.Max(s, float64(vals[i*G+j]))
			}
			score[i] = s * float64(c.invT)
			top = math.Max(top, score[i])
		}
		var sum float64
		for i := range c.k {
			want[i] = math.Exp(score[i] - top)
			sum += want[i]
		}
		pmax, expect := 0.0, 0.0
		for i := range c.k {
			want[i] = (want[i]/sum + float64(c.w)*float64(prior[i])) / (1 + float64(c.w))
			pmax = math.Max(pmax, want[i])
			expect += want[i] * float64(i)
		}
		mode := 0
		for i := range c.k {
			if want[i] == pmax {
				mode = i
				break
			}
		}
		var md, ud, h float64
		for i := range c.k {
			md += want[i] * math.Abs(float64(i-mode))
			ud += math.Abs(float64(i) - float64(c.k-1)/2)
			h -= want[i] * (score[i] - top - math.Log(sum))
		}
		ud /= float64(c.k)
		u := 1 / float64(c.k)
		choice, scoreConf, ent := max(0, (pmax-u)/(1-u)), max(0, 1-md/ud), 1-h/math.Log(float64(c.k))
		if c.k == 1 {
			choice, scoreConf, ent = 1, 1, 1
		}
		p, st := run(keep)
		for i := range N {
			if math.Abs(float64(p[i])-want[i]) > 1e-5 {
				t.Fatalf("%d options at 1/T %v: probability %d is %v, want %v", c.k, c.invT, i, p[i], want[i])
			}
		}
		for i, w := range []float64{pmax, expect, float64(mode), choice, scoreConf, ent} {
			if math.Abs(float64(st[i])-w) > 1e-4*math.Max(1, math.Abs(w)) {
				t.Fatalf("%d options at 1/T %v: statistic %d is %v, want %v", c.k, c.invT, i, st[i], w)
			}
		}
		if c.k < N {
			all := make([]float32, N)
			for i := range all {
				all[i] = 1
			}
			if q, _ := run(all); math.Abs(float64(q[0])-want[0]) < 1e-5 {
				t.Errorf("%d options with every option kept: probability 0 is %v as with the mask: the gate cannot see the mask", c.k, q[0])
			}
		}
	}
}
