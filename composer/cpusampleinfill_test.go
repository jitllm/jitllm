package composer

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"
	"time"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/tok"
)

// TestComposedCPUSampleInfill holds graph.SampleWalk's infill form on the CPU
// device to its definition (oracle.SampleInfill, llama.cpp's infill sampler
// over the walk's survivors): the token of every draw whose comparisons clear
// float32 rounding, over a vocabulary built to be full of prefixes -- every
// piece of one to four letters over two, repeats of them, empty pieces, and
// end-of-generation tokens with and without a piece -- and random logits, k,
// temperatures, cuts, top-n-sigma and uniforms. Every branch of the sampler
// is taken many times: the merge, the end-of-generation survivors alone, and
// the end-of-turn token where nothing else is left.
func TestComposedCPUSampleInfill(t *testing.T) {
	for _, K := range []int{16, 64} {
		t.Run(fmt.Sprintf("K%d", K), func(t *testing.T) { infillGate(t, K) })
	}
}

func infillGate(t *testing.T, K int) {
	const n = 300
	rng := rand.New(rand.NewSource(int64(31 + K)))
	// The 16 pieces of four letters first: no one prefixes another.
	var pieces []string
	for l := 4; l >= 1; l-- {
		for b := range 1 << l {
			s := make([]byte, l)
			for j := range s {
				s[j] = "ab"[b>>(l-1-j)&1]
			}
			pieces = append(pieces, string(s))
		}
	}
	for len(pieces) < n {
		if rng.Intn(9) == 0 {
			pieces = append(pieces, "")
			continue
		}
		pieces = append(pieces, pieces[rng.Intn(30)])
	}
	eogIDs := []int{290, 295, 299}
	pieces[290], pieces[299] = "", ""
	eog := func(id int) bool { return slices.Contains(eogIDs, id) }
	const eot = 299
	class, parent, depth := tok.InfillOf(pieces)
	g := (&graph.SampleWalk{SampleTopK: graph.SampleTopK{NVocab: n, K: K}, Infill: true, Depth: depth, EOG: len(eogIDs)}).Build()
	if err := graph.Verify(g, graph.Facts{}); err != nil {
		t.Fatal(err)
	}
	in := map[string]graph.ValueID{}
	for _, v := range g.In {
		in[g.Values[v].Name] = v
	}
	tokV := g.Out[0]
	start := time.Now()
	_, codes, err := Default().Form(Request{Graph: g, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: 1, Page: 64, Keys: 1}, Device: cpuDevice(hostArm())})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("K %d: %d ops formed in %v", K, len(g.Ops), time.Since(start))
	f32 := func(x float64) uint32 { return math.Float32bits(float32(x)) }
	var eogBits []uint32
	for _, id := range eogIDs {
		eogBits = append(eogBits, f32(float64(id)))
	}
	draw := func(l []float32, k int, invT, topp, minp, sigma, u float64) uint32 {
		lw := make([]uint32, n)
		for i, v := range l {
			lw[i] = math.Float32bits(v)
		}
		vals := map[graph.ValueID][]uint32{
			in["logits"]: lw, in["sample.window"]: make([]uint32, n), in["sample.penalty"]: {f32(1), 0, 0},
			in["sample.walk"]: {f32(float64(k)), f32(invT), f32(topp), f32(minp), f32(u), f32(math.Inf(1)), f32(math.Inf(1)), 0, 0,
				f32(1), f32(sigma)},
			in["sample.infill.class"]: append([]uint32(nil), class...), in["sample.infill.parent"]: append([]uint32(nil), parent...),
			in["sample.eog"]: eogBits, in["sample.eot"]: {eot},
			tokV: {0},
		}
		if err := runCPUValues(hostArm(), codes, g, vals, [3][]uint32{}); err != nil {
			t.Fatal(err)
		}
		return vals[tokV][0]
	}
	piece := func(id int) string { return pieces[id] }
	branches := [3]int{}
	decisive, cases := 0, 0
	for c := range 600 {
		l := make([]float32, n)
		for i := range l {
			l[i] = float32(rng.NormFloat64() * 2)
		}
		k := 1 + rng.Intn(K)
		switch c % 10 {
		case 0:
			// The pieces no one prefixes, alone on top and alike: nothing
			// merges, none reaches 0.2, and the draw is the end-of-turn token.
			for i := range 16 {
				l[i] = 9
			}
			k = 16
		case 1:
			// An end-of-generation token well up: its survivors alone.
			l[eogIDs[c/10%len(eogIDs)]] = 6
		}
		invT := 1 / (0.3 + 1.5*rng.Float64())
		topp, minp, sigma := math.Inf(1), 0.0, 0.0
		if c%3 == 2 {
			topp = 0.5 + 0.5*rng.Float64()
		}
		if c%4 == 3 {
			minp = 0.05 * rng.Float64()
		}
		if c%5 == 4 {
			sigma = 0.5 + 2*rng.Float64()
		}
		u := float64(float32(rng.Float64()))
		want, branch, margin := oracle.SampleInfill(l, K, k, invT, topp, minp, math.Inf(1), math.Inf(1), sigma, [2]float64{}, u, piece, eog, eot)
		got := draw(l, k, invT, topp, minp, sigma, u)
		cases++
		if margin < 1e-4 {
			continue
		}
		decisive++
		branches[branch]++
		if got != want {
			t.Fatalf("case %d (k %d, 1/T %.3f, top-p %.3f, min-p %.3f, sigma %.2f, u %.6f, branch %d): drew %d, want %d (margin %.3g)",
				c, k, invT, topp, minp, sigma, u, branch, got, want, margin)
		}
	}
	if decisive < cases*8/10 {
		t.Fatalf("%d of %d draws decisive: the comparison sees too little", decisive, cases)
	}
	for b, m := range branches {
		if m < 20 {
			t.Fatalf("branch %d taken by %d decisive draws (merge, end-of-generation, end-of-turn: %v): the gate does not cover it", b, m, branches)
		}
	}
	t.Logf("K %d, depth %d: %d of %d draws decisive and right; branches %v", K, depth, decisive, cases, branches)
}
