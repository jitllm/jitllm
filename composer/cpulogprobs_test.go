package composer

import (
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/oracle"
)

// TestComposedCPULogprobs holds graph.Logprobs on the CPU device to its
// definition: over random logits (some with values repeated, so the order's
// ties fall to the id, and some far apart), every token's log-probability
// to the oracle's log-softmax within the exp's and the log's rounding, and
// the K most likely ids exactly, the larger logit first and the lower id on
// a tie. A row not shifted by its maximum, or one missing the log of the
// sum, is far outside the bound.
func TestComposedCPULogprobs(t *testing.T) {
	const n, K = 300, 20
	g := (&graph.Logprobs{NVocab: n, K: K}).Build()
	if err := graph.Verify(g, graph.Facts{}); err != nil {
		t.Fatal(err)
	}
	// Apart, as the engine forms it: the log row is the caller's own
	// buffer.
	_, codes, err := Default().Form(Request{Graph: g, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: 1, Page: 64, Keys: 1}, Device: cpuDevice(), Apart: true})
	if err != nil {
		t.Fatal(err)
	}
	rowV, idsV := g.Out[0], g.Out[1]
	run := func(logits []float32) ([]float32, []uint32) {
		lw := make([]uint32, n)
		for i, v := range logits {
			lw[i] = math.Float32bits(v)
		}
		vals := map[graph.ValueID][]uint32{g.In[0]: lw, rowV: make([]uint32, n), idsV: make([]uint32, K)}
		if err := runCPUValues(codes, g, vals, [3][]uint32{}); err != nil {
			t.Fatal(err)
		}
		row := make([]float32, n)
		for i, w := range vals[rowV] {
			row[i] = math.Float32frombits(w)
		}
		return row, vals[idsV]
	}
	rng := rand.New(rand.NewSource(37))
	worst := 0.0
	for c := range 200 {
		l := make([]float32, n)
		scale := 3.0
		if c%7 == 3 {
			scale = 40
		}
		for i := range l {
			l[i] = float32(rng.NormFloat64()*scale) + 50*float32(c%3)
		}
		if c%4 == 0 {
			for i := 0; i < n; i += 3 {
				l[i] = l[(i*7)%n]
			}
		}
		row, ids := run(l)
		want := oracle.LogSoftmax32(l)
		for i, w := range want {
			d := math.Abs(float64(row[i]) - w)
			worst = max(worst, d)
			if !(d <= 1e-4+2e-6*math.Abs(w)) {
				t.Fatalf("case %d token %d: log-probability %v, the oracle's %v", c, i, row[i], w)
			}
		}
		order := make([]int, n)
		for i := range order {
			order[i] = i
		}
		sort.SliceStable(order, func(a, b int) bool { return l[order[a]] > l[order[b]] })
		for i, id := range ids {
			if int(id) != order[i] {
				t.Fatalf("case %d: alternative %d is token %d, want %d", c, i, id, order[i])
			}
		}
	}
	t.Logf("200 rows: worst log-probability difference %.3g, every top-%d exact", worst, K)
}

// TestComposedCPUMaskRow holds graph.MaskRow on the CPU device to its
// definition, bit for bit: an allowed token's logit unchanged, a refused
// one's -inf, over rows of every sign and a mask of either kind.
func TestComposedCPUMaskRow(t *testing.T) {
	const n = 300
	g := (&graph.MaskRow{NVocab: n}).Build()
	if err := graph.Verify(g, graph.Facts{}); err != nil {
		t.Fatal(err)
	}
	// Apart, as the engine forms it: the masked row is the caller's own
	// buffer, never the logits'.
	_, codes, err := Default().Form(Request{Graph: g, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: 1, Page: 64, Keys: 1}, Device: cpuDevice(), Apart: true})
	if err != nil {
		t.Fatal(err)
	}
	ninf := math.Float32bits(float32(math.Inf(-1)))
	rng := rand.New(rand.NewSource(41))
	refused := 0
	for c := range 50 {
		lw, mw := make([]uint32, n), make([]uint32, n)
		for i := range lw {
			lw[i] = math.Float32bits(float32(rng.NormFloat64() * 5))
			if rng.Intn(3) == 0 {
				mw[i] = ninf
			}
		}
		vals := map[graph.ValueID][]uint32{g.In[0]: lw, g.In[1]: mw, g.Out[0]: make([]uint32, n)}
		if err := runCPUValues(codes, g, vals, [3][]uint32{}); err != nil {
			t.Fatal(err)
		}
		for i, got := range vals[g.Out[0]] {
			want := lw[i]
			if mw[i] == ninf {
				want = ninf
				refused++
			}
			if got != want {
				t.Fatalf("case %d token %d: %v, want %v", c, i, math.Float32frombits(got), math.Float32frombits(want))
			}
		}
	}
	if refused == 0 {
		t.Fatal("no token refused: the gate saw nothing")
	}
	t.Logf("50 rows exact, %d tokens refused", refused)
}
