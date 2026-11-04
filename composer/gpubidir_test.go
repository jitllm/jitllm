package composer_test

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUAttendBidirRows forms a picture's attention as a tower
// block states it -- its keys and values appended at the call's rows into
// one page, every row attending every row (graph.Attrs.Bidir) -- at a run
// of rows below a workgroup's chunk of positions and past it, so a row's
// span takes one partition or several, at a vision tower's head width (72),
// in the general form (no sequence count) and, where the device states a
// matrix unit, the single-sequence form, held at sampled rows to the
// definition: every row, where the call's live elements, counted in 32
// bits, once wrapped past a flattened space of 4096 rows of 1152 and left
// the rows from 455 on unwritten. Two violations must fail: each row's span cut to the rows up
// to its own (a causal row) and the span's first position moved by one. A
// symmetric window the graph states (EmbeddingGemma's sliding layers: 512,
// half either side) is held the same way, and its form without the window
// must fail too.
func TestComposedGPUAttendBidirRows(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	if len(devs) == 0 {
		t.Skip("no GPU")
	}
	ran := 0
	for _, d := range devs {
		tile, _ := backend.LowersTile(d, backend.TileShape{M: 16, N: 16, K: 16, Elem: ir.TileF16, Acc: ir.TileF32})
		for _, c := range [][3]int{{1024, 72, 0}, {2100, 72, 0}, {4096, 72, 0}, {2100, 64, 0}, {700, 256, 512}, {2100, 72, 512}} {
			for _, seqs := range []int{0, 1} {
				if seqs == 1 && !tile {
					continue
				}
				ran++
				t.Run(fmt.Sprintf("%s/rows%d/dim%d/win%d/seqs%d", d.API(), c[0], c[1], c[2], seqs), func(t *testing.T) { gpuAttendBidir(t, d, c[0], c[1], c[2], seqs) })
			}
		}
	}
	if ran == 0 {
		t.Error("no case ran")
	}
}

// gpuAttendBidir is a run of R rows at head width dim on d, the device's
// matrix unit stated where seqs is 1, each row within win/2 of it either
// side where win is a window. Past 2048 rows -- a workgroup's chunk of
// positions at most (Workgroups.Chunk) -- the form must split the range.
func gpuAttendBidir(t *testing.T, d backend.Device, R, dim, win, seqs int) {
	const heads = 16
	keys := (R + 127) / 128 * 128
	page := R + 1
	scale := float32(1 / math.Sqrt(float64(dim)))
	dev := gpuDevice(d)
	if seqs == 1 {
		dev.Matrix = [3]int{16, 16, 16}
	}
	form := func(win int) (*graph.Graph, []composer.Code) {
		b := graph.New()
		n := b.KVAppend(b.In("k", heads*dim), b.In("v", heads*dim), 0)
		b.Out(b.Attend(b.In("q", heads*dim), n, graph.Attrs{NHead: heads, NKVHead: heads, HeadDim: dim, Scale: float64(scale), Bidir: true, Window: win}))
		_, codes, err := composer.Default().Form(composer.Request{Graph: b.G, Step: composer.Step{Rows: R, Page: page, Keys: keys, Seqs: seqs},
			Device: dev, History: composer.History{Keys: composer.Pool{ByDim: true}, Values: composer.Pool{Half: true}}})
		if err != nil {
			t.Fatal(err)
		}
		return b.G, codes
	}
	g, codes := form(win)
	var passes []string
	for _, cd := range codes {
		for _, p := range cd.Passes {
			if !slices.Contains(passes, p) {
				passes = append(passes, p)
			}
		}
	}
	t.Logf("%d kernels, passes %v", len(codes), passes)
	if R > 2048 && !slices.Contains(passes, "splitrange") && !slices.Contains(passes, "flash") {
		t.Errorf("%d rows past a workgroup's chunk and the range is not split: the gate did not select the partitions", R)
	}

	rng := rand.New(rand.NewSource(13))
	w := heads * dim
	rnd := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		return x
	}
	q, k, v := rnd(R*w), rnd(R*w), rnd(R*w)
	round := func(x float32) float64 { return float64(quant.DecodeHalf(quant.EncodeHalf(x))) }
	sample := []int{0, 1, R / 3, R / 2, R - 2, R - 1}
	for r := 7; r < R; r += 97 {
		sample = append(sample, r)
	}
	ref := func(r int) []float64 {
		lo, hi := 0, R
		if win > 0 {
			lo, hi = max(0, r-win/2), min(R, r+win/2+1)
		}
		want := make([]float64, w)
		for h := range heads {
			sc := make([]float64, hi-lo)
			for p := lo; p < hi; p++ {
				var s float64
				for e := range dim {
					s += float64(q[r*w+h*dim+e]) * float64(k[p*w+h*dim+e])
				}
				sc[p-lo] = s * float64(scale)
			}
			oracle.Softmax(sc)
			for e := range dim {
				var o float64
				for p := lo; p < hi; p++ {
					o += sc[p-lo] * round(v[p*w+h*dim+e])
				}
				want[h*dim+e] = o
			}
		}
		return want
	}
	pgs := make([]uint32, (keys+page-1)/page)
	for i := range pgs {
		pgs[i] = uint32(i)
	}
	run := func(codes []composer.Code, causal bool, shift uint32) []float32 {
		rows := make([]uint32, R*4)
		for r := range R {
			end := uint32(R)
			if causal {
				end = uint32(r + 1)
			}
			copy(rows[r*4:], []uint32{0, min(shift, uint32(r)), end, uint32(r)})
		}
		vals := map[graph.ValueID][]uint32{g.In[0]: f32w(k), g.In[1]: f32w(v), g.In[2]: f32w(q), g.Out[0]: poisoned(R * w)}
		tables := map[string][]uint32{"rows": rows, "rows.win": rows, "rows.n": {uint32(R)}, "pages": pgs,
			"kv.k": poisoned(len(pgs) * page * w), "kv.v": poisoned(len(pgs) * page * w / 2)}
		out, err := launchAttend(d, codes, vals, tables, nil, g.Out[0])
		if err != nil {
			t.Fatal(err)
		}
		return wf32(out)
	}
	nmse := func(got []float32, want []float64, r int) float64 {
		var num, den float64
		for i := range w {
			e := float64(got[r*w+i]) - want[i]
			num += e * e
			den += want[i] * want[i]
		}
		return num / den
	}
	wants := make(map[int][]float64, len(sample))
	for _, r := range sample {
		wants[r] = ref(r)
	}
	got := run(codes, false, 0)
	worst, bad := 0.0, 0
	for _, r := range sample {
		e := nmse(got, wants[r], r)
		if math.IsNaN(e) || e > 1e-6 {
			if bad < 8 {
				t.Errorf("row %d: NMSE %.3e against the definition", r, e)
			}
			bad++
		}
		if !math.IsNaN(e) {
			worst = max(worst, e)
		}
	}
	t.Logf("%d sampled rows of %d: worst NMSE %.3e, %d over the band", len(sample), R, worst, bad)
	if e := nmse(run(codes, true, 0), wants[0], 0); !(e > 1e-6) {
		t.Errorf("with each row's span cut to its own the NMSE is %.3e: the gate cannot see a causal row", e)
	}
	// The last row's span starts at the first position, a windowed one's
	// past it: row 1's window reaches the first position.
	vr := R - 1
	if win > 0 {
		vr = 1
	}
	if e := nmse(run(codes, false, 1), wants[vr], vr); !(e > 1e-6) {
		t.Errorf("with the span's first position moved the NMSE is %.3e: the gate cannot see a wrong span", e)
	}
	if win > 0 {
		_, open := form(0)
		if e := nmse(run(open, false, 0), wants[R/2], R/2); !(e > 1e-6) {
			t.Errorf("formed without its window the NMSE is %.3e: the gate cannot see the window", e)
		}
	}
}
