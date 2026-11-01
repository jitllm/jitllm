package jit_test

import (
	"math"
	"math/rand"
	"slices"
	"sort"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testlock"
	"github.com/jitllm/jitllm/jit"
)

// TestComposedGPUPhasesExpertsSum runs a mixture's tail -- the experts'
// outputs weighted, summed and scaled by their sum of squares -- on every GPU
// device present: its launches reading the part's inputs run as phases of
// one, and each output is the host's within 1e-5 of the largest.
func TestComposedGPUPhasesExpertsSum(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate runs on one")
	}
	const K, W = 2, 64
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			g, down, wt, out := expertsSum(K, W)
			_, codes, err := jit.Default().Form(jit.Request{Graph: g, Formats: map[jlm.Role]jlm.Type{}, Step: jit.Step{Rows: 1, Page: 64, Keys: 1}, Device: gpuDevice(d)})
			if err != nil {
				t.Fatal(err)
			}
			phased := 0
			for _, c := range codes {
				if slices.Contains(c.Passes, "phases") {
					phased++
				}
			}
			if phased == 0 {
				t.Fatalf("no launch runs phases (%d launches)", len(codes))
			}
			op, err := compileLone(d, codes)
			if err != nil {
				t.Fatal(err)
			}
			defer op.close()
			rng := rand.New(rand.NewSource(7))
			word := func(n int) ([]uint32, []float64) {
				w, f := make([]uint32, n), make([]float64, n)
				for i := range w {
					v := float32(rng.NormFloat64())
					w[i], f[i] = math.Float32bits(v), float64(v)
				}
				return w, f
			}
			dw, df := word(K * W)
			ww, wf := word(K)
			vals := map[graph.ValueID][]uint32{down: dw, wt: ww}
			if err := runGraphValues(d, codes, op.kerns, g, vals); err != nil {
				t.Fatal(err)
			}
			want := make([]float64, W)
			var ss, top float64
			for i := range want {
				for j := range K {
					want[i] += df[j*W+i] * wf[j]
				}
				ss += want[i] * want[i]
			}
			for i := range want {
				want[i] /= ss
				top = max(top, math.Abs(want[i]))
			}
			for i, w := range want {
				if got := float64(math.Float32frombits(vals[out][i])); !(math.Abs(got-w) <= 1e-5*top) {
					t.Fatalf("output %d is %v, want %v", i, got, w)
				}
			}
			t.Logf("%d launches, %d of them phases", len(codes), phased)
		})
	}
}

// TestComposedGPURouterRows runs a router's chain -- a softmax over the
// experts' scores, their top k, the picked scores gathered and renormalised --
// formed at 512 rows and called at 3 on every GPU device present: the top
// k's kernel takes the softmax's division as its prologue, read at
// coordinates and gathered, and each live row's weights are the host's
// within 1e-6.
func TestComposedGPURouterRows(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate runs on one")
	}
	const E, K, capacity, rows, guard = 8, 2, 512, 3, 64
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			b := graph.New()
			x := b.In("logits", E)
			op := func(kind graph.OpKind, fn graph.MapFn, name string, width int, in ...graph.ValueID) graph.ValueID {
				return b.Emit(kind, in, []graph.Result{{Name: name, Width: width}}, nil, graph.Attrs{Fn: fn}, nil, nil)[0]
			}
			m := op(graph.OpReduce, graph.MapMax, "max", 1, x)
			e := op(graph.OpMap, graph.MapExp, "exp", E, op(graph.OpMap, graph.MapSub, "sub", E, x, m))
			p := op(graph.OpMap, graph.MapDiv, "p", E, e, op(graph.OpReduce, graph.MapAdd, "sum", 1, e))
			top := b.Emit(graph.OpTopK, []graph.ValueID{p}, []graph.Result{{Name: "sel", Width: K, Elem: graph.ElemU32}, {Name: "best", Width: K}},
				nil, graph.Attrs{TopK: K}, nil, nil)
			w := b.Emit(graph.OpTake, []graph.ValueID{p, top[0]}, []graph.Result{{Name: "w", Width: K}}, nil, graph.Attrs{}, nil, nil)[0]
			wn := op(graph.OpMap, graph.MapDiv, "wn", K, w, op(graph.OpReduce, graph.MapAdd, "wsum", 1, w))
			b.Out(top[0])
			b.Out(wn)
			_, codes, err := poisonedRegistry().Form(jit.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{}, Step: jit.Step{Rows: capacity, Page: 64, Keys: 1}, Device: gpuDevice(d)})
			if err != nil {
				t.Fatal(err)
			}
			if !slices.ContainsFunc(codes, func(c jit.Code) bool {
				return slices.Contains(c.Passes, "prologue") && slices.Contains(c.Passes, "epilogue")
			}) {
				t.Fatalf("no launch takes a prologue and an epilogue (%d launches)", len(codes))
			}
			lone, err := compileLone(d, codes)
			if err != nil {
				t.Fatal(err)
			}
			defer lone.close()
			rng := rand.New(rand.NewSource(11))
			xs := make([]float32, capacity*E)
			for i := range xs {
				xs[i] = math.Float32frombits(guardWord)
			}
			for i := range rows * E {
				xs[i] = float32(rng.NormFloat64() * 3)
			}
			res, _, err := runGraphRows(d, codes, lone.kerns, x, wn, xs, capacity*K, guard, [3][]uint32{}, rows)
			if err != nil {
				t.Fatal(err)
			}
			for r := range rows {
				row := xs[r*E : (r+1)*E]
				order := make([]int, E)
				for i := range order {
					order[i] = i
				}
				sort.SliceStable(order, func(a, b int) bool { return row[order[a]] > row[order[b]] })
				var kept float64
				for _, i := range order[:K] {
					kept += math.Exp(float64(row[i]))
				}
				for j := range K {
					want := math.Exp(float64(row[order[j]])) / kept
					if got := float64(res[r*K+j]); !(math.Abs(got-want) <= 1e-6) {
						t.Fatalf("row %d weight %d is %v, want %v", r, j, got, want)
					}
				}
			}
			t.Logf("%d launches", len(codes))
		})
	}
}

// TestComposedGPUPhases runs a router's chain at one row -- a softmax over
// the experts' scores, their top k, the kept weights renormalised -- on every
// GPU device present: its one-workgroup kernels must run as phases of one
// launch, the selection must be the host's and the weights within 1e-6 of
// them. The selection one place off must part.
func TestComposedGPUPhases(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate runs on one")
	}
	const K = 4
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			ran, parted := 0, 0
			for _, E := range []int{8, 32, 128} {
				b := graph.New()
				x := b.In("scores", E)
				m := b.Emit(graph.OpReduce, []graph.ValueID{x}, []graph.Result{{Name: "max", Width: 1}}, nil, graph.Attrs{Fn: graph.MapMax}, nil, nil)[0]
				s := b.Emit(graph.OpMap, []graph.ValueID{x, m}, []graph.Result{{Name: "sub", Width: E}}, nil, graph.Attrs{Fn: graph.MapSub}, nil, nil)[0]
				e := b.Emit(graph.OpMap, []graph.ValueID{s}, []graph.Result{{Name: "exp", Width: E}}, nil, graph.Attrs{Fn: graph.MapExp}, nil, nil)[0]
				z := b.Emit(graph.OpReduce, []graph.ValueID{e}, []graph.Result{{Name: "sum", Width: 1}}, nil, graph.Attrs{Fn: graph.MapAdd}, nil, nil)[0]
				p := b.Emit(graph.OpMap, []graph.ValueID{e, z}, []graph.Result{{Name: "p", Width: E}}, nil, graph.Attrs{Fn: graph.MapDiv}, nil, nil)[0]
				top := b.Emit(graph.OpTopK, []graph.ValueID{p}, []graph.Result{{Name: "sel", Width: K}, {Name: "w", Width: K}}, nil, graph.Attrs{TopK: K}, nil, nil)
				b.G.Values[top[0]].Elem = graph.ElemU32
				ws := b.Emit(graph.OpReduce, []graph.ValueID{top[1]}, []graph.Result{{Name: "wsum", Width: 1}}, nil, graph.Attrs{Fn: graph.MapAdd}, nil, nil)[0]
				wn := b.Emit(graph.OpMap, []graph.ValueID{top[1], ws}, []graph.Result{{Name: "wn", Width: K}}, nil, graph.Attrs{Fn: graph.MapDiv}, nil, nil)[0]
				b.Out(top[0])
				b.Out(wn)
				_, codes, err := jit.Default().Form(jit.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{}, Step: jit.Step{Rows: 1, Page: 64, Keys: 1}, Device: gpuDevice(d)})
				if err != nil {
					t.Fatalf("%d experts: %v", E, err)
				}
				phased := 0
				for _, c := range codes {
					if slices.Contains(c.Passes, "phases") {
						phased++
					}
				}
				if phased == 0 {
					t.Fatalf("%d experts: no launch runs phases (%d launches)", E, len(codes))
				}
				op, err := compileLone(d, codes)
				if err != nil {
					t.Fatal(err)
				}
				rng := rand.New(rand.NewSource(int64(E)))
				xs := make([]uint32, E)
				fs := make([]float64, E)
				for i := range xs {
					v := float32(rng.NormFloat64() * 3)
					xs[i], fs[i] = math.Float32bits(v), float64(v)
				}
				vals := map[graph.ValueID][]uint32{x: xs}
				err = runGraphValues(d, codes, op.kerns, b.G, vals)
				op.close()
				if err != nil {
					t.Fatalf("%d experts: %v", E, err)
				}
				order := make([]int, E)
				for i := range order {
					order[i] = i
				}
				sort.SliceStable(order, func(a, b int) bool { return fs[order[a]] > fs[order[b]] })
				hi := math.Inf(-1)
				for _, f := range fs {
					hi = max(hi, f)
				}
				var kept float64
				for _, i := range order[:K] {
					kept += math.Exp(fs[i] - hi)
				}
				for j := range K {
					if got := vals[top[0]][j]; got != uint32(order[j]) {
						t.Fatalf("%d experts: place %d selects %d, want %d", E, j, got, order[j])
					}
					if vals[top[0]][j] != uint32(order[(j+1)%K]) {
						parted++
					}
					want := math.Exp(fs[order[j]]-hi) / kept
					if got := float64(math.Float32frombits(vals[wn][j])); math.Abs(got-want) > 1e-6 {
						t.Fatalf("%d experts: weight %d is %v, want %v", E, j, got, want)
					}
				}
				ran++
				t.Logf("%d experts: %d launches, %d of them phases", E, len(codes), phased)
			}
			if parted == 0 {
				t.Fatal("the selection one place off agrees with the form: the gate cannot see it")
			}
			if ran != 3 {
				t.Fatalf("%d of 3 widths ran", ran)
			}
		})
	}
}
