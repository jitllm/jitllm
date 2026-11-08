package composer

import (
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPURouterOverManyExperts forms a mixture's router over 4096
// experts on the CPU as the front end states it (softmax, the top K, the
// picks' weights over their sum) and holds the picks to the definition and the
// weights to a float64 softmax, on the profile as it ships, at one row and
// at five (the scatter of the kept, past one row, outgrew the vector
// registers held as one map), and with no loop across lanes: the form a register refusal leaves (Kernel.Unvector), on an
// x86 device the top-k's orderable keys then a float's bits in the general
// registers (Kernel.BitsGPR). Each arm asserts the form it ran.
func TestComposedCPURouterOverManyExperts(t *testing.T) {
	const E = 4096
	rng := rand.New(rand.NewSource(9))
	x0 := jlm.Operand(0)
	term := func(i int) jlm.Arg { return jlm.Arg{Kind: jlm.ArgTerm, Index: uint32(i)} }
	softmax := &jlm.Expr{Terms: []jlm.Term{
		{Fn: jlm.FnMax, Reduce: true, Args: [3]jlm.Arg{x0}},
		{Fn: jlm.FnSub, Args: [3]jlm.Arg{x0, term(0)}},
		{Fn: jlm.FnExp, Args: [3]jlm.Arg{term(1)}},
		{Fn: jlm.FnAdd, Reduce: true, Args: [3]jlm.Arg{term(2)}},
		{Fn: jlm.FnDiv, Args: [3]jlm.Arg{term(2), term(3)}},
	}}
	noLanes := Default()
	p := noLanes.Profiles[DeviceCPU]
	p.Off = map[string]bool{Vectorize{}.Name(): true}
	noLanes.Profiles[DeviceCPU] = p
	for _, arm := range []struct {
		name   string
		r      *Registry
		lanes  bool
		shapes [][2]int
	}{
		{"as shipped", Default(), true, [][2]int{{1, 1}, {1, 2}, {1, 8}, {5, 2}, {5, 8}}},
		{"no lanes", noLanes, false, [][2]int{{1, 1}, {1, 2}, {1, 8}}},
	} {
		for _, rk := range arm.shapes {
			rows, K := rk[0], rk[1]
			b := graph.New()
			in := b.In("logits", E)
			score, err := b.Stated("router.score", softmax, []graph.ValueID{in}, E, graph.Facts{})
			if err != nil {
				t.Fatal(err)
			}
			o := b.Emit(graph.OpTopK, []graph.ValueID{score}, []graph.Result{{Name: "router.sel", Width: K, Elem: graph.ElemU32},
				{Name: "router.best", Width: K}}, nil, graph.Attrs{TopK: K}, nil, nil)
			w := b.Emit(graph.OpTake, []graph.ValueID{score, o[0]}, []graph.Result{{Name: "router.w", Width: K}}, nil, graph.Attrs{}, nil, nil)[0]
			sum := b.Emit(graph.OpReduce, []graph.ValueID{w}, []graph.Result{{Name: "router.wsum", Width: 1}}, nil, graph.Attrs{Fn: graph.MapAdd}, nil, nil)[0]
			sum = b.Emit(graph.OpMap, []graph.ValueID{sum}, []graph.Result{{Name: "router.sum", Width: 1}}, nil, graph.Attrs{Fn: graph.MapAdd, Const: 1e-20}, nil, nil)[0]
			wn := b.Emit(graph.OpMap, []graph.ValueID{w, sum}, []graph.Result{{Name: "router.wn", Width: K}}, nil, graph.Attrs{Fn: graph.MapDiv}, nil, nil)[0]
			b.Out(o[0])
			b.Out(wn)
			_, codes, err := arm.r.Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: rows, Page: 64, Keys: 1}, Device: cpuDevice(hostArm()), Apart: true})
			if err != nil {
				t.Fatalf("%s/%d rows/top %d: forming: %v", arm.name, rows, K, err)
			}
			lanes := slices.ContainsFunc(codes, func(c Code) bool { return slices.Contains(c.Passes, Vectorize{}.Name()) })
			if lanes != arm.lanes {
				t.Fatalf("%s/%d rows/top %d: a code across lanes %v, want %v", arm.name, rows, K, lanes, arm.lanes)
			}
			x := make([]float32, rows*E)
			xs := make([]uint32, rows*E)
			for i := range x {
				x[i] = float32(4 * rng.NormFloat64())
				xs[i] = math.Float32bits(x[i])
			}
			vals := map[graph.ValueID][]uint32{in: xs}
			if err := runCPUGridValues(hostArm(), codes, b.G, vals, [3][]uint32{}, nil, func(graph.ValueID) int { return rows }, nil); err != nil {
				t.Fatalf("%s/%d rows/top %d: %v", arm.name, rows, K, err)
			}
			for r := range rows {
				xr := x[r*E : (r+1)*E]
				order := sampleOrder(xr)
				var den, picked float64
				for _, v := range xr {
					den += math.Exp(float64(v - xr[order[0]]))
				}
				for p := range K {
					picked += math.Exp(float64(xr[order[p]]-xr[order[0]])) / den
				}
				for p := range K {
					if got := vals[o[0]][r*K+p]; int(got) != order[p] {
						t.Fatalf("%s/%d rows/top %d: row %d place %d holds expert %d, want %d", arm.name, rows, K, r, p, got, order[p])
					}
					want := math.Exp(float64(xr[order[p]]-xr[order[0]])) / den / picked
					if got := float64(math.Float32frombits(vals[wn][r*K+p])); math.IsNaN(got) || math.Abs(got-want) > 1e-5 {
						t.Fatalf("%s/%d rows/top %d: row %d place %d weighs %g, want %g", arm.name, rows, K, r, p, got, want)
					}
				}
			}
		}
	}
}
