package jit

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sort"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/x86"
)

// TestComposedCPUGroupedRouter runs a grouped router on the CPU device as
// the front end states it (engine/model/front.go): each group's two best
// scores summed, the groups ranked, the experts of all but the kept groups
// masked, the best K of the rest taken, at a token's row and a prompt's,
// against the choice made in Go.
func TestComposedCPUGroupedRouter(t *testing.T) {
	const E, G, kept, K = 32, 8, 3, 4
	rng := rand.New(rand.NewSource(17))
	ran := 0
	for _, vnni := range []bool{true, false} {
		for _, slots := range []int{1, 6} {
			for _, R := range []int{1, 5, 64} {
				name := fmt.Sprintf("vnni %v/%d slots/%d rows", vnni, slots, R)
				b := graph.New()
				score := b.In("score", E)
				span := E / G
				top := b.Emit(graph.OpTopK, []graph.ValueID{score}, []graph.Result{{Name: "group.sel", Width: G * 2, Elem: graph.ElemU32},
					{Name: "group.best", Width: G * 2}}, nil, graph.Attrs{TopK: 2, Span: span}, nil, nil)
				gs := b.Emit(graph.OpReduce, []graph.ValueID{top[1]}, []graph.Result{{Name: "group.score", Width: G}}, nil,
					graph.Attrs{Fn: graph.MapAdd, Span: 2}, nil, nil)[0]
				rank := b.Emit(graph.OpRank, []graph.ValueID{gs}, []graph.Result{{Name: "group.rank", Width: G, Elem: graph.ElemU32}},
					nil, graph.Attrs{}, nil, nil)[0]
				keep := b.Emit(graph.OpMap, []graph.ValueID{rank}, []graph.Result{{Name: "group.kept", Width: G}}, nil,
					graph.Attrs{Fn: graph.MapLess, Const: kept}, nil, nil)[0]
				masked := b.Emit(graph.OpMap, []graph.ValueID{keep, score}, []graph.Result{{Name: "kept", Width: E}}, nil,
					graph.Attrs{Fn: graph.MapWhere, Const: math.Inf(-1)}, nil, nil)[0]
				sel := b.Emit(graph.OpTopK, []graph.ValueID{masked}, []graph.Result{{Name: "sel", Width: K, Elem: graph.ElemU32},
					{Name: "best", Width: K}}, nil, graph.Attrs{TopK: K}, nil, nil)[0]
				b.Out(sel)
				dev := cpuDevice()
				dev.Slots = slots
				if !vnni {
					dev.ISA &^= uint64(x86.ISAVNNI)
				}
				_, codes, err := Default().Form(Request{Graph: b.G, Step: Step{Rows: R, Page: 64, Keys: 64, ActWindow: 32, Seqs: 1}, Device: dev})
				if err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				sw := make([]uint32, R*E)
				want := make([]uint32, 0, R*K)
				for r := range R {
					row := make([]float32, E)
					for i, v := range rng.Perm(E) {
						row[i] = float32(v)/4 - 3
						sw[r*E+i] = math.Float32bits(row[i])
					}
					want = append(want, groupedChoice(row, G, kept, K)...)
				}
				vals := map[graph.ValueID][]uint32{score: sw}
				if err := runCPUGridValues(codes, b.G, vals, [3][]uint32{{0}, {0}, {0}}, nil, func(graph.ValueID) int { return R }, nil); err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				if got := vals[sel][:R*K]; !slices.Equal(got, want) {
					t.Errorf("%s: chose %v, want %v", name, got, want)
				}
				ran++
			}
		}
	}
	if ran == 0 {
		t.Fatal("no arm ran: this gate proved nothing")
	}
}

// groupedChoice is the K experts a grouped router keeps, best first: the
// kept groups by the sum of each group's two best scores, then the best K
// experts among them.
func groupedChoice(row []float32, groups, kept, k int) []uint32 {
	span := len(row) / groups
	gs := make([]float32, groups)
	for g := range groups {
		s := slices.Clone(row[g*span : (g+1)*span])
		sort.Slice(s, func(i, j int) bool { return s[i] > s[j] })
		gs[g] = s[0] + s[1]
	}
	order := make([]int, groups)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool { return gs[order[i]] > gs[order[j]] })
	keep := map[int]bool{}
	for _, g := range order[:kept] {
		keep[g] = true
	}
	var ids []int
	for e := range row {
		if keep[e/span] {
			ids = append(ids, e)
		}
	}
	sort.SliceStable(ids, func(i, j int) bool { return row[ids[i]] > row[ids[j]] })
	out := make([]uint32, k)
	for i := range k {
		out[i] = uint32(ids[i])
	}
	return out
}
