package composer

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/graph"
)

// TestCPUStoreReadsItsIteration forms on the CPU the group ranking of
// DeepSeek-V3's router: each group's score is the sum of its best two
// experts, and each group's rank is counted over the others, so the
// outer loop's iteration is read by the rank's store alone. The binder
// gives that iteration a register; without one the x86 lowering has no
// address to store at.
func TestCPUStoreReadsItsIteration(t *testing.T) {
	if !LowersCPU(uint64(cpu.CPU().ISA())) {
		t.Skip("no CPU lowering on this host")
	}
	for _, c := range []struct{ e, g int }{{8, 2}, {16, 4}, {64, 8}} {
		b := graph.New()
		x := b.In("x", c.e)
		o := b.Emit(graph.OpTopK, []graph.ValueID{x}, []graph.Result{{Name: "g.sel", Width: 2 * c.g, Elem: graph.ElemU32},
			{Name: "g.best", Width: 2 * c.g}}, nil, graph.Attrs{TopK: 2, Span: c.e / c.g}, nil, nil)
		s := b.Emit(graph.OpReduce, []graph.ValueID{o[1]}, []graph.Result{{Name: "g.score", Width: c.g}}, nil,
			graph.Attrs{Fn: graph.MapAdd, Span: 2}, nil, nil)[0]
		rank := b.Emit(graph.OpRank, []graph.ValueID{s}, []graph.Result{{Name: "g.rank", Width: c.g, Elem: graph.ElemU32}}, nil,
			graph.Attrs{}, nil, nil)[0]
		mapc := func(name string, fn graph.MapFn, k float64, width int, in ...graph.ValueID) graph.ValueID {
			return b.Emit(graph.OpMap, in, []graph.Result{{Name: name, Width: width}}, nil, graph.Attrs{Fn: fn, Const: k}, nil, nil)[0]
		}
		kept := mapc("g.kept", graph.MapLess, 1, c.g, rank)
		choose := mapc("kept", graph.MapWhere, math.Inf(-1), c.e, kept, x)
		sel := b.Emit(graph.OpTopK, []graph.ValueID{choose}, []graph.Result{{Name: "sel", Width: 2, Elem: graph.ElemU32},
			{Name: "best", Width: 2}}, nil, graph.Attrs{TopK: 2}, nil, nil)
		b.Out(b.Emit(graph.OpTake, []graph.ValueID{x, sel[0]}, []graph.Result{{Name: "w", Width: 2}}, nil, graph.Attrs{}, nil, nil)[0])
		req := Request{Graph: b.G, Step: Step{Rows: 1, Page: 32, Keys: 128, ActWindow: 32},
			Device: Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}}
		if _, _, err := Default().Form(req); err != nil {
			t.Errorf("%d experts in %d groups: %v", c.e, c.g, err)
		}
	}
}
