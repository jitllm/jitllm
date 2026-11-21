package composer_test

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/graph"
)

// TestFitGivesAKernelItsElements forms, for a GPU, an iteration of a few
// rows and of many: a kernel of one workgroup whose threads are its
// elements launches as many threads as its elements, rounded up to a
// subgroup (Fit), and one of more elements than the profile's workgroup
// (Threads, 128) keeps that width. A fitted code names the pass.
func TestFitGivesAKernelItsElements(t *testing.T) {
	const group = 128
	dev := composer.Device{Kind: composer.DeviceGPU, Slots: 16384, Lanes: 32, Units: 10, Shared: 32 << 10}
	for _, c := range []struct{ n, groups, width int }{{1, 1, 32}, {7, 1, 32}, {32, 1, 32}, {33, 1, 64}, {100, 1, group}, {256, 2, group}} {
		b := graph.New()
		x := b.In("x", c.n)
		terms := []graph.Term{
			{Fn: graph.MapAdd, Args: [3]graph.Arg{{Kind: graph.ArgIn}, {Kind: graph.ArgConst, Const: 1}}},
			{Fn: graph.MapFma, Args: [3]graph.Arg{{Kind: graph.ArgState}, {Kind: graph.ArgState}, {Kind: graph.ArgIn}}},
		}
		b.Out(b.Emit(graph.OpIterate, []graph.ValueID{x}, []graph.Result{{Name: "y", Width: c.n}}, nil,
			graph.Attrs{Taps: 1, Span: 64, Expr: terms}, nil, nil)[0])
		_, codes, err := composer.Default().Form(composer.Request{Graph: b.G, Step: composer.Step{Rows: 1, Page: 64, Keys: 1}, Device: dev, Apart: true})
		if err != nil {
			t.Fatal(err)
		}
		if len(codes) != 1 || codes[0].GPU == nil {
			t.Fatalf("an iteration of %d rows formed %d codes", c.n, len(codes))
		}
		k := codes[0]
		if k.Groups != c.groups || k.GPU.Group[0] != c.width {
			t.Errorf("an iteration of %d rows launches %d workgroups of %d, want %d of %d", c.n, k.Groups, k.GPU.Group[0], c.groups, c.width)
		}
		if fitted := slices.Contains(k.Passes, "fit"); fitted != (c.width < group) {
			t.Errorf("an iteration of %d rows: fit among its passes %v, want %v", c.n, fitted, c.width < group)
		}
	}
}
