package composer_test

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestPhasesTakeEdgeReadersOrWriters forms a router's chain on a V100's
// statement: the one-workgroup kernels reading the part's input run as
// phases of one launch, and every read of the input still launches before
// the first write of an output, the order the tier writes an output over
// its input by.
func TestPhasesTakeEdgeReadersOrWriters(t *testing.T) {
	v100 := composer.Device{Kind: composer.DeviceGPU, Slots: 80 * 2048, Lanes: 32, Units: 80, ExactDiv: true, ExactArith: true, Regs: 32, Shared: 96 << 10, VectorLoads: true}
	const E, K = 32, 4
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
	if !edgeOrder(t, b.G, x, []graph.ValueID{top[0], wn}, v100) {
		t.Errorf("the router: the first launch reading the input runs no phases")
	}
	// The input's maximum, then its exponential handed back: a reader and a
	// writer, which one launch would order apart only by its barrier.
	b = graph.New()
	x = b.In("scores", E)
	m = b.Emit(graph.OpReduce, []graph.ValueID{x}, []graph.Result{{Name: "max", Width: 1}}, nil, graph.Attrs{Fn: graph.MapMax}, nil, nil)[0]
	y := b.Emit(graph.OpMap, []graph.ValueID{m}, []graph.Result{{Name: "exp", Width: 1}}, nil, graph.Attrs{Fn: graph.MapExp}, nil, nil)[0]
	b.Out(y)
	edgeOrder(t, b.G, x, []graph.ValueID{y}, v100)
}

// TestPhasesFormTheExpertsSum forms a mixture's tail on a V100's statement:
// the picked experts' outputs weighted, summed and scaled by their sum of
// squares, reading the part's inputs in one-workgroup launches that run as
// phases of one.
func TestPhasesFormTheExpertsSum(t *testing.T) {
	v100 := composer.Device{Kind: composer.DeviceGPU, Slots: 80 * 2048, Lanes: 32, Units: 80, ExactDiv: true, ExactArith: true, Regs: 32, Shared: 96 << 10, VectorLoads: true}
	g, _, _, _ := expertsSum(2, 64)
	phased := 0
	for _, c := range formWith(t, composer.Default(), g, map[jlm.Role]jlm.Type{}, v100) {
		if slices.Contains(c.Passes, "phases") {
			phased++
		}
	}
	if phased == 0 {
		t.Errorf("no launch runs phases")
	}
}

// expertsSum is k experts' outputs of width w weighted by their router
// weights and summed, then divided by the sum's sum of squares.
func expertsSum(k, w int) (g *graph.Graph, down, wt, out graph.ValueID) {
	b := graph.New()
	down, wt = b.In("experts.down", k*w), b.In("router.w", k)
	y := b.Emit(graph.OpMap, []graph.ValueID{down, wt}, []graph.Result{{Name: "experts.weighted", Width: k * w}}, nil, graph.Attrs{Fn: graph.MapMul}, nil, nil)[0]
	var sum graph.ValueID
	for j := range k {
		s := b.Emit(graph.OpSlice, []graph.ValueID{y}, []graph.Result{{Name: "experts.out", Width: w}}, nil, graph.Attrs{Offset: j * w, Width: w}, nil, nil)[0]
		if j == 0 {
			sum = s
		} else {
			sum = b.Add(sum, s)
		}
	}
	sq := b.Emit(graph.OpMap, []graph.ValueID{sum, sum}, []graph.Result{{Name: "sq", Width: w}}, nil, graph.Attrs{Fn: graph.MapMul}, nil, nil)[0]
	ss := b.Emit(graph.OpReduce, []graph.ValueID{sq}, []graph.Result{{Name: "sumsq", Width: 1}}, nil, graph.Attrs{Fn: graph.MapAdd}, nil, nil)[0]
	out = b.Emit(graph.OpMap, []graph.ValueID{sum, ss}, []graph.Result{{Name: "scaled", Width: w}}, nil, graph.Attrs{Fn: graph.MapDiv}, nil, nil)[0]
	b.Out(out)
	return b.G, down, wt, out
}

// edgeOrder checks that g's input x is last read in a launch before the
// first that writes one of outs, and says whether the first launch reading
// x runs phases.
func edgeOrder(t *testing.T, g *graph.Graph, x graph.ValueID, outs []graph.ValueID, d composer.Device) bool {
	t.Helper()
	codes := formWith(t, composer.Default(), g, map[jlm.Role]jlm.Type{}, d)
	lastRead, firstWrite, phased := -1, len(codes), false
	for i, c := range codes {
		for _, bd := range c.Binds {
			o := bd.Operand
			if o.Kind != composer.OperandValue {
				continue
			}
			if o.Value == x && bd.Read {
				if lastRead < 0 {
					phased = slices.Contains(c.Passes, "phases")
				}
				lastRead = i
			}
			if bd.Writes && slices.Contains(outs, o.Value) {
				firstWrite = min(firstWrite, i)
			}
		}
	}
	if lastRead < 0 || firstWrite == len(codes) {
		t.Fatalf("%d launches: the input read at %d, an output first written at %d", len(codes), lastRead, firstWrite)
	}
	if lastRead >= firstWrite {
		t.Errorf("%d launches: the input last read at launch %d, an output first written at %d", len(codes), lastRead, firstWrite)
	}
	return phased
}
