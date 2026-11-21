package composer

import (
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUReduceIndexWords forms OpReduce over index rows
// (graph.ElemU32) on the CPU device -- a sum, a max and a min, in sequence
// and in tree order -- at ragged widths, each held to the words' own
// unsigned reduction: a sum that wraps, a max and a min by unsigned order.
// A min in sequence starts at the largest word, a constant the CPU lowering
// does not start a vector sum from yet: it is held to the right word or a
// refusal when formed, never a wrong word. Words added as floats part from
// it. A sum in sequence of a row wider than a vector is held to a vector
// loop: a word a lane, not a word a trip.
func TestComposedCPUReduceIndexWords(t *testing.T) { gateComposedCPUReduceIndexWords(t, hostArm()) }
func gateComposedCPUReduceIndexWords(t *testing.T, cpuOn cpuArm) {
	rng := rand.New(rand.NewSource(11))
	ran, refused, vectored := 0, 0, 0
	for _, order := range []graph.Order{graph.OrderSeq, graph.OrderTree} {
		for _, fn := range []graph.MapFn{graph.MapAdd, graph.MapMax, graph.MapMin} {
			for _, n := range []int{1, 2, 3, 7, 8, 9, 15, 16, 17, 33, 100} {
				b := graph.New()
				x := b.In("x", n)
				b.G.Values[x].Elem = graph.ElemU32
				s := b.Emit(graph.OpReduce, []graph.ValueID{x}, []graph.Result{{Name: "r", Width: 1, Elem: graph.ElemU32}}, nil,
					graph.Attrs{Fn: fn, Span: n, Order: order}, nil, nil)[0]
				b.Out(s)
				ks, codes, err := Default().FormKernels(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: 1, Page: 64, Keys: 1}, Device: cpuDevice(cpuOn)})
				if err != nil && fn == graph.MapMin && order == graph.OrderSeq {
					refused++
					continue
				}
				if err != nil {
					t.Fatalf("%v in order %d at width %d: %v", fn, order, n, err)
				}
				if fn == graph.MapAdd && order == graph.OrderSeq && n >= 16 {
					lanes := 0
					for _, k := range ks {
						for _, l := range k.Loops {
							lanes = max(lanes, l.Lanes)
						}
					}
					if lanes == 0 {
						t.Fatalf("a sum of %d words formed no vector loop: a word a trip", n)
					}
					vectored++
				}
				xs := make([]uint32, n)
				for i := range xs {
					xs[i] = rng.Uint32()
				}
				vals := map[graph.ValueID][]uint32{x: xs}
				if err := runCPUValues(cpuOn, codes, b.G, vals, [3][]uint32{}); err != nil {
					t.Fatalf("%v in order %d at width %d: %v", fn, order, n, err)
				}
				if got, want := vals[s][0], reduceWords(fn, xs); got != want {
					t.Fatalf("%v in order %d at width %d: %#x, want %#x", fn, order, n, got, want)
				}
				ran++
			}
		}
	}
	if ran < 50 {
		t.Fatalf("%d reductions ran: the gate held too few", ran)
	}
	t.Logf("%d reductions of index words held on the CPU device, %d sums of them a vector loop, %d word mins refused when formed", ran, vectored, refused)
}

// reduceWords is fn over xs as unsigned words.
func reduceWords(fn graph.MapFn, xs []uint32) uint32 {
	r := xs[0]
	for _, v := range xs[1:] {
		switch {
		case fn == graph.MapAdd:
			r += v
		case fn == graph.MapMax && v > r, fn == graph.MapMin && v < r:
			r = v
		}
	}
	return r
}
