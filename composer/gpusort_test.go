package composer_test

import (
	"math"
	"math/rand"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUTopKOrdersAWholeRow is TestComposedCPUTopKOrdersAWholeRow
// on every GPU device present: a top-K of the whole row and of a selection's
// kept, ordered by the radix sort, every place the larger value first and
// the lower index of two equal, the value its element's bits, guard words
// past every output.
func TestComposedGPUTopKOrdersAWholeRow(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate runs on one")
	}
	rng := rand.New(rand.NewSource(19))
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			for _, c := range []struct{ S, K int }{{2000, 2000}, {4096, 4096}, {5000, 5000}, {6000, 3000}} {
				for _, ties := range []bool{true, false} {
					b := graph.New()
					in := b.In("x", c.S)
					o := b.Emit(graph.OpTopK, []graph.ValueID{in}, []graph.Result{{Name: "top.ids", Width: c.K, Elem: graph.ElemU32},
						{Name: "top.vals", Width: c.K}}, nil, graph.Attrs{TopK: c.K}, nil, nil)
					b.Out(o[0], o[1])
					_, codes, err := composer.Default().Form(composer.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{},
						Step: composer.Step{Rows: 1, Page: 64, Keys: 1}, Device: gpuDevice(d)})
					if err != nil {
						t.Fatalf("top %d of %d: forming: %v", c.K, c.S, err)
					}
					if !slices.ContainsFunc(codes, func(k composer.Code) bool {
						return slices.ContainsFunc(k.Binds, func(b composer.Binding) bool { return strings.Contains(b.Operand.Name, ".sort.") })
					}) {
						t.Fatalf("top %d of %d: no kernel of the sort formed", c.K, c.S)
					}
					op, err := compileLone(d, codes)
					if err != nil {
						t.Fatalf("top %d of %d: %v", c.K, c.S, err)
					}
					x := make([]float32, c.S)
					xs := make([]uint32, c.S)
					for i := range x {
						if ties {
							x[i] = float32(rng.Intn(40) - 20)
							if x[i] == 0 && rng.Intn(2) == 0 {
								x[i] = float32(math.Copysign(0, -1))
							}
						} else {
							x[i] = float32(3 * rng.NormFloat64())
						}
						xs[i] = math.Float32bits(x[i])
					}
					vals := map[graph.ValueID][]uint32{in: xs}
					err = runGraphValues(d, codes, op.kerns, b.G, vals)
					op.close()
					if err != nil {
						t.Fatalf("top %d of %d: %v", c.K, c.S, err)
					}
					order := make([]int, c.S)
					for i := range order {
						order[i] = i
					}
					slices.SortStableFunc(order, func(a, b int) int {
						switch {
						case x[a] > x[b]:
							return -1
						case x[a] < x[b]:
							return 1
						}
						return 0
					})
					for p := range c.K {
						if got := vals[o[0]][p]; int(got) != order[p] {
							t.Fatalf("top %d of %d (ties %v): place %d holds %d, want %d (%g)", c.K, c.S, ties, p, got, order[p], x[order[p]])
						}
						if got, want := vals[o[1]][p], xs[order[p]]; got != want {
							t.Fatalf("top %d of %d (ties %v): place %d values %#x, want %#x", c.K, c.S, ties, p, got, want)
						}
					}
				}
			}
		})
	}
}
