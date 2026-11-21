package composer

import (
	"math"
	"math/rand"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUTopKOrdersAWholeRow forms a top-K too wide to rank pairwise
// -- the whole row, and a selection's kept -- on the CPU and holds every
// place to the definition: the larger value first, the lower index of two
// equal (sampleOrder), the value its element's own bits. Rows of few
// distinct values and of signed zeros make the ties; a ragged last block and
// several rows make the edges. The sort's kernels ran (sortedPlaces).
func TestComposedCPUTopKOrdersAWholeRow(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	for _, c := range []struct{ rows, S, K int }{
		{1, 2000, 2000}, {1, 4096, 4096}, {3, 4096, 4096}, {1, 5000, 5000}, {2, 6000, 3000},
	} {
		if !sortPays(c.K) {
			t.Fatalf("top %d of %d: the sort does not pay, and the gate would hold the ranking", c.K, c.S)
		}
		b := graph.New()
		in := b.In("x", c.S)
		o := b.Emit(graph.OpTopK, []graph.ValueID{in}, []graph.Result{{Name: "top.ids", Width: c.K, Elem: graph.ElemU32},
			{Name: "top.vals", Width: c.K}}, nil, graph.Attrs{TopK: c.K}, nil, nil)
		b.Out(o[0], o[1])
		_, codes, err := Default().Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: c.rows, Page: 64, Keys: 1},
			Device: cpuDevice(hostArm())})
		if err != nil {
			t.Fatalf("top %d of %d, %d rows: forming: %v", c.K, c.S, c.rows, err)
		}
		if !slices.ContainsFunc(codes, func(k Code) bool {
			return slices.ContainsFunc(k.Binds, func(b Binding) bool { return strings.Contains(b.Operand.Name, ".sort.") })
		}) {
			t.Fatalf("top %d of %d: no kernel of the sort formed", c.K, c.S)
		}
		x := make([]float32, c.rows*c.S)
		for i := range x {
			switch r := i / c.S; {
			case r%2 == 0:
				x[i] = float32(rng.Intn(40) - 20)
				if x[i] == 0 && rng.Intn(2) == 0 {
					x[i] = float32(math.Copysign(0, -1))
				}
			default:
				x[i] = float32(3 * rng.NormFloat64())
			}
		}
		xs := make([]uint32, len(x))
		for i, v := range x {
			xs[i] = math.Float32bits(v)
		}
		vals := map[graph.ValueID][]uint32{in: xs}
		if err := runCPUGridValues(hostArm(), codes, b.G, vals, [3][]uint32{}, nil, func(graph.ValueID) int { return c.rows }, nil); err != nil {
			t.Fatalf("top %d of %d, %d rows: %v", c.K, c.S, c.rows, err)
		}
		for r := range c.rows {
			xr := x[r*c.S : (r+1)*c.S]
			order := sampleOrder(xr)
			for p := range c.K {
				if got := vals[o[0]][r*c.K+p]; int(got) != order[p] {
					t.Fatalf("top %d of %d: row %d place %d holds %d (%g), want %d (%g)", c.K, c.S, r, p, got, xr[min(int(got), c.S-1)],
						order[p], xr[order[p]])
				}
				if got, want := vals[o[1]][r*c.K+p], math.Float32bits(xr[order[p]]); got != want {
					t.Fatalf("top %d of %d: row %d place %d values %#x, want %#x", c.K, c.S, r, p, got, want)
				}
			}
		}
		t.Logf("top %d of %d, %d rows: %d kernels, every place the definition's", c.K, c.S, c.rows, len(codes))
	}
}
