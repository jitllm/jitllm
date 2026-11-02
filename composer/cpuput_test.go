package composer

import (
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// PutGraph is OpPut over a row of w floats with k indices and values.
func PutGraph(w, k int) (g *graph.Graph, row, idx, vals, out graph.ValueID) {
	b := graph.New()
	row, idx, vals = b.In("row", w), b.In("idx", k), b.In("vals", k)
	b.G.Values[idx].Elem = graph.ElemU32
	out = b.Emit(graph.OpPut, []graph.ValueID{row, idx, vals}, []graph.Result{{Name: "put", Width: w}}, nil, graph.Attrs{}, nil, nil)[0]
	b.Out(out)
	return b.G, row, idx, vals, out
}

// PutRef is OpPut's definition: row with each value at its index, an index
// past the row naming none.
func PutRef(row []float32, idx []uint32, vals []float32) []float32 {
	out := slices.Clone(row)
	for j, ix := range idx {
		if int(ix) < len(row) {
			out[ix] = vals[j]
		}
	}
	return out
}

// PutCase is one row of PutGraph's inputs: indices across the row and past
// it (some repeated, each repeat with its value), and the values an index's.
func PutCase(rng *rand.Rand, w, k int) (row []float32, idx []uint32, vals []float32) {
	row, idx, vals = make([]float32, w), make([]uint32, k), make([]float32, k)
	for i := range row {
		row[i] = float32(rng.NormFloat64())
	}
	for j := range idx {
		idx[j] = uint32(rng.Intn(w + 3))
		if j > 0 && rng.Intn(4) == 0 {
			idx[j] = idx[rng.Intn(j)]
		}
		if rng.Intn(9) == 0 {
			idx[j] = math.MaxUint32
		}
		vals[j] = -7 - float32(idx[j]%1000)/4
	}
	return row, idx, vals
}

// TestComposedCPUPut holds OpPut on the CPU device to its definition, bit
// for bit, at ragged widths, index counts and rows, indices past the row and
// repeated among them. The violation: the row unput, which the comparison
// must tell apart.
func TestComposedCPUPut(t *testing.T) {
	rng := rand.New(rand.NewSource(23))
	ran, parted := 0, 0
	for _, w := range []int{1, 3, 8, 40, 129, 1000} {
		for _, k := range []int{1, 5, 16, 33} {
			g, row, idx, vals, out := PutGraph(w, k)
			if err := graph.Verify(g, graph.Facts{}); err != nil {
				t.Fatal(err)
			}
			for _, R := range []int{1, 3} {
				_, codes, err := Default().Form(Request{Graph: g, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: R, Page: 64, Keys: 1}, Device: cpuDevice()})
				if err != nil {
					t.Fatalf("%d indices into %d at %d rows: %v", k, w, R, err)
				}
				var rows, vs []float32
				var ix []uint32
				for range R {
					r, i, v := PutCase(rng, w, k)
					rows, ix, vs = append(rows, r...), append(ix, i...), append(vs, v...)
				}
				got := make([]uint32, R*w)
				words := func(f []float32) []uint32 {
					u := make([]uint32, len(f))
					for i, x := range f {
						u[i] = math.Float32bits(x)
					}
					return u
				}
				m := map[graph.ValueID][]uint32{row: words(rows), idx: slices.Clone(ix), vals: words(vs), out: got}
				if err := runCPUGridValues(codes, g, m, [3][]uint32{}, nil, func(graph.ValueID) int { return R }, nil); err != nil {
					t.Fatalf("%d indices into %d at %d rows: %v", k, w, R, err)
				}
				for r := range R {
					want := PutRef(rows[r*w:(r+1)*w], ix[r*k:(r+1)*k], vs[r*k:(r+1)*k])
					for i := range w {
						if m[out][r*w+i] != math.Float32bits(want[i]) {
							t.Fatalf("%d indices into %d, row %d of %d: element %d is %v, want %v", k, w, r, R, i,
								math.Float32frombits(m[out][r*w+i]), want[i])
						}
					}
					if !slices.Equal(want, rows[r*w:(r+1)*w]) {
						parted++
					}
				}
				ran++
			}
		}
	}
	if ran == 0 || parted == 0 {
		t.Fatalf("%d shapes, %d rows parted from the row unput: the gate cannot see a put", ran, parted)
	}
	t.Logf("%d shapes held bit for bit; %d rows parted from the row unput", ran, parted)
}
