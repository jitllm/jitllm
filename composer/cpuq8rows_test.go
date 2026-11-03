package composer

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// q8RowsCase is a step's rows for TestComposedCPUQ8Rows: random values at
// spread exponents, a row whose first block is all zeros, and a block whose
// scale is exactly one holding halves, which a rounding half to even settles
// the other way.
func q8RowsCase(R, W int) []float32 {
	rng := rand.New(rand.NewSource(11))
	f := make([]float32, R*W)
	for i := range f {
		f[i] = float32(rng.NormFloat64() * math.Pow(2, float64(rng.Intn(20)-10)))
	}
	clear(f[:min(W, q8Block)])
	tie := f[W : W+min(W, q8Block)]
	for i := range tie {
		tie[i] = float32(i%9) - 4.5
	}
	tie[0] = 127
	return f
}

// TestComposedCPUQ8Rows executes OpQ8Rows and OpWidenQ8Rows on the CPU at
// head widths whole blocks and ragged, over a step of several rows: each row
// word for word as oracle.QuantizeQ8 writes the zero-padded row (llama.cpp's
// quantize_row_q8_0: amax/127, halves away from zero) with the pair's sum
// term at Q8_0's bias, and each element widened as d*q.
func TestComposedCPUQ8Rows(t *testing.T) {
	const R = 5
	away := 0
	for _, W := range []int{32, 40, 80, 128, 200} {
		nb := (W + q8Block - 1) / q8Block
		rs := graph.Q8RowWords(W)
		floats := q8RowsCase(R, W)
		form := func(kind graph.OpKind) ([]Code, *graph.Graph, graph.ValueID, graph.ValueID) {
			b := graph.New()
			var in, out graph.ValueID
			if kind == graph.OpQ8Rows {
				in = b.In("f", W)
				out = b.Emit(kind, []graph.ValueID{in}, []graph.Result{{Name: "q", Width: rs, Elem: graph.ElemU32}}, nil, graph.Attrs{}, nil, nil)[0]
			} else {
				in = b.In("q", rs)
				b.G.Values[in].Elem = graph.ElemU32
				out = b.Emit(kind, []graph.ValueID{in}, []graph.Result{{Name: "f", Width: W}}, nil, graph.Attrs{}, nil, nil)[0]
			}
			b.Out(out)
			if err := graph.Verify(b.G, graph.Facts{}); err != nil {
				t.Fatal(err)
			}
			_, codes, err := Default().Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: R, Page: 1, Keys: 1},
				Device: cpuDevice(), Apart: true})
			if err != nil {
				t.Fatalf("%v at width %d: %v", kind, W, err)
			}
			return codes, b.G, in, out
		}
		run := func(kind graph.OpKind, in []uint32, n int) []uint32 {
			codes, g, iv, ov := form(kind)
			vals := map[graph.ValueID][]uint32{iv: in, ov: make([]uint32, n)}
			if err := runCPUGridValues(codes, g, vals, [3][]uint32{}, nil, func(graph.ValueID) int { return R }, nil); err != nil {
				t.Fatalf("%v at width %d: %v", kind, W, err)
			}
			return vals[ov]
		}
		rows := run(graph.OpQ8Rows, f32Words(floats), R*rs)
		pad := make([]float32, nb*q8Block)
		q := make([]int8, len(pad))
		pairs := make([]float32, 2*nb)
		for r := 0; r < R; r++ {
			clear(pad[copy(pad, floats[r*W:(r+1)*W]):])
			if err := oracle.QuantizeQ8(q, pairs, pad, 128); err != nil {
				t.Fatal(err)
			}
			for i, v := range pad {
				if s := v / pairs[2*(i/q8Block)]; math.Abs(float64(s)-math.Trunc(float64(s))) == 0.5 {
					away++
				}
			}
			want := make([]uint32, rs)
			for i, c := range q {
				want[i/4] |= uint32(uint8(c)) << (8 * (i % 4))
			}
			for b, p := range pairs {
				want[nb*q8Words+b] = math.Float32bits(p)
			}
			for j, w := range want {
				if got := rows[r*rs+j]; got != w {
					t.Fatalf("width %d: row %d word %d is %#x, want %#x", W, r, j, got, w)
				}
			}
		}
		back := run(graph.OpWidenQ8Rows, rows, R*W)
		for r := 0; r < R; r++ {
			row := rows[r*rs : (r+1)*rs]
			for i := 0; i < W; i++ {
				c := int8(row[i/4] >> (8 * (i % 4)))
				want := math.Float32frombits(row[nb*q8Words+2*(i/q8Block)]) * float32(c)
				if got := back[r*W+i]; got != math.Float32bits(want) {
					t.Fatalf("width %d: row %d element %d is %v, want %v", W, r, i, math.Float32frombits(got), want)
				}
			}
		}
	}
	if away == 0 {
		t.Fatal("no element is a half: the gate cannot see the rounding")
	}
}
