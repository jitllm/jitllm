package composer

import (
	"fmt"
	"math"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// blockFmts is every llama.cpp block format a history row is held in.
func blockFmts() []graph.BlockFmt {
	return []graph.BlockFmt{graph.BlockQ4_0, graph.BlockQ4_1, graph.BlockQ5_0, graph.BlockQ5_1}
}

// TestComposedCPUBlockRows executes OpBlockRows and OpWidenBlockRows on the
// CPU in each block format at head widths whole blocks and ragged, over a
// step of several rows (q8RowsCase's: a zero block and a block of halves):
// each row byte for byte as oracle.QuantizeBlocks writes the zero-padded row
// (ggml's reference quantizers), and each element widened as the
// container's decoder reads the same bytes.
func TestComposedCPUBlockRows(t *testing.T) { gateComposedCPUBlockRows(t, hostArm()) }
func gateComposedCPUBlockRows(t *testing.T, cpuOn cpuArm) {
	const R = 5
	for _, f := range blockFmts() {
		for _, W := range []int{32, 40, 80, 128, 200} {
			nb := (W + 31) / 32
			rw := graph.BlockRowWords(f, W)
			floats := q8RowsCase(R, W)
			form := func(kind graph.OpKind) ([]Code, *graph.Graph, graph.ValueID, graph.ValueID) {
				b := graph.New()
				var in, out graph.ValueID
				at := graph.Attrs{Block: f}
				if kind == graph.OpBlockRows {
					in = b.In("f", W)
					out = b.Emit(kind, []graph.ValueID{in}, []graph.Result{{Name: "q", Width: rw, Elem: graph.ElemU32}}, nil, at, nil, nil)[0]
				} else {
					in = b.In("q", rw)
					b.G.Values[in].Elem = graph.ElemU32
					out = b.Emit(kind, []graph.ValueID{in}, []graph.Result{{Name: "f", Width: W}}, nil, at, nil, nil)[0]
				}
				b.Out(out)
				if err := graph.Verify(b.G, graph.Facts{}); err != nil {
					t.Fatal(err)
				}
				_, codes, err := Default().Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: R, Page: 1, Keys: 1},
					Device: cpuDevice(cpuOn), Apart: true})
				if err != nil {
					t.Fatalf("%v %v at width %d: %v", f, kind, W, err)
				}
				return codes, b.G, in, out
			}
			run := func(kind graph.OpKind, in []uint32, n int) []uint32 {
				codes, g, iv, ov := form(kind)
				vals := map[graph.ValueID][]uint32{iv: in, ov: make([]uint32, n)}
				if err := runCPUGridValues(cpuOn, codes, g, vals, [3][]uint32{}, nil, func(graph.ValueID) int { return R }, nil); err != nil {
					t.Fatalf("%v %v at width %d: %v", f, kind, W, err)
				}
				return vals[ov]
			}
			rows := run(graph.OpBlockRows, f32Words(floats), R*rw)
			pad := make([]float32, nb*32)
			ggml := make([][]byte, R)
			for r := 0; r < R; r++ {
				clear(pad[copy(pad, floats[r*W:(r+1)*W]):])
				ggml[r] = make([]byte, nb*f.Bytes())
				if err := oracle.QuantizeBlocks(ggml[r], pad, f); err != nil {
					t.Fatal(err)
				}
				for j, w := range oracle.BlockRow(ggml[r], f, W) {
					if got := rows[r*rw+j]; got != w {
						t.Fatalf("%v width %d: row %d word %d is %#x, want %#x", f, W, r, j, got, w)
					}
				}
			}
			back := run(graph.OpWidenBlockRows, rows, R*W)
			deq := make([]float32, nb*32)
			for r := 0; r < R; r++ {
				if err := oracle.DequantBlocks(deq, ggml[r], f); err != nil {
					t.Fatal(err)
				}
				for i := 0; i < W; i++ {
					if got := back[r*W+i]; got != math.Float32bits(deq[i]) {
						t.Fatalf("%v width %d: row %d element %d is %v, want %v", f, W, r, i, math.Float32frombits(got), deq[i])
					}
				}
			}
		}
	}
}

// TestComposedCPUAttendBlocks is TestComposedCPUAttendQ8 over pools of each
// block format: the history's rows as oracle.QuantizeBlocks writes them, the
// row's own appended through the composed quantizer, held to the definition
// over every row as the container's decoder reads it.
func TestComposedCPUAttendBlocks(t *testing.T) {
	for _, f := range blockFmts() {
		for _, sh := range []struct{ heads, kvh, dim int }{{32, 8, 64}, {16, 8, 128}, {15, 5, 64}, {3, 1, 6}, {4, 2, 40}} {
			t.Run(fmt.Sprintf("%v_%dq%dkv%dd", f, sh.heads, sh.kvh, sh.dim), func(t *testing.T) { composedCPUAttendQ8(t, hostArm(), sh.heads, sh.kvh, sh.dim, f) })
		}
	}
}
