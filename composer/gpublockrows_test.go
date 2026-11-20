package composer_test

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUBlockRows executes OpBlockRows and OpWidenBlockRows on each
// GPU backend, held as TestComposedCPUBlockRows holds the CPU's: each row
// byte for byte as oracle.QuantizeBlocks writes the zero-padded row, and
// each element widened as the container's decoder reads the same bytes. A
// card's history in a GGUF block type is written and read through these, so
// a card that narrows a row another way than the CPU device holds a
// different history.
func TestComposedGPUBlockRows(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate proved nothing")
	}
	const R = 5
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			dev := gpuDevice(d)
			for _, f := range composer.BlockFmts() {
				for _, W := range []int{32, 40, 64, 80, 128, 200} {
					nb := (W + 31) / 32
					rw := graph.BlockRowWords(f, W)
					floats := composer.Q8RowsCase(R, W)
					run := func(kind graph.OpKind, x []float32) []float32 {
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
						_, codes, err := composer.Default().Form(composer.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{},
							Step: composer.Step{Rows: R, Page: 1, Keys: 1}, Device: dev, Apart: true})
						if err != nil {
							t.Fatalf("%v %v at width %d: %v", f, kind, W, err)
						}
						got, err := runGPUGraph(d, codes, b.G, func(graph.ValueID) int { return R }, x, nil, nil)
						if err != nil {
							t.Fatalf("%v %v at width %d: %v", f, kind, W, err)
						}
						return got
					}
					rows := run(graph.OpBlockRows, floats)
					pad := make([]float32, nb*32)
					ref := make([][]byte, R)
					for r := 0; r < R; r++ {
						clear(pad[copy(pad, floats[r*W:(r+1)*W]):])
						ref[r] = make([]byte, nb*f.Bytes())
						if err := oracle.QuantizeBlocks(ref[r], pad, f); err != nil {
							t.Fatal(err)
						}
						for j, w := range oracle.BlockRow(ref[r], f, W) {
							got := math.Float32bits(rows[r*rw+j])
							if got != w && !(f == graph.BlockIQ4NL && j < (nb+1)/2 && composer.HalvesWithinAnUlp(got, w)) {
								t.Fatalf("%v width %d: row %d word %d is %#x, want %#x", f, W, r, j, got, w)
							}
						}
					}
					back := run(graph.OpWidenBlockRows, rows[:R*rw])
					deq := make([]float32, nb*32)
					for r := 0; r < R; r++ {
						if err := oracle.DequantBlocks(deq, ref[r], f); err != nil {
							t.Fatal(err)
						}
						for i := 0; i < W; i++ {
							if got := back[r*W+i]; math.Float32bits(got) != math.Float32bits(deq[i]) {
								t.Fatalf("%v width %d: row %d element %d is %v, want %v", f, W, r, i, got, deq[i])
							}
						}
					}
				}
			}
		})
	}
}
