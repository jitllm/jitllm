package composer

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// TestComposedCPUBF16Halves executes OpHalves and OpWidenHalves under
// Attrs.BF16 on the CPU at ragged widths over several rows: each word the
// two bfloat16s oracle.BF16 rounds its floats to (ggml's rounding, ties to
// even, a NaN quieted), and each element widened back as the float whose top
// half it is. The rows hold ties both ways, a NaN and the infinities.
func TestComposedCPUBF16Halves(t *testing.T) {
	cpuOn := hostArm()
	const R = 5
	ties := 0
	for _, W := range []int{2, 6, 18, 40, 256} {
		floats := HalvesCase(R * W)
		for i := 0; i < len(floats); i += 7 {
			floats[i] = math.Float32frombits(math.Float32bits(floats[i])&0xffff0000 | 0x8000)
		}
		floats[1] = float32(math.NaN())
		floats[3%len(floats)] = float32(math.Inf(1))
		floats[len(floats)-1] = float32(math.Inf(-1))
		form := func(kind graph.OpKind) ([]Code, *graph.Graph, graph.ValueID, graph.ValueID) {
			b := graph.New()
			var in, out graph.ValueID
			at := graph.Attrs{BF16: true}
			if kind == graph.OpHalves {
				in = b.In("f", W)
				out = b.Emit(kind, []graph.ValueID{in}, []graph.Result{{Name: "h", Width: W / 2, Elem: graph.ElemU32}}, nil, at, nil, nil)[0]
			} else {
				in = b.In("h", W/2)
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
				t.Fatalf("%v at width %d: %v", kind, W, err)
			}
			return codes, b.G, in, out
		}
		run := func(kind graph.OpKind, in []uint32, n int) []uint32 {
			codes, g, iv, ov := form(kind)
			vals := map[graph.ValueID][]uint32{iv: in, ov: make([]uint32, n)}
			if err := runCPUGridValues(cpuOn, codes, g, vals, [3][]uint32{}, nil, func(graph.ValueID) int { return R }, nil); err != nil {
				t.Fatalf("%v at width %d: %v", kind, W, err)
			}
			return vals[ov]
		}
		words := run(graph.OpHalves, f32Words(floats), R*W/2)
		for j, w := range words {
			lo, hi := oracle.BF16(floats[2*j]), oracle.BF16(floats[2*j+1])
			if want := uint32(lo) | uint32(hi)<<16; w != want {
				t.Fatalf("width %d: word %d is %#x, want %#x (from %v, %v)", W, j, w, want, floats[2*j], floats[2*j+1])
			}
		}
		for _, f := range floats {
			if math.Float32bits(f)&0xffff == 0x8000 {
				ties++
			}
		}
		back := run(graph.OpWidenHalves, words, R*W)
		for i, b := range back {
			if want := words[i/2] >> (16 * (i % 2)) << 16; b != want {
				t.Fatalf("width %d: element %d is %#x, want %#x", W, i, b, want)
			}
		}
	}
	if ties == 0 {
		t.Fatal("no element is a tie: the gate cannot see the rounding")
	}
}
