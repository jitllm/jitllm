package composer_test

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUHalves executes OpHalves and OpWidenHalves on each GPU
// backend at ragged widths over a step of several rows, held as
// TestComposedCPUHalves holds the CPU's: each word two halves rounded to
// nearest even, the lower element low (quant.EncodeHalf), and each half
// widened as quant.DecodeHalf reads it. GLSL.std.450's PackHalf2x16 leaves
// its rounding to the driver, and one rounds toward zero; the ties must part
// from that, so the gate sees the rounding on every driver.
func TestComposedGPUHalves(t *testing.T) {
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
			ties := 0
			for _, W := range []int{2, 6, 18, 40} {
				floats := composer.HalvesCase(R * W)
				run := func(kind graph.OpKind, x []float32) []float32 {
					b := graph.New()
					var in, out graph.ValueID
					if kind == graph.OpHalves {
						in = b.In("f", W)
						out = b.Emit(kind, []graph.ValueID{in}, []graph.Result{{Name: "h", Width: W / 2, Elem: graph.ElemU32}}, nil, graph.Attrs{}, nil, nil)[0]
					} else {
						in = b.In("h", W/2)
						b.G.Values[in].Elem = graph.ElemU32
						out = b.Emit(kind, []graph.ValueID{in}, []graph.Result{{Name: "f", Width: W}}, nil, graph.Attrs{}, nil, nil)[0]
					}
					b.Out(out)
					if err := graph.Verify(b.G, graph.Facts{}); err != nil {
						t.Fatal(err)
					}
					_, codes, err := composer.Default().Form(composer.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{},
						Step: composer.Step{Rows: R, Page: 1, Keys: 1}, Device: dev, Apart: true})
					if err != nil {
						t.Fatalf("%v at width %d: %v", kind, W, err)
					}
					got, err := runGPUGraph(d, codes, b.G, func(graph.ValueID) int { return R }, x, nil, nil)
					if err != nil {
						t.Fatalf("%v at width %d: %v", kind, W, err)
					}
					return got
				}
				words := run(graph.OpHalves, floats)
				for j := range R * W / 2 {
					w := math.Float32bits(words[j])
					lo, hi := quant.EncodeHalf(floats[2*j]), quant.EncodeHalf(floats[2*j+1])
					if want := uint32(lo) | uint32(hi)<<16; w != want {
						t.Fatalf("width %d: word %d is %#x, want %#x (from %v, %v)", W, j, w, want, floats[2*j], floats[2*j+1])
					}
				}
				for _, f := range floats {
					if quant.EncodeHalf(f) != quant.EncodeHalf(math.Float32frombits(math.Float32bits(f)&^0x1fff)) {
						ties++
					}
				}
				back := run(graph.OpWidenHalves, words[:R*W/2])
				for i := range R * W {
					h := uint16(math.Float32bits(words[i/2]) >> (16 * (i % 2)))
					if want := float32(quant.DecodeHalf(h)); math.Float32bits(back[i]) != math.Float32bits(want) {
						t.Fatalf("width %d: element %d is %v, want %v (half %#x)", W, i, back[i], want, h)
					}
				}
			}
			if ties == 0 {
				t.Fatal("no value rounds away from zero: the gate cannot see the rounding")
			}
		})
	}
}
