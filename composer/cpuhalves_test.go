package composer

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
)

// HalvesCase is a step's floats for TestComposedCPUHalves: random values,
// ties a rounding to even settles both ways, binary16 subnormals, values
// past its range and zeros of both signs.
func HalvesCase(n int) []float32 {
	rng := rand.New(rand.NewSource(9))
	special := []float32{1 + 1.0/2048, 1 + 3.0/2048, -(1 + 1.0/2048), 6e-8, -3e-6, 7e4, -1e6, 0, float32(math.Copysign(0, -1)), 65504, 65520}
	f := make([]float32, n)
	for i := range f {
		if i < len(special) {
			f[i] = special[i]
			continue
		}
		f[i] = float32(rng.NormFloat64() * math.Pow(2, float64(rng.Intn(30)-15)))
	}
	return f
}

// TestComposedCPUHalves executes OpHalves and OpWidenHalves on the CPU at
// ragged widths over a step of several rows: each word two halves rounded
// to nearest even, the lower element low, as quant.EncodeHalf rounds them,
// and each half widened as quant.DecodeHalf reads it. The ties must part
// from a rounding toward zero, so the gate sees the rounding.
func TestComposedCPUHalves(t *testing.T) { gateComposedCPUHalves(t, hostArm()) }
func gateComposedCPUHalves(t *testing.T, cpuOn cpuArm) {
	const R = 5
	ties := 0
	for _, W := range []int{2, 6, 18, 40} {
		floats := HalvesCase(R * W)
		form := func(kind graph.OpKind) ([]Code, *graph.Graph, graph.ValueID, graph.ValueID) {
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
			lo, hi := quant.EncodeHalf(floats[2*j]), quant.EncodeHalf(floats[2*j+1])
			if want := uint32(lo) | uint32(hi)<<16; w != want {
				t.Fatalf("width %d: word %d is %#x, want %#x (from %v, %v)", W, j, w, want, floats[2*j], floats[2*j+1])
			}
		}
		for _, f := range floats {
			// Toward zero: the mantissa's bits below binary16's dropped.
			if quant.EncodeHalf(f) != quant.EncodeHalf(math.Float32frombits(math.Float32bits(f)&^0x1fff)) {
				ties++
			}
		}
		back := run(graph.OpWidenHalves, words, R*W)
		for i, v := range back {
			h := uint16(words[i/2] >> (16 * (i % 2)))
			if want := float32(quant.DecodeHalf(h)); math.Float32bits(math.Float32frombits(v)) != math.Float32bits(want) {
				t.Fatalf("width %d: element %d is %v, want %v (half %#x)", W, i, math.Float32frombits(v), want, h)
			}
		}
	}
	if ties == 0 {
		t.Fatal("no value rounds away from zero: the gate cannot see the rounding")
	}
}
