package composer

import (
	"math"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// TestComposedCPUPicturePILPadded resizes a picture of its own size by
// Pillow's bicubic filter and pads it to whole units (graph.Attrs.PicPad):
// a 97x61 picture to 40x26 inside a 64-pixel class, laid out over a 48x32
// grid of 16-pixel units -- every resized byte normalised by a mean and std
// of 0.5, every pad cell a zero sample normalised (-1), and zero past the
// grid. The same picture unpadded, laid out 40 wide, is the violation.
func TestComposedCPUPicturePILPadded(t *testing.T) {
	const (
		W, w, h, ow, oh, side, unit = 48, 97, 61, 40, 26, 128, 16
		pw, ph                      = 48, 32
	)
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	rgb := oracle.PILPattern(w, h)
	rgba := make([]byte, w*h*4)
	for i := range w * h {
		copy(rgba[4*i:], rgb[3*i:3*i+3])
		rgba[4*i+3] = 255
	}
	words := unsafe.Slice((*uint32)(unsafe.Pointer(&rgba[0])), len(rgba)/4)
	K := max(PILTapCount(graph.FilterCubic, side, ow), PILTapCount(graph.FilterCubic, side, oh))
	rows := (pw*ph*3 + W - 1) / W
	run := func(pad int) []float32 {
		b := graph.NewBlock(-1)
		b.In("unused", 1)
		b.Out(b.Emit(graph.OpPicture, nil, []graph.Result{{Name: "pixels", Width: W}}, nil,
			graph.Attrs{Pixels: graph.PixRGBA8, PicW: 64, PicH: 64, PicSide: side, PicLive: true, PicPad: pad,
				Chain: graph.ChainPIL, Filter: graph.FilterCubic, Taps: K,
				Mean: [3]float64{0.5, 0.5, 0.5}, Std: [3]float64{0.5, 0.5, 0.5}}, nil, nil)[0])
		if err := graph.Verify(b.G, graph.Facts{}); err != nil {
			t.Fatal(err)
		}
		_, codes, err := r.Form(Request{Graph: b.G, Step: Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
		if err != nil {
			t.Fatal(err)
		}
		req := make([]uint32, RequestPicTaps+(64+64)*PILEntryWords(K))
		req[RequestPicW], req[RequestPicH], req[RequestPicStride] = w, h, w*4
		req[RequestPicOW], req[RequestPicOH] = ow, oh
		PILTaps(graph.FilterCubic, w, ow, 0, ow, K, req[RequestPicTaps:])
		PILTaps(graph.FilterCubic, h, oh, 0, oh, K, req[RequestPicTaps+ow*PILEntryWords(K):])
		got, err := runCPUBound(hostArm(), codes, []float32{0}, rows*W, [3][]uint32{}, nil, func(o Operand) []uint32 {
			switch o.Name {
			case "request":
				return req
			case "picture":
				return words
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		return got
	}
	full := oracle.PILResize(rgb, w, h, ow, oh, true)
	want := make([]float64, pw*ph*3)
	for y := range ph {
		for x := range pw {
			for c := range 3 {
				v := -1.0
				if x < ow && y < oh {
					v = (float64(full[(y*ow+x)*3+c])/255 - 0.5) / 0.5
				}
				want[(y*pw+x)*3+c] = v
			}
		}
	}
	got := run(unit)
	for i, v := range want {
		if math.IsNaN(float64(got[i])) || math.Abs(float64(got[i])-v) > 1e-5 {
			t.Fatalf("element %d (pixel %d of the %dx%d grid, channel %d) reads %v, want %v", i, i/3, pw, ph, i%3, got[i], v)
		}
	}
	for i := len(want); i < rows*W; i++ {
		if got[i] != 0 {
			t.Fatalf("element %d past the padded grid is %v", i, got[i])
		}
	}
	bad, diff := run(0), 0
	for i, v := range want {
		if math.Abs(float64(bad[i])-v) > 1e-3 {
			diff++
		}
	}
	if diff == 0 {
		t.Fatal("the picture unpadded writes the padded grid")
	}
	t.Logf("a %dx%d picture resized to %dx%d and padded to %dx%d: every element; unpadded it parts on %d", w, h, ow, oh, pw, ph, diff)
}
