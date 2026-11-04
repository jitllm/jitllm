package composer

import (
	"math"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// TestComposedCPUPicture resamples decoded pictures on the CPU as a tower
// reads its pixels (graph.OpPicture) and holds every value to the oracle's:
// RGBA and gray samples, rows padded past their pixels, pictures shrunk,
// grown and of one pixel, a target not a whole number of rows. A picture
// stated a pixel narrower than it is, which reads its rows skewed, is the
// violation it must see.
func TestComposedCPUPicture(t *testing.T) {
	const W = 48 // a row of 4x4 patches, as OpPatchify reads them
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	mean, std := [3]float64{0.48145466, 0.4578275, 0.40821073}, [3]float64{0.26862954, 0.26130258, 0.27577711}
	cases := []struct {
		w, h, pad, ow, oh int
		gray              bool
	}{
		{1, 1, 0, 4, 4, false}, {7, 5, 0, 16, 16, false}, {33, 17, 3, 16, 16, false},
		{64, 64, 0, 16, 16, false}, {300, 200, 8, 48, 32, false}, {5, 9, 1, 20, 12, true}, {64, 48, 0, 32, 24, true},
	}
	ran := 0
	for _, c := range cases {
		px := 4
		if c.gray {
			px = 1
		}
		stride := c.w*px + c.pad
		pix := make([]byte, (c.h*stride+3)/4*4)
		for i := range pix {
			pix[i] = byte(i*131 + i/7)
		}
		rows := (c.ow*c.oh*3 + W - 1) / W
		b := graph.NewBlock(-1)
		b.In("unused", 1)
		format := graph.PixRGBA8
		if c.gray {
			format = graph.PixGray8
		}
		b.Out(b.Emit(graph.OpPicture, nil, []graph.Result{{Name: "pixels", Width: W}}, nil,
			graph.Attrs{Pixels: format, PicW: c.ow, PicH: c.oh, Chain: graph.ChainHalfPixel, Filter: graph.FilterLinear,
				Mean: mean, Std: std}, nil, nil)[0])
		if err := graph.Verify(b.G, graph.Facts{}); err != nil {
			t.Fatal(err)
		}
		_, codes, err := r.Form(Request{Graph: b.G, Step: Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
		if err != nil {
			t.Fatalf("%dx%d to %dx%d: %v", c.w, c.h, c.ow, c.oh, err)
		}
		words := unsafe.Slice((*uint32)(unsafe.Pointer(&pix[0])), len(pix)/4)
		run := func(w int) []float32 {
			req := make([]uint32, RequestPicStride+1)
			req[RequestPicW], req[RequestPicH], req[RequestPicStride] = uint32(w), uint32(c.h), uint32(stride)
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
				t.Fatalf("%dx%d to %dx%d: %v", c.w, c.h, c.ow, c.oh, err)
			}
			return got
		}
		want := oracle.Picture(pix, c.gray, c.w, c.h, stride, c.ow, c.oh, mean, std)
		got := run(c.w)
		worst := 0.0
		for i, g := range got {
			wv := float32(0)
			if i < len(want) {
				wv = want[i]
			}
			if math.IsNaN(float64(g)) {
				t.Fatalf("%dx%d to %dx%d: element %d is NaN", c.w, c.h, c.ow, c.oh, i)
			}
			worst = max(worst, math.Abs(float64(g-wv)))
		}
		if worst > 2e-5 {
			t.Fatalf("%dx%d to %dx%d: %g from the oracle", c.w, c.h, c.ow, c.oh, worst)
		}
		if c.w > 1 {
			bad := run(c.w - 1)
			diff := 0.0
			for i := range want {
				diff = max(diff, math.Abs(float64(bad[i]-want[i])))
			}
			if diff < 1e-3 {
				t.Fatalf("%dx%d to %dx%d: a picture stated a pixel narrower reads the same pixels", c.w, c.h, c.ow, c.oh)
			}
		}
		ran++
		t.Logf("%dx%d (stride %d, gray %v) to %dx%d: %.1e from the oracle", c.w, c.h, stride, c.gray, c.ow, c.oh, worst)
	}
	if ran != len(cases) {
		t.Fatalf("%d of %d pictures ran", ran, len(cases))
	}
}
