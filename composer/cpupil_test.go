package composer

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// TestComposedCPUPicturePIL resizes pictures on the CPU by Pillow's
// antialiased filters (graph.ChainPIL), their taps PILTaps', and holds every
// byte to oracle.PILResize, itself held to Pillow's (TestPILResizeMatchesPillow):
// bilinear and bicubic, shrunk, grown and of another aspect, RGBA and a
// JPEG-shaped YCbCr picture's pixels, a form reaching further than the
// picture needs. The other filter's taps are the violation it must see.
func TestComposedCPUPicturePIL(t *testing.T) {
	const W = 48
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	type pcase struct {
		w, h, ow, oh, side int
		filter             uint8
		ycbcr              bool
	}
	cases := []pcase{
		{97, 61, 48, 48, 128, graph.FilterBilinear, false}, {20, 33, 48, 48, 64, graph.FilterBilinear, false},
		{97, 61, 48, 48, 128, graph.FilterCubic, false}, {20, 33, 48, 48, 64, graph.FilterCubic, false},
		{300, 200, 64, 64, 512, graph.FilterBilinear, true}, {300, 200, 64, 64, 300, graph.FilterCubic, false},
	}
	for _, c := range cases {
		name := fmt.Sprintf("%dx%d to %dx%d filter %d ycbcr %v", c.w, c.h, c.ow, c.oh, c.filter, c.ycbcr)
		rgb := oracle.PILPattern(c.w, c.h)
		var words []uint32
		var px uint8
		stride, cstride, p2, p3, sub := 0, 0, 0, 0, 0
		if c.ycbcr {
			img := image.NewYCbCr(image.Rect(0, 0, c.w, c.h), image.YCbCrSubsampleRatio444)
			for y := range c.h {
				for x := range c.w {
					i := (y*c.w + x) * 3
					yy, cb, cr := color.RGBToYCbCr(rgb[i], rgb[i+1], rgb[i+2])
					img.Y[img.YOffset(x, y)], img.Cb[img.COffset(x, y)], img.Cr[img.COffset(x, y)] = yy, cb, cr
					rr, gg, bb := color.YCbCrToRGB(yy, cb, cr)
					rgb[i], rgb[i+1], rgb[i+2] = rr, gg, bb
				}
			}
			var at []int
			words, at = tableOf(img.Y, img.Cb, img.Cr)
			px, stride, cstride, p2, p3 = graph.PixYCbCr8, img.YStride, img.CStride, at[1], at[2]
		} else {
			rgba := make([]byte, c.w*c.h*4)
			for i := range c.w * c.h {
				copy(rgba[4*i:], rgb[3*i:3*i+3])
				rgba[4*i+3] = 255
			}
			words = unsafe.Slice((*uint32)(unsafe.Pointer(&rgba[0])), len(rgba)/4)
			px, stride = graph.PixRGBA8, c.w*4
		}
		K := max(PILTapCount(c.filter, c.side, c.ow), PILTapCount(c.filter, c.side, c.oh))
		rows := (c.ow*c.oh*3 + W - 1) / W
		b := graph.NewBlock(-1)
		b.In("unused", 1)
		b.Out(b.Emit(graph.OpPicture, nil, []graph.Result{{Name: "pixels", Width: W}}, nil,
			graph.Attrs{Pixels: px, PicW: c.ow, PicH: c.oh, PicSide: c.side, Chain: graph.ChainPIL, Filter: c.filter, Taps: K,
				Std: [3]float64{1, 1, 1}}, nil, nil)[0])
		if err := graph.Verify(b.G, graph.Facts{}); err != nil {
			t.Fatal(err)
		}
		_, codes, err := r.Form(Request{Graph: b.G, Step: Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		run := func(filter uint8) []float32 {
			req := make([]uint32, RequestPicTaps+(c.ow+c.oh)*(K+1))
			req[RequestPicW], req[RequestPicH], req[RequestPicStride] = uint32(c.w), uint32(c.h), uint32(stride)
			req[RequestPicCStride], req[RequestPicP2], req[RequestPicP3], req[RequestPicSub] = uint32(cstride), uint32(p2), uint32(p3), uint32(sub)
			PILTaps(filter, c.w, c.ow, K, req[RequestPicTaps:])
			PILTaps(filter, c.h, c.oh, K, req[RequestPicTaps+c.ow*(K+1):])
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
				t.Fatalf("%s: %v", name, err)
			}
			return got
		}
		want := oracle.PILResize(rgb, c.w, c.h, c.ow, c.oh, c.filter == graph.FilterCubic)
		got := run(c.filter)
		for i, w := range want {
			v := float64(got[i]) * 255
			if math.IsNaN(v) || math.Abs(v-float64(w)) > 1e-3 {
				t.Fatalf("%s: byte %d (pixel %d channel %d) reads %v, Pillow's %d", name, i, i/3, i%3, v, w)
			}
		}
		for i := len(want); i < rows*W; i++ {
			if got[i] != 0 {
				t.Fatalf("%s: element %d past the picture is %v", name, i, got[i])
			}
		}
		other := graph.FilterCubic
		if c.filter == graph.FilterCubic {
			other = graph.FilterBilinear
		}
		bad, diff := run(other), 0
		for i, w := range want {
			if math.Abs(float64(bad[i])*255-float64(w)) > 0.5 {
				diff++
			}
		}
		if diff == 0 {
			t.Fatalf("%s: the other filter's taps write the same bytes", name)
		}
		t.Logf("%s (%d taps reaching a side of %d): Pillow's %d bytes; the other filter's taps part on %d", name, K, c.side, len(want), diff)
	}
}
