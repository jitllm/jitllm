package composer

import (
	"fmt"
	"math"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// TestComposedCPUPicturePILRectangles resizes pictures of their own size
// through one formed band pass and finish of an area's class
// (graph.Attrs.PicArea): a wide output, its transpose, a square and one
// whose bands hold fewer source rows than PicSide (PicBandRows), each laid
// at its live width, every byte Pillow's (oracle.PILResize). Advancing a
// band past the rows it holds, and skipping a band, are the violations it
// must see.
func TestComposedCPUPicturePILRectangles(t *testing.T) {
	const W, side, area, most = 48, 32, 3600, 128
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	type pcase struct{ w, h, ow, oh int }
	cases := []pcase{{150, 97, 90, 40}, {97, 150, 40, 90}, {61, 61, 60, 60}, {300, 40, 120, 30}}
	K := 0
	for _, c := range cases {
		K = max(K, PILTapCount(graph.FilterCubic, c.w, c.ow), PILTapCount(graph.FilterCubic, c.h, c.oh))
	}
	a := graph.Attrs{Pixels: graph.PixRGBA8, PicW: most, PicH: most, PicArea: area, PicLive: true, PicSide: side,
		Chain: graph.ChainPIL, Filter: graph.FilterCubic, Taps: K, Std: [3]float64{1, 1, 1}}
	rows := (area*3 + W - 1) / W
	form := func(pass uint8) []Code {
		pa := a
		pa.PicPass = pass
		b := graph.NewBlock(-1)
		b.In("unused", 1)
		b.Out(b.Emit(graph.OpPicture, nil, []graph.Result{{Name: "pixels", Width: W}}, nil, pa, nil, nil)[0])
		if err := graph.Verify(b.G, graph.Facts{}); err != nil {
			t.Fatal(err)
		}
		_, codes, err := r.Form(Request{Graph: b.G, Step: Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
		if err != nil {
			t.Fatalf("pass %d: %v", pass, err)
		}
		return codes
	}
	band, fin := form(graph.PicBand), form(graph.PicFinish)
	E := PILEntryWords(K)
	req := make([]uint32, RequestPicTaps+PicEntries(a)*E)
	acc := make([]uint32, area*3)
	short := false
	for _, c := range cases {
		name := fmt.Sprintf("%dx%d to %dx%d", c.w, c.h, c.ow, c.oh)
		rgb := oracle.PILPattern(c.w, c.h)
		rgba := make([]byte, c.w*c.h*4)
		for i := range c.w * c.h {
			copy(rgba[4*i:], rgb[3*i:3*i+3])
			rgba[4*i+3] = 255
		}
		words := unsafe.Slice((*uint32)(unsafe.Pointer(&rgba[0])), len(rgba)/4)
		for i := range req {
			req[i] = 0xdeadbeef
		}
		req[RequestPicW], req[RequestPicH], req[RequestPicStride] = uint32(c.w), uint32(c.h), uint32(c.w*4)
		req[RequestPicOW], req[RequestPicOH], req[RequestPicWide] = uint32(c.ow), uint32(c.oh), 0
		B := PicBandRows(a, c.ow)
		req[RequestPicBandRows] = uint32(B)
		short = short || B < side
		PILTaps(graph.FilterCubic, c.w, c.ow, 0, c.ow, K, req[RequestPicTaps:])
		PILTaps(graph.FilterCubic, c.h, c.oh, 0, c.oh, K, req[RequestPicTaps+c.ow*E:])
		run := func(codes []Code) []float32 {
			got, err := runCPUBound(hostArm(), codes, []float32{0}, rows*W, [3][]uint32{}, nil, func(o Operand) []uint32 {
				switch o.Name {
				case "request":
					return req
				case "picture":
					return words
				case "picture.acc", "picture.acc.out":
					return acc
				}
				return nil
			})
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			return got
		}
		banded := func(step, skip int) []float32 {
			for i := range acc {
				acc[i] = 0xdeadbeef
			}
			for b, i := 0, 0; b < c.h; b, i = b+step, i+1 {
				if i == skip {
					continue
				}
				req[RequestPicBand] = uint32(b)
				run(band)
			}
			return run(fin)
		}
		got := banded(B, -1)
		full := oracle.PILResize(rgb, c.w, c.h, c.ow, c.oh, true)
		for i := range c.ow * c.oh * 3 {
			if math.Abs(float64(got[i])*255-float64(full[i])) > 1e-3 {
				t.Fatalf("%s: byte %d reads %v in bands of %d, Pillow's %d", name, i, got[i]*255, B, full[i])
			}
		}
		parts := func(v []float32) int {
			n := 0
			for i := range c.ow * c.oh * 3 {
				if math.Abs(float64(v[i])*255-float64(full[i])) > 1e-3 {
					n++
				}
			}
			return n
		}
		if c.h > B && parts(banded(B, 1)) == 0 {
			t.Fatalf("%s: a band skipped writes Pillow's bytes", name)
		}
		if B < side && parts(banded(side, -1)) == 0 {
			t.Fatalf("%s: bands advanced by %d past the %d rows each holds write Pillow's bytes", name, side, B)
		}
		t.Logf("%s: %d bands of %d rows, every one of %d bytes Pillow's", name, (c.h+B-1)/B, B, c.ow*c.oh*3)
	}
	if !short {
		t.Fatal("no case's bands hold fewer rows than PicSide")
	}
}
