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

// TestComposedCPUPicturePILBands resizes pictures taller than a band
// (graph.Attrs.PicPass): each band of side source rows through the band
// pass in order, the accumulator poisoned before the first, then the
// finish -- every byte the whole pass's and Pillow's (oracle.PILResize),
// bilinear and bicubic, a picture of several bands and one whose last band
// is short. Skipping a band is the violation it must see.
func TestComposedCPUPicturePILBands(t *testing.T) {
	const W = 48
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	type pcase struct {
		w, h, ow, oh, side int
		filter             uint8
	}
	cases := []pcase{
		{97, 200, 32, 48, 64, graph.FilterBilinear}, {97, 200, 32, 48, 64, graph.FilterCubic},
		{61, 150, 40, 40, 48, graph.FilterCubic}, {300, 97, 64, 24, 32, graph.FilterBilinear},
	}
	for _, c := range cases {
		name := fmt.Sprintf("%dx%d to %dx%d in bands of %d filter %d", c.w, c.h, c.ow, c.oh, c.side, c.filter)
		rgb := oracle.PILPattern(c.w, c.h)
		rgba := make([]byte, c.w*c.h*4)
		for i := range c.w * c.h {
			copy(rgba[4*i:], rgb[3*i:3*i+3])
			rgba[4*i+3] = 255
		}
		words := unsafe.Slice((*uint32)(unsafe.Pointer(&rgba[0])), len(rgba)/4)
		K := max(PILTapCount(c.filter, c.w, c.ow), PILTapCount(c.filter, c.h, c.oh))
		rows := (c.ow*c.oh*3 + W - 1) / W
		form := func(pass uint8, side int) []Code {
			b := graph.NewBlock(-1)
			b.In("unused", 1)
			b.Out(b.Emit(graph.OpPicture, nil, []graph.Result{{Name: "pixels", Width: W}}, nil,
				graph.Attrs{Pixels: graph.PixRGBA8, PicW: c.ow, PicH: c.oh, PicSide: side, Chain: graph.ChainPIL, Filter: c.filter,
					Taps: K, Std: [3]float64{1, 1, 1}, PicPass: pass}, nil, nil)[0])
			if err := graph.Verify(b.G, graph.Facts{}); err != nil {
				t.Fatal(err)
			}
			_, codes, err := r.Form(Request{Graph: b.G, Step: Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
			if err != nil {
				t.Fatalf("%s, pass %d: %v", name, pass, err)
			}
			return codes
		}
		whole, band, fin := form(graph.PicWhole, max(c.w, c.h)), form(graph.PicBand, c.side), form(graph.PicFinish, c.side)
		req := make([]uint32, RequestPicTaps+(c.ow+c.oh)*PILEntryWords(K))
		req[RequestPicW], req[RequestPicH], req[RequestPicStride] = uint32(c.w), uint32(c.h), uint32(c.w*4)
		PILTaps(c.filter, c.w, c.ow, 0, c.ow, K, req[RequestPicTaps:])
		PILTaps(c.filter, c.h, c.oh, 0, c.oh, K, req[RequestPicTaps+c.ow*PILEntryWords(K):])
		acc := make([]uint32, c.ow*c.oh*3)
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
		banded := func(skip int) []float32 {
			for i := range acc {
				acc[i] = 0xdeadbeef
			}
			for b, i := 0, 0; b < c.h; b, i = b+c.side, i+1 {
				if i == skip {
					continue
				}
				req[RequestPicBand] = uint32(b)
				run(band)
			}
			return run(fin)
		}
		req[RequestPicBand] = 0
		want, got := run(whole), banded(-1)
		full := oracle.PILResize(rgb, c.w, c.h, c.ow, c.oh, c.filter == graph.FilterCubic)
		for i := range c.ow * c.oh * 3 {
			if got[i] != want[i] || math.Abs(float64(got[i])*255-float64(full[i])) > 1e-3 {
				t.Fatalf("%s: byte %d reads %v in bands, %v whole, Pillow's %d", name, i, got[i]*255, want[i]*255, full[i])
			}
		}
		bad, diff := banded(1), 0
		for i := range c.ow * c.oh * 3 {
			if bad[i] != want[i] {
				diff++
			}
		}
		if diff == 0 {
			t.Fatalf("%s: a band skipped writes the same bytes", name)
		}
		t.Logf("%s: %d bands, every one of %d bytes the whole pass's and Pillow's; a band skipped parts on %d",
			name, (c.h+c.side-1)/c.side, c.ow*c.oh*3, diff)
	}
}
