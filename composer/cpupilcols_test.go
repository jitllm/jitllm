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

// TestComposedCPUPicturePILColumns resizes pictures whose windows outrun
// the entries' taps: each band of source rows runs its bands of K source
// columns (graph.PicCols), each column band's taps its own share of every
// output's window (PILChunk), their sums added in the horizontal
// accumulator, then the row band's pass over those sums into the vertical
// one, its taps clipped to the band; then the finish -- every byte
// Pillow's (oracle.PILResize), bilinear and bicubic, both accumulators
// poisoned first. Skipping a column band is the violation it must see.
func TestComposedCPUPicturePILColumns(t *testing.T) {
	const W = 48
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	type pcase struct {
		w, h, ow, oh, k int
		filter          uint8
	}
	cases := []pcase{
		{300, 97, 24, 13, 9, graph.FilterCubic}, {300, 97, 24, 13, 9, graph.FilterBilinear},
		{97, 300, 13, 24, 11, graph.FilterCubic}, {211, 211, 7, 9, 5, graph.FilterCubic},
	}
	for _, c := range cases {
		name := fmt.Sprintf("%dx%d to %dx%d in chunks of %d taps filter %d", c.w, c.h, c.ow, c.oh, c.k, c.filter)
		rgb := oracle.PILPattern(c.w, c.h)
		rgba := make([]byte, c.w*c.h*4)
		for i := range c.w * c.h {
			copy(rgba[4*i:], rgb[3*i:3*i+3])
			rgba[4*i+3] = 255
		}
		words := unsafe.Slice((*uint32)(unsafe.Pointer(&rgba[0])), len(rgba)/4)
		if PILTapCount(c.filter, c.w, c.ow) <= c.k && PILTapCount(c.filter, c.h, c.oh) <= c.k {
			t.Fatalf("%s: every window fits the taps; the case runs no column band it needs", name)
		}
		rows := (c.ow*c.oh*3 + W - 1) / W
		side := c.k
		form := func(pass uint8, columns bool) []Code {
			b := graph.NewBlock(-1)
			b.In("unused", 1)
			b.Out(b.Emit(graph.OpPicture, nil, []graph.Result{{Name: "pixels", Width: W}}, nil,
				graph.Attrs{Pixels: graph.PixRGBA8, PicW: c.ow, PicH: c.oh, PicSide: side, Chain: graph.ChainPIL, Filter: c.filter,
					Taps: c.k, Std: [3]float64{1, 1, 1}, PicPass: pass, PicColumns: columns}, nil, nil)[0])
			if err := graph.Verify(b.G, graph.Facts{}); err != nil {
				t.Fatal(err)
			}
			_, codes, err := r.Form(Request{Graph: b.G, Step: Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
			if err != nil {
				t.Fatalf("%s, pass %d: %v", name, pass, err)
			}
			return codes
		}
		cols, band, fin := form(graph.PicCols, false), form(graph.PicBand, true), form(graph.PicFinish, false)
		E := PILEntryWords(c.k)
		req := make([]uint32, RequestPicTaps+(c.ow+c.oh)*E)
		req[RequestPicW], req[RequestPicH], req[RequestPicStride] = uint32(c.w), uint32(c.h), uint32(c.w*4)
		req[RequestPicBandRows] = uint32(side)
		acc, hacc := make([]uint32, c.ow*c.oh*3), make([]uint32, side*c.ow*3)
		run := func(codes []Code) []float32 {
			got, err := runCPUBound(hostArm(), codes, []float32{0}, rows*W, [3][]uint32{}, nil, func(o Operand) []uint32 {
				switch o.Name {
				case "request":
					return req
				case "picture":
					return words
				case "picture.acc", "picture.acc.out":
					return acc
				case "picture.hacc", "picture.hacc.out":
					return hacc
				}
				return nil
			})
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			return got
		}
		resize := func(skip int) []float32 {
			for i := range acc {
				acc[i] = 0xdeadbeef
			}
			for i := range hacc {
				hacc[i] = 0xdeadbeef
			}
			for rb := 0; rb < c.h; rb += side {
				req[RequestPicBand] = uint32(rb)
				PILChunk(c.filter, PILPrec(c.filter, false, c.h, c.oh), c.h, c.oh, 0, c.oh, c.k, rb, min(rb+side, c.h), req[RequestPicTaps+c.ow*E:])
				for cb, j := 0, 0; cb < c.w; cb, j = cb+c.k, j+1 {
					if j == skip {
						continue
					}
					req[RequestPicCol] = uint32(cb)
					PILChunk(c.filter, PILPrec(c.filter, false, c.w, c.ow), c.w, c.ow, 0, c.ow, c.k, cb, min(cb+c.k, c.w), req[RequestPicTaps:])
					run(cols)
				}
				run(band)
			}
			return run(fin)
		}
		got := resize(-1)
		full := oracle.PILResize(rgb, c.w, c.h, c.ow, c.oh, c.filter == graph.FilterCubic)
		for i := range c.ow * c.oh * 3 {
			if math.Abs(float64(got[i])*255-float64(full[i])) > 1e-3 {
				t.Fatalf("%s: byte %d reads %v in column bands, Pillow's %d", name, i, got[i]*255, full[i])
			}
		}
		bad, diff := resize(1), 0
		for i := range c.ow * c.oh * 3 {
			if bad[i] != got[i] {
				diff++
			}
		}
		if diff == 0 {
			t.Fatalf("%s: a column band skipped writes the same bytes", name)
		}
		t.Logf("%s: %d row bands of %d column bands, every one of %d bytes Pillow's; a column band skipped parts on %d",
			name, (c.h+side-1)/side, (c.w+c.k-1)/c.k, c.ow*c.oh*3, diff)
	}
}
