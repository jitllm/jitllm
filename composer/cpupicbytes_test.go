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

// TestComposedCPUPictureBytes resizes a picture twice, as a processor
// resizing its own uint8 output does: the first resize writes its bytes
// (graph.Attrs.PicBytes), a word a pixel and 255 above, and the second reads
// those words as the step's picture (PixRGBA8). Both are held to the
// oracle's two resizes byte for byte, torchvision's and Pillow's precision,
// and the alpha byte and the words past the picture are held too. The
// violation is one resize to the second's size, which parts from the chain.
func TestComposedCPUPictureBytes(t *testing.T) {
	const W = 48
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	for _, c := range []struct {
		w, h, mw, mh, ow, oh int
		torch                bool
	}{
		{300, 200, 160, 107, 64, 64, true}, {300, 200, 160, 107, 64, 64, false}, {61, 97, 128, 203, 40, 40, true},
	} {
		name := fmt.Sprintf("%dx%d to %dx%d to %dx%d torch %v", c.w, c.h, c.mw, c.mh, c.ow, c.oh, c.torch)
		form := func(ow, oh, iw, ih, width int, bytes bool) ([]Code, int) {
			side := max(iw, ih, ow, oh)
			K := max(PILTapCount(graph.FilterLanczos, side, ow), PILTapCount(graph.FilterLanczos, side, oh))
			b := graph.NewBlock(-1)
			b.In("unused", 1)
			res := graph.Result{Name: "pixels", Width: width}
			if bytes {
				res.Elem = graph.ElemU32
			}
			b.Out(b.Emit(graph.OpPicture, nil, []graph.Result{res}, nil,
				graph.Attrs{Pixels: graph.PixRGBA8, PicW: ow, PicH: oh, PicSide: side, Chain: graph.ChainPIL, Filter: graph.FilterLanczos,
					Taps: K, Std: [3]float64{1, 1, 1}, PicTorch: c.torch, PicBytes: bytes}, nil, nil)[0])
			if err := graph.Verify(b.G, graph.Facts{}); err != nil {
				t.Fatal(err)
			}
			rows := (ow*oh*3 + width - 1) / width
			if bytes {
				rows = oh
			}
			_, codes, err := r.Form(Request{Graph: b.G, Step: Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			return codes, K
		}
		run := func(codes []Code, K, iw, ih, ow, oh, n int, words []uint32) []float32 {
			req := make([]uint32, RequestPicTaps+(ow+oh)*PILEntryWords(K))
			req[RequestPicW], req[RequestPicH], req[RequestPicStride] = uint32(iw), uint32(ih), uint32(iw*4)
			PILTapsAt(graph.FilterLanczos, c.torch, iw, ow, 0, ow, K, req[RequestPicTaps:])
			PILTapsAt(graph.FilterLanczos, c.torch, ih, oh, 0, oh, K, req[RequestPicTaps+ow*PILEntryWords(K):])
			got, err := runCPUBound(hostArm(), codes, []float32{0}, n, [3][]uint32{}, nil, func(o Operand) []uint32 {
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
		resize := oracle.PILResizeBy
		if c.torch {
			resize = oracle.TorchResizeBy
		}
		rgb := oracle.PILPattern(c.w, c.h)
		rgba := make([]byte, c.w*c.h*4)
		for i := range c.w * c.h {
			copy(rgba[4*i:], rgb[3*i:3*i+3])
			rgba[4*i+3] = 255
		}
		src := unsafe.Slice((*uint32)(unsafe.Pointer(&rgba[0])), len(rgba)/4)

		codes1, K1 := form(c.mw, c.mh, c.w, c.h, c.mw, true)
		out1 := run(codes1, K1, c.w, c.h, c.mw, c.mh, c.mw*c.mh, src)
		mid := resize(rgb, c.w, c.h, c.mw, c.mh, oracle.PILLanczos)
		words := make([]uint32, c.mw*c.mh)
		for i := range words {
			words[i] = math.Float32bits(out1[i])
			want := uint32(mid[3*i]) | uint32(mid[3*i+1])<<8 | uint32(mid[3*i+2])<<16 | 0xff000000
			if words[i] != want {
				t.Fatalf("%s: the first resize's pixel %d reads %#08x, the oracle's %#08x", name, i, words[i], want)
			}
		}

		codes2, K2 := form(c.ow, c.oh, c.mw, c.mh, W, false)
		n := (c.ow*c.oh*3 + W - 1) / W * W
		got := run(codes2, K2, c.mw, c.mh, c.ow, c.oh, n, words)
		want := resize(mid, c.mw, c.mh, c.ow, c.oh, oracle.PILLanczos)
		for i, w := range want {
			if v := float64(got[i]) * 255; math.IsNaN(v) || math.Abs(v-float64(w)) > 1e-3 {
				t.Fatalf("%s: the second resize's byte %d reads %v, the oracle's %d", name, i, v, w)
			}
		}
		once := resize(rgb, c.w, c.h, c.ow, c.oh, oracle.PILLanczos)
		parts := 0
		for i := range want {
			if once[i] != want[i] {
				parts++
			}
		}
		if parts == 0 {
			t.Fatalf("%s: one resize writes the chain's bytes: the gate cannot see a stage dropped", name)
		}
		t.Logf("%s: the chain's %d bytes the oracle's; one resize parts on %d", name, len(want), parts)
	}
}
