package composer

import (
	"math"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// TestComposedCPUPictureSecondFrame reads a video's second frame
// (graph.Attrs.PicFrame): two RGBA frames back to back in the step's picture,
// rows padded past their pixels, the second at the byte the request's frame
// word names, resampled as the first would be and held to the oracle's
// resample of the second frame alone. Reading the first frame instead is the
// violation.
func TestComposedCPUPictureSecondFrame(t *testing.T) {
	const W, w, h, pad, ow, oh = 48, 33, 17, 3, 16, 16
	mean, std := [3]float64{0.5, 0.5, 0.5}, [3]float64{0.5, 0.5, 0.5}
	stride := w*4 + pad
	size := (h*stride + 3) / 4 * 4
	pix := make([]byte, 2*size)
	for i := range pix {
		pix[i] = byte(i*131 + i/7)
	}
	rows := (ow*oh*3 + W - 1) / W
	for _, frame := range []bool{true, false} {
		b := graph.NewBlock(-1)
		b.In("unused", 1)
		b.Out(b.Emit(graph.OpPicture, nil, []graph.Result{{Name: "pixels", Width: W}}, nil,
			graph.Attrs{Pixels: graph.PixRGBA8, PicW: ow, PicH: oh, Chain: graph.ChainHalfPixel, Filter: graph.FilterLinear,
				Mean: mean, Std: std, PicFrame: frame}, nil, nil)[0])
		_, codes, err := Default().WithoutInt8(DeviceCPU).Form(Request{Graph: b.G, Step: Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32},
			Device: cpuDevice(hostArm()), Apart: true})
		if err != nil {
			t.Fatal(err)
		}
		req := make([]uint32, RequestPicFrame+1)
		req[RequestPicW], req[RequestPicH], req[RequestPicStride], req[RequestPicFrame] = w, h, uint32(stride), uint32(size)
		words := unsafe.Slice((*uint32)(unsafe.Pointer(&pix[0])), len(pix)/4)
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
		want := oracle.Picture(pix[size:], false, w, h, stride, ow, oh, mean, std)
		worst := 0.0
		for i, v := range want {
			worst = max(worst, math.Abs(float64(got[i]-v)))
		}
		if frame && !(worst < 1e-4) {
			t.Fatalf("the second frame reads %.3g from the oracle's", worst)
		}
		if !frame && !(worst > 1e-2) {
			t.Fatalf("the first frame read as the second (%.3g): the gate cannot tell them apart", worst)
		}
	}
}
