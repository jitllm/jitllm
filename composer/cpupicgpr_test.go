//go:build amd64

package composer

import (
	"image/color"
	"math"
	"os/exec"
	"strings"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/objdump"
	"github.com/jitllm/jitllm/internal/oracle"
)

// TestComposedCPUScalarPictureKeepsItsWordsInGeneralRegisters composes the
// tower entry's picture an earlier converter stated for SmolVLM -- premultiplied
// RGBA resampled half-pixel bilinear to 512x512, in rows of 768 -- and holds it
// to the oracle. Its map binds only with no loop across lanes, and there its
// words -- the call's geometry read from a table, the pixel coordinates its
// floats convert to -- are the general registers', which spill: no table word
// is loaded into a vector (vpbroadcastd from memory) and no float converts in
// one (cvttps2dq, cvtps2dq); each conversion is a scalar one to a general
// register. In vector registers they crowded the map out of the sixteen there
// are, and the entry was refused.
func TestComposedCPUScalarPictureKeepsItsWordsInGeneralRegisters(t *testing.T) {
	if _, err := exec.LookPath("objdump"); err != nil {
		t.Fatal("objdump is not installed: this gate reads the machine code through it")
	}
	const W, side = 768, 512
	rows := side * side * 3 / W
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	mean, std := [3]float64{0.5, 0.5, 0.5}, [3]float64{0.5, 0.5, 0.5}
	b := graph.NewBlock(-1)
	b.In("unused", 1)
	b.Out(b.Emit(graph.OpPicture, nil, []graph.Result{{Name: "pixels", Width: W}}, nil,
		graph.Attrs{Pixels: graph.PixRGBAPremul, PicW: side, PicH: side, Chain: graph.ChainHalfPixel, Filter: graph.FilterLinear,
			Mean: mean, Std: std}, nil, nil)[0])
	if err := graph.Verify(b.G, graph.Facts{}); err != nil {
		t.Fatal(err)
	}
	_, codes, err := r.Form(Request{Graph: b.G, Step: Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32}, Device: dev})
	if err != nil {
		t.Fatal(err)
	}
	converts := 0
	for _, code := range codes {
		for _, in := range objdump.Disasm(t, t.TempDir(), code.Name, code.CPU) {
			op, _, _ := strings.Cut(in.Text, " ")
			switch op = strings.TrimPrefix(op, "v"); {
			case op == "pbroadcastd" && strings.Contains(in.Text, "ptr"), op == "cvttps2dq", op == "cvtps2dq":
				t.Fatalf("%s holds a word in a vector register: %s", code.Name, in.Text)
			case op == "cvttss2si", op == "cvtss2si":
				converts++
			}
		}
	}
	if converts == 0 {
		t.Fatal("no float converted to a general register: the scalar conversions this gate holds never ran")
	}
	for _, c := range []struct{ w, h, pad int }{{640, 480, 12}, {300, 200, 0}, {517, 389, 5}} {
		stride := c.w*4 + c.pad
		pix := make([]byte, (c.h*stride+3)/4*4)
		for i := range pix {
			pix[i] = byte(i*131 + i/7)
		}
		// A premultiplied pixel's colour is at most its alpha.
		for y := range c.h {
			for x := range c.w {
				p := pix[y*stride+x*4:]
				p[0], p[1], p[2] = min(p[0], p[3]), min(p[1], p[3]), min(p[2], p[3])
			}
		}
		req := make([]uint32, RequestPicOW)
		req[RequestPicW], req[RequestPicH], req[RequestPicStride] = uint32(c.w), uint32(c.h), uint32(stride)
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
			t.Fatalf("%dx%d: %v", c.w, c.h, err)
		}
		want := oracle.PictureOf(func(y, x int) [3]byte {
			p := pix[y*stride+x*4:]
			s := color.NRGBAModel.Convert(color.RGBA{R: p[0], G: p[1], B: p[2], A: p[3]}).(color.NRGBA)
			return [3]byte{s.R, s.G, s.B}
		}, c.w, c.h, side, side, mean, std)
		worst := 0.0
		for i, g := range got[:len(want)] {
			if math.IsNaN(float64(g)) {
				t.Fatalf("%dx%d: element %d is NaN", c.w, c.h, i)
			}
			worst = max(worst, math.Abs(float64(g-want[i])))
		}
		if worst > 2e-5 {
			t.Fatalf("%dx%d: %g from the oracle", c.w, c.h, worst)
		}
		t.Logf("%dx%d: %.1e from the oracle", c.w, c.h, worst)
	}
	t.Logf("%d scalar conversions to a general register", converts)
}
