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

// pictureCase is one decoded picture as the step's picture table and
// request words hold it, and the oracle's red, green and blue for a pixel.
type pictureCase struct {
	name                 string
	px                   uint8
	w, h, stride         int
	cstride, p2, p3, sub int
	words                []uint32
	rgb                  func(y, x int) [3]byte
}

// tableOf is planes one after another, each from a word boundary, as words,
// and each plane's byte offset.
func tableOf(planes ...[]byte) ([]uint32, []int) {
	var b []byte
	var at []int
	for _, p := range planes {
		for len(b)%4 != 0 {
			b = append(b, 0)
		}
		at = append(at, len(b))
		b = append(b, p...)
	}
	for len(b)%4 != 0 {
		b = append(b, 0)
	}
	return unsafe.Slice((*uint32)(unsafe.Pointer(&b[0])), len(b)/4), at
}

// ycbcrCase is a sub-picture of a random YCbCr picture at ratio, its corner
// at an odd place so the chroma phase is read; its chroma shifts and phase
// are read off the picture's own COffset.
func ycbcrCase(t *testing.T, name string, ratio image.YCbCrSubsampleRatio, seed int) pictureCase {
	t.Helper()
	full := image.NewYCbCr(image.Rect(0, 0, 37, 29), ratio)
	for i := range full.Y {
		full.Y[i] = byte(i*37 + seed)
	}
	for i := range full.Cb {
		full.Cb[i], full.Cr[i] = byte(i*53+seed*3), byte(i*71+seed*5)
	}
	img := full.SubImage(image.Rect(3, 5, 34, 28)).(*image.YCbCr)
	r := img.Rect
	ylen := img.YOffset(r.Max.X-1, r.Max.Y-1) + 1
	clen := img.COffset(r.Max.X-1, r.Max.Y-1) + 1
	words, at := tableOf(img.Y[:ylen], img.Cb[:clen], img.Cr[:clen])
	shift := func(n, step int, off func(int) int) (int, int) {
		for hs := 0; hs < 3; hs++ {
			for m := 0; m < 1<<hs; m++ {
				ok := true
				for k := 0; k < min(n, 8); k++ {
					ok = ok && off(k) == ((k+m)>>hs)*step
				}
				if ok {
					return hs, m
				}
			}
		}
		t.Fatalf("%s: no shift reads its chroma", name)
		return 0, 0
	}
	hs, mx := shift(r.Dx(), 1, func(k int) int { return img.COffset(r.Min.X+k, r.Min.Y) })
	vs, my := shift(r.Dy(), img.CStride, func(k int) int { return img.COffset(r.Min.X, r.Min.Y+k) })
	return pictureCase{name: name, px: graph.PixYCbCr8, w: r.Dx(), h: r.Dy(), stride: img.YStride,
		cstride: img.CStride, p2: at[1], p3: at[2], sub: hs | vs<<8 | mx<<16 | my<<24, words: words,
		rgb: func(y, x int) [3]byte {
			X, Y := r.Min.X+x, r.Min.Y+y
			c := img.COffset(X, Y)
			rr, gg, bb := color.YCbCrToRGB(img.Y[img.YOffset(X, Y)], img.Cb[c], img.Cr[c])
			return [3]byte{rr, gg, bb}
		}}
}

// palettedCase is a random paletted picture, its palette's entries as the
// engine lays them: four bytes an entry, the colour's straight red, green
// and blue.
func palettedCase(w, h, n int) pictureCase {
	pal := make(color.Palette, n)
	for i := range pal {
		pal[i] = color.NRGBA{R: byte(i * 29), G: byte(i*61 + 7), B: byte(i*97 + 3), A: byte(255 - i%3*40)}
	}
	img := image.NewPaletted(image.Rect(0, 0, w, h), pal)
	for i := range img.Pix {
		img.Pix[i] = byte((i*13 + i/5) % n)
	}
	entries := make([]byte, 4*256)
	for i, c := range pal {
		v := color.NRGBAModel.Convert(c).(color.NRGBA)
		entries[4*i], entries[4*i+1], entries[4*i+2], entries[4*i+3] = v.R, v.G, v.B, v.A
	}
	words, at := tableOf(img.Pix[:(h-1)*img.Stride+w], entries)
	return pictureCase{name: fmt.Sprintf("paletted %dx%d of %d", w, h, n), px: graph.PixPaletted8, w: w, h: h,
		stride: img.Stride, p2: at[1], words: words, rgb: func(y, x int) [3]byte {
			v := color.NRGBAModel.Convert(img.Palette[img.ColorIndexAt(x, y)]).(color.NRGBA)
			return [3]byte{v.R, v.G, v.B}
		}}
}

// TestComposedCPUPictureFormats resamples YCbCr pictures at every chroma
// subsampling Go's decoders leave -- sub-pictures at odd corners, so the
// chroma phase is read -- and paletted ones on the CPU, and holds every value
// to the oracle's of the pixels image/color gives. A chroma phase off by one
// and a palette read a byte off are the violations it must see.
func TestComposedCPUPictureFormats(t *testing.T) {
	const W = 48
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	mean, std := [3]float64{0.5, 0.5, 0.5}, [3]float64{0.5, 0.5, 0.5}
	cases := []pictureCase{
		ycbcrCase(t, "4:4:4", image.YCbCrSubsampleRatio444, 1), ycbcrCase(t, "4:2:2", image.YCbCrSubsampleRatio422, 2),
		ycbcrCase(t, "4:2:0", image.YCbCrSubsampleRatio420, 3), ycbcrCase(t, "4:4:0", image.YCbCrSubsampleRatio440, 4),
		ycbcrCase(t, "4:1:1", image.YCbCrSubsampleRatio411, 5), ycbcrCase(t, "4:1:0", image.YCbCrSubsampleRatio410, 6),
		palettedCase(19, 11, 256), palettedCase(5, 3, 7),
	}
	const ow, oh = 16, 16
	rows := (ow*oh*3 + W - 1) / W
	forms := map[uint8][]Code{}
	for _, c := range cases {
		codes, ok := forms[c.px]
		if !ok {
			b := graph.NewBlock(-1)
			b.In("unused", 1)
			b.Out(b.Emit(graph.OpPicture, nil, []graph.Result{{Name: "pixels", Width: W}}, nil,
				graph.Attrs{Pixels: c.px, PicW: ow, PicH: oh, Chain: graph.ChainHalfPixel, Filter: graph.FilterLinear,
					Mean: mean, Std: std}, nil, nil)[0])
			if err := graph.Verify(b.G, graph.Facts{}); err != nil {
				t.Fatal(err)
			}
			var err error
			if _, codes, err = r.Form(Request{Graph: b.G, Step: Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32},
				Device: dev, Apart: true}); err != nil {
				t.Fatalf("%s: %v", c.name, err)
			}
			forms[c.px] = codes
		}
		run := func(c pictureCase) []float32 {
			req := make([]uint32, RequestPicSub+1)
			req[RequestPicW], req[RequestPicH], req[RequestPicStride] = uint32(c.w), uint32(c.h), uint32(c.stride)
			req[RequestPicCStride], req[RequestPicP2], req[RequestPicP3] = uint32(c.cstride), uint32(c.p2), uint32(c.p3)
			req[RequestPicSub] = uint32(c.sub)
			got, err := runCPUBound(hostArm(), codes, []float32{0}, rows*W, [3][]uint32{}, nil, func(o Operand) []uint32 {
				switch o.Name {
				case "request":
					return req
				case "picture":
					return c.words
				}
				return nil
			})
			if err != nil {
				t.Fatalf("%s: %v", c.name, err)
			}
			return got
		}
		want := oracle.PictureOf(c.rgb, c.w, c.h, ow, oh, mean, std)
		got := run(c)
		worst := 0.0
		for i := range want {
			if math.IsNaN(float64(got[i])) {
				t.Fatalf("%s: element %d is NaN", c.name, i)
			}
			worst = max(worst, math.Abs(float64(got[i]-want[i])))
		}
		if worst > 2e-5 {
			t.Fatalf("%s: %g from the oracle", c.name, worst)
		}
		bad := c
		if c.px == graph.PixYCbCr8 {
			bad.sub ^= 1 << 16
			if c.sub&0xff == 0 {
				bad.sub = c.sub ^ 1<<24
			}
			if c.sub&0xffff == 0 {
				bad.p2, bad.p3 = c.p3, c.p2
			}
		} else {
			bad.p2 = c.p2 + 1
		}
		diff := 0.0
		for i, g := range run(bad) {
			if i < len(want) {
				diff = max(diff, math.Abs(float64(g-want[i])))
			}
		}
		if diff < 1e-3 {
			t.Fatalf("%s: the violation reads the same pixels", c.name)
		}
		t.Logf("%s (%dx%d, sub %#x): %.1e from the oracle, the violation %.2g", c.name, c.w, c.h, c.sub, worst, diff)
	}
}
