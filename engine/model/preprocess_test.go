package model

import (
	"image"
	"image/color"
	"io"
	"math"
	"testing"
)

// preprocess runs a tower's Preprocess on a state of its own.
func preprocess(tw *Tower, r io.Reader) ([]float32, error) {
	px, _, err := preprocessSized(tw, r)
	return px, err
}

// preprocessSized is preprocess with the picture's resized grid, which a
// dynamic-resolution tower's Encode on another state needs
// (Instance.SetImageSize) and its prompt is laid out by.
func preprocessSized(tw *Tower, r io.Reader) ([]float32, [2]int, error) {
	ts := tw.testState()
	defer ts.Close()
	px, err := ts.Preprocess(r)
	gh, gw := ts.patchGrid()
	return px, [2]int{gw * tw.Cfg.PatchSz, gh * tw.Cfg.PatchSz}, err
}

// preprocessRef is the bilinear resize and normalisation Preprocess replaced,
// kept as the oracle for it: sample at pixel centres, clamp at the edges,
// squash to square, then (v - mean) / std.
func preprocessRef(img image.Image, c TowerConfig) []float32 {
	b := img.Bounds()
	sw, sh := b.Dx(), b.Dy()
	n := c.ImageSz
	out := make([]float32, n*n*3)
	at := func(x, y, ch int) float64 {
		r, g, bl, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
		return float64([3]uint32{r, g, bl}[ch]) / 65535
	}
	for y := 0; y < n; y++ {
		fy := (float64(y)+0.5)*float64(sh)/float64(n) - 0.5
		y0 := int(math.Floor(fy))
		wy := fy - float64(y0)
		y1 := clampi(y0+1, 0, sh-1)
		y0 = clampi(y0, 0, sh-1)
		for x := 0; x < n; x++ {
			fx := (float64(x)+0.5)*float64(sw)/float64(n) - 0.5
			x0 := int(math.Floor(fx))
			wx := fx - float64(x0)
			x1 := clampi(x0+1, 0, sw-1)
			x0 = clampi(x0, 0, sw-1)
			for ch := 0; ch < 3; ch++ {
				top := at(x0, y0, ch) + (at(x1, y0, ch)-at(x0, y0, ch))*wx
				bot := at(x0, y1, ch) + (at(x1, y1, ch)-at(x0, y1, ch))*wx
				v := top + (bot-top)*wy
				out[(y*n+x)*3+ch] = float32((v - c.Mean[ch]) / c.Std[ch])
			}
		}
	}
	return out
}

// TestPreprocessMatchesTheBilinearReference holds the generated resize (two
// passes of the F32 matvec) to the Go resampler it replaced, on images
// that shrink, grow and change aspect, with an offset origin and a 16-bit
// channel, so a half-pixel shift, a swapped axis, a dropped normalisation or an
// edge clamp all show.
func TestPreprocessMatchesTheBilinearReference(t *testing.T) {
	m, tw := openTower(t)
	defer m.Close()
	ts := tw.testState()
	defer ts.Close()
	c := tw.Cfg
	c.ImageSz = 48
	c.Mean = [3]float64{0.48, 0.46, 0.41}
	c.Std = [3]float64{0.27, 0.26, 0.28}
	saved := ts.vis.t.Cfg
	ts.vis.t.Cfg = c
	defer func() { ts.vis.t.Cfg = saved }()
	for _, sz := range [][2]int{{97, 61}, {20, 33}, {48, 48}, {1, 5}, {130, 7}} {
		img := image.NewRGBA64(image.Rect(3, 5, 3+sz[0], 5+sz[1]))
		for y := 0; y < sz[1]; y++ {
			for x := 0; x < sz[0]; x++ {
				img.SetRGBA64(3+x, 5+y, color.RGBA64{
					R: uint16((x*7919 + y*104729) % 65536), G: uint16((x * y * 131) % 65536),
					B: uint16((x + 3*y) * 997 % 65536), A: 0xFFFF})
			}
		}
		got, err := ts.PreprocessImage(img)
		if err != nil {
			t.Fatal(err)
		}
		want := preprocessRef(img, c)
		var num, den, worst float64
		for i := range want {
			d := float64(got[i]) - float64(want[i])
			if math.IsNaN(d) || math.IsInf(d, 0) {
				t.Fatalf("%v: element %d is %v", sz, i, got[i])
			}
			num += d * d
			den += float64(want[i]) * float64(want[i])
			worst = math.Max(worst, math.Abs(d))
		}
		t.Logf("%dx%d -> %d: NMSE %.3e, worst |d| %.2e", sz[0], sz[1], c.ImageSz, num/den, worst)
		if worst > 1e-4 {
			t.Fatalf("%dx%d: worst |d| %.3e against the bilinear reference", sz[0], sz[1], worst)
		}
	}
}
