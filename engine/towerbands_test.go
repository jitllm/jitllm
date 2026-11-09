package engine

import (
	"image"
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/device/cpudev"
)

// TestATallPictureRunsInBands encodes a picture wider and taller than a band of the
// tower's Pillow resize (composer.PictureSide) through synth-qwen2vl's
// tower on the CPU device: it is not refused, its every band runs (the
// tower's band count), its rows are finite, and a black picture of its
// size encodes to other rows -- the picture is read. The bands' sums are
// held to the whole pass and to Pillow by composer's
// TestComposedCPUPicturePILBands.
func TestATallPictureRunsInBands(t *testing.T) {
	m, _, opts := openQVLTower(t, "synth-qwen2vl")
	defer m.Close()
	s := testInstance(t, m, 64, opts...)
	defer s.Close()
	const w, h = 8300, 8300
	if sq, _ := m.Tower.Square(w, h, 0); sq.Pw > m.Tower.Classes[sq.Class].Pic.PicW || sq.Ph > m.Tower.Classes[sq.Class].Pic.PicH {
		t.Fatalf("a %dx%d picture resizes to %dx%d, past its class", w, h, sq.Pw, sq.Ph)
	}
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			i := img.PixOffset(x, y)
			img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = uint8(x*7+y), uint8(y/37), uint8((x^y)*3), 255
		}
	}
	got, err := s.EncodePicture(img)
	if err != nil {
		t.Fatalf("a %dx%d picture: %v", w, h, err)
	}
	got = slices.Clone(got)
	tw, ok := s.tower.parts.(*cpudev.Tower)
	if !ok {
		t.Fatalf("the tower's parts are %T, not the CPU device's", s.tower.parts)
	}
	bands := (h + composer.PictureSide - 1) / composer.PictureSide
	if tw.Bands < bands {
		t.Fatalf("%d bands ran for a picture of %d rows in bands of %d", tw.Bands, h, composer.PictureSide)
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("row value %d is %v", i, v)
		}
	}
	black, err := s.EncodePicture(image.NewNRGBA(image.Rect(0, 0, w, h)))
	if err != nil {
		t.Fatal(err)
	}
	if slices.Equal(got, black) {
		t.Fatal("the tall picture encodes as a black one: its pixels are not read")
	}
	t.Logf("a %dx%d picture in %d bands of %d rows: %d values", w, h, tw.Bands, composer.PictureSide, len(got))
}
