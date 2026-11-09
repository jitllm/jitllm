package engine

import (
	"image"
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/device/cpudev"
)

// TestATallPictureRunsInBands encodes pictures taller, wider, and both, than a band of the
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
	for _, sz := range [][2]int{{320, 9000}, {9000, 320}, {8300, 8300}} {
		tallPicture(t, s, sz[0], sz[1])
	}
}

// tallPicture encodes a w x h picture through instance s's tower: not
// refused, every band its class's resize holds at the picture's width run,
// its rows finite and a black picture's other.
func tallPicture(t *testing.T, s *Instance, w, h int) {
	t.Helper()
	tg := s.m.Tower
	sq, _ := tg.Square(w, h, 0)
	a := tg.Classes[sq.Class].Pic
	if !a.Holds(sq.Pw, sq.Ph) {
		t.Fatalf("a %dx%d picture resizes to %dx%d, past class %d", w, h, sq.Pw, sq.Ph, sq.Class)
	}
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			i := img.PixOffset(x, y)
			img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = uint8(x*7+y), uint8(y/37), uint8((x^y)*3), 255
		}
	}
	var ran int
	if s.tower != nil {
		if tw, ok := s.tower.parts.(*cpudev.Tower); ok {
			ran = tw.Bands
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
	B := composer.PicBandRows(a, sq.Pw)
	if want := (h + B - 1) / B; tw.Bands-ran != want {
		t.Fatalf("a %dx%d picture resized to %dx%d ran %d bands, its %d rows in bands of %d", w, h, sq.Pw, sq.Ph, tw.Bands-ran, h, B)
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
	t.Logf("a %dx%d picture resized to %dx%d in bands of %d rows: %d values", w, h, sq.Pw, sq.Ph, B, len(got))
}
