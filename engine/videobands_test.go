package engine

import (
	"image"
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/device/cpudev"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestATallVideoRunsInBands encodes a video of two frames taller than a
// band of the tower's Pillow resize (composer.PictureSide) through Clef's
// VL fixture on the CPU device: not refused, each frame's bands run into
// its own accumulator (the tower's band count), the pair's rows are
// finite, and black frames encode to other rows.
func TestATallVideoRunsInBands(t *testing.T) {
	src, ok := existingModel(testmodels.Path("clef/fixture-vl/clef-fixture-vl.gguf"))
	if !ok {
		testmodels.Missing(t, "MODEL MISSING: %s (scripts/clefgold.py makes it)", testmodels.Path("clef/fixture-vl/clef-fixture-vl.gguf"))
	}
	m, err := model.Open(jlmOfPair(t, src, towerOf(src)))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const w, h = 4400, 4400
	frame := func(seed int, black bool) image.Image {
		img := image.NewNRGBA(image.Rect(0, 0, w, h))
		for y := range h {
			for x := range w {
				i := img.PixOffset(x, y)
				if !black {
					img.Pix[i], img.Pix[i+1], img.Pix[i+2] = uint8(x*7+y+seed), uint8(y/37+seed), uint8((x^y)*3)
				}
				img.Pix[i+3] = 255
			}
		}
		return img
	}
	s := testInstance(t, m, 4096)
	defer s.Close()
	encode := func(black bool) []float32 {
		spans, err := m.VideoSpans([]image.Image{frame(0, black), frame(11, black)}, 0)
		if err != nil {
			t.Fatal(err)
		}
		enc, err := s.encodePictures(spans, 0)
		if err != nil {
			t.Fatalf("a video of %dx%d frames: %v", w, h, err)
		}
		var rows []float32
		for _, sp := range enc {
			if sp.Frame != nil {
				rows = append(rows, sp.Embd...)
			}
		}
		if len(rows) == 0 {
			t.Fatal("the video encoded no pair")
		}
		return rows
	}
	got := encode(false)
	tw, ok := s.tower.parts.(*cpudev.Tower)
	if !ok {
		t.Fatalf("the tower's parts are %T, not the CPU device's", s.tower.parts)
	}
	bands := 2 * ((h + composer.PictureSide - 1) / composer.PictureSide)
	if tw.Bands < bands {
		t.Fatalf("%d bands ran for a pair of frames of %d rows in bands of %d", tw.Bands, h, composer.PictureSide)
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("row value %d is %v", i, v)
		}
	}
	if slices.Equal(got, encode(true)) {
		t.Fatal("the tall video encodes as a black one: its frames are not read")
	}
	t.Logf("a pair of %dx%d frames in %d bands: %d values", w, h, tw.Bands, len(got))
}
