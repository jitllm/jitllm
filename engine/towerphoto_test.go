package engine

import (
	"image"
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestAPhotoOfFortyEightMegapixelsEncodes encodes an 8000x6000 4:2:0 YCbCr
// picture -- a phone camera's JPEG, decoded -- through synth-gemma3nv's
// tower. Its three planes are 72 MB, copied into one table the step reads
// at U32 offsets: it encodes, its rows are finite, and a picture of
// its size with other samples encodes to other rows.
func TestAPhotoOfFortyEightMegapixelsEncodes(t *testing.T) {
	src, ok := existingModel(testmodels.Path("synth-gemma3nv.gguf"))
	if !ok {
		testmodels.Missing(t, "MODEL MISSING: %s (see docs/testing.md) -- this gate proved nothing", testmodels.Path("synth-gemma3nv.gguf"))
	}
	m, err := model.Open(jlmOfPair(t, src, towerOf(src)))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	s := testInstance(t, m, 64)
	defer s.Close()
	const w, h = 8000, 6000
	photo := func(f func(x, y int) (uint8, uint8, uint8)) *image.YCbCr {
		img := image.NewYCbCr(image.Rect(0, 0, w, h), image.YCbCrSubsampleRatio420)
		for y := range h {
			for x := range w {
				yy, cb, cr := f(x, y)
				img.Y[img.YOffset(x, y)] = yy
				c := img.COffset(x, y)
				img.Cb[c], img.Cr[c] = cb, cr
			}
		}
		return img
	}
	img := photo(func(x, y int) (uint8, uint8, uint8) { return uint8(x*7 + y), uint8(x / 31), uint8((x ^ y) * 3) })
	if planes := len(img.Y) + len(img.Cb) + len(img.Cr); planes <= 4*4096*4096 {
		t.Fatalf("the photo's planes are %d bytes, a 4096-square picture's: the case proves nothing", planes)
	}
	got, err := s.EncodePicture(img)
	if err != nil {
		t.Fatalf("a %dx%d photo: %v", w, h, err)
	}
	got = slices.Clone(got)
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("row value %d is %v", i, v)
		}
	}
	flat, err := s.EncodePicture(photo(func(int, int) (uint8, uint8, uint8) { return 90, 128, 128 }))
	if err != nil {
		t.Fatal(err)
	}
	if slices.Equal(got, flat) {
		t.Fatal("the photo encodes as a flat grey one: its samples are not read")
	}
	sq, _ := m.Tower.Square(w, h, 0)
	t.Logf("a %dx%d photo of %d MiB of planes resized to %dx%d, %d values (the table holds %d words)",
		w, h, (len(img.Y)+len(img.Cb)+len(img.Cr))>>20, sq.Ow, sq.Oh, len(got), composer.PictureWords)
}
