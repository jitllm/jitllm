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

// TestAPictureWiderThanItsTapsRunsInColumns encodes a picture past 65535
// pixels wide through synth-gemma3nv's tower, a fixed-size class whose
// resize windows then outrun its taps: it is not refused, each band of
// source rows runs its bands of Taps source columns and then the row band
// over their sums (the tower's band count), its rows are finite, and a black
// picture of its size encodes to other rows. The column bands' sums are held
// to Pillow by composer's TestComposedCPUPicturePILColumns.
func TestAPictureWiderThanItsTapsRunsInColumns(t *testing.T) {
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
	const w, h = 66500, 16
	tg := m.Tower
	sq, _ := tg.Square(w, h, 0)
	a := tg.Classes[sq.Class].Pic
	if composer.PILTapCount(a.Filter, w, sq.Ow) <= a.Taps {
		t.Fatalf("a %dx%d picture resized to %dx%d reads %d taps an output, which its %d hold: the case runs no column band",
			w, h, sq.Ow, sq.Oh, composer.PILTapCount(a.Filter, w, sq.Ow), a.Taps)
	}
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			i := img.PixOffset(x, y)
			img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = uint8(x*7+y), uint8(x/251), uint8((x^y)*3), 255
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
	side := min(composer.PicBandRows(model.ColumnPic(a), sq.Pw), a.Taps)
	rows, cols := (h+side-1)/side, (w+a.Taps-1)/a.Taps
	if want := rows * (cols + 1); tw.Bands != want {
		t.Fatalf("a %dx%d picture ran %d bands, its %d row bands of %d column bands and the row band's pass each", w, h, tw.Bands, rows, cols)
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
		t.Fatal("the wide picture encodes as a black one: its pixels are not read")
	}
	t.Logf("a %dx%d picture resized to %dx%d: %d row bands of %d column bands of %d taps, %d values", w, h, sq.Ow, sq.Oh, rows, cols, a.Taps, len(got))
}
