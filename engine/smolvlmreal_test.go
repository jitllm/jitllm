package engine

import (
	"encoding/json"
	"image"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestSmolVLMRealMatchesTransformers holds SmolVLM-256M's stated tower to
// transformers' Idefics3ImageProcessor, vision model and connector in
// float32 on the photo (scripts/smolvlmreal.py), the container converted
// with the publisher's preprocessor_config.json: the photo cut into the
// processor's 4 x 4 tiles and its global view, in its order, after its two
// whole-picture resizes (jlm.PictureGeometry.Stages); each square's pixels
// the processor's (a resize fewer, as the engine made them before the
// stages, parts by up to sixteen levels); each square's 64 rows within 3e-2
// of transformers' (the tower's matrices the converter's, its activations
// int8). The tiles read in reverse order and a grey picture of the photo's
// size are the violations.
func TestSmolVLMRealMatchesTransformers(t *testing.T) {
	dir := testmodels.Path("vlmb/hf/SmolVLM-256M-Instruct")
	gold := filepath.Join(dir, "goldens")
	if _, err := os.Stat(filepath.Join(gold, "photo.features.f32")); err != nil {
		testmodels.Missing(t, "GOLDEN MISSING: %v -- run scripts/smolvlmreal.py (RULE 11)", err)
	}
	text, proj := testmodels.Path("SmolVLM-256M-Instruct-Q8_0.gguf"), testmodels.Path("mmproj-SmolVLM-256M-Instruct-Q8_0.gguf")
	if !sourcePresent(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and %s (RULE 11)", text, proj)
	}
	m, err := model.Open(jlmWithConfig(t, text, proj, filepath.Join(dir, "config.json")))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	f, err := os.Open(testmodels.Path("vlm/goldens/photo.png"))
	if err != nil {
		testmodels.Missing(t, "PICTURE MISSING: %v", err)
	}
	img, err := png.Decode(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(gold, "photo.json"))
	if err != nil {
		t.Fatal(err)
	}
	var meta struct{ Rows, Cols, Squares, Side int }
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	wantPx := readF32File(t, filepath.Join(gold, "photo.pixels.f32"))
	want := readF32File(t, filepath.Join(gold, "photo.features.f32"))
	b := img.Bounds()
	_, n, err := m.Tower.Square(b.Dx(), b.Dy(), 0)
	if err != nil || n != meta.Squares || meta.Rows*meta.Cols+1 != n {
		t.Fatalf("the photo is %d squares (%v), the processor's %d (%dx%d tiles and the global view)", n, err, meta.Squares,
			meta.Rows, meta.Cols)
	}
	s := testInstance(t, m, 64)
	defer s.Close()
	per := 3 * meta.Side * meta.Side
	for i := range n {
		px, _ := towerPixelsAt(t, s, img, i)
		pn, pw := nmseOf(px, wantPx[i*per:(i+1)*per])
		if math.IsNaN(pn) || pn > 1e-10 || pw > 1e-5 {
			t.Errorf("square %d's pixels read NMSE %.3e (max|d| %.3e) against the processor's", i, pn, pw)
		}
		t.Logf("square %d: pixels NMSE %.3e, max|d| %.3f", i, pn, pw)
	}
	rows, err := s.EncodePicture(img)
	if err != nil {
		t.Fatal(err)
	}
	D := m.Cfg.NEmbd
	squares := func(rows []float32) [][]float32 {
		var out [][]float32
		at := 0
		for i := range n {
			sq, _, err := m.Tower.Square(b.Dx(), b.Dy(), i)
			if err != nil {
				t.Fatal(err)
			}
			// A square's rows are its markers' and its own.
			k := m.Tower.SquareOut(sq) - sq.MarkN - sq.TrailN
			at += sq.MarkN * D
			out = append(out, rows[at:at+k*D])
			at += (k + sq.TrailN) * D
		}
		if at != len(rows) {
			t.Fatalf("the squares and their markers are %d values of the picture's %d", at, len(rows))
		}
		return out
	}
	got := squares(rows)
	side := len(want) / n
	worst := 0.0
	for i, g := range got {
		e, w := nmseOf(g, want[i*side:(i+1)*side])
		worst = max(worst, e)
		t.Logf("square %d: %d rows, NMSE %.3e, max|d| %.3f against transformers' f32 tower", i, len(g)/D, e, w)
		if len(g) != side || math.IsNaN(e) || e > 3e-2 {
			t.Errorf("square %d's %d values read NMSE %.3e against transformers' %d", i, len(g), e, side)
		}
	}
	rev := slices.Clone(got[:n-1])
	slices.Reverse(rev)
	if e, _ := nmseOf(slices.Concat(rev...), want[:(n-1)*side]); !(e > 10*worst) {
		t.Errorf("VIOLATION: the tiles in reverse order read NMSE %.3e against the worst square's %.3e", e, worst)
	}
	grey := image.NewGray(image.Rect(0, 0, b.Dx(), b.Dy()))
	for i := range grey.Pix {
		grey.Pix[i] = 128
	}
	gr, err := s.EncodePicture(grey)
	if err != nil {
		t.Fatal(err)
	}
	if e, _ := nmseOf(slices.Concat(squares(gr)...), want); !(e > 10*worst) {
		t.Errorf("VIOLATION: a grey picture reads NMSE %.3e against the worst square's %.3e", e, worst)
	}
}
