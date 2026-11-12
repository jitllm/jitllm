package engine

import (
	"encoding/json"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestLFM2VLRealMatchesTransformers holds LiquidAI's LFM2-VL-450M to
// transformers on the photo (scripts/lfm2vlreal.py: its own
// Lfm2VlForConditionalGeneration and Lfm2VlImageProcessor in float32): every
// piece's pixels the processor's, the rows after each piece's marker row
// within 3e-2 of get_image_features, then every block on each card and half
// of them within twice the CPU device's distance, a grey picture of the
// photo's size the violation.
func TestLFM2VLRealMatchesTransformers(t *testing.T) {
	dir := testmodels.Path("vlmb/hf/LFM2-VL-450M/goldens")
	raw, err := os.ReadFile(filepath.Join(dir, "photo.json"))
	if err != nil {
		testmodels.Missing(t, "GOLDEN MISSING: %v -- run scripts/lfm2vlreal.py (RULE 11)", err)
	}
	g := &lfm2Golden{}
	if err := json.Unmarshal(raw, g); err != nil {
		t.Fatal(err)
	}
	text, proj := testmodels.Path("vlmb/LFM2-VL-450M-F16.gguf"), testmodels.Path("vlmb/mmproj-LFM2-VL-450M-F16.gguf")
	if !sourcePresent(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and %s (RULE 11)", text, proj)
	}
	f, err := os.Open(testmodels.Path("vlm/goldens/photo.png"))
	if err != nil {
		testmodels.Missing(t, "PICTURE MISSING: %v", err)
	}
	img, err := png.Decode(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	px := readF32File(t, filepath.Join(dir, "photo.px"))
	// The violation: the mmproj alone names no filter and resamples as the
	// class defaults, bilinear.
	bare, err := model.Open(jlmOfPair(t, text, proj))
	if err != nil {
		t.Fatal(err)
	}
	s := testInstance(t, bare, 64, WithKVF16(false))
	barePx, _ := towerPixelsAt(t, s, img, 0)
	s.Close()
	bare.Close()
	off := 0
	for j, v := range barePx[:min(len(barePx), len(px))] {
		if math.Abs(float64(v-px[j])) >= 2e-4 {
			off++
		}
	}
	if off == 0 {
		t.Fatal("VIOLATION: the mmproj converted alone resamples as the processor does: the gate cannot see the config")
	}
	t.Logf("violation (the mmproj alone, bilinear): %d of the first piece's %d samples part", off, len(barePx))
	m, err := model.Open(jlmWithConfig(t, text, proj, testmodels.Path("vlmb/hf/LFM2-VL-450M")))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	tg := m.Tower
	b := img.Bounds()
	if _, n := squareOf(t, tg, b.Dx(), b.Dy(), 0); n != len(g.Pieces) {
		t.Fatalf("the photo is %d pieces, the processor's %d", n, len(g.Pieces))
	}
	opts := []Option{WithKVF16(false)}
	s = testInstance(t, m, 64, opts...)
	at, differ, worst := 0, 0, 0.0
	for i, pc := range g.Pieces {
		got, sq := towerPixelsAt(t, s, img, i)
		if [2]int{sq.Gh, sq.Gw} != pc {
			s.Close()
			t.Fatalf("piece %d is %dx%d patches, the processor's %dx%d", i, sq.Gh, sq.Gw, pc[0], pc[1])
		}
		for j, v := range got {
			d := math.Abs(float64(v - px[at+j]))
			if !(d < 2e-4) {
				differ++
			}
			worst = math.Max(worst, d)
		}
		at += len(got)
	}
	if at != len(px) || differ > 0 {
		s.Close()
		t.Fatalf("%d of %d samples differ from the processor's, worst by %.5f (%d of the golden's %d read)",
			differ, at, worst, at, len(px))
	}
	rows, err := s.EncodePicture(img)
	s.Close()
	if err != nil {
		t.Fatal(err)
	}
	want := readF32File(t, filepath.Join(dir, "photo.features.f32"))
	D := int(m.Container.Vision().ProjDim)
	ref := slices.Clone(rows)
	var got []float32
	pos, fed := 0, 0
	for i := range g.Pieces {
		sq, _ := squareOf(t, tg, b.Dx(), b.Dy(), i)
		marks := 0
		if sq.Mark > 0 {
			marks = 1
		}
		n := tg.SquareOut(sq) - marks
		if (pos+marks+n)*D > len(rows) || (fed+n)*D > len(want) {
			t.Fatalf("piece %d overruns: the engine's %d rows, transformers' %d", i, len(rows)/D, len(want)/D)
		}
		got = append(got, rows[(pos+marks)*D:(pos+marks+n)*D]...)
		copy(ref[(pos+marks)*D:], want[fed*D:(fed+n)*D])
		pos, fed = pos+marks+n, fed+n
	}
	if pos*D != len(rows) || fed*D != len(want) {
		t.Fatalf("%d rows read of %d, %d of transformers' %d", pos, len(rows)/D, fed, len(want)/D)
	}
	nm, wd := nmseOf(got, want)
	t.Logf("%d pieces, every sample the processor's; %d rows: NMSE %.3e, max|d| %.4f against transformers' f32 tower",
		len(g.Pieces), fed, nm, wd)
	if math.IsNaN(nm) || nm > 3e-2 {
		t.Fatalf("the real tower's rows read NMSE %.3e against transformers'", nm)
	}
	towerRunsOnTheCard(t, m, img, ref, 0, opts...)
}
