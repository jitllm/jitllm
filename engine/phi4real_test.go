package engine

import (
	"encoding/json"
	"image"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestPhi4RealMatchesTransformers holds Phi-4-reasoning-vision-15B -- the
// published Q4_K_M text model and f16 mmproj, one container -- to the
// checkpoint's own processor, Siglip2VisionModel and mlp2x_gelu projector in
// float32 on the photo and the square (scripts/phi4visiongold.py DIR
// IMAGE...): the NaFlex grid the processor's, every sample within one 8-bit
// step, the projector's rows within 1e-2; then each card, every block and
// half of them, within twice the CPU device's distance on the photo
// (towerRunsOnTheCard, whose violation is a grey picture). The violation:
// the position table resized with its corners aligned rather than at
// half-pixel centres. 15B of text: scripts/cap 24G.
func TestPhi4RealMatchesTransformers(t *testing.T) {
	text := testmodels.Path("minicpmv/Phi-4-reasoning-vision-15B-Q4_K_M.gguf")
	proj := testmodels.Path("minicpmv/mmproj-Phi-4-reasoning-vision-15B-f16.gguf")
	gold := testmodels.Path("minicpmv/hf/Phi-4-reasoning-vision-15B/gold")
	if !sourcePresent(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (RULE 11)", text)
	}
	m, err := model.Open(jlmOfPair(t, text, proj))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	opts := []Option{WithKVF16(false)}
	readPNG := func(path string) image.Image {
		f, err := os.Open(path)
		if err != nil {
			testmodels.Missing(t, "PICTURE MISSING: %v", err)
		}
		defer f.Close()
		img, err := png.Decode(f)
		if err != nil {
			t.Fatal(err)
		}
		return img
	}
	tg := m.Tower
	rec := m.Container.Vision().Tower
	pos := -1
	for i, op := range rec.Entry.Ops {
		if op.Code == jlm.TOpPosResample {
			pos = i
		}
	}
	if pos < 0 || rec.Entry.Ops[pos].Chain != jlm.ResampleHalfPixel {
		t.Fatalf("the tower's entry resamples no position table at half-pixel centres: %+v", rec.Entry.Ops)
	}
	aligned := func() *model.TowerGraphs {
		// Plain bilinear: on a grid past the table's side, where antialiasing
		// changes nothing, the corners alone move it.
		op := &rec.Entry.Ops[pos]
		filter := op.Filter
		op.Chain, op.Filter = jlm.ResampleAligned, jlm.TapsBilinear
		bad, err := m.BuildTowerGraphs()
		op.Chain, op.Filter = jlm.ResampleHalfPixel, filter
		if err != nil {
			t.Fatal(err)
		}
		return bad
	}()
	photo := testmodels.Path("minicpmv/photo896x448.png")
	var photoEmb []float32
	for _, pic := range []struct{ name, path string }{
		{"photo896x448", photo},
		{"quad-and-disc", filepath.Join("testdata", "quad-and-disc.png")},
	} {
		t.Run(pic.name, func(t *testing.T) {
			base := filepath.Join(gold, pic.name)
			raw, err := os.ReadFile(base + ".json")
			if err != nil {
				testmodels.Missing(t, "GOLDEN MISSING: %v -- run scripts/phi4visiongold.py (RULE 11)", err)
			}
			var g struct{ GH, GW int }
			if err := json.Unmarshal(raw, &g); err != nil {
				t.Fatal(err)
			}
			img := readPNG(pic.path)
			s := testInstance(t, m, 64, opts...)
			defer s.Close()
			px, sq := towerPixels(t, s, img)
			if sq.Gw != g.GW || sq.Gh != g.GH {
				t.Fatalf("a %dx%d grid, the processor's %dx%d", sq.Gw, sq.Gh, g.GW, g.GH)
			}
			want := readF32(t, base+".px")
			if len(px) != len(want) {
				t.Fatalf("%d samples, the processor's %d", len(px), len(want))
			}
			worst := 0.0
			for i, w := range want {
				worst = math.Max(worst, math.Abs(float64(px[i]-w)))
			}
			// Pillow's BILINEAR at its own fixed point: one 8-bit step at most.
			if worst > 1.01/127.5 {
				t.Fatalf("the pixels are %.5f from the processor's, past one 8-bit step", worst)
			}
			rows, err := s.EncodePicture(img)
			if err != nil {
				t.Fatal(err)
			}
			emb := readF32(t, base+".emb")
			ne, nw := nmseOf(rows, emb)
			t.Logf("%dx%d patches: pixels within %.2e, the projector's rows NMSE %.3e, max|d| %.3g", sq.Gw, sq.Gh, worst, ne, nw)
			if len(rows) != len(emb) || math.IsNaN(ne) || ne > 1e-2 {
				t.Fatalf("%d values at NMSE %.3e from transformers' %d (bound 1e-2)", len(rows), ne, len(emb))
			}
			m.Tower = aligned
			bs := testInstance(t, m, 64, opts...)
			bad, err := bs.EncodePicture(img)
			bs.Close()
			m.Tower = tg
			if err != nil {
				t.Fatal(err)
			}
			nb, _ := nmseOf(bad, emb)
			t.Logf("violation (the table resized with its corners aligned): NMSE %.3e", nb)
			if !(nb > 10*ne) {
				t.Fatalf("VIOLATION: a table resized with its corners aligned reads %.3e against a clean %.3e", nb, ne)
			}
			if pic.path == photo {
				photoEmb = emb
			}
		})
	}
	if photoEmb == nil {
		t.Fatal("the photo's arm did not run")
	}
	towerRunsOnTheCard(t, m, readPNG(photo), photoEmb, 0, opts...)
}
