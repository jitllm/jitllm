package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestPixtralRealMatchesTransformers holds Ministral 3's real tower and head
// to transformers' PixtralVisionModel and Mistral3MultiModalProjector on a
// real photograph (scripts/pixtralreal.py): the processor's pixels, and the
// rows the text model reads. The container is the published Q4_K_M text
// model and BF16 mmproj, one file; the tower's matrices are the converter's
// Q8_0 and its activations int8, which is what separates the two. Then every
// tower block on each device backend against the host.
func TestPixtralRealMatchesTransformers(t *testing.T) {
	dir := testmodels.Path(filepath.Join("vlmb", "hf", "Ministral-3-3B", "goldens"))
	raw, err := os.ReadFile(filepath.Join(dir, "photo.json"))
	if err != nil {
		t.Skipf("GOLDEN MISSING: %v -- run scripts/pixtralreal.py (RULE 11); this gate proved nothing", err)
	}
	var meta struct{ H, W int }
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	text := testmodels.Path(filepath.Join("newarch", "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"))
	proj := testmodels.Path(filepath.Join("vlmb", "mmproj-Ministral-3-3B-BF16.gguf"))
	if !vlmAvailable(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and %s (RULE 11)", text, proj)
	}
	m, err := Open(jlmOfPair(t, text, proj), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	tw := m.Tower()
	c := tw.Cfg
	wantPx := readF32(t, filepath.Join(dir, "photo.pixels.f32"))
	want := readF32(t, filepath.Join(dir, "photo.features.f32"))
	img := goldPicture(t, "photo")
	encode := func(s *State) ([]float32, []float32) {
		px, err := s.PreprocessImage(img)
		if err != nil {
			t.Fatal(err)
		}
		px = append([]float32(nil), px...)
		out, err := s.Encode(px)
		if err != nil {
			t.Fatal(err)
		}
		return px, append([]float32(nil), out...)
	}
	host := tw.testState()
	px, rows := encode(host)
	gh, gw := host.patchGrid()
	host.Close()
	if gh*c.PatchSz != meta.H || gw*c.PatchSz != meta.W {
		t.Fatalf("the photo resized to %dx%d, the processor's %dx%d", gw*c.PatchSz, gh*c.PatchSz, meta.W, meta.H)
	}
	pn, pw := nmse32(wantPx, px)
	t.Logf("pixels: %dx%d, NMSE %.3e, max|d| %.2e against the processor", meta.W, meta.H, pn, pw)
	if math.IsNaN(pn) || pn > 1e-10 || pw > 1e-5 {
		t.Fatalf("the photo's pixels are not the processor's: NMSE %.3e, max|d| %.2e", pn, pw)
	}
	sc := c.Scale
	g := &pixGolden{}
	g.Image.Grid = [3]int{1, gh / sc, gw / sc}
	grid, breaks := pixSplit(t, g, rows, c.ProjDim)
	nm, worst := nmse32(want, grid)
	t.Logf("%d rows of %d and %d break rows: NMSE %.3e, max|d| %.4f against transformers' f32 tower",
		len(grid)/c.ProjDim, c.ProjDim, len(breaks)/c.ProjDim, nm, worst)
	if math.IsNaN(nm) || nm > 3e-2 {
		t.Fatalf("the real tower's rows read NMSE %.3e against transformers", nm)
	}
	// The device arms run with the picture's own ceiling: a tower sized for
	// image_size 1540 asks a card for scratch over 12,100 rows, which a 4 GB
	// card does not hold and declines by budget, as placement should. The
	// photo's grid is 46x36 either way.
	defer func(sz, mp int) { tw.Cfg.ImageSz, tw.Cfg.MaxPixels = sz, mp }(tw.Cfg.ImageSz, tw.Cfg.MaxPixels)
	tw.Cfg.ImageSz = 700
	pixtralPixels(&tw.Cfg)
	ran := 0
	for _, spec := range stepDevices() {
		gpu, err := openTier(t, tier.WithDevices(spec))
		if err != nil || gpu == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer gpu.Close()
			s := tw.testState()
			defer s.Close()
			s.SetDevice(gpu)
			if n := s.GPUBlocks(); n != c.NLayer {
				t.Fatalf("%s took %d of %d tower blocks (%s)", spec, n, c.NLayer, gpu.Err())
			}
			_, got := encode(s)
			dn, dw := nmseOf(got, rows)
			gd, _ := pixSplit(t, g, got, c.ProjDim)
			tn, _ := nmse32(want, gd)
			t.Logf("%s, all %d blocks: NMSE %.3e (max|d| %.4f) against the host, %.3e against transformers",
				spec, c.NLayer, dn, dw, tn)
			if math.IsNaN(dn) || dn > 2e-2 || tn > 3e-2 {
				t.Fatalf("%s: NMSE %.3e against the host, %.3e against transformers", spec, dn, tn)
			}
		})
	}
	if ran == 0 {
		t.Log("no GPU backend on this host")
	}
}
