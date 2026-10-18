package model

import (
	"encoding/json"
	"image"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestGemma4RealMatchesTransformers holds Gemma 4 E2B's real tower and head
// to transformers' Gemma4VisionModel and Gemma4MultimodalEmbedder on a real
// photograph (scripts/gemma4real.py): the processor's pixels, and the rows
// the text model reads. E2B is the clipped-linear tower, so every block's
// clamps are the checkpoint's own. The container is the published Q4_K_M text
// model and BF16 mmproj, one file; the tower's matrices are the converter's
// Q8_0 and its activations int8, which is what separates the two. Then every
// tower block on each device backend against the host.
func TestGemma4RealMatchesTransformers(t *testing.T) {
	dir := testmodels.Path(filepath.Join("vlmb", "hf", "gemma-4-E2B-it", "goldens"))
	raw, err := os.ReadFile(filepath.Join(dir, "photo.json"))
	if err != nil {
		t.Skipf("GOLDEN MISSING: %v -- run scripts/gemma4real.py (RULE 11); this gate proved nothing", err)
	}
	var meta struct{ H, W int }
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	text := testmodels.Path(filepath.Join("vlmb", "gemma-4-E2B-it-Q4_K_M.gguf"))
	proj := testmodels.Path(filepath.Join("vlmb", "mmproj-gemma-4-E2B-it-BF16.gguf"))
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
	clamped := 0
	for i := range c.NLayer {
		if m.layers[m.Cfg.NLayer+i].clamp != nil {
			clamped++
		}
	}
	if clamped != c.NLayer {
		t.Fatalf("%d of %d tower blocks carry their clamps", clamped, c.NLayer)
	}
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
	nm, worst := nmse32(want, rows)
	t.Logf("%d rows of %d: NMSE %.3e, max|d| %.4f against transformers' f32 tower", len(rows)/c.ProjDim, c.ProjDim, nm, worst)
	if math.IsNaN(nm) || nm > 3e-2 {
		t.Fatalf("the real tower's rows read NMSE %.3e against transformers", nm)
	}
	// The violation on the real weights: the clamps off.
	tw.Cfg.fault = faultGemma4NoClamp
	hv := tw.testState()
	_, bad := encode(hv)
	hv.Close()
	tw.Cfg.fault = towerFaultNone
	bn, _ := nmse32(want, bad)
	t.Logf("without the checkpoint's clamps: NMSE %.3e", bn)
	if !(bn > 10*nm) {
		t.Errorf("the real tower without its clamps reads %.3e against a clean %.3e: the gate cannot see them", bn, nm)
	}
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
			tn, _ := nmse32(want, got)
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

// TestGemma4RealDescribesAnImage asks Gemma 4 E2B about the photo and a blank
// picture of the same size, through its own chat template.
func TestGemma4RealDescribesAnImage(t *testing.T) {
	text := testmodels.Path(filepath.Join("vlmb", "gemma-4-E2B-it-Q4_K_M.gguf"))
	proj := testmodels.Path(filepath.Join("vlmb", "mmproj-gemma-4-E2B-it-BF16.gguf"))
	if !vlmAvailable(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and %s (RULE 11)", text, proj)
	}
	m, err := Open(jlmOfPair(t, text, proj), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	describe := func(img image.Image) string {
		msgs := []ChatMessage{{Role: "user", Content: "Describe this image in one sentence.", Images: 1}}
		spans := pictureSpans(t, m, msgs, img)
		st := m.NewState(SpanPositions(spans, m.Cfg.NEmbd) + 48)
		defer st.Close()
		logits, err := st.PrefillMixed(spans...)
		if err != nil {
			t.Fatal(err)
		}
		var out []int32
		for i := 0; i < 40; i++ {
			next := Greedy(logits)
			if m.Vocab.IsEOG(next) {
				break
			}
			out = append(out, next)
			if logits, err = st.Forward(next); err != nil {
				t.Fatal(err)
			}
		}
		return strings.ToLower(m.Vocab.Decode(out))
	}
	img := goldPicture(t, "photo")
	got := describe(img)
	t.Logf("jitllm: %q", got)
	none := describe(image.NewRGBA(img.Bounds()))
	t.Logf("blank:  %q", none)
	if !namesThePhoto(got) {
		t.Errorf("the answer names nothing in the photo (a newspaper, the New York Times, the moon): %q", got)
	}
	if namesThePhoto(none) {
		t.Errorf("a BLANK picture is described as the photo too (%q): the embeddings are not what produced the answer", none)
	}
}
