package model

import (
	"image"
	"math"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestGemma3nRealMatchesTransformers holds Gemma 3n E2B's real MobileNet-V5
// and embedder to transformers' (scripts/gemma3nreal.py) on a real
// photograph: the processor's pixels, and the rows the text model reads. The
// container is the published Q4_K_M text model and an f16 mmproj written by
// llama.cpp's own converter from the checkpoint's vision tensors, one file;
// the tower's matrices are the converter's Q8_0 and its activations int8,
// which is what separates the two. Then every tower block on each device
// backend against the host.
func TestGemma3nRealMatchesTransformers(t *testing.T) {
	dir := testmodels.Path(filepath.Join("vlmb", "hf", "gemma-3n-E2B-it", "goldens"))
	want := readF32(t, filepath.Join(dir, "photo.features.f32"))
	wantPx := readF32(t, filepath.Join(dir, "photo.pixels.f32"))
	text := testmodels.Path(filepath.Join("gemma3n", "gemma-3n-E2B-it-Q4_K_M.gguf"))
	proj := testmodels.Path(filepath.Join("gemma3n", "mmproj-gemma-3n-E2B-it-f16.gguf"))
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
	host.Close()
	pn, pw := nmse32(wantPx, px)
	t.Logf("pixels: NMSE %.3e, max|d| %.2e against the processor", pn, pw)
	if math.IsNaN(pn) || pw > 1e-5 {
		t.Fatalf("the photo's pixels are not the processor's: NMSE %.3e, max|d| %.2e", pn, pw)
	}
	nm, worst := nmse32(want, rows)
	t.Logf("%d blocks, %d rows of %d: NMSE %.3e, max|d| %.4f against transformers' f32 tower",
		c.NLayer, len(rows)/c.ProjDim, c.ProjDim, nm, worst)
	if math.IsNaN(nm) || nm > 3e-2 {
		t.Fatalf("the real tower's rows read NMSE %.3e against transformers", nm)
	}
	// The violation on the real weights: PyTorch's symmetric padding.
	tw.Cfg.fault = mnFaultSymmetric
	hv := tw.testState()
	_, bad := encode(hv)
	hv.Close()
	tw.Cfg.fault = towerFaultNone
	bn, _ := nmse32(want, bad)
	t.Logf("with symmetric padding: NMSE %.3e", bn)
	if !(bn > 10*nm) {
		t.Errorf("the real tower with symmetric padding reads %.3e against a clean %.3e: the gate cannot see it", bn, nm)
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

// TestGemma3nRealDescribesAnImage asks Gemma 3n E2B about the photo and a
// blank picture of the same size, through its own chat template.
func TestGemma3nRealDescribesAnImage(t *testing.T) {
	text := testmodels.Path(filepath.Join("gemma3n", "gemma-3n-E2B-it-Q4_K_M.gguf"))
	proj := testmodels.Path(filepath.Join("gemma3n", "mmproj-gemma-3n-E2B-it-f16.gguf"))
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
