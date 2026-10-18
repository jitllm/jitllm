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

// Phi-4-reasoning-vision-15B itself: the published Q4_K_M text model and f16
// mmproj, one container, against the checkpoint's own tower and projector on
// two real pictures (scripts/phi4visiongold.py DIR IMAGE...). 15B of text:
// scripts/cap 24G.

var (
	phi4vText = testmodels.Path("minicpmv/Phi-4-reasoning-vision-15B-Q4_K_M.gguf")
	phi4vProj = testmodels.Path("minicpmv/mmproj-Phi-4-reasoning-vision-15B-f16.gguf")
	phi4vGold = testmodels.Path("minicpmv/hf/Phi-4-reasoning-vision-15B/gold")
)

func openPhi4VReal(t testing.TB) *Model {
	t.Helper()
	if !vlmAvailable(phi4vText, phi4vProj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (RULE 11)", phi4vText)
	}
	m, err := Open(jlmOfPair(t, phi4vText, phi4vProj), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	if m.Tower() == nil {
		m.Close()
		t.Fatal("the container carries no tower, so this gate proves nothing")
	}
	return m
}

// TestPhi4RealMatchesTransformers pins the tower the container describes,
// then holds the picture, the tower's hidden_states[-2] and the projector's
// rows to the checkpoint's own code on the photo and the square; every tower
// block on each device backend against the host.
func TestPhi4RealMatchesTransformers(t *testing.T) {
	m := openPhi4VReal(t)
	defer m.Close()
	tw := m.Tower()
	c := tw.Cfg
	for _, tc := range []struct {
		name      string
		got, want int
	}{
		// 27 blocks in the checkpoint; the converter keeps 26, the layer the
		// model reads (hidden_states[-2]).
		{"NLayer", c.NLayer, 26}, {"NEmbd", c.NEmbd, 1152},
		{"PatchSz", c.PatchSz, 16}, {"table side", c.ImageSz / c.PatchSz, 16},
		{"MinPixels", c.MinPixels, 256 * 256}, {"MaxPixels", c.MaxPixels, 3600 * 256},
		{"ProjDim", c.ProjDim, 5120}, {"text NEmbd", m.Cfg.NEmbd, 5120}, {"text window", int(m.Cfg.SWAWindow), 0},
	} {
		if tc.got != tc.want {
			t.Errorf("%s = %d, want %d", tc.name, tc.got, tc.want)
		}
	}
	for _, pic := range []struct{ name, path string }{{"photo896x448", mcvPhoto}, {"quad-and-disc", testImage}} {
		t.Run(pic.name, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join(phi4vGold, pic.name+".json"))
			if err != nil {
				t.Skipf("GOLDEN MISSING: %v -- run scripts/phi4visiongold.py (RULE 11); this gate proved nothing", err)
			}
			var g struct{ W, H, GH, GW int }
			if err := json.Unmarshal(raw, &g); err != nil {
				t.Fatal(err)
			}
			img := decodeImage(t, pic.path)
			var vit []float32
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
			s := tw.testState()
			s.vis.beforeProj = func(x []float32) { vit = append(vit[:0], x...) }
			px, emb := encode(s)
			gh, gw := s.patchGrid()
			s.Close()
			if gh != g.GH || gw != g.GW {
				t.Fatalf("a %dx%d grid, the processor's %dx%d", gw, gh, g.GW, g.GH)
			}
			base := filepath.Join(phi4vGold, pic.name)
			pn, pw := nmse32(readF32(t, base+".px"), px)
			nv, _ := nmse32(readF32(t, base+".vit"), vit)
			wantEmb := readF32(t, base+".emb")
			ne, _ := nmse32(wantEmb, emb)
			t.Logf("%dx%d patches: pixels NMSE %.3e (max|d| %.2e), tower NMSE %.3e, projector NMSE %.3e",
				gw, gh, pn, pw, nv, ne)
			// Pillow's BILINEAR at its own fixed point against the checkpoint's
			// processor: one 8-bit step at most.
			if math.IsNaN(pn) || pw > 1.01/127.5 || nv > 1e-2 || ne > 1e-2 {
				t.Fatalf("pixels %.3e (max %.2e), tower %.3e, projector %.3e", pn, pw, nv, ne)
			}
			if gh != gw {
				tw.Cfg.fault = faultPhi4PosTransposed
				s := tw.testState()
				_, bad := encode(s)
				s.Close()
				tw.Cfg.fault = towerFaultNone
				nb, _ := nmse32(wantEmb, bad)
				t.Logf("violation, the table resized to the grid's transpose: NMSE %.3e", nb)
				if !(nb > 10*ne) {
					t.Errorf("violation: a transposed position table reads %.3e against a clean %.3e", nb, ne)
				}
			}
			// The device arms run with the picture's own ceiling: a tower sized
			// for 3600 patches asks a 4 GB card, already holding the text
			// model's share, for scratch it does not have, and the placement
			// declines by budget, as it should. The grid is the same either
			// way: the picture's own patch count is under its ceiling.
			defer func(mp int) { tw.Cfg.MaxPixels = mp }(tw.Cfg.MaxPixels)
			tw.Cfg.MaxPixels = gh * gw * c.PatchSz * c.PatchSz
			for _, spec := range stepDevices() {
				gpu, err := tier.OpenWith(tier.WithDevices(spec))
				if err != nil || gpu == nil {
					t.Logf("%s: not present (%v)", spec, err)
					continue
				}
				t.Run(spec, func(t *testing.T) {
					defer gpu.Close()
					s := tw.testState()
					defer s.Close()
					s.SetDevice(gpu)
					if n := s.GPUBlocks(); n != c.NLayer {
						t.Fatalf("%s took %d of %d tower blocks (%s)", spec, n, c.NLayer, gpu.Err())
					}
					_, got := encode(s)
					dn, _ := nmseOf(got, emb)
					tn, _ := nmse32(wantEmb, got)
					t.Logf("%s, all %d blocks: NMSE %.3e against the host, %.3e against transformers", spec, c.NLayer, dn, tn)
					if math.IsNaN(dn) || dn > 1e-2 || tn > 1e-2 {
						t.Fatalf("%s: NMSE %.3e against the host, %.3e against transformers", spec, dn, tn)
					}
				})
			}
		})
	}
}

// TestPhi4RealDescribesAnImage asks about the photo and a blank picture of
// the same size, through the model's own chat template.
func TestPhi4RealDescribesAnImage(t *testing.T) {
	m := openPhi4VReal(t)
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
	img := decodeImage(t, mcvPhoto)
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
