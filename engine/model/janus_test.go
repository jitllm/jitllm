package model

import (
	"image"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// Janus-Pro-1B: a SigLIP-L tower on one 384 square, llava's two-matrix
// aligner with its second matrix named mm.1, a letterboxed picture, and the
// processor's <begin_of_image>/<end_of_image> markers. The goldens are
// scripts/janusgold.py's, transformers' own JanusVisionModel and aligner.

var (
	janusText = testmodels.Path("minicpmv/Janus-Pro-1B-Q4_K_M.gguf")
	janusProj = testmodels.Path("minicpmv/mmproj-Janus-Pro-1B-Q8_0.gguf")
	janusGold = testmodels.Path("minicpmv/hf/Janus-Pro-1B/gold")
)

func openJanus(t testing.TB, opts ...Option) *Model {
	t.Helper()
	if !vlmAvailable(janusText, janusProj) {
		t.Skipf("MODEL MISSING: %s, its mmproj and its merged container -- this gate proved nothing", janusText)
	}
	m, err := Open(jlmOfPair(t, janusText, janusProj), opts...)
	if err != nil {
		t.Fatal(err)
	}
	if m.Tower() == nil {
		m.Close()
		t.Fatal("the merged container carries no tower, so this gate proves nothing")
	}
	return m
}

// TestJanusTowerShape pins the container's description of the tower, and that
// the aligner's second matrix -- mm.1 in the file -- reached the container as
// the projector's second matrix rather than being dropped or refused.
func TestJanusTowerShape(t *testing.T) {
	m := openJanus(t)
	defer m.Close()
	tw := m.Tower()
	c := tw.Cfg
	for _, tc := range []struct {
		name      string
		got, want int
	}{
		{"NLayer", c.NLayer, 24}, {"NEmbd", c.NEmbd, 1024}, {"NHead", c.NHead, 16}, {"NFFN", c.NFFN, 4096},
		{"ImageSz", c.ImageSz, 384}, {"PatchSz", c.PatchSz, 16}, {"Patches", c.Patches(), 576},
		{"Tokens", c.Tokens(), 576}, {"ProjDim", c.ProjDim, 2048}, {"text NEmbd", m.Cfg.NEmbd, 2048},
	} {
		if tc.got != tc.want {
			t.Errorf("%s = %d, want %d", tc.name, tc.got, tc.want)
		}
	}
	if c.Projector != jlm.ProjJanus.String() || c.CLS || c.Rope || c.PosBuckets != 0 {
		t.Errorf("projector %s, class token %v, rope %v, buckets %d", c.Projector, c.CLS, c.Rope, c.PosBuckets)
	}
	if tw.projW2.e == nil || tw.projW2.rows != 2048 || tw.projW2.k != 2048 {
		t.Errorf("the aligner's second matrix is missing or misshapen: %+v", tw.projW2.e)
	}
}

// TestJanusMatchesTransformers holds the letterbox and the tower and aligner
// to transformers' own on the photo (a letterboxed 2:1 picture) and on the
// repository's square, which no letterbox touches.
func TestJanusMatchesTransformers(t *testing.T) {
	m := openJanus(t)
	defer m.Close()
	tw := m.Tower()
	c := tw.Cfg
	for _, pic := range []struct{ name, path string }{{"photo896x448", mcvPhoto}, {"quad-and-disc", testImage}} {
		t.Run(pic.name, func(t *testing.T) {
			if _, err := os.Stat(filepath.Join(janusGold, pic.name+".json")); err != nil {
				t.Skipf("GOLDEN MISSING: %v -- run scripts/janusgold.py; this gate proved nothing", err)
			}
			img := decodeImage(t, pic.path)
			s := tw.testState()
			defer s.Close()
			px, err := s.PreprocessImage(img)
			if err != nil {
				t.Fatal(err)
			}
			base := filepath.Join(janusGold, pic.name)
			want := readF32n(t, base+".px", c.ImageSz*c.ImageSz*3)
			off, worst := 0, float32(0)
			for j := range want {
				d := px[j] - want[j]
				if d < 0 {
					d = -d
				}
				if d > 1e-6 {
					off++
				}
				worst = max(worst, d)
			}
			t.Logf("%d of %d pixel values off the processor's, worst by %.5f", off, len(want), worst)
			if worst > 1.01/127.5 || off*10000 > len(want) {
				t.Errorf("the letterbox is not the processor's")
			}
			var vit []float32
			s.vis.beforeProj = func(x []float32) { vit = append(vit[:0], x...) }
			emb, err := s.Encode(want)
			if err != nil {
				t.Fatal(err)
			}
			nv := visNMSE(t, "tower", vit, readF32n(t, base+".vit", c.Patches()*c.NEmbd))
			ne := visNMSE(t, "aligner", emb, readF32n(t, base+".emb", c.Tokens()*c.ProjDim))
			t.Logf("tower NMSE %.3e, aligner NMSE %.3e", nv, ne)
			// Three things part the tower from transformers' float32, measured
			// apart on transformers' own tower: its MLP's erf GELU is tanh here,
			// as in llama.cpp's use_gelu (7e-4); the mmproj's weights are Q8_0
			// (with the tanh, 2.7e-3 and 6.3e-3); and the engine's int8
			// activations over 24 blocks, the rest. The bound is the sum of
			// the three on these two pictures, with room.
			if nv > 6e-2 || ne > 5e-2 {
				t.Errorf("tower %.3e, aligner %.3e", nv, ne)
			}
		})
	}
}

// TestJanusOnDeviceMatchesHost places every tower block on each backend.
func TestJanusOnDeviceMatchesHost(t *testing.T) {
	m := openJanus(t)
	defer m.Close()
	tw := m.Tower()
	c := tw.Cfg
	px := make([]float32, c.ImageSz*c.ImageSz*3)
	for i := range px {
		px[i] = float32((i*37)%255)/127.5 - 1
	}
	host := tw.testState()
	want, err := host.Encode(px)
	if err != nil {
		t.Fatal(err)
	}
	want = append([]float32(nil), want...)
	host.Close()
	ran := 0
	for _, spec := range stepDevices() {
		g, err := openTier(t, tier.WithDevices(spec))
		if err != nil || g == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer g.Close()
			s := tw.testState()
			defer s.Close()
			s.SetDevice(g)
			if s.GPUBlocks() != c.NLayer {
				t.Fatalf("%s took %d of %d tower blocks (%s)", spec, s.GPUBlocks(), c.NLayer, g.Err())
			}
			got, err := s.Encode(px)
			if err != nil {
				t.Fatal(err)
			}
			n := visNMSE(t, "embeddings", got, want)
			// One block alone, where both tiers' int8 roundings have not yet
			// compounded: the bound that sees a wrong graph (vision_test.go's).
			after0 := func(s *State) []float32 {
				var x0 []float32
				s.vis.afterBlock = func(li int, x []float32) {
					if li == 0 && x0 == nil {
						x0 = append([]float32(nil), x...)
					}
				}
				if _, err := s.Encode(px); err != nil {
					t.Fatal(err)
				}
				return x0
			}
			hs := tw.testState()
			h0 := after0(hs)
			hs.Close()
			defer func(was int) { tw.opt.towerGPULayers = was }(tw.opt.towerGPULayers)
			tw.opt.towerGPULayers = 1
			one := tw.testState()
			defer one.Close()
			one.SetDevice(g)
			n0 := visNMSE(t, "block 0", after0(one), h0)
			t.Logf("%s, every block placed: NMSE %.3e against the host; after one block %.3e", spec, n, n0)
			// Metal reads ~4e-5 after one block where CUDA reads ~1e-6: its matvec
			// rounds the activation its own way, as sm_70's does (vision_test.go).
			if n > 2e-2 || n0 > 1e-4 {
				t.Fatalf("%s: NMSE %.3e, after one block %.3e -- the device is not running this tower", spec, n, n0)
			}
		})
	}
	if ran == 0 {
		t.Skip("no GPU backend on this host")
	}
}

// TestJanusDescribesAnImage asks about the photo and, as the violation, a
// blank picture through the identical path.
func TestJanusDescribesAnImage(t *testing.T) {
	m := openJanus(t)
	defer m.Close()
	tw := m.Tower()
	describe := func(img image.Image) string {
		s := tw.testState()
		px, err := s.PreprocessImage(img)
		if err != nil {
			t.Fatal(err)
		}
		emb, err := s.Encode(px)
		if err != nil {
			t.Fatal(err)
		}
		emb = append([]float32(nil), emb...)
		s.Close()
		msgs := []ChatMessage{{Role: "user", Content: "Describe this image in one sentence.", Images: 1}}
		spans, err := m.ChatSpans(msgs, [][]float32{emb}, true)
		if err != nil {
			t.Fatal(err)
		}
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
		t.Errorf("a BLANK picture is described as the photo too (%q): the embeddings are not what "+
			"produced the answer", none)
	}
}
