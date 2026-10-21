package model

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// Gemma 3n's vision against transformers' own Gemma3nForConditionalGeneration
// (scripts/gemma3nvisiongold.py): timm's MobileNet-V5 at a quarter of its
// channels, every one of its 84 blocks, the fusion adapter and the embedder.
// The golden's large half -- the picture's pixels and every block's output
// at the engine's arithmetic -- sits beside the GGUFs in synth-gemma3nv.gold.

type g3nvGolden struct {
	Pre   []int32 `json:"pre"`
	Post  []int32 `json:"post"`
	Cont  []int32 `json:"cont"`
	BOI   int32   `json:"boi"`
	EOI   int32   `json:"eoi"`
	Image struct {
		H, W   int
		RGB    []byte    `json:"rgb"`
		Soft   int       `json:"soft"`
		Embd   []float32 `json:"embd"`
		EmbdQ8 []float32 `json:"embd_q8"`
		EmbdA8 []float32 `json:"embd_a8"`
	} `json:"image"`
	Logits   []pixLogits `json:"logits"`
	LogitsA8 []pixLogits `json:"logits_a8"`
}

const g3nvFixture = "synth-gemma3nv"

func openGemma3nV(t testing.TB, opts ...Option) (*Model, *g3nvGolden) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "gemma3nv", g3nvFixture+".json"))
	if err != nil {
		t.Fatal(err)
	}
	g := &g3nvGolden{}
	if err := json.Unmarshal(raw, g); err != nil {
		t.Fatal(err)
	}
	text, proj := testmodels.Path(g3nvFixture+".gguf"), testmodels.Path("mmproj-"+g3nvFixture+".gguf")
	if !vlmAvailable(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (set JITLLM_MODELS) -- run scripts/gemma3nvisiongold.py (RULE 11)", text)
	}
	m, err := Open(jlmOfPair(t, text, proj), append([]Option{noTune, WithKVF16(false)}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	return m, g
}

// g3nvGold reads one of the fixture's large goldens.
func g3nvGold(t testing.TB, name string) []float32 {
	t.Helper()
	return readF32(t, testmodels.Path(filepath.Join(g3nvFixture+".gold", name)))
}

// g3nvEncode is the tower's rows for the golden's picture, on a fresh vision
// State; tap, when set, sees each block's output by index.
func g3nvEncode(t *testing.T, tw *Tower, g *g3nvGolden, tap func(i int, x []float32)) []float32 {
	t.Helper()
	s := tw.testState()
	defer s.Close()
	if tap != nil {
		s.vis.afterBlock = tap
	}
	px, err := s.PreprocessImage(goldenImage(g.Image.W, g.Image.H, g.Image.RGB))
	if err != nil {
		t.Fatal(err)
	}
	out, err := s.Encode(px)
	if err != nil {
		t.Fatal(err)
	}
	return append([]float32(nil), out...)
}

// TestGemma3nPreprocessMatchesTheProcessor holds the picture -- squashed to
// 768x768 by the processor's antialiased bilinear, [0, 1] -- to
// SiglipImageProcessorFast's, value for value.
func TestGemma3nPreprocessMatchesTheProcessor(t *testing.T) {
	m, g := openGemma3nV(t)
	defer m.Close()
	s := m.Tower().testState()
	defer s.Close()
	got, err := s.PreprocessImage(goldenImage(g.Image.W, g.Image.H, g.Image.RGB))
	if err != nil {
		t.Fatal(err)
	}
	want := g3nvGold(t, "pixels.f32")
	nm, worst := nmse32(want, got)
	t.Logf("%dx%d -> 768x768: NMSE %.3e, max|d| %.2e against the processor", g.Image.W, g.Image.H, nm, worst)
	if math.IsNaN(nm) || worst > 1e-6 {
		t.Fatalf("the picture is not the processor's: NMSE %.3e, max|d| %.2e", nm, worst)
	}
}

// TestGemma3nTowerMatchesTransformers holds every block's output and the
// embedder's rows to transformers' at the engine's arithmetic (Q8_0 weights,
// int8 activations on every matrix's input).
func TestGemma3nTowerMatchesTransformers(t *testing.T) {
	m, g := openGemma3nV(t)
	defer m.Close()
	tw := m.Tower()
	if tw.Cfg.NLayer != 84 {
		t.Fatalf("the tower has %d blocks, timm's encoder 84", tw.Cfg.NLayer)
	}
	var worstBlock float64
	worstAt := -1
	got := g3nvEncode(t, tw, g, func(i int, x []float32) {
		if i < 0 {
			return
		}
		want := g3nvGold(t, fmt.Sprintf("blk%d.a8.f32", i))
		if len(want) != len(x) {
			t.Fatalf("block %d: %d values, transformers' %d", i, len(x), len(want))
		}
		nm, _ := nmse32(want, x)
		if i < 4 || i%10 == 0 || nm > 1e-3 {
			t.Logf("block %d: NMSE %.3e", i, nm)
		}
		if nm > worstBlock || math.IsNaN(nm) {
			worstBlock, worstAt = nm, i
		}
	})
	clean, worst := nmse32(g.Image.EmbdA8, got)
	f32, _ := nmse32(g.Image.Embd, got)
	t.Logf("worst block %d at %.3e; %d rows of %d: NMSE %.3e, max|d| %.4f at the engine's arithmetic (%.3e against the f32 tower)",
		worstAt, worstBlock, len(got)/tw.Cfg.ProjDim, tw.Cfg.ProjDim, clean, worst, f32)
	if math.IsNaN(clean) || clean > towerBound {
		t.Fatalf("the rows are not transformers': NMSE %.3e (bound %.0e)", clean, towerBound)
	}
	// Each piece of the family's taken out moves the rows past the bound and
	// ten times the clean arm. (The embedder's sqrt(NEmbd) before its norm is
	// invisible but for the norm's epsilon, so no gate can see it; it is the
	// reference's, and kept.)
	for _, v := range []struct {
		name string
		f    mnFaultKind
	}{
		{"symmetric padding in place of SAME", mnFaultSymmetric},
		{"no residual adds", mnFaultNoResidual},
		{"the coarser stage upsampled transposed", mnFaultUpsampleT},
		{"each pooling window's first row", mnFaultPoolFirst},
	} {
		tw.Cfg.fault = v.f
		bad := g3nvEncode(t, tw, g, nil)
		tw.Cfg.fault = towerFaultNone
		nb, _ := nmse32(g.Image.EmbdA8, bad)
		t.Logf("violation %q: NMSE %.3e", v.name, nb)
		if !(nb > 10*clean && nb > towerBound) {
			t.Errorf("%s reads %.3e against a clean %.3e: the gate cannot see it", v.name, nb, clean)
		}
	}
}

// g3nvRun prefills the golden's prompt, the picture a span between its
// markers, and decodes the continuation, with the tower and the text model
// on gpu when it is not nil.
func g3nvRun(t *testing.T, m *Model, g *g3nvGolden, gpu *tier.GPU) [][]float32 {
	t.Helper()
	st := m.NewState(8)
	pic, err := st.Picture(goldenImage(g.Image.W, g.Image.H, g.Image.RGB))
	st.Close()
	if err != nil {
		t.Fatal(err)
	}
	if pic.Rows != g.Image.Soft {
		t.Fatalf("the picture takes %d positions, the processor's %d soft tokens", pic.Rows, g.Image.Soft)
	}
	spans := []Span{{Tokens: append(append([]int32(nil), g.Pre...), g.BOI)},
		{Picture: pic}, {Tokens: append([]int32{g.EOI}, g.Post...)}}
	s := m.NewState(SpanPositions(spans, m.Cfg.NEmbd) + len(g.Cont) + 1)
	defer s.Close()
	var before int64
	if gpu != nil {
		if err := s.SetDevice(gpu); err != nil {
			t.Fatal(err)
		}
		if s.GPULayers() != m.Cfg.NLayer {
			t.Fatalf("the device took %d of %d text blocks (%s)", s.GPULayers(), m.Cfg.NLayer, gpu.Err())
		}
		before = gpu.Stats().ConvBlocks
		// The host arm's rows are cached under the picture's key.
		m.imgCache.clearForTest()
	}
	lg, err := s.PrefillMixed(spans...)
	if err != nil {
		t.Fatal(err)
	}
	if gpu != nil {
		if n := gpu.Stats().ConvBlocks - before; n != int64(m.Tower().Cfg.NLayer) {
			t.Fatalf("the device ran %d of %d tower blocks (%s)", n, m.Tower().Cfg.NLayer, gpu.Err())
		}
	}
	out := [][]float32{append([]float32(nil), lg...)}
	for _, id := range g.Cont[:len(g.Cont)-1] {
		if lg, err = s.Forward(id); err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	return out
}

// TestGemma3nMatchesTransformersEndToEnd holds the whole model from the
// pixels to the logits: the picture's 256 rows between its markers, causal,
// each image position's per-layer input the reference's token 0.
func TestGemma3nMatchesTransformersEndToEnd(t *testing.T) {
	m, g := openGemma3nV(t)
	defer m.Close()
	got := g3nvRun(t, m, g, nil)
	w, flips := pixWorst(g.LogitsA8, got)
	wf, ff := pixWorst(g.Logits, got)
	t.Logf("from the pixels: worst logit NMSE %.3e, %d argmax flips (%.3e and %d against the f32 tower)", w, flips, wf, ff)
	for i := range got {
		if r := g.LogitsA8[i]; Greedy(got[i]) != r.Argmax {
			t.Logf("row %d: argmax %d against %d, a tie of %.3f", i, Greedy(got[i]), r.Argmax, r.Margin)
			if r.Margin >= tieMargin {
				t.Fatalf("row %d parts from transformers at a margin of %.3f", i, r.Margin)
			}
		}
	}
	if math.IsNaN(w) || w > 2e-2 {
		t.Fatalf("end to end: worst NMSE %.3e", w)
	}
	// The violation: the end-of-image marker embedded from the text table, as
	// llama.cpp embeds it, in place of the vision embedder's hard token.
	h := m.hardEmbd
	m.hardEmbd = nil
	bad := g3nvRun(t, m, g, nil)
	m.hardEmbd = h
	nb, _ := pixWorst(g.LogitsA8, bad)
	t.Logf("violation, the end-of-image marker from the text table: worst logit NMSE %.3e", nb)
	if !(nb > 10*w) {
		t.Fatalf("the text table's marker reads %.3e against a clean %.3e: the gate cannot see it", nb, w)
	}
	// Both halves on each device: the tower's blocks and the text model's.
	for _, spec := range stepDevices() {
		gpu, err := openTier(t, tier.WithDevices(spec))
		if err != nil || gpu == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		t.Run(spec, func(t *testing.T) {
			defer gpu.Close()
			dev := g3nvRun(t, m, g, gpu)
			var worst float64
			for i := range dev {
				nm, _ := nmseOf(dev[i], got[i])
				worst = math.Max(worst, nm)
			}
			t.Logf("%s, every block of both: worst logit NMSE %.3e against the host", spec, worst)
			if math.IsNaN(worst) || worst > 1e-3 {
				t.Fatalf("%s: %.3e against the host", spec, worst)
			}
		})
	}
}
