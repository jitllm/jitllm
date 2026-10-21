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

// Gemma 4's vision against transformers' own Gemma4ForConditionalGeneration
// (scripts/gemma4visiongold.py): the processor's pixels, the tower and
// head's rows with the clipped linears binding, and the whole model, whose
// image rows see each other on the sliding layers only.

type g4vGolden struct {
	Pre   []int32 `json:"pre"`
	Post  []int32 `json:"post"`
	Cont  []int32 `json:"cont"`
	BOI   int32   `json:"boi"`
	EOI   int32   `json:"eoi"`
	Image struct {
		H, W   int
		RGB    []byte    `json:"rgb"`
		Grid   [2]int    `json:"grid"` // patches, rows then columns
		Soft   int       `json:"soft"`
		PX     []float32 `json:"px"` // the processor's patch rows, (ky, kx, c) each
		Embd   []float32 `json:"embd"`
		EmbdQ8 []float32 `json:"embd_q8"`
		EmbdA8 []float32 `json:"embd_a8"`
	} `json:"image"`
	Logits       []pixLogits `json:"logits"`
	LogitsA8     []pixLogits `json:"logits_a8"`
	LogitsCausal []pixLogits `json:"logits_causal"`
}

const g4vFixture = "synth-gemma4v"

func openGemma4V(t testing.TB, opts ...Option) (*Model, *g4vGolden) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "gemma4v", g4vFixture+".json"))
	if err != nil {
		t.Fatal(err)
	}
	g := &g4vGolden{}
	if err := json.Unmarshal(raw, g); err != nil {
		t.Fatal(err)
	}
	text, proj := testmodels.Path(g4vFixture+".gguf"), testmodels.Path("mmproj-"+g4vFixture+".gguf")
	if !vlmAvailable(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (set JITLLM_MODELS) -- run scripts/gemma4visiongold.py (RULE 11)", text)
	}
	m, err := Open(jlmOfPair(t, text, proj), append([]Option{noTune, WithKVF16(false)}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	return m, g
}

// g4vPixels lays the processor's patch rows out as Preprocess's picture,
// [y][x][channel], after the model's own 2 * (x - 0.5) in float32 (the
// processor rescales only).
func g4vPixels(g *g4vGolden, patch int) []float32 {
	gh, gw := g.Image.Grid[0], g.Image.Grid[1]
	w := gw * patch
	out := make([]float32, gh*gw*patch*patch*3)
	for p := 0; p < gh*gw; p++ {
		py, px := p/gw, p%gw
		for ky := 0; ky < patch; ky++ {
			for kx := 0; kx < patch; kx++ {
				for c := 0; c < 3; c++ {
					v := g.Image.PX[((p*patch+ky)*patch+kx)*3+c]
					out[((py*patch+ky)*w+px*patch+kx)*3+c] = 2 * (v - 0.5)
				}
			}
		}
	}
	return out
}

func g4vEncode(t *testing.T, tw *Tower, g *g4vGolden) []float32 {
	t.Helper()
	s := tw.testState()
	defer s.Close()
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

// TestGemma4PreprocessMatchesTheProcessor holds the picture to transformers'
// Gemma4ImageProcessorPil: the size its budget picks (here UP from 192x144)
// and Pillow's BICUBIC, then 2x - 1.
func TestGemma4PreprocessMatchesTheProcessor(t *testing.T) {
	m, g := openGemma4V(t)
	defer m.Close()
	tw := m.Tower()
	s := tw.testState()
	defer s.Close()
	got, err := s.PreprocessImage(goldenImage(g.Image.W, g.Image.H, g.Image.RGB))
	if err != nil {
		t.Fatal(err)
	}
	gh, gw := s.patchGrid()
	if [2]int{gh, gw} != g.Image.Grid {
		t.Fatalf("a %dx%d picture became %dx%d patches, the processor's %v", g.Image.W, g.Image.H, gh, gw, g.Image.Grid)
	}
	want := g4vPixels(g, tw.Cfg.PatchSz)
	nm, worst := nmse32(want, got)
	t.Logf("%dx%d -> %dx%d patches: NMSE %.3e, max|d| %.2e against the processor", g.Image.W, g.Image.H, gw, gh, nm, worst)
	if math.IsNaN(nm) || nm > 1e-10 || worst > 1e-5 {
		t.Fatalf("the picture is not the processor's: NMSE %.3e, max|d| %.2e", nm, worst)
	}
	// The violation: Pixtral's rule, which never resizes up.
	if w, h, err := tw.Cfg.pixtralResize(g.Image.W, g.Image.H); err == nil && h/tw.Cfg.PatchSz == gh && w/tw.Cfg.PatchSz == gw {
		t.Fatal("another family's resize lands on the same grid: the gate cannot see the rule")
	}
}

// TestGemma4TowerMatchesTransformers holds the head's rows to transformers'
// tower and embedder at the engine's arithmetic, with the clipped linears'
// bounds binding on a tenth of every matrix's values; each piece of the
// family's taken out moves it far past the bound.
func TestGemma4TowerMatchesTransformers(t *testing.T) {
	m, g := openGemma4V(t)
	defer m.Close()
	tw := m.Tower()
	got := g4vEncode(t, tw, g)
	if len(got) != len(g.Image.EmbdA8) {
		t.Fatalf("%d values, transformers' %d", len(got), len(g.Image.EmbdA8))
	}
	clean, worst := nmse32(g.Image.EmbdA8, got)
	f32, _ := nmse32(g.Image.Embd, got)
	q8, _ := nmse32(g.Image.Embd, g.Image.EmbdA8)
	t.Logf("%d rows of %d: NMSE %.3e, max|d| %.4f at the engine's arithmetic (%.3e against the f32 tower, "+
		"where the arithmetic alone moves transformers by %.3e)", len(got)/tw.Cfg.ProjDim, tw.Cfg.ProjDim,
		clean, worst, f32, q8)
	if math.IsNaN(clean) || clean > towerBound {
		t.Fatalf("the head's rows are not transformers': NMSE %.3e (bound %.0e)", clean, towerBound)
	}
	// The weightless v norm moves rows whose v heads are near unit RMS
	// already by little; it is held to ten times the clean arm.
	for _, v := range []struct {
		name string
		set  func() func()
		weak bool
	}{
		{"no clamps", func() func() {
			tw.Cfg.fault = faultGemma4NoClamp
			return func() { tw.Cfg.fault = towerFaultNone }
		}, false},
		{"no pool (each group's first patch)", func() func() {
			tw.Cfg.fault = faultGemma4NoPool
			return func() { tw.Cfg.fault = towerFaultNone }
		}, false},
		{"no standardisation", func() func() {
			b, s := tw.g4.bias, tw.g4.std
			tw.g4.bias, tw.g4.std = nil, nil
			return func() { tw.g4.bias, tw.g4.std = b, s }
		}, false},
		{"no weightless v norm", func() func() {
			tw.cfg.VNorm = false
			return func() { tw.cfg.VNorm = true }
		}, true},
		{"the row turning the first half", func() func() {
			tw.Cfg.fault = faultGemma4RowFirst
			return func() { tw.Cfg.fault = towerFaultNone }
		}, false},
		{"scores at 1/sqrt(head_dim)", func() func() {
			tw.cfg.AttnScale = 1 / math.Sqrt(float64(tw.Cfg.HeadDim))
			return func() { tw.cfg.AttnScale = 1 }
		}, false},
	} {
		undo := v.set()
		bad := g4vEncode(t, tw, g)
		undo()
		nb, _ := nmse32(g.Image.EmbdA8, bad)
		t.Logf("violation %q: NMSE %.3e", v.name, nb)
		if !(nb > 10*towerBound) && !(v.weak && nb > 10*clean) {
			t.Errorf("%s reads %.3e against a clean %.3e: the gate cannot see it", v.name, nb, clean)
		}
	}
}

// g4vRun prefills the golden's prompt, the picture a span (bidir as given),
// and decodes the continuation.
func g4vRun(t *testing.T, m *Model, g *g4vGolden, bidir bool) [][]float32 {
	return g4vRunOn(t, m, g, bidir, nil)
}

// g4vRunOn is g4vRun with the session on gpu (nil is the host): the text
// model's blocks and the tower's.
func g4vRunOn(t *testing.T, m *Model, g *g4vGolden, bidir bool, gpu *tier.GPU) [][]float32 {
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
		{Picture: pic, Bidir: bidir}, {Tokens: append([]int32{g.EOI}, g.Post...)}}
	s := m.NewState(SpanPositions(spans, m.Cfg.NEmbd) + len(g.Cont) + 1)
	defer s.Close()
	if gpu != nil {
		if err := s.SetDevice(gpu); err != nil {
			t.Fatal(err)
		}
		if s.GPULayers() != m.Cfg.NLayer {
			t.Fatalf("the device took %d of %d text blocks (%s)", s.GPULayers(), m.Cfg.NLayer, gpu.Err())
		}
	}
	lg, err := s.PrefillMixed(spans...)
	if err != nil {
		t.Fatal(err)
	}
	out := [][]float32{append([]float32(nil), lg...)}
	for _, id := range g.Cont[:len(g.Cont)-1] {
		if lg, err = s.Forward(id); err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	// A device that refused the prompt hands its blocks home and the host
	// answers: the arm proves nothing unless every block is still placed.
	if gpu != nil && s.GPULayers() != m.Cfg.NLayer {
		t.Fatalf("after the prompt the device holds %d of %d text blocks (%s)", s.GPULayers(), m.Cfg.NLayer, gpu.Err())
	}
	return out
}

// TestGemma4MatchesTransformersEndToEnd is the whole model from the pixels,
// the soft tokens a bidirectional run on the sliding layers and causal on
// the global ones (use_bidirectional_attention "vision"), against
// transformers' logits with the tower at the engine's arithmetic. The
// violations: the run causal everywhere, and bidirectional on every layer
// (Gemma 3's).
func TestGemma4MatchesTransformersEndToEnd(t *testing.T) {
	m, g := openGemma4V(t)
	defer m.Close()
	if !m.Cfg.BidirSWA {
		t.Fatal("the fixture's text model is not bidirectional on its sliding layers: this gate proves nothing")
	}
	got := g4vRun(t, m, g, true)
	w, flips := pixWorst(g.LogitsA8, got)
	wf, ff := pixWorst(g.Logits, got)
	t.Logf("from the pixels: worst logit NMSE %.3e, %d argmax flips (%.3e and %d against the f32 tower)", w, flips, wf, ff)
	// The toy text model's top two are within 0.05-0.11 on three rows; a
	// flip there is a tie the int8 decides, and is named rather than counted.
	// A flip past tieMargin fails.
	for i := range got {
		if r := g.LogitsA8[i]; Greedy(got[i]) != r.Argmax {
			a := Greedy(got[i])
			t.Logf("row %d: argmax %d against %d, a tie of %.3f", i, a, r.Argmax, r.Margin)
			if r.Margin >= tieMargin {
				t.Fatalf("row %d parts from transformers at a margin of %.3f", i, r.Margin)
			}
		}
	}
	if math.IsNaN(w) || w > 2e-2 {
		t.Fatalf("end to end: worst NMSE %.3e", w)
	}
	causal, _ := pixWorst(g.LogitsA8, g4vRun(t, m, g, false))
	m.Cfg.BidirSWA = false
	every, _ := pixWorst(g.LogitsA8, g4vRun(t, m, g, true))
	m.Cfg.BidirSWA = true
	t.Logf("violations: causal everywhere %.3e, bidirectional on every layer %.3e", causal, every)
	if !(causal > 10*w) || !(every > 10*w) {
		t.Fatalf("the mask's violations read %.3e and %.3e against %.3e: the gate cannot see them", causal, every, w)
	}
	// On each device backend, the text model's blocks and the tower's
	// placed: the image run bidirectional on the sliding layers only there
	// too (LayerPlan.BidirOff), held to the host's logits, and the violation
	// -- every layer bidirectional -- far from them.
	for _, spec := range stepDevices() {
		gpu, err := openTier(t, tier.WithDevices(spec))
		if err != nil || gpu == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		t.Run(spec, func(t *testing.T) {
			dev := g4vRunOn(t, m, g, true, gpu)
			gpu.Close()
			var worst float64
			for i := range dev {
				nm, _ := nmseOf(dev[i], got[i])
				worst = math.Max(worst, nm)
				if Greedy(dev[i]) != Greedy(got[i]) {
					t.Fatalf("%s: row %d's argmax is %d, the host's %d", spec, i, Greedy(dev[i]), Greedy(got[i]))
				}
			}
			// The violation on a device of its own, the first closed so the
			// two do not split the card's budget: the blocks the first
			// placement prepared carry the plan they were prepared with.
			gpu2, err := openTier(t, tier.WithDevices(spec))
			if err != nil {
				t.Fatal(err)
			}
			defer gpu2.Close()
			m.Cfg.BidirSWA = false
			bad := g4vRunOn(t, m, g, true, gpu2)
			m.Cfg.BidirSWA = true
			var nb float64
			for i := range bad {
				x, _ := nmseOf(bad[i], got[i])
				nb = math.Max(nb, x)
			}
			t.Logf("%s, every block: worst logit NMSE %.3e against the host; every layer bidirectional %.3e", spec, worst, nb)
			if math.IsNaN(worst) || worst > 1e-3 || !(nb > 10*worst) {
				t.Fatalf("%s: %.3e against the host, the violation %.3e", spec, worst, nb)
			}
		})
	}
}

// TestGemma4OnDeviceMatchesHost places every tower block on each device
// backend present, holds the rows to the host's and brings them home.
func TestGemma4OnDeviceMatchesHost(t *testing.T) {
	m, g := openGemma4V(t)
	defer m.Close()
	tw := m.Tower()
	want := g4vEncode(t, tw, g)
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
			if n := s.GPUBlocks(); n != tw.Cfg.NLayer {
				t.Fatalf("%s took %d of %d tower blocks (%s)", spec, n, tw.Cfg.NLayer, gpu.Err())
			}
			enc := func() []float32 {
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
			nm, worst := nmseOf(enc(), want)
			t.Logf("%s, all %d blocks: NMSE %.3e, max|d| %.4f against the host", spec, tw.Cfg.NLayer, nm, worst)
			if math.IsNaN(nm) || math.IsInf(nm, 0) || nm > 1e-3 {
				t.Fatalf("%s: the tower on the device reads NMSE %.3e against the host", spec, nm)
			}
			if err := s.SetDevice(nil); err != nil {
				t.Fatal(err)
			}
			if nb, _ := nmseOf(enc(), want); nb != 0 {
				t.Fatalf("%s: home again, the host reads NMSE %.3e against itself", spec, nb)
			}
		})
	}
	if ran == 0 {
		t.Skip("no GPU backend on this host")
	}
}
