package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// Phi-4-reasoning-vision against transformers' own classes
// (scripts/phi4visiongold.py): Siglip2VisionModel at hidden_states[-2], the
// mlp2x_gelu projector and Phi3ForCausalLM, both GGUFs by llama.cpp's
// converter. And the picture arithmetic against the reference on its own
// (--model-free): the processor's resample size and SigLIP2's position-table
// resize onto grids up, down and unequal.

type p4vGolden struct {
	Pre   []int32 `json:"pre"`
	Post  []int32 `json:"post"`
	Cont  []int32 `json:"cont"`
	Image struct {
		H, W   int
		RGB    []byte    `json:"rgb"`
		Grid   [2]int    `json:"grid"` // patches, rows then columns
		PX     []float32 `json:"px"`   // the processor's picture, [y][x][c]
		Vit    []float32 `json:"vit"`
		VitA8  []float32 `json:"vit_a8"`
		Embd   []float32 `json:"embd"`
		EmbdA8 []float32 `json:"embd_a8"`
	} `json:"image"`
	Logits   []pixLogits `json:"logits"`
	LogitsA8 []pixLogits `json:"logits_a8"`
}

const p4vFixture = "synth-phi4v"

func openPhi4VFixture(t testing.TB, opts ...Option) (*Model, *p4vGolden) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "phi4v", p4vFixture+".json"))
	if err != nil {
		t.Fatal(err)
	}
	g := &p4vGolden{}
	if err := json.Unmarshal(raw, g); err != nil {
		t.Fatal(err)
	}
	text, proj := testmodels.Path(p4vFixture+".gguf"), testmodels.Path("mmproj-"+p4vFixture+".gguf")
	if !vlmAvailable(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (set JITLLM_MODELS) -- run scripts/phi4visiongold.py (RULE 11)", text)
	}
	m, err := Open(jlmOfPair(t, text, proj), append([]Option{noTune, WithKVF16(false)}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	return m, g
}

type phi4Gold struct {
	Patch, Lo, Hi int
	Sizes         []struct{ W, H, TW, TH int }
	Grids         []struct {
		GH, GW int
		V      []float64
	}
}

func readPhi4Gold(t *testing.T) phi4Gold {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "phi4-naflex.json"))
	if err != nil {
		t.Fatalf("GOLDEN MISSING: %v (scripts/phi4visiongold.py --model-free)", err)
	}
	var g phi4Gold
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	if len(g.Sizes) == 0 || len(g.Grids) == 0 {
		t.Fatal("the golden is empty")
	}
	return g
}

// TestNaflexSizeMatchesProcessor holds the resample size to transformers'
// binary search, for pictures below the minimum, inside the range and above
// the maximum.
func TestNaflexSizeMatchesProcessor(t *testing.T) {
	g := readPhi4Gold(t)
	for _, s := range g.Sizes {
		tw, th := naflexSize(s.W, s.H, g.Patch, g.Lo, g.Hi)
		if tw != s.TW || th != s.TH {
			t.Errorf("%dx%d resamples to %dx%d, the processor's %dx%d", s.W, s.H, tw, th, s.TW, s.TH)
		}
	}
	t.Logf("%d sizes, every one the processor's", len(g.Sizes))
}

// TestPositionTableResizeMatchesTorch resizes a 16x16 table of a known
// formula onto each grid and holds it to F.interpolate(bilinear,
// antialias=True). A reduction is where the antialiasing window widens, so
// the golden must hold one.
func TestPositionTableResizeMatchesTorch(t *testing.T) {
	g := readPhi4Gold(t)
	const side, ch = 16, 2
	tab := make([]float32, side*side*ch)
	for y := 0; y < side; y++ {
		for x := 0; x < side; x++ {
			for c := 0; c < ch; c++ {
				tab[(y*side+x)*ch+c] = float32(math.Sin(0.37*float64(y) + 0.11*float64(x*c) + float64(c)))
			}
		}
	}
	tw := &Tower{posW: tab, Cfg: TowerConfig{ImageSz: side * 16, PatchSz: 16, NEmbd: ch, Kind: jlm.ProjPhi4}}
	s := &State{vis: &visRun{t: tw}, jit: nn.NewJIT(64, 64, []quant.Type{quant.F32})}
	defer s.jit.Close()
	reduced := 0
	for _, gr := range g.Grids {
		got := s.phi4Table(gr.GH, gr.GW)
		if len(got) != len(gr.V) {
			t.Fatalf("%dx%d: %d values, want %d", gr.GH, gr.GW, len(got), len(gr.V))
		}
		worst := 0.0
		for i := range got {
			worst = math.Max(worst, math.Abs(float64(got[i])-gr.V[i]))
		}
		t.Logf("16x16 -> %dx%d: worst |d| %.2e", gr.GH, gr.GW, worst)
		if worst > 1e-5 {
			t.Errorf("16x16 -> %dx%d: worst |d| %.2e against torch", gr.GH, gr.GW, worst)
		}
		if gr.GH < side || gr.GW < side {
			reduced++
		}
	}
	if reduced == 0 {
		t.Fatal("no golden grid reduces the table, so the antialiasing window was never exercised")
	}
}

// TestPhi4PreprocessMatchesTheProcessor holds the picture -- resized UP to
// the minimum patch count, Pillow BILINEAR -- to the processor's, value for
// value; Pillow BICUBIC is the violation.
func TestPhi4PreprocessMatchesTheProcessor(t *testing.T) {
	m, g := openPhi4VFixture(t)
	defer m.Close()
	tw := m.Tower()
	enc := func() ([]float32, [2]int) {
		s := tw.testState()
		defer s.Close()
		got, err := s.PreprocessImage(goldenImage(g.Image.W, g.Image.H, g.Image.RGB))
		if err != nil {
			t.Fatal(err)
		}
		gh, gw := s.patchGrid()
		return got, [2]int{gh, gw}
	}
	got, grid := enc()
	if grid != g.Image.Grid {
		t.Fatalf("a %dx%d picture became %v patches, the processor's %v", g.Image.W, g.Image.H, grid, g.Image.Grid)
	}
	nm, worst := nmse32(g.Image.PX, got)
	t.Logf("%dx%d -> %v patches: NMSE %.3e, max|d| %.2e against the processor", g.Image.W, g.Image.H, grid, nm, worst)
	if math.IsNaN(nm) || nm > 1e-10 || worst > 1e-5 {
		t.Fatalf("the picture is not the processor's: NMSE %.3e, max|d| %.2e", nm, worst)
	}
	tw.Cfg.fault = faultPhi4Bicubic
	bad, _ := enc()
	tw.Cfg.fault = towerFaultNone
	nb, _ := nmse32(g.Image.PX, bad)
	t.Logf("violation: BICUBIC, NMSE %.3e", nb)
	if !(nb > 1e-6) {
		t.Fatalf("BICUBIC reads %.3e: the gate cannot see the filter", nb)
	}
}

func p4vEncode(t *testing.T, tw *Tower, g *p4vGolden) []float32 {
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

// TestPhi4TowerMatchesTransformers holds the projector's rows to
// transformers' tower and projector at the engine's arithmetic, one per
// patch; the table read at the grid's transpose, or not resized at all, moves
// it far past the bound.
func TestPhi4TowerMatchesTransformers(t *testing.T) {
	m, g := openPhi4VFixture(t)
	defer m.Close()
	tw := m.Tower()
	c := tw.Cfg
	if c.NLayer != 2 {
		t.Fatalf("the tower has %d blocks; the checkpoint's 3 less the one hidden_states[-2] skips is 2", c.NLayer)
	}
	got := p4vEncode(t, tw, g)
	if len(got) != len(g.Image.EmbdA8) {
		t.Fatalf("%d values, transformers' %d", len(got), len(g.Image.EmbdA8))
	}
	clean, worst := nmse32(g.Image.EmbdA8, got)
	f32, _ := nmse32(g.Image.Embd, got)
	a8, _ := nmse32(g.Image.Embd, g.Image.EmbdA8)
	t.Logf("%d rows of %d: NMSE %.3e, max|d| %.4f at the engine's arithmetic (%.3e against the f32 tower, "+
		"where the arithmetic alone moves transformers by %.3e)", len(got)/c.ProjDim, c.ProjDim, clean, worst, f32, a8)
	if math.IsNaN(clean) || clean > towerBound {
		t.Fatalf("the rows are not transformers': NMSE %.3e (bound %.0e)", clean, towerBound)
	}
	// The projector alone, on transformers' own tower rows at the engine's
	// arithmetic: its two matrices and the GELU between, without the tower's
	// int8 in front of them.
	ps := tw.testState()
	if _, err := ps.PreprocessImage(goldenImage(g.Image.W, g.Image.H, g.Image.RGB)); err != nil {
		t.Fatal(err)
	}
	pj, err := ps.project(append([]float32(nil), g.Image.VitA8...))
	if err != nil {
		t.Fatal(err)
	}
	pn, _ := nmse32(g.Image.EmbdA8, pj)
	ps.Close()
	t.Logf("the projector alone on transformers' tower rows: NMSE %.3e", pn)
	if math.IsNaN(pn) || pn > 1e-4 {
		t.Fatalf("the projector alone reads NMSE %.3e against transformers'", pn)
	}
	for _, v := range []struct {
		name string
		set  func() func()
	}{
		{"the table at the grid's transpose", func() func() {
			tw.Cfg.fault = faultPhi4PosTransposed
			return func() { tw.Cfg.fault = towerFaultNone }
		}},
		{"the table's first rows, not resized", func() func() {
			tw.Cfg.fault = faultPhi4NoResize
			return func() { tw.Cfg.fault = towerFaultNone }
		}},
	} {
		undo := v.set()
		bad := p4vEncode(t, tw, g)
		undo()
		nb, _ := nmse32(g.Image.EmbdA8, bad)
		t.Logf("violation %q: NMSE %.3e", v.name, nb)
		// The table is a small term beside the patch rows in this fixture,
		// as in the checkpoint; each violation is held to ten times the clean
		// arm and past the bound.
		if !(nb > 10*clean && nb > towerBound) {
			t.Errorf("%s reads %.3e against a clean %.3e: the gate cannot see it", v.name, nb, clean)
		}
	}
}

// p4vRun prefills the golden's prompt, the picture where "<image>" stood with
// no markers, and decodes the continuation, on gpu when it is not nil.
func p4vRun(t *testing.T, m *Model, g *p4vGolden, gpu *tier.GPU) [][]float32 {
	t.Helper()
	st := m.NewState(8)
	pic, err := st.Picture(goldenImage(g.Image.W, g.Image.H, g.Image.RGB))
	st.Close()
	if err != nil {
		t.Fatal(err)
	}
	if want := g.Image.Grid[0] * g.Image.Grid[1]; pic.Rows != want {
		t.Fatalf("the picture takes %d positions, the processor's %d patches", pic.Rows, want)
	}
	spans := []Span{{Tokens: g.Pre}, {Picture: pic}, {Tokens: g.Post}}
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
	if gpu != nil && s.GPULayers() != m.Cfg.NLayer {
		t.Fatalf("after the prompt the device holds %d of %d text blocks (%s)", s.GPULayers(), m.Cfg.NLayer, gpu.Err())
	}
	return out
}

// TestPhi4MatchesTransformersEndToEnd holds the whole model from the pixels
// to the logits: the picture's rows where "<image>" stood, then the
// continuation, at the engine's arithmetic. The text model is phi3 with a
// stated sliding_window of zero, full attention on every layer.
func TestPhi4MatchesTransformersEndToEnd(t *testing.T) {
	m, g := openPhi4VFixture(t)
	defer m.Close()
	if m.Cfg.SWAWindow != 0 {
		t.Fatalf("the text model slides at %d; a stated window of zero is full attention", m.Cfg.SWAWindow)
	}
	got := p4vRun(t, m, g, nil)
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
	for _, spec := range stepDevices() {
		gpu, err := openTier(t, tier.WithDevices(spec))
		if err != nil || gpu == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		t.Run(spec, func(t *testing.T) {
			defer gpu.Close()
			dev := p4vRun(t, m, g, gpu)
			var worst float64
			for i := range dev {
				nm, _ := nmseOf(dev[i], got[i])
				worst = math.Max(worst, nm)
			}
			t.Logf("%s, every block: worst logit NMSE %.3e against the host", spec, worst)
			if math.IsNaN(worst) || worst > 1e-3 {
				t.Fatalf("%s: %.3e against the host", spec, worst)
			}
		})
	}
}

// TestPhi4TowerOnDeviceMatchesHost runs every tower block on each device
// backend and holds the rows to the host's.
func TestPhi4TowerOnDeviceMatchesHost(t *testing.T) {
	m, g := openPhi4VFixture(t)
	defer m.Close()
	tw := m.Tower()
	want := p4vEncode(t, tw, g)
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
			px, err := s.PreprocessImage(goldenImage(g.Image.W, g.Image.H, g.Image.RGB))
			if err != nil {
				t.Fatal(err)
			}
			out, err := s.Encode(px)
			if err != nil {
				t.Fatal(err)
			}
			nm, worst := nmseOf(out, want)
			t.Logf("%s, all %d blocks: NMSE %.3e, max|d| %.4f against the host", spec, tw.Cfg.NLayer, nm, worst)
			if math.IsNaN(nm) || math.IsInf(nm, 0) || nm > 1e-3 {
				t.Fatalf("%s: the tower on the device reads NMSE %.3e against the host", spec, nm)
			}
		})
	}
	if ran == 0 {
		t.Skip("no GPU backend on this host")
	}
}
