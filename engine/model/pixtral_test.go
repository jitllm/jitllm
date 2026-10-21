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

// Pixtral / Mistral 3's vision against transformers' own
// Mistral3ForConditionalGeneration (scripts/pixtralgold.py): the processor's
// resize, the tower and head's rows, and the whole model from the pixels on.

// pixGolden is scripts/pixtralgold.py's record.
type pixGolden struct {
	Pre   []int32 `json:"pre"`
	Post  []int32 `json:"post"`
	Cont  []int32 `json:"cont"`
	Image struct {
		H      int       `json:"h"`
		W      int       `json:"w"`
		RGB    []byte    `json:"rgb"`
		Grid   [3]int    `json:"grid"`
		Embd   []float32 `json:"embd"`
		EmbdQ8 []float32 `json:"embd_q8"`
		EmbdA8 []float32 `json:"embd_a8"`
		// Break is the [IMG_BREAK] token's embedding in the text model.
		Break []float32 `json:"break"`
	} `json:"image"`
	Prep struct {
		H, W    int
		Longest int       `json:"longest"`
		RGB     []byte    `json:"rgb"`
		OutH    int       `json:"out_h"`
		OutW    int       `json:"out_w"`
		PX      []float32 `json:"px"`
	} `json:"prep"`
	Logits []pixLogits `json:"logits"`
	// LogitsA8 is the same prompt with transformers' tower rows at the
	// engine's arithmetic (Image.EmbdA8) spliced in.
	LogitsA8 []pixLogits `json:"logits_a8"`
}

type pixLogits struct {
	Argmax int32     `json:"argmax"`
	Head   []float64 `json:"head"`
	// Margin is the reference's top-two gap over the whole vocabulary, where
	// a golden records it (zero where not).
	Margin float64 `json:"margin"`
}

const pixFixture = "synth-pixtral"

// openPixtral opens the fixture's one container and its golden.
func openPixtral(t testing.TB, opts ...Option) (*Model, *pixGolden) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "pixtral", pixFixture+".json"))
	if err != nil {
		t.Fatal(err)
	}
	g := &pixGolden{}
	if err := json.Unmarshal(raw, g); err != nil {
		t.Fatal(err)
	}
	text, proj := testmodels.Path(pixFixture+".gguf"), testmodels.Path("mmproj-"+pixFixture+".gguf")
	if !vlmAvailable(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (set JITLLM_MODELS) -- run scripts/pixtralgold.py (RULE 11)", text)
	}
	m, err := Open(jlmOfPair(t, text, proj), append([]Option{noTune, WithKVF16(false)}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	return m, g
}

// pixEncode preprocesses and encodes the golden's picture on a fresh state.
func pixEncode(t *testing.T, tw *Tower, g *pixGolden, mut func(*State)) []float32 {
	t.Helper()
	ts := tw.testState()
	defer ts.Close()
	px, err := ts.PreprocessImage(goldenImage(g.Image.W, g.Image.H, g.Image.RGB))
	if err != nil {
		t.Fatal(err)
	}
	if mut != nil {
		mut(ts)
	}
	out, err := ts.Encode(px)
	if err != nil {
		t.Fatal(err)
	}
	return append([]float32(nil), out...)
}

// pixSplit is the head's rows split into the merged grid's (the rows
// transformers' projector emits) and the break rows between them, after
// checking there are as many of each as the processor writes.
func pixSplit(t *testing.T, g *pixGolden, rows []float32, d int) (grid, breaks []float32) {
	t.Helper()
	mh, mw := g.Image.Grid[1], g.Image.Grid[2]
	if want := mh*mw + mh - 1; len(rows) != want*d {
		t.Fatalf("the head emitted %d rows, the processor lays out %d (%dx%d and %d breaks)",
			len(rows)/d, want, mh, mw, mh-1)
	}
	o := 0
	for y := 0; y < mh; y++ {
		grid = append(grid, rows[o*d:(o+mw)*d]...)
		o += mw
		if y+1 < mh {
			breaks = append(breaks, rows[o*d:(o+1)*d]...)
			o++
		}
	}
	return grid, breaks
}

// TestPixtralPreprocessMatchesTheProcessor holds Preprocess to transformers'
// PixtralImageProcessorPil on a picture it must resize: the longest side
// brought under the edge and each side rounded up to a merge unit, Pillow's
// BICUBIC, then rescaled and normalised.
func TestPixtralPreprocessMatchesTheProcessor(t *testing.T) {
	m, g := openPixtral(t)
	defer m.Close()
	tw := m.Tower()
	p := g.Prep
	if tw.Cfg.ImageSz != p.Longest {
		t.Fatalf("the tower's longest edge is %d, the processor's %d", tw.Cfg.ImageSz, p.Longest)
	}
	ts := tw.testState()
	defer ts.Close()
	got, err := ts.PreprocessImage(goldenImage(p.W, p.H, p.RGB))
	if err != nil {
		t.Fatal(err)
	}
	gh, gw := ts.patchGrid()
	if gh*tw.Cfg.PatchSz != p.OutH || gw*tw.Cfg.PatchSz != p.OutW {
		t.Fatalf("a %dx%d picture resized to %dx%d, the processor says %dx%d",
			p.W, p.H, gw*tw.Cfg.PatchSz, gh*tw.Cfg.PatchSz, p.OutW, p.OutH)
	}
	nm, worst := nmse32(p.PX, got)
	t.Logf("%dx%d -> %dx%d: NMSE %.3e, max|d| %.4f against the processor", p.W, p.H, p.OutW, p.OutH, nm, worst)
	if math.IsNaN(nm) || nm > 1e-10 || worst > 1e-5 {
		t.Fatalf("the preprocessed picture is not the processor's: NMSE %.3e, max|d| %.4f", nm, worst)
	}
	// The violation: Qwen-VL's smart_resize, which rounds to nearest where
	// Pixtral rounds up.
	if w, h, err := tw.Cfg.qwenResize(p.W, p.H); err == nil && w == p.OutW && h == p.OutH {
		t.Fatalf("smart_resize lands on the processor's %dx%d too: the gate cannot see the rule", w, h)
	}
	bl, err := ts.preprocessWith(goldenImage(p.W, p.H, p.RGB), bilinearTaps)
	if err != nil {
		t.Fatal(err)
	}
	bad, _ := nmse32(p.PX, bl)
	if !(bad > 10*nm) {
		t.Fatalf("a bilinear resize reads %.3e against bicubic's %.3e: the gate cannot see the filter", bad, nm)
	}
	t.Logf("bilinear (the violation): NMSE %.3e", bad)
}

// TestPixtralTowerMatchesTransformers holds the tower and head's rows to
// transformers' get_image_features on the same picture -- the merged grid's
// rows in the order the language model reads them, a break row after each
// grid row but the last -- and each of the family's own pieces taken out must
// move it far past the bound: the 2-D rotary, its interleaved pairing, the
// odd-frequency half, the merger's column order, the break rows.
func TestPixtralTowerMatchesTransformers(t *testing.T) {
	m, g := openPixtral(t)
	defer m.Close()
	tw := m.Tower()
	d := tw.Cfg.ProjDim
	rows := pixEncode(t, tw, g, nil)
	grid, breaks := pixSplit(t, g, rows, d)
	clean, worst := nmse32(g.Image.EmbdA8, grid)
	f32, _ := nmse32(g.Image.Embd, grid)
	q8, _ := nmse32(g.Image.Embd, g.Image.EmbdA8)
	t.Logf("%d grid rows of %d: NMSE %.3e, max|d| %.4f against transformers at the engine's int8 "+
		"(%.3e against its f32 tower, where the int8 alone moves transformers by %.3e)",
		len(grid)/d, d, clean, worst, f32, q8)
	if math.IsNaN(clean) || clean > towerBound {
		t.Fatalf("the head's rows are not transformers': NMSE %.3e (bound %.0e)", clean, towerBound)
	}
	for i := 0; i < len(breaks); i += d {
		for j := 0; j < d; j++ {
			// The golden is written to eight decimals.
			if math.Abs(float64(breaks[i+j]-g.Image.Break[j])) > 1e-7 {
				t.Fatalf("break row %d element %d is %v, [IMG_BREAK]'s embedding is %v", i/d, j, breaks[i+j], g.Image.Break[j])
			}
		}
	}
	t.Logf("%d break rows, each [IMG_BREAK]'s embedding exactly", len(breaks)/d)
	for _, v := range []struct {
		name string
		mut  func(*State)
		set  func() func()
	}{
		{"no 2-D rotary", func(s *State) {
			gh, gw := s.patchGrid()
			s.vis.ropeSC = make([]float32, gh*gw*tw.Cfg.HeadDim)
			for i := 0; i < len(s.vis.ropeSC); i += 2 {
				s.vis.ropeSC[i] = 1
			}
			s.vis.ropeAt = [2]int{gh, gw}
		}, nil},
		{"NEOX pairs where the GGUF's are interleaved", nil, func() func() {
			tw.cfg.RopeNeox = true
			return func() { tw.cfg.RopeNeox = false }
		}},
		{"Qwen-VL's frequencies (each half from the first)", nil, func() func() {
			tw.Cfg.fault = faultPixtralFreq
			return func() { tw.Cfg.fault = towerFaultNone }
		}},
		{"the merger read channel-major", nil, func() func() {
			tw.Cfg.fault = faultPixtralMerge
			return func() { tw.Cfg.fault = towerFaultNone }
		}},
	} {
		undo := func() {}
		if v.set != nil {
			undo = v.set()
		}
		bad := pixEncode(t, tw, g, v.mut)
		undo()
		bg, _ := pixSplit(t, g, bad, d)
		nb, _ := nmse32(g.Image.EmbdA8, bg)
		t.Logf("violation %q: NMSE %.3e", v.name, nb)
		if !(nb > 10*towerBound) {
			t.Errorf("%s reads %.3e against a clean %.3e: the gate cannot see it", v.name, nb, clean)
		}
	}
}

// pixWorst is the worst logit NMSE of rows against the golden's and the
// argmax flips.
func pixWorst(want []pixLogits, rows [][]float32) (float64, int) {
	worst, flips := 0.0, 0
	for i, lg := range rows {
		nmse := llama4Cmp(want[i].Head, lg)
		if math.IsNaN(nmse) {
			return math.Inf(1), len(rows)
		}
		worst = math.Max(worst, nmse)
		if Greedy(lg) != want[i].Argmax {
			flips++
		}
	}
	return worst, flips
}

// pixRun prefills the golden's prompt with img in the picture's place and
// decodes its continuation, returning the logits the golden records.
func pixRun(t *testing.T, m *Model, g *pixGolden, img Span, rows int) [][]float32 {
	t.Helper()
	st := m.NewState(len(g.Pre) + rows + len(g.Post) + len(g.Cont) + 1)
	defer st.Close()
	lg, err := st.PrefillMixed(Span{Tokens: g.Pre}, img, Span{Tokens: g.Post})
	if err != nil {
		t.Fatal(err)
	}
	out := [][]float32{append([]float32(nil), lg...)}
	for _, id := range g.Cont[:len(g.Cont)-1] {
		if lg, err = st.Forward(id); err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	return out
}

// TestPixtralMatchesTransformersEndToEnd is the whole model from the pixels:
// the picture a span of the prompt, encoded by the prefill on the State's own
// vision segment, against Mistral3ForConditionalGeneration's logits with its
// tower at the engine's arithmetic (the toy text model amplifies the int8's
// 6e-3 on the rows to 0.2 of the logits, so the f32 reference is reported
// and the arithmetic-matched one is the bound). Given transformers' own rows
// the text model reads 5e-12.
func TestPixtralMatchesTransformersEndToEnd(t *testing.T) {
	m, g := openPixtral(t)
	defer m.Close()
	st := m.NewState(8)
	pic, err := st.Picture(goldenImage(g.Image.W, g.Image.H, g.Image.RGB))
	st.Close()
	if err != nil {
		t.Fatal(err)
	}
	mh, mw := g.Image.Grid[1], g.Image.Grid[2]
	if pic.Rows != mh*mw+mh-1 {
		t.Fatalf("the picture takes %d positions, the processor's layout %d", pic.Rows, mh*mw+mh-1)
	}
	got := pixRun(t, m, g, Span{Picture: pic}, pic.Rows)
	w, flips := pixWorst(g.LogitsA8, got)
	wf, ff := pixWorst(g.Logits, got)
	t.Logf("the engine's tower and text model from the pixels: worst logit NMSE %.3e, %d argmax flips "+
		"(%.3e and %d against the f32 tower)", w, flips, wf, ff)
	// A flip is held against the reference's own control: transformers with
	// the tower's rows at the engine's int8 arithmetic and at float32 pick
	// different tokens at rows 3 and 5, so the int8 rounding alone decides
	// there, and a host whose int8 rounds otherwise (the M4: 1 flip against
	// the int8 arm, 1 against the f32) lands on either. A row where the two
	// arms agree must be matched; the engine's token must be one of the two
	// everywhere.
	outside := 0
	for i, lg := range got {
		if a := Greedy(lg); a != g.LogitsA8[i].Argmax && a != g.Logits[i].Argmax {
			outside++
		}
	}
	if math.IsNaN(w) || w > 2e-2 || outside > 0 {
		t.Fatalf("end to end: worst NMSE %.3e, %d argmax flips outside both of transformers' arms", w, outside)
	}
	// The violation: the break rows left out, so every grid row after the
	// first sits one position early.
	m.Tower().Cfg.fault = faultPixtralNoBreaks
	vs := m.NewState(8)
	bad, err := vs.Picture(goldenImage(g.Image.W, g.Image.H, g.Image.RGB))
	vs.Close()
	if err != nil {
		m.Tower().Cfg.fault = towerFaultNone
		t.Fatal(err)
	}
	m.imgCache.clearForTest()
	bad.Rows = mh * mw
	nb, _ := pixWorst(g.LogitsA8, pixRun(t, m, g, Span{Picture: bad}, bad.Rows))
	m.Tower().Cfg.fault = towerFaultNone
	m.imgCache.clearForTest()
	t.Logf("no break rows (the violation): %.3e", nb)
	if !(nb > 20*w) {
		t.Fatalf("leaving the breaks out reads %.3e against %.3e: the gate cannot see them", nb, w)
	}
}

// TestPixtralOnDeviceMatchesHost places every tower block on each device
// backend present and holds the head's rows to the host's, then brings the
// blocks home and demands the host's own answer again (relocation).
func TestPixtralOnDeviceMatchesHost(t *testing.T) {
	m, g := openPixtral(t)
	defer m.Close()
	tw := m.Tower()
	img := goldenImage(g.Image.W, g.Image.H, g.Image.RGB)
	encode := func(s *State) []float32 {
		px, err := s.PreprocessImage(img)
		if err != nil {
			t.Fatal(err)
		}
		out, err := s.Encode(px)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), out...)
	}
	host := tw.testState()
	want := encode(host)
	host.Close()
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
			got := encode(s)
			nm, worst := nmseOf(got, want)
			t.Logf("%s, all %d blocks: NMSE %.3e, max|d| %.4f against the host", spec, tw.Cfg.NLayer, nm, worst)
			if math.IsNaN(nm) || math.IsInf(nm, 0) || nm > 1e-3 {
				t.Fatalf("%s: the tower on the device reads NMSE %.3e against the host", spec, nm)
			}
			if err := s.SetDevice(nil); err != nil {
				t.Fatal(err)
			}
			if nb, _ := nmseOf(encode(s), want); nb != 0 {
				t.Fatalf("%s: home again, the host reads NMSE %.3e against itself", spec, nb)
			}
		})
	}
	if ran == 0 {
		t.Skip("no GPU backend on this host")
	}
}
