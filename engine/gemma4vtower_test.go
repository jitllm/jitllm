package engine

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// gemma4vGolden is scripts/gemma4visiongold.py's reference for
// synth-gemma4v: a picture resized up to the soft-token budget, its patch
// grid, the processor's pixels -- its patch rows in raster order, each
// (ky, kx, channel), in [0, 1], which the reference's patch embedder takes
// to 2x - 1 -- and the head's rows at the engine's arithmetic (embd_a8).
type gemma4vGolden struct {
	Image struct {
		H, W   int
		RGB    []byte    `json:"rgb"`
		Grid   [2]int    `json:"grid"`
		Soft   int       `json:"soft"`
		PX     []float32 `json:"px"`
		Embd   []float32 `json:"embd"`
		EmbdQ8 []float32 `json:"embd_q8"`
		EmbdA8 []float32 `json:"embd_a8"`
	} `json:"image"`
}

func openGemma4V(t *testing.T) (*model.Model, *gemma4vGolden) {
	t.Helper()
	const name = "synth-gemma4v"
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "gemma4v", name+".json"))
	if err != nil {
		t.Fatalf("GOLDEN MISSING: %v -- run scripts/gemma4visiongold.py", err)
	}
	g := &gemma4vGolden{}
	if err := json.Unmarshal(raw, g); err != nil {
		t.Fatal(err)
	}
	text, proj := testmodels.Path(name+".gguf"), testmodels.Path("mmproj-"+name+".gguf")
	if !sourcePresent(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (set JITLLM_MODELS) -- run scripts/gemma4visiongold.py (RULE 11)", text)
	}
	m, err := model.Open(jlmOfPair(t, text, proj))
	if err != nil {
		t.Fatal(err)
	}
	return m, g
}

// TestGemma4VisionTowerMatchesTransformers holds synth-gemma4v's stated
// tower to transformers' Gemma4ImageProcessor and Gemma4VisionModel: the
// picture resized up to the soft-token budget at its aspect, its grid and
// every sample; and the head's rows -- the blocks' clipped linears, q/k/v
// norms and sandwich norms, the 3x3 pool, the standardisation and the
// projection -- at the engine's arithmetic within the tower bound. The 2-D
// rotary turning the row's quarter first is the violation.
func TestGemma4VisionTowerMatchesTransformers(t *testing.T) {
	m, g := openGemma4V(t)
	defer m.Close()
	tg := m.Tower
	s := testInstance(t, m, 64, WithKVF16(false))
	defer s.Close()
	img := rgbPicture(g.Image.RGB, g.Image.W, g.Image.H)
	px, sq := towerPixels(t, s, img)
	if [2]int{sq.Gh, sq.Gw} != g.Image.Grid {
		t.Fatalf("a %dx%d picture became %dx%d patches, the processor's %v", g.Image.W, g.Image.H, sq.Gh, sq.Gw, g.Image.Grid)
	}
	if len(px) != len(g.Image.PX) {
		t.Fatalf("%d samples, the processor's %d", len(px), len(g.Image.PX))
	}
	// The processor's patch rows as the picture op's raster, at 2x - 1.
	P, W := tg.Patch, sq.Gw*tg.Patch
	want := make([]float32, len(px))
	for i, v := range g.Image.PX {
		c, kx, ky, p := i%3, i/3%P, i/(3*P)%P, i/(3*P*P)
		want[((p/sq.Gw*P+ky)*W+p%sq.Gw*P+kx)*3+c] = 2*v - 1
	}
	differ, first, worst := 0, -1, 0.0
	for i, v := range px {
		d := math.Abs(float64(v - want[i]))
		if !(d < 2e-4) {
			if first < 0 {
				first = i
			}
			differ++
		}
		worst = math.Max(worst, d)
	}
	if differ > 0 {
		t.Fatalf("%d of %d samples differ from the processor's, worst by %.4f; the first, sample %d: %v, the processor's %v",
			differ, len(px), worst, first, px[first], want[first])
	}
	t.Logf("a %dx%d picture resized to %v patches: every one of %d samples the processor's", g.Image.W, g.Image.H,
		g.Image.Grid, len(px))
	rows, err := s.EncodePicture(img)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != len(g.Image.EmbdA8) {
		t.Fatalf("%d values, the reference's %d (%d soft tokens)", len(rows), len(g.Image.EmbdA8), g.Image.Soft)
	}
	clean, worst := nmseOf(rows, g.Image.EmbdA8)
	f32, _ := nmseOf(rows, g.Image.Embd)
	q8, _ := nmseOf(rows, g.Image.EmbdQ8)
	a8, _ := nmseOf(g.Image.Embd, g.Image.EmbdA8)
	t.Logf("the head's %d rows: NMSE %.3e, max|d| %.4f at the engine's arithmetic (%.3e from the f32 tower, %.3e "+
		"from its Q8_0 matrices, where the arithmetic alone moves it %.3e)", g.Image.Soft, clean, worst, f32, q8, a8)
	if math.IsNaN(clean) || clean > 2e-3 {
		t.Fatalf("the rows are NMSE %.3e from transformers' (bound 2e-3)", clean)
	}

	// The violation: the row's quarter turned first.
	runs := m.Container.Vision().Tower.Block.Rope.Runs
	swap := func() {
		for i := range runs {
			runs[i].Axis = 1 - runs[i].Axis
		}
	}
	swap()
	bad, err := m.BuildTowerGraphs()
	swap()
	if err != nil {
		t.Fatal(err)
	}
	m.Tower = bad
	bs := testInstance(t, m, 64, WithKVF16(false))
	brows, err := bs.EncodePicture(img)
	bs.Close()
	m.Tower = tg
	if err != nil {
		t.Fatal(err)
	}
	nb, _ := nmseOf(brows, g.Image.EmbdA8)
	t.Logf("violation (the row's quarter first): NMSE %.3e", nb)
	if !(nb > 10*clean && nb > 2e-3) {
		t.Fatalf("VIOLATION: the row's quarter first reads NMSE %.3e against a clean %.3e", nb, clean)
	}
}

// TestGemma4VisionTowerRunsOnTheCard places synth-gemma4v's tower on each
// device, every block and then half of them, and holds the head's rows to
// the CPU device's.
func TestGemma4VisionTowerRunsOnTheCard(t *testing.T) {
	m, g := openGemma4V(t)
	defer m.Close()
	towerRunsOnTheCard(t, m, rgbPicture(g.Image.RGB, g.Image.W, g.Image.H), nil, 0, WithKVF16(false))
}
