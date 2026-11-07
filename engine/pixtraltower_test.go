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

// pixtralGolden is scripts/pixtralgold.py's reference for synth-pixtral:
// a picture the processor resizes, and a picture's projector rows at the
// engine's arithmetic (embd_a8) with the [IMG_BREAK] row apart.
type pixtralGolden struct {
	Image struct {
		H, W   int
		RGB    []byte    `json:"rgb"`
		EmbdA8 []float32 `json:"embd_a8"`
		Break  []float32 `json:"break"`
	} `json:"image"`
	Prep struct {
		H, W, Longest int
		RGB           []byte    `json:"rgb"`
		OutH          int       `json:"out_h"`
		OutW          int       `json:"out_w"`
		PX            []float32 `json:"px"`
	} `json:"prep"`
}

func openPixtral(t *testing.T) (*model.Model, *pixtralGolden) {
	t.Helper()
	const name = "synth-pixtral"
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "pixtral", name+".json"))
	if err != nil {
		t.Fatalf("GOLDEN MISSING: %v -- run scripts/pixtralgold.py", err)
	}
	g := &pixtralGolden{}
	if err := json.Unmarshal(raw, g); err != nil {
		t.Fatal(err)
	}
	text, proj := testmodels.Path(name+".gguf"), testmodels.Path("mmproj-"+name+".gguf")
	if !sourcePresent(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (set JITLLM_MODELS) -- run scripts/pixtralgold.py (RULE 11)", text)
	}
	m, err := model.Open(jlmOfPair(t, text, proj))
	if err != nil {
		t.Fatal(err)
	}
	return m, g
}

// TestPixtralTowerMatchesTransformers holds synth-pixtral's stated tower to
// transformers' PixtralImageProcessor, PixtralVisionModel and
// Mistral3MultiModalProjector: a picture past the longest side resized to
// the processor's size, every sample; and a picture's rows -- the merged
// grid's, the [IMG_BREAK] row after each merged row but the last -- at the
// engine's arithmetic within the tower bound. The 2-D rotary's axes swapped
// is the violation.
func TestPixtralTowerMatchesTransformers(t *testing.T) {
	m, g := openPixtral(t)
	defer m.Close()
	tg := m.Tower
	s := testInstance(t, m, 64, WithKVF16(false))
	defer s.Close()
	was := tg.Fit.Cap
	tg.Fit.Cap = g.Prep.Longest
	px, sq := towerPixels(t, s, rgbPicture(g.Prep.RGB, g.Prep.W, g.Prep.H))
	tg.Fit.Cap = was
	if sq.Gw*tg.Patch != g.Prep.OutW || sq.Gh*tg.Patch != g.Prep.OutH {
		t.Fatalf("a %dx%d picture resized to %dx%d, the processor's %dx%d", g.Prep.W, g.Prep.H,
			sq.Gw*tg.Patch, sq.Gh*tg.Patch, g.Prep.OutW, g.Prep.OutH)
	}
	if len(px) != len(g.Prep.PX) {
		t.Fatalf("%d samples, the processor's %d", len(px), len(g.Prep.PX))
	}
	differ := 0
	for i, v := range px {
		if !(math.Abs(float64(v-g.Prep.PX[i])) < 2e-4) {
			differ++
		}
	}
	if differ > 0 {
		t.Fatalf("%d of %d samples differ from the processor's", differ, len(px))
	}
	t.Logf("a %dx%d picture resized to %dx%d: every one of %d samples the processor's", g.Prep.W, g.Prep.H,
		g.Prep.OutW, g.Prep.OutH, len(px))

	img := rgbPicture(g.Image.RGB, g.Image.W, g.Image.H)
	_, isq := towerPixels(t, s, img)
	D := len(g.Image.Break)
	W, H := isq.Gw/tg.Frame, isq.Gh/tg.Frame
	if D == 0 || W*H*D != len(g.Image.EmbdA8) {
		t.Fatalf("a %dx%d merged grid of %d, the reference's %d values", W, H, D, len(g.Image.EmbdA8))
	}
	var want []float32
	for y := range H {
		want = append(want, g.Image.EmbdA8[y*W*D:(y+1)*W*D]...)
		if y+1 < H {
			want = append(want, g.Image.Break...)
		}
	}
	rows, err := s.EncodePicture(img)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != len(want) {
		t.Fatalf("%d values, the reference's %d with its breaks", len(rows), len(want))
	}
	clean, worst := nmseOf(rows, want)
	t.Logf("the tower's %d rows (%dx%d merged, %d breaks): NMSE %.3e, max|d| %.4f at the engine's arithmetic",
		len(rows)/D, H, W, H-1, clean, worst)
	if math.IsNaN(clean) || clean > 2e-3 {
		t.Fatalf("the rows are NMSE %.3e from transformers' (bound 2e-3)", clean)
	}

	// The violation: the rotary's axes swapped.
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
	nb, _ := nmseOf(brows, want)
	t.Logf("violation (the rotary's axes swapped): NMSE %.3e", nb)
	if !(nb > 10*clean && nb > 2e-3) {
		t.Fatalf("VIOLATION: the rotary's axes swapped read NMSE %.3e against a clean %.3e", nb, clean)
	}
}

// TestPixtralTowerRunsOnTheCard places synth-pixtral's tower on each device,
// every block and then half of them, and holds the picture's rows, breaks
// and all, to the CPU device's.
func TestPixtralTowerRunsOnTheCard(t *testing.T) {
	m, g := openPixtral(t)
	defer m.Close()
	towerRunsOnTheCard(t, m, rgbPicture(g.Image.RGB, g.Image.W, g.Image.H), nil, 0, WithKVF16(false))
}
