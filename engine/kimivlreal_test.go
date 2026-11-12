package engine

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestKimiVLRealMatchesMoonshot holds the real Kimi-VL-A3B-Instruct's
// MoonViT and projector -- its published tensors, converted by llama.cpp's
// converter -- to Moonshot's own modeling_kimi_vl.py on the picture the
// golden carries (scripts/kimivlreal.py): the rows within 5e-3 of Moonshot's
// at the engine's arithmetic (Q8_0 matrices, int8 activations), and on each
// card, every block and then half of them, within twice the CPU device's
// distance from Moonshot's float32 (towerRunsOnTheCard, whose violation is a
// grey picture). The text model is Kimi-VL-A3B-Thinking-2506's, whose
// vocabulary spells the tower's markers; nothing past the tower is read. The
// violation: the rotary's column and row pairs swapped.
func TestKimiVLRealMatchesMoonshot(t *testing.T) {
	g := qvlGoldenOf(t, "real-Kimi-VL-A3B-Instruct-tower")
	text, proj := testmodels.Path("Kimi-VL-A3B-Thinking-2506-Q4_K_M.gguf"), testmodels.Path("mmproj-Kimi-VL-A3B-Instruct-f16.gguf")
	if !sourcePresent(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and %s -- scripts/kimivlreal.py writes the mmproj (RULE 11)", text, proj)
	}
	m, err := model.Open(jlmOfPair(t, text, proj))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	opts := []Option{WithKVF16(false)}
	img := rgbPicture(g.Image.RGB, g.Image.W, g.Image.H)
	encode := func() []float32 {
		t.Helper()
		s := testInstance(t, m, 64, opts...)
		defer s.Close()
		rows, err := s.EncodePicture(img)
		if err != nil {
			t.Fatal(err)
		}
		return slices.Clone(rows)
	}
	rows := encode()
	if len(rows) != len(g.Image.EmbdA8) {
		t.Fatalf("%d values, Moonshot's tower %d", len(rows), len(g.Image.EmbdA8))
	}
	a8, worst := nmseOf(rows, g.Image.EmbdA8)
	f32, _ := nmseOf(rows, g.Image.Embd)
	sim, _ := nmseOf(g.Image.EmbdA8, g.Image.Embd)
	t.Logf("%d rows: NMSE %.3e, max|d| %.3g against Moonshot's tower at the engine's arithmetic, %.3e against its float32 "+
		"(where that arithmetic alone moves it %.3e)", len(rows)/int(m.Container.Vision().ProjDim), a8, worst, f32, sim)
	if math.IsNaN(a8) || a8 > 5e-3 {
		t.Fatalf("the real tower reads %.3e against Moonshot's (bound 5e-3)", a8)
	}
	tg := m.Tower
	rec := m.Container.Vision().Tower
	runs := slices.Clone(rec.Block.Rope.Runs)
	for i := range rec.Block.Rope.Runs {
		rec.Block.Rope.Runs[i].Axis = 1 - rec.Block.Rope.Runs[i].Axis
	}
	swapped, err := m.BuildTowerGraphs()
	copy(rec.Block.Rope.Runs, runs)
	if err != nil {
		t.Fatal(err)
	}
	m.Tower = swapped
	bad := encode()
	m.Tower = tg
	nb, _ := nmseOf(bad, g.Image.EmbdA8)
	if !(nb > 10*5e-3) {
		t.Fatalf("VIOLATION: the rotary's axes swapped read NMSE %.3e against a clean %.3e", nb, a8)
	}
	t.Logf("violation (axes swapped): NMSE %.3e", nb)
	towerRunsOnTheCard(t, m, img, g.Image.Embd, 0, opts...)
}
