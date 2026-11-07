package engine

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestPhi4TowerMatchesTransformers holds synth-phi4v's stated tower to its
// own processor and Siglip2VisionModel at hidden_states[-2] with the
// mlp2x_gelu projector (scripts/phi4visiongold.py): the picture resized up
// to the least patch count by Pillow's BILINEAR filter, to every sample and
// its grid; and the rows, the position table resized to the grid by
// torch's antialiased bilinear, at the engine's arithmetic (embd_a8) within
// the tower bound. Each part has its violation: Pillow's BICUBIC filter,
// and the table's first rows read as a fixed grid's, not resized.
func TestPhi4TowerMatchesTransformers(t *testing.T) {
	const name = "synth-phi4v"
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "phi4v", name+".json"))
	if err != nil {
		t.Fatalf("GOLDEN MISSING: %v -- run scripts/phi4visiongold.py", err)
	}
	var g struct {
		Image struct {
			H, W   int
			RGB    []byte    `json:"rgb"`
			Grid   [2]int    `json:"grid"`
			PX     []float32 `json:"px"`
			EmbdA8 []float32 `json:"embd_a8"`
		} `json:"image"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	text, proj := testmodels.Path(name+".gguf"), testmodels.Path("mmproj-"+name+".gguf")
	if !sourcePresent(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (set JITLLM_MODELS) -- run scripts/phi4visiongold.py (RULE 11)", text)
	}
	m, err := model.Open(jlmOfPair(t, text, proj))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	tg := m.Tower
	img := rgbPicture(g.Image.RGB, g.Image.W, g.Image.H)
	s := testInstance(t, m, 64, WithKVF16(false))
	defer s.Close()
	px, sq := towerPixels(t, s, img)
	if [2]int{sq.Gh, sq.Gw} != g.Image.Grid {
		t.Fatalf("a %dx%d picture became %dx%d patches, the processor's %v", g.Image.W, g.Image.H, sq.Gh, sq.Gw, g.Image.Grid)
	}
	if len(px) != len(g.Image.PX) {
		t.Fatalf("%d samples, the processor's %d", len(px), len(g.Image.PX))
	}
	pn, pw := nmseOf(px, g.Image.PX)
	t.Logf("%dx%d -> %v patches: NMSE %.3e, max|d| %.2e against the processor", g.Image.W, g.Image.H, g.Image.Grid, pn, pw)
	if math.IsNaN(pn) || pn > 1e-10 || pw > 1e-5 {
		t.Fatalf("the picture is not the processor's: NMSE %.3e, max|d| %.2e", pn, pw)
	}
	rows, err := s.EncodePicture(img)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != len(g.Image.EmbdA8) {
		t.Fatalf("%d values, the reference's %d", len(rows), len(g.Image.EmbdA8))
	}
	clean, worst := nmseOf(rows, g.Image.EmbdA8)
	t.Logf("the tower's %d rows: NMSE %.3e, max|d| %.4f at the engine's arithmetic", sq.Gh*sq.Gw, clean, worst)
	if math.IsNaN(clean) || clean > 2e-3 {
		t.Fatalf("the rows are NMSE %.3e from transformers' (bound 2e-3)", clean)
	}

	rec := m.Container.Vision().Tower
	rebuilt := func(edit func()) *Instance {
		t.Helper()
		edit()
		bad, err := m.BuildTowerGraphs()
		edit()
		if err != nil {
			t.Fatal(err)
		}
		m.Tower = bad
		return testInstance(t, m, 64, WithKVF16(false))
	}
	// The violations: BICUBIC in place of BILINEAR, and the table not
	// resized.
	filter := &rec.Pre[0].Ints[3]
	was := *filter
	bs := rebuilt(func() {
		if *filter == was {
			*filter = int32(jlm.FilterPILBicubic)
		} else {
			*filter = was
		}
	})
	bad, _ := towerPixels(t, bs, img)
	bs.Close()
	m.Tower = tg
	nb, _ := nmseOf(bad, g.Image.PX)
	t.Logf("violation (BICUBIC): NMSE %.3e", nb)
	if !(nb > 1e-6) {
		t.Fatalf("VIOLATION: BICUBIC reads NMSE %.3e: the gate cannot see the filter", nb)
	}
	var res *jlm.TowerOp
	for i := range rec.Entry.Ops {
		if rec.Entry.Ops[i].Code == jlm.TOpPosResample {
			res = &rec.Entry.Ops[i]
		}
	}
	stated := *res
	ts := rebuilt(func() {
		if res.Code == jlm.TOpPosResample {
			*res = jlm.TowerOp{Code: jlm.TOpPosTable, Bind: stated.Bind}
		} else {
			*res = stated
		}
	})
	brows, err := ts.EncodePicture(img)
	if err != nil {
		t.Fatal(err)
	}
	nt, _ := nmseOf(brows, g.Image.EmbdA8)
	ts.Close()
	m.Tower = tg
	t.Logf("violation (the table not resized): NMSE %.3e", nt)
	if !(nt > 10*clean && nt > 2e-3) {
		t.Fatalf("VIOLATION: the table not resized reads NMSE %.3e against a clean %.3e", nt, clean)
	}
}

// TestPhi4TowerRunsOnTheCard places synth-phi4v's tower on each device,
// every block and then half of them, and holds the picture's rows to the
// CPU device's.
func TestPhi4TowerRunsOnTheCard(t *testing.T) {
	const name = "synth-phi4v"
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "phi4v", name+".json"))
	if err != nil {
		t.Fatalf("GOLDEN MISSING: %v -- run scripts/phi4visiongold.py", err)
	}
	var g struct {
		Image struct {
			H, W int
			RGB  []byte `json:"rgb"`
		} `json:"image"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	text, proj := testmodels.Path(name+".gguf"), testmodels.Path("mmproj-"+name+".gguf")
	if !sourcePresent(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (set JITLLM_MODELS) -- run scripts/phi4visiongold.py (RULE 11)", text)
	}
	m, err := model.Open(jlmOfPair(t, text, proj))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	towerRunsOnTheCard(t, m, rgbPicture(g.Image.RGB, g.Image.W, g.Image.H), nil, 0, WithKVF16(false))
}
