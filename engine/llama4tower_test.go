package engine

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// llama4Golden is scripts/llama4visiongold.py's reference for synth-llama4v:
// a picture on its canvas, the processor's pixels per tile ([tile][y][x]
// [channel], the whole picture's tile last), the tower's rows per tile at
// the engine's arithmetic (embd_a8), and the picture's run of the prompt as
// the processor lays it out.
type llama4Golden struct {
	Rows   []int32 `json:"rows"`
	Aspect [2]int  `json:"aspect"`
	Image  struct {
		H, W   int
		RGB    []byte    `json:"rgb"`
		Tiles  int       `json:"tiles"`
		PX     []float32 `json:"px"`
		EmbdA8 []float32 `json:"embd_a8"`
	} `json:"image"`
}

func openLlama4V(t *testing.T) (*model.Model, *llama4Golden) {
	t.Helper()
	const name = "synth-llama4v"
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "llama4v", name+".json"))
	if err != nil {
		t.Fatalf("GOLDEN MISSING: %v -- run scripts/llama4visiongold.py", err)
	}
	g := &llama4Golden{}
	if err := json.Unmarshal(raw, g); err != nil {
		t.Fatal(err)
	}
	text, proj := testmodels.Path(name+".gguf"), testmodels.Path("mmproj-"+name+".gguf")
	if !sourcePresent(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (set JITLLM_MODELS) -- run scripts/llama4visiongold.py (RULE 11)", text)
	}
	m, err := model.Open(jlmOfPair(t, text, proj))
	if err != nil {
		t.Fatal(err)
	}
	return m, g
}

// markerTable is the container's marker rows (jlm.RoleVTileMarks), D wide.
func markerTable(t *testing.T, m *model.Model, D int) []float32 {
	t.Helper()
	for i := range m.Container.Entries() {
		e := &m.Container.Entries()[i]
		if e.Role != jlm.RoleVTileMarks {
			continue
		}
		typ, ok := jlm.SourceType(e.Type)
		qs, err := m.Container.ReadPayload(e)
		if !ok || err != nil {
			t.Fatalf("the marker table of %v: %v", e.Type, err)
		}
		if int(e.Dims[0]) != D {
			t.Fatalf("the marker table's rows are %d wide, not %d", e.Dims[0], D)
		}
		out := make([]float32, int(e.Dims[0])*int(e.Dims[1]))
		if err := quant.Dequant32(typ, qs, out); err != nil {
			t.Fatal(err)
		}
		return out
	}
	t.Fatal("the container has no marker table")
	return nil
}

// TestLlama4TowerMatchesTransformers holds synth-llama4v's stated tower to
// transformers' Llama4ImageProcessor, Llama4VisionModel and its projector: a
// 224x168 picture on the processor's 2x2 canvas, not resized and padded
// black at the bottom, then squashed to the whole picture's tile -- every
// sample of every tile; each tile's rows at the engine's arithmetic within
// the tower bound, each marker row its table's, in the order the processor
// lays the separators and <|image|> out. The rotary's positions counted
// from zero is the violation.
func TestLlama4TowerMatchesTransformers(t *testing.T) {
	m, g := openLlama4V(t)
	defer m.Close()
	tg := m.Tower
	s := testInstance(t, m, 64, WithKVF16(false))
	defer s.Close()
	img := rgbPicture(g.Image.RGB, g.Image.W, g.Image.H)
	W, H := g.Image.W, g.Image.H
	_, n := tg.Square(W, H, 0)
	if n != g.Image.Tiles || len(g.Image.PX)%n != 0 {
		t.Fatalf("%d squares, the processor's %d tiles (%d samples)", n, g.Image.Tiles, len(g.Image.PX))
	}
	one, differ, worst := len(g.Image.PX)/n, 0, 0.0
	for i := range n {
		px, _ := towerPixelsAt(t, s, img, i)
		if len(px) != one {
			t.Fatalf("tile %d is %d samples, the processor's %d", i, len(px), one)
		}
		for j, v := range px {
			d := math.Abs(float64(v - g.Image.PX[i*one+j]))
			if !(d < 1e-6) {
				differ++
			}
			worst = math.Max(worst, d)
		}
	}
	if differ > 0 {
		t.Fatalf("%d of %d samples differ from the processor's, worst by %.5f", differ, len(g.Image.PX), worst)
	}
	t.Logf("a %dx%d picture on a %dx%d canvas: every one of %d samples of %d tiles the processor's", W, H,
		g.Aspect[0], g.Aspect[1], len(g.Image.PX), n)

	rows, err := s.EncodePicture(img)
	if err != nil {
		t.Fatal(err)
	}
	D := int(m.Container.Vision().ProjDim)
	marks := markerTable(t, m, D)
	per := len(g.Image.EmbdA8) / D / n
	// The processor's run between <|image_start|> and <|image_end|>: each
	// marker as its row of the table plus one, a tile's rows as 0.
	span := m.Container.Vision().Tower.Span
	if len(g.Rows) < 2 || !slices.Equal(span.Before, g.Rows[:1]) || !slices.Equal(span.After, g.Rows[len(g.Rows)-1:]) {
		t.Fatalf("the span is %v ... %v, the processor's %v ... %v", span.Before, span.After, g.Rows[:1], g.Rows[len(g.Rows)-1:])
	}
	var want []int32
	var got, wantRows []float32
	pos, worstMark := 0, 0.0
	mark := func(k int) {
		for j, v := range rows[pos*D : (pos+1)*D] {
			worstMark = math.Max(worstMark, math.Abs(float64(v-marks[(k-1)*D+j])))
		}
		pos++
	}
	for i := range n {
		sq, _ := tg.Square(W, H, i)
		if sq.Mark > 0 {
			want = append(want, int32(sq.Mark))
			mark(sq.Mark)
		}
		want = append(want, 0)
		got = append(got, rows[pos*D:(pos+per)*D]...)
		wantRows = append(wantRows, g.Image.EmbdA8[i*per*D:(i+1)*per*D]...)
		pos += per
		if sq.Trail > 0 {
			want = append(want, int32(sq.Trail))
			mark(sq.Trail)
		}
	}
	if pos*D != len(rows) {
		t.Fatalf("%d rows read of %d", pos, len(rows)/D)
	}
	if worstMark != 0 {
		t.Fatalf("a marker row is off its table row by %.3g", worstMark)
	}
	// The processor's layout, each marker id as its table row plus one and
	// a tile's run of rows as 0.
	var ref []int32
	kinds := map[int32]int32{}
	for _, id := range g.Rows[1 : len(g.Rows)-1] {
		if len(ref) > 0 && ref[len(ref)-1] == 0 && isPatchRow(g, id) {
			continue
		}
		if isPatchRow(g, id) {
			ref = append(ref, 0)
			continue
		}
		if _, ok := kinds[id]; !ok {
			kinds[id] = int32(len(kinds) + 1)
		}
		ref = append(ref, kinds[id])
	}
	if !slices.Equal(ref, want) {
		t.Fatalf("the picture's run is %v, the processor's %v", want, ref)
	}
	clean, worstRow := nmseOf(got, wantRows)
	t.Logf("%d tiles' %d rows: NMSE %.3e, max|d| %.4f at the engine's arithmetic; the run %v, every marker row its table's",
		n, n*per, clean, worstRow, want)
	if math.IsNaN(clean) || clean > 2e-3 {
		t.Fatalf("the rows are NMSE %.3e from transformers' (bound 2e-3)", clean)
	}

	// The violation: a cell's positions counted from zero.
	rope := &m.Container.Vision().Tower.Block.Rope
	rope.Origin = 0
	bad, err := m.BuildTowerGraphs()
	rope.Origin = 1
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
	var bgot []float32
	pos = 0
	for i := range n {
		sq, _ := tg.Square(W, H, i)
		if sq.Mark > 0 {
			pos++
		}
		bgot = append(bgot, brows[pos*D:(pos+per)*D]...)
		pos += per
		if sq.Trail > 0 {
			pos++
		}
	}
	nb, _ := nmseOf(bgot, wantRows)
	t.Logf("violation (positions from zero): NMSE %.3e", nb)
	if !(nb > 2e-3) {
		t.Fatalf("VIOLATION: positions counted from zero still read NMSE %.3e", nb)
	}
}

// isPatchRow says id is the processor's placeholder for a tile's rows: the
// id the run repeats.
func isPatchRow(g *llama4Golden, id int32) bool {
	n := 0
	for _, x := range g.Rows {
		if x == id {
			n++
		}
	}
	return n > len(g.Rows)/4
}

// TestLlama4TowerRunsOnTheCard places synth-llama4v's tower on each device,
// every block and half of them, within the CPU device's band.
func TestLlama4TowerRunsOnTheCard(t *testing.T) {
	m, g := openLlama4V(t)
	defer m.Close()
	towerRunsOnTheCard(t, m, rgbPicture(g.Image.RGB, g.Image.W, g.Image.H), nil, 0)
}
