package engine

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image/color"
	"image/png"
	"math"
	"os"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// lfm2Golden is scripts/lfm2visiongold.py --fixture's reference for
// synth-lfm2vl: a picture cut into tiles and a thumbnail, each piece's patch
// grid (rows, columns),
// and beside it the pieces' pixels (.px, [y][x][channel] each, in order) and
// the projector's rows at the engine's arithmetic (.emb_a8).
type lfm2Golden struct {
	W, H, Rows, Cols int
	Pieces           [][2]int
}

func openLFM2VL(t *testing.T) (*model.Model, *lfm2Golden, string) {
	t.Helper()
	const name = "synth-lfm2vl"
	text, proj := testmodels.Path(name+".gguf"), testmodels.Path("mmproj-"+name+".gguf")
	if !sourcePresent(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj -- run scripts/lfm2visiongold.py --fixture (RULE 11)", text)
	}
	base := testmodels.Path(name + "-gold/picture")
	raw, err := os.ReadFile(base + ".json")
	if err != nil {
		testmodels.Missing(t, "GOLDEN MISSING: %v -- run scripts/lfm2visiongold.py --fixture (RULE 11)", err)
	}
	g := &lfm2Golden{}
	if err := json.Unmarshal(raw, g); err != nil {
		t.Fatal(err)
	}
	m, err := model.Open(jlmOfPair(t, text, proj))
	if err != nil {
		t.Fatal(err)
	}
	return m, g, base
}

func readF32File(t *testing.T, path string) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		testmodels.Missing(t, "GOLDEN MISSING: %v", err)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return out
}

// TestLFM2VLTowerMatchesTransformers holds synth-lfm2vl's stated tower to
// transformers' Lfm2VlImageProcessor, Siglip2VisionModel and projector: the
// picture cut into its tiles and thumbnail as the processor cuts it, every
// piece's every sample; and the picture's rows -- each piece's marker row,
// the text table's row of its <|img_row_R_col_C|> or <|img_thumbnail|>,
// then its rows at the engine's arithmetic within the tower bound. A
// tolerance that keeps the picture whole is the violation.
func TestLFM2VLTowerMatchesTransformers(t *testing.T) {
	m, g, base := openLFM2VL(t)
	defer m.Close()
	tg := m.Tower
	f, err := os.Open(base + ".png")
	if err != nil {
		testmodels.Missing(t, "PICTURE MISSING: %v", err)
	}
	img, err := png.Decode(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	b := img.Bounds()
	if _, n := squareOf(t, tg, b.Dx(), b.Dy(), 0); n != len(g.Pieces) || g.Cols*g.Rows < 2 {
		t.Fatalf("the picture is %d pieces, the processor's %d (%dx%d tiles)", n, len(g.Pieces), g.Cols, g.Rows)
	}
	s := testInstance(t, m, 64, WithKVF16(false))
	defer s.Close()
	px := readF32File(t, base+".px")
	at, differ, worst := 0, 0, 0.0
	for i, pc := range g.Pieces {
		got, sq := towerPixelsAt(t, s, img, i)
		if [2]int{sq.Gh, sq.Gw} != pc {
			t.Fatalf("piece %d is %dx%d patches, the processor's %dx%d", i, sq.Gh, sq.Gw, pc[0], pc[1])
		}
		for j, v := range got {
			d := math.Abs(float64(v - px[at+j]))
			if !(d < 2e-4) {
				differ++
			}
			worst = math.Max(worst, d)
		}
		at += len(got)
	}
	if at != len(px) || differ > 0 {
		t.Fatalf("%d of %d samples differ from the processor's, worst by %.5f (%d of the golden's %d read)",
			differ, at, worst, at, len(px))
	}
	t.Logf("%dx%d tiles and a thumbnail: every one of %d samples the processor's", g.Cols, g.Rows, at)

	rows, err := s.EncodePicture(img)
	if err != nil {
		t.Fatal(err)
	}
	emb := readF32File(t, base+".emb_a8")
	D := int(m.Container.Vision().ProjDim)
	marks := tileMarks(t, m)
	var got, want []float32
	pos, ref := 0, 0
	worstMark := 0.0
	for i := range g.Pieces {
		sq, _ := squareOf(t, tg, b.Dx(), b.Dy(), i)
		n := tg.SquareOut(sq) - 1
		mark := marks[(sq.Mark-1)*D : sq.Mark*D]
		for j, v := range rows[pos*D : (pos+1)*D] {
			worstMark = math.Max(worstMark, math.Abs(float64(v-mark[j])))
		}
		if (ref+n)*D > len(emb) || (pos+1+n)*D > len(rows) {
			var each []string
			for j := range g.Pieces {
				q, _ := squareOf(t, tg, b.Dx(), b.Dy(), j)
				each = append(each, fmt.Sprintf("%dx%d patches -> %d", q.Gh, q.Gw, tg.SquareOut(q)))
			}
			t.Fatalf("piece %d overruns: the reference has %d rows, the engine %d; the pieces %v (the processor's %v)",
				i, len(emb)/D, len(rows)/D, each, g.Pieces)
		}
		got = append(got, rows[(pos+1)*D:(pos+1+n)*D]...)
		want = append(want, emb[ref*D:(ref+n)*D]...)
		pos, ref = pos+1+n, ref+n
	}
	if pos*D != len(rows) || ref*D != len(emb) {
		t.Fatalf("%d rows read of %d, %d of the reference's %d", pos, len(rows)/D, ref, len(emb)/D)
	}
	if worstMark != 0 {
		t.Fatalf("a marker row is off its table row by %.3g", worstMark)
	}
	clean, worst := nmseOf(got, want)
	t.Logf("%d pieces' %d rows: NMSE %.3e, max|d| %.4f at the engine's arithmetic; %d marker rows their table's",
		len(g.Pieces), ref, clean, worst, len(g.Pieces))
	if math.IsNaN(clean) || clean > 2e-3 {
		t.Fatalf("the rows are NMSE %.3e from transformers' (bound 2e-3)", clean)
	}

	// The violation: a tolerance that keeps the picture whole.
	stated := tg.Geometry
	restateStep(t, m, "synth-lfm2vl", convert.PreTileSmart, func(s *jlm.PreStep) { s.Floats[0] = 1e6 })
	_, n := squareOf(t, tg, b.Dx(), b.Dy(), 0)
	tg.Geometry = stated
	if n == len(g.Pieces) {
		t.Fatal("VIOLATION: a tolerance of a million still cuts the picture into the processor's pieces")
	}
}

// tileMarks is m's marker table: a row for each piece marker the tower
// writes before a piece's rows, the text table's row of its token.
func tileMarks(t *testing.T, m *model.Model) []float32 {
	t.Helper()
	D := int(m.Container.Vision().ProjDim)
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
		marks := make([]float32, int(e.Dims[0])*int(e.Dims[1]))
		if err := quant.Dequant32(typ, qs, marks); err != nil {
			t.Fatal(err)
		}
		if int(e.Dims[0]) != D {
			t.Fatalf("the marker table's rows are %d wide, not %d", e.Dims[0], D)
		}
		return marks
	}
	t.Fatal("the container has no marker table")
	return nil
}

// lfm2E2EGolden is synth-lfm2vl's end-to-end record from what
// scripts/lfm2visiongold.py --fixture writes beside the fixture: the
// processor's ids around the picture's run and the run itself, the
// continuation and transformers' logits after the prompt, the picture, and
// every piece's projector rows (f32 and at the engine's arithmetic).
func lfm2E2EGolden(t *testing.T) *towerE2EGolden {
	t.Helper()
	base := testmodels.Path("synth-lfm2vl-gold/picture")
	raw, err := os.ReadFile(base + ".json")
	if err != nil {
		testmodels.Missing(t, "GOLDEN MISSING: %v -- run scripts/lfm2visiongold.py --fixture (RULE 11)", err)
	}
	var r struct {
		IDs        []int32    `json:"ids"`
		ImageToken int32      `json:"image_token"`
		Cont       []int32    `json:"cont"`
		Logits     []e2eLogit `json:"logits"`
	}
	if err := json.Unmarshal(raw, &r); err != nil {
		t.Fatal(err)
	}
	first, last := slices.Index(r.IDs, r.ImageToken), -1
	for i, id := range r.IDs {
		if id == r.ImageToken {
			last = i
		}
	}
	if first < 1 {
		t.Fatalf("the golden's ids hold no picture run after a marker (the first <image> at %d)", first)
	}
	f, err := os.Open(base + ".png")
	if err != nil {
		testmodels.Missing(t, "PICTURE MISSING: %v", err)
	}
	img, err := png.Decode(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	g := &towerE2EGolden{Pre: r.IDs[:first-1], Rows: r.IDs[first-1 : last+1], Post: r.IDs[last+1:], Cont: r.Cont, Logits: r.Logits}
	b := img.Bounds()
	g.Image.W, g.Image.H = b.Dx(), b.Dy()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			c := color.NRGBAModel.Convert(img.At(x, y)).(color.NRGBA)
			g.Image.RGB = append(g.Image.RGB, c.R, c.G, c.B)
		}
	}
	g.Image.Embd, g.Image.EmbdA8 = readF32File(t, base+".emb"), readF32File(t, base+".emb_a8")
	return g
}

// lfm2Lay lays synth-lfm2vl's projector rows as its picture's span: each
// piece's marker row, then the piece's rows. TestLFM2VLTowerMatchesTransformers
// holds the marker rows to the table exactly: seven rows among 1783 move
// the logits by less than the tower's own error does.
func lfm2Lay(t *testing.T, m *model.Model, g *towerE2EGolden, rows []float32, _ []int) []float32 {
	t.Helper()
	D, tg, marks := m.Cfg.NEmbd, m.Tower, tileMarks(t, m)
	var out []float32
	at := 0
	_, n := squareOf(t, tg, g.Image.W, g.Image.H, 0)
	for i := range n {
		sq, _ := squareOf(t, tg, g.Image.W, g.Image.H, i)
		k := tg.SquareOut(sq) - 1
		if (at+k)*D > len(rows) {
			t.Fatalf("piece %d's %d rows overrun the reference's %d", i, k, len(rows)/D)
		}
		out = append(out, marks[(sq.Mark-1)*D:sq.Mark*D]...)
		out = append(out, rows[at*D:(at+k)*D]...)
		at += k
	}
	if at*D != len(rows) || len(out) != len(g.Rows)*D {
		t.Fatalf("the pieces lay %d of the reference's %d rows into %d span rows, the processor's run %d",
			at, len(rows)/D, len(out)/D, len(g.Rows))
	}
	return out
}

// TestLFM2VLTowerRunsOnTheCard places synth-lfm2vl's tower on each device,
// every block and then half of them, and holds the tiled picture's rows,
// markers and all, to the CPU device's.
func TestLFM2VLTowerRunsOnTheCard(t *testing.T) {
	m, _, base := openLFM2VL(t)
	defer m.Close()
	f, err := os.Open(base + ".png")
	if err != nil {
		testmodels.Missing(t, "PICTURE MISSING: %v", err)
	}
	img, err := png.Decode(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	towerRunsOnTheCard(t, m, img, nil, 0, WithKVF16(false))
}
