package engine

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// mcvGolden is one picture's manifest from scripts/minicpmvgold.py: the
// picture's size, its slice grid, and each piece's size and patch grid.
type mcvGolden struct {
	W, H   int
	Grid   []int
	Pieces []struct {
		W, H, GH, GW int
	}
}

// mcvPictures are the pictures the goldens were written for: the photo, cut
// into an overview and two slices, and the repository's 384 square, an
// overview alone.
func mcvPictures() []struct{ name, path string } {
	return []struct{ name, path string }{
		{"photo896x448", testmodels.Path("minicpmv/photo896x448.png")},
		{"quad-and-disc", filepath.Join("testdata", "quad-and-disc.png")},
	}
}

func openMiniCPMV(t *testing.T) *model.Model {
	t.Helper()
	text, proj := testmodels.Path("minicpmv/MiniCPM-V-4-Q4_K_M.gguf"), testmodels.Path("minicpmv/mmproj-MiniCPM-V-4-f16.gguf")
	if !sourcePresent(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (set JITLLM_MODELS) (RULE 11)", text)
	}
	m, err := model.Open(jlmOfPair(t, text, proj))
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func readMcv(t *testing.T, name string) (mcvGolden, string) {
	t.Helper()
	dir := testmodels.Path("minicpmv/hf/MiniCPM-V-4/gold")
	b, err := os.ReadFile(filepath.Join(dir, name+".json"))
	if err != nil {
		testmodels.Missing(t, "GOLDEN MISSING: %v -- scripts/minicpmvgold.py writes it", err)
	}
	var g mcvGolden
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	if len(g.Pieces) == 0 {
		t.Fatal("the golden has no pieces")
	}
	return g, dir
}

func readF32n(t *testing.T, path string, n int) []float32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		testmodels.Missing(t, "GOLDEN MISSING: %v -- scripts/minicpmvgold.py writes it", err)
	}
	if len(b) != 4*n {
		t.Fatalf("%s is %d bytes, want %d floats", path, len(b), n)
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func decodePNG(t *testing.T, path string) image.Image {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		testmodels.Missing(t, "IMAGE MISSING: %v", err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	return img
}

// TestMiniCPMVTowerMatchesTransformers holds MiniCPM-V 4's stated tower to
// the checkpoint's own code (scripts/minicpmvgold.py), piece by piece: the
// processor's cut and Pillow's resize, every sample; the tower's rows after
// its post-LayerNorm (the head read alone); and the resampler's 64 rows, its
// marker runs their table's, in the processor's order, and the resampler
// alone on the reference's tower rows. The tower's bound is
// the int8 activations': from block 13 nine channels of the residual sit at
// ~2000 against a median of ~0.3, and a 32-wide quantization window holding
// one rounds its neighbours away. Each violation is a rewrite of the stated
// graph: positions unbucketed, keys without their 2-D position, queries
// without their LayerNorm.
func TestMiniCPMVTowerMatchesTransformers(t *testing.T) {
	m := openMiniCPMV(t)
	defer m.Close()
	const vitBound, embBound = 3e-2, 3e-2
	for _, pic := range mcvPictures() {
		t.Run(pic.name, func(t *testing.T) { miniCPMVPicture(t, m, pic.name, pic.path, vitBound, embBound) })
	}
}

func miniCPMVPicture(t *testing.T, m *model.Model, name, path string, vitBound, embBound float64) {
	g, dir := readMcv(t, name)
	img := decodePNG(t, path)
	if b := img.Bounds(); b.Dx() != g.W || b.Dy() != g.H {
		t.Fatalf("a %dx%d picture, the golden's %dx%d", b.Dx(), b.Dy(), g.W, g.H)
	}
	tg := m.Tower
	_, n := tg.Square(g.W, g.H, 0)
	if n != len(g.Pieces) {
		t.Fatalf("%d pieces, the processor cut %d", n, len(g.Pieces))
	}
	v := m.Container.Vision()
	E, D := int(v.NEmbd), int(v.ProjDim)
	s := testInstance(t, m, 64, WithKVF16(false))
	defer s.Close()
	for i, gp := range g.Pieces {
		px, sq := towerPixelsAt(t, s, img, i)
		if sq.Pw != gp.W || sq.Ph != gp.H || sq.Gw != gp.GW || sq.Gh != gp.GH {
			t.Fatalf("piece %d is %dx%d (%dx%d patches), the processor's %dx%d (%dx%d)", i, sq.Pw, sq.Ph, sq.Gw, sq.Gh,
				gp.W, gp.H, gp.GW, gp.GH)
		}
		want := readF32n(t, filepath.Join(dir, fmt.Sprintf("%s.p%d.px", name, i)), gp.W*gp.H*3)
		off, worst := 0, 0.0
		for j := range want {
			d := math.Abs(float64(px[j] - want[j]))
			if d > 1e-6 {
				off++
			}
			worst = math.Max(worst, d)
		}
		t.Logf("piece %d %dx%d: %d of %d samples off the processor's, worst by %.5f", i, gp.W, gp.H, off, len(want), worst)
		// One 8-bit step, normalised, is 1/127.5; a downscale's sums sit
		// within f32 noise of a half on a few samples in ten thousand.
		if worst > 1.01/127.5 || off*1000 > len(want) {
			t.Fatalf("piece %d: the preprocessing is not the processor's", i)
		}
	}

	// The tower alone: the head its post-LayerNorm.
	rec := v.Tower
	full := rec.Head
	rec.Head = jlm.Program{NIn: 1, Out: 1, Ops: []jlm.TowerOp{full.Ops[0]}}
	vt, err := m.BuildTowerGraphs()
	rec.Head = full
	if err != nil {
		t.Fatal(err)
	}
	m.Tower = vt
	vs := testInstance(t, m, 64, WithKVF16(false))
	vrows, err := vs.EncodePicture(img)
	vs.Close()
	m.Tower = tg
	if err != nil {
		t.Fatal(err)
	}
	// The encode lays each piece's rows at the projection's width apart.
	at := 0
	for i, gp := range g.Pieces {
		k := gp.GW * gp.GH
		wantVit := readF32n(t, filepath.Join(dir, fmt.Sprintf("%s.p%d.vit", name, i)), k*E)
		nv, _ := nmseOf(vrows[at*D:at*D+k*E], wantVit)
		at += k
		t.Logf("piece %d (%dx%d patches): the tower's rows NMSE %.3e", i, gp.GW, gp.GH, nv)
		if math.IsNaN(nv) || nv > vitBound {
			t.Fatalf("piece %d: the tower's rows are NMSE %.3e from the reference (bound %.0e)", i, nv, vitBound)
		}
	}

	// The whole tower: each piece's marker runs and its 64 rows.
	marks := markerTable(t, m, D)
	encode := func() (pieces [][]float32, worstMark float64) {
		ps := testInstance(t, m, 64, WithKVF16(false))
		defer ps.Close()
		rows, err := ps.EncodePicture(img)
		if err != nil {
			t.Fatal(err)
		}
		pos := 0
		run := func(first, n int) {
			for r := range n {
				for j, x := range rows[(pos+r)*D : (pos+r+1)*D] {
					worstMark = math.Max(worstMark, math.Abs(float64(x-marks[(first-1+r)*D+j])))
				}
			}
			pos += n
		}
		for i := range g.Pieces {
			sq, _ := tg.Square(g.W, g.H, i)
			if sq.Mark > 0 {
				run(sq.Mark, sq.MarkN)
			}
			pieces = append(pieces, rows[pos*D:(pos+tg.Fixed)*D])
			pos += tg.Fixed
			if sq.Trail > 0 {
				run(sq.Trail, sq.TrailN)
			}
		}
		if pos*D != len(rows) {
			t.Fatalf("%d rows read of %d", pos, len(rows)/D)
		}
		return pieces, worstMark
	}
	wantEmb := make([][]float32, len(g.Pieces))
	for i := range g.Pieces {
		wantEmb[i] = readF32n(t, filepath.Join(dir, fmt.Sprintf("%s.p%d.emb", name, i)), tg.Fixed*D)
	}
	reading := func(pieces [][]float32) float64 {
		worst := 0.0
		for i, p := range pieces {
			e, _ := nmseOf(p, wantEmb[i])
			worst = math.Max(worst, e)
		}
		return worst
	}
	pieces, worstMark := encode()
	if worstMark != 0 {
		t.Fatalf("a marker row is off its table row by %.3g", worstMark)
	}
	clean := reading(pieces)
	t.Logf("%d pieces of %d rows: the worst NMSE %.3e; every marker row its table's", len(pieces), tg.Fixed, clean)
	if math.IsNaN(clean) || clean > embBound {
		t.Fatalf("the resampler's rows are NMSE %.3e from the reference (bound %.0e)", clean, embBound)
	}

	// The resampler alone, on the reference's own tower rows: with the
	// tower's int8 rows out of it the only error is the resampler's own --
	// its int8 activations, ~3e-4 -- so a feature the whole tower's rows
	// barely show (the keys' 2-D position moves them from 6.2e-3 to 7.1e-3)
	// stands five times past it. A violation is held past three times the
	// clean reading.
	last := len(rec.Head.Ops) - 1
	if rec.Head.Ops[last].Code != jlm.TOpMarkRow {
		t.Fatalf("the head ends in op %d, not its marker rows", rec.Head.Ops[last].Code)
	}
	resample := func() float64 {
		worst := 0.0
		for i, gp := range g.Pieces {
			vit := readF32n(t, filepath.Join(dir, fmt.Sprintf("%s.p%d.vit", name, i)), gp.GW*gp.GH*E)
			got := projectorFrom(t, s, 1, last, vit, gp.GW, gp.GH)
			if len(got) != len(wantEmb[i]) {
				t.Fatalf("piece %d: the resampler wrote %d values, the reference %d", i, len(got), len(wantEmb[i]))
			}
			e, _ := nmseOf(got, wantEmb[i])
			worst = math.Max(worst, e)
		}
		return worst
	}
	alone := resample()
	t.Logf("the resampler alone: the worst NMSE %.3e", alone)
	if math.IsNaN(alone) || alone > 1e-3 {
		t.Fatalf("the resampler alone reads NMSE %.3e from the reference (bound 1e-3)", alone)
	}

	// The violations, each a rewrite of the stated graph: the entry's read
	// through the whole tower, the resampler's through the resampler alone.
	for _, f := range []struct {
		name  string
		edit  func(r *jlm.Tower) func()
		alone bool
	}{
		{"positions unbucketed", func(r *jlm.Tower) func() {
			op := r.Entry.Ops[2]
			r.Entry.Ops[2] = jlm.TowerOp{Code: jlm.TOpPosTable, Bind: op.Bind}
			return func() { r.Entry.Ops[2] = op }
		}, false},
		{"keys without their position", func(r *jlm.Tower) func() {
			in := r.Head.Ops[5].In
			r.Head.Ops[5].In = []uint16{3}
			return func() { r.Head.Ops[5].In = in }
		}, true},
		{"queries without their LayerNorm", func(r *jlm.Tower) func() {
			in := r.Head.Ops[9].In
			r.Head.Ops[9].In = []uint16{8}
			return func() { r.Head.Ops[9].In = in }
		}, true},
	} {
		undo := f.edit(rec)
		if f.alone {
			nb := resample()
			undo()
			t.Logf("violation (%s): the resampler alone reads NMSE %.3e", f.name, nb)
			if !(nb > 3*alone) {
				t.Fatalf("VIOLATION: %s reads %.3e against a clean %.3e: the gate cannot see it", f.name, nb, alone)
			}
			continue
		}
		bad, err := m.BuildTowerGraphs()
		undo()
		if err != nil {
			t.Fatalf("%s: %v", f.name, err)
		}
		m.Tower = bad
		bp, _ := encode()
		m.Tower = tg
		nb := reading(bp)
		t.Logf("violation (%s): NMSE %.3e", f.name, nb)
		if !(nb > 10*embBound) {
			t.Fatalf("VIOLATION: %s reads %.3e, not past %.3e", f.name, nb, 10*embBound)
		}
	}
}

// TestMiniCPMVPicturesAreNumbered lays two pictures into one prompt as the
// processor does: each picture's run begins with its number, counted from
// 0, in <image_id> and </image_id> (use_image_id), before <image>. The
// numbers dropped -- the span stating no index text -- is the violation.
func TestMiniCPMVPicturesAreNumbered(t *testing.T) {
	m := openMiniCPMV(t)
	defer m.Close()
	img := decodePNG(t, filepath.Join("testdata", "quad-and-disc.png"))
	sp, err := m.PictureSpan(img)
	if err != nil {
		t.Fatal(err)
	}
	msgs := []model.ChatMessage{{Role: "user", Content: "What differs?", Images: 2}}
	// before is the token run ahead of each picture's rows.
	before := func() [][]int32 {
		spans, err := m.ChatSpans(msgs, []model.Span{sp, sp}, true)
		if err != nil {
			t.Fatal(err)
		}
		var out [][]int32
		for i, s := range spans {
			if s.Rows() && i > 0 {
				out = append(out, spans[i-1].Tokens)
			}
		}
		return out
	}
	got := before()
	if len(got) != 2 {
		t.Fatalf("%d pictures laid out, want 2", len(got))
	}
	for i, ids := range got {
		want := m.Vocab.EncodeSpecial(fmt.Sprintf("<image_id>%d</image_id>", i), false)
		if len(want) < 3 || !slices.Equal(ids[max(0, len(ids)-len(want)):], want) {
			t.Fatalf("picture %d follows %v, which does not end in <image_id>%d</image_id>'s %v", i, ids, i, want)
		}
	}
	t.Logf("two pictures, each after its number's ids")
	span := &m.Container.Vision().Tower.Span
	ib, ia := span.IndexBefore, span.IndexAfter
	span.IndexBefore, span.IndexAfter = "", ""
	bad := before()
	span.IndexBefore, span.IndexAfter = ib, ia
	open := m.Vocab.EncodeSpecial("<image_id>", false)
	for _, ids := range bad {
		if len(open) == 1 && slices.Contains(ids, open[0]) {
			t.Fatal("VIOLATION: a span stating no index text still numbers its pictures")
		}
	}
}
