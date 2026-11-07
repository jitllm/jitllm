package engine

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/device/cpudev"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// g3nvGolden is scripts/gemma3nvisiongold.py's picture and the soft rows
// transformers' head writes for it, at f32 and at the engine's arithmetic.
type g3nvGolden struct {
	Image struct {
		H, W   int
		RGB    string
		Soft   int
		Embd   []float32
		EmbdA8 []float32 `json:"embd_a8"`
	}
}

func readG3nv(t *testing.T) (g3nvGolden, image.Image) {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "gemma3nv", "synth-gemma3nv.json"))
	if err != nil {
		t.Fatalf("the golden: %v", err)
	}
	var g g3nvGolden
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	rgb, err := base64.StdEncoding.DecodeString(g.Image.RGB)
	if err != nil || len(rgb) != g.Image.W*g.Image.H*3 {
		t.Fatalf("the golden's picture is %d bytes for %dx%d: %v", len(rgb), g.Image.W, g.Image.H, err)
	}
	img := image.NewRGBA(image.Rect(0, 0, g.Image.W, g.Image.H))
	for i := range g.Image.W * g.Image.H {
		copy(img.Pix[4*i:], rgb[3*i:3*i+3])
		img.Pix[4*i+3] = 255
	}
	return g, img
}

// g3nvTap is the golden's rows at tap name (.a8: at the engine's int8
// activations and Q8_0 weights), [y][x][channel].
func g3nvTap(t *testing.T, name string, n int) []float32 {
	t.Helper()
	return readF32n(t, testmodels.Path(filepath.Join("synth-gemma3nv.gold", name+".f32")), n)
}

// chainPixels is the picture op of a chained tower's entry alone over img:
// its 768x768 pixels, [y][x][channel].
func chainPixels(t *testing.T, s *Instance, img image.Image) []float32 {
	t.Helper()
	if err := s.towerReady(); err != nil {
		t.Fatal(err)
	}
	tr, tg := s.tower, s.m.Tower
	ts := tr.s
	words, pl, err := model.PictureWords(img, &tr.pad, &tr.ent)
	if err != nil {
		t.Fatal(err)
	}
	pl.Request(tr.req)
	sq, squares := tg.Square(pl.W, pl.H, 0)
	cl := tg.Classes[sq.Class]
	a := cl.Pic
	a.Pixels = pl.Px
	tr.squareRequest(pl, sq, a, squares)
	b := graph.NewBlock(-1)
	b.Out(b.Emit(graph.OpPicture, nil, []graph.Result{{Name: "pixels", Width: 3}}, nil, a, nil, nil)[0])
	f, err := ts.towerForm(b.G, cl.Rows, nil)
	if err != nil {
		t.Fatal(err)
	}
	tr.size(cl.Rows, 0)
	out := make([]float32, cl.Rows*3)
	bd := new(cpudev.Bound)
	*bd = cpudev.Bound{Li: -1, Own: execOf(ts).Device().Own(f), N: cl.Rows, Req: tr.req, Pos: tr.posTab, Pic: words}
	bd.Weights, _ = execOf(ts).Bindings().Bind(f, -1, &ts.dense.TowerDense)
	bd.IO[0] = unsafe.Pointer(&out[0])
	execOf(ts).Runner().Run(bd, f)
	return out
}

// TestGemma3nTowerMatchesTransformers holds Gemma 3n's stated MobileNet-V5
// tower, run as a chain of blocks on the CPU device, to transformers' own
// (scripts/gemma3nvisiongold.py, synth-gemma3nv: every one of
// mobilenetv5_300m_enc's 84 blocks at a quarter of its channels, every
// parameter randomised): the processor's pixels; the stem; each block ALONE,
// fed the reference's rows before it, against the reference's at the
// engine's own arithmetic (its int8 activations and Q8_0 weights), so a
// block's error is its own; and the whole tower's 256 soft rows against the
// reference's, with the end-of-image marker's row after them the embedder's
// hard row. Each violation is a rewrite of the stated graph: the fusion
// adapter's two inputs swapped, and an inverted residual's skip dropped
// (its add reading the branch twice).
func TestGemma3nTowerMatchesTransformers(t *testing.T) {
	m := openGemma3nV(t)
	defer m.Close()
	g, img := readG3nv(t)
	shapes := gemma3nvShapes(t)
	tg, v := m.Tower, m.Container.Vision()
	D := int(v.ProjDim)
	s := testInstance(t, m, 64, WithKVF16(false))
	defer s.Close()

	px := chainPixels(t, s, img)
	want := g3nvTap(t, "pixels", len(px))
	off, worst := 0, 0.0
	for i := range want {
		d := math.Abs(float64(px[i] - want[i]))
		if d > 1e-6 {
			off++
		}
		worst = math.Max(worst, d)
	}
	t.Logf("the picture: %d of %d samples off the processor's, worst by %.5f", off, len(want), worst)
	if worst > 1.01/255 || off*1000 > len(want) {
		t.Fatal("the preprocessing is not the processor's")
	}

	tr := s.tower
	ch := tr.parts.(nn.TowerChain)
	words, pl, err := model.PictureWords(img, &tr.pad, &tr.ent)
	if err != nil {
		t.Fatal(err)
	}
	pl.Request(tr.req)
	sq, squares := tg.Square(pl.W, pl.H, 0)
	tr.squareRequest(pl, sq, tg.Classes[sq.Class].Pic, squares)
	cl := tg.Classes[0]
	tr.size(cl.Rows, 0)
	size := func(name string) int {
		s := shapes[name]
		return s[0] * s[1] * s[2]
	}
	stem := make([]float32, size("stem"))
	if err := tr.parts.Entry(0, pl.Px, words, tr.req, tr.posTab, cl.Rows, stem); err != nil {
		t.Fatal(err)
	}
	e, _ := nmseOf(stem, g3nvTap(t, "stem.a8", len(stem)))
	t.Logf("the stem: NMSE %.3e", e)
	if math.IsNaN(e) || e > 1e-4 {
		t.Fatalf("the stem is NMSE %.3e from the reference (bound 1e-4)", e)
	}
	prev, worstBlock, at := g3nvTap(t, "stem.a8", len(stem)), 0.0, -1
	for i := range tg.Blocks {
		name := fmt.Sprintf("blk%d", i)
		out := make([]float32, size(name))
		w, err := tr.lease.Ensure(m.TowerBase + i)
		if err != nil {
			t.Fatal(err)
		}
		if err := ch.ChainBlock(i, m.TowerBase+i, w, tr.req, tr.posTab, prev, out); err != nil {
			t.Fatal(err)
		}
		ref := g3nvTap(t, name+".a8", len(out))
		e, _ := nmseOf(out, ref)
		if math.IsNaN(e) || e > worstBlock {
			worstBlock, at = e, i
		}
		prev = ref
	}
	t.Logf("each of the %d blocks alone: the worst NMSE %.3e (block %d)", len(tg.Blocks), worstBlock, at)
	if math.IsNaN(worstBlock) || worstBlock > 1e-3 {
		t.Fatalf("block %d alone is NMSE %.3e from the reference (bound 1e-3)", at, worstBlock)
	}

	// The whole tower, through EncodePicture: the soft rows, and the
	// marker's row after them.
	encode := func() []float32 {
		x := testInstance(t, m, 64, WithKVF16(false))
		defer x.Close()
		rows, err := x.EncodePicture(img)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), rows...)
	}
	rows := encode()
	soft := g.Image.Soft
	if soft != 256 || len(rows) != (soft+1)*D || len(g.Image.EmbdA8) != soft*D {
		t.Fatalf("%d rows of %d for %d soft rows of a golden of %d", len(rows)/D, D, soft, len(g.Image.EmbdA8)/max(D, 1))
	}
	clean, _ := nmseOf(rows[:soft*D], g.Image.EmbdA8)
	f32, _ := nmseOf(rows[:soft*D], g.Image.Embd)
	t.Logf("the tower's %d soft rows: NMSE %.3e from the reference at the engine's arithmetic, %.3e from its f32", soft, clean, f32)
	if math.IsNaN(clean) || clean > 1e-3 {
		t.Fatalf("the soft rows are NMSE %.3e from the reference (bound 1e-3)", clean)
	}
	rec := v.Tower
	last := rec.Head.Ops[len(rec.Head.Ops)-1]
	var hard []float32
	for _, x := range m.TowerDense.Vecs {
		if x.Role == jlm.RoleVMNHardEmbd {
			hard = x.V
		}
	}
	mark := hard[int(last.Offset)*D : (int(last.Offset)+1)*D]
	for j, x := range rows[soft*D:] {
		if x != mark[j] {
			t.Fatalf("the marker's row's element %d is %v, the hard table's %v", j, x, mark[j])
		}
	}

	// The violations, each a rewrite of the stated graph.
	uir := -1
	for i, cb := range rec.Chain {
		ops := cb.Prog.Ops
		o := ops[len(ops)-1]
		plain := true
		for _, op := range ops {
			plain = plain && op.Code != jlm.TOpUnfold && op.Code != jlm.TOpRowsAttend
		}
		if plain && o.Code == jlm.TOpAdd && o.In[0] == 0 && uir < 0 {
			uir = i
		}
	}
	if uir < 0 {
		t.Fatal("the chain has no inverted residual with a skip")
	}
	for _, f := range []struct {
		name string
		edit func() func()
	}{
		{"the fusion's inputs swapped", func() func() {
			in := rec.Head.Ops[1].In
			rec.Head.Ops[1].In = []uint16{in[1], in[0]}
			return func() { rec.Head.Ops[1].In = in }
		}},
		{fmt.Sprintf("block %d's skip dropped", uir), func() func() {
			op := &rec.Chain[uir].Prog.Ops[len(rec.Chain[uir].Prog.Ops)-1]
			in := op.In
			op.In = []uint16{in[1], in[1]}
			return func() { op.In = in }
		}},
	} {
		undo := f.edit()
		bad, err := m.BuildTowerGraphs()
		undo()
		if err != nil {
			t.Fatalf("%s: %v", f.name, err)
		}
		m.Tower = bad
		br := encode()
		m.Tower = tg
		nb, _ := nmseOf(br[:soft*D], g.Image.EmbdA8)
		t.Logf("violation (%s): NMSE %.3e", f.name, nb)
		if !(nb > 100*clean) || !(nb > 1e-3) {
			t.Fatalf("VIOLATION: %s reads %.3e against a clean %.3e", f.name, nb, clean)
		}
	}
}
