package engine

import (
	"encoding/json"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// towerE2EGolden is the end-to-end record a stated tower's golden carries:
// the reference's ids around one picture's run, the continuation decoded
// after it, the picture and the tower's rows for it (f32 and at the engine's
// arithmetic), and transformers' logits at each decoded position -- and,
// where the model's picture attends both ways, with its rows causal.
type towerE2EGolden struct {
	Pre  []int32 `json:"pre"`
	Post []int32 `json:"post"`
	Cont []int32 `json:"cont"`
	// BOI and EOI are the ids that open and close the picture's run where
	// the golden keeps them apart from Pre and Post (Gemma's).
	BOI int32 `json:"boi"`
	EOI int32 `json:"eoi"`
	// Rows is the run between Pre and Post as the processor lays it out
	// (Llama 4's).
	Rows  []int32 `json:"rows"`
	Image struct {
		H, W   int
		RGB    []byte    `json:"rgb"`
		Embd   []float32 `json:"embd"`
		EmbdA8 []float32 `json:"embd_a8"`
		// Break is the [IMG_BREAK] token's embedding (Pixtral's).
		Break []float32 `json:"break"`
	} `json:"image"`
	Logits       []e2eLogit `json:"logits"`
	LogitsCausal []e2eLogit `json:"logits_causal"`
	LogitsEvery  []e2eLogit `json:"logits_every"`
}

type e2eLogit struct {
	Argmax int32     `json:"argmax"`
	Head   []float64 `json:"head"`
}

// towerE2E is one fixture of TestStatedTowerMatchesTransformersEndToEnd.
type towerE2E struct {
	name, dir, script string
	// load reads a golden kept beside the fixture rather than in
	// testdata/golden/dir (nil is testdata's).
	load func(t *testing.T) *towerE2EGolden
	// around is the reference's ids before and after the picture's rows.
	around func(g *towerE2EGolden) (pre, post []int32)
	// lay is the picture's span rows for the tower's rows in the reference's
	// order: the rows with the marker rows the tower writes between them, in
	// the order tiles gives (nil is the reference's).
	lay func(t *testing.T, m *model.Model, g *towerE2EGolden, rows []float32, tiles []int) []float32
	// violations are the fixture's own, beside the rows reversed.
	violations func(x *e2eArm)
}

// TestStatedTowerMatchesTransformersEndToEnd holds Gemma 4's, Gemma 3n's,
// Llama 4's, Phi-4's, Pixtral's and LFM2-VL's models to transformers from
// the pixels:
// the reference's prompt around the picture, the picture encoded by the
// prefill through the stated tower, its rows attending as the container
// states (jlm.TowerSpan.Bidir, jlm.Config.BidirLayers), then the
// continuation decoded -- on the CPU device and on each card with every text
// and tower block placed. The bound is the Qwen2-VL gate's: twice the worst
// of the reference's int8 rows and eight draws of the tower's own error
// through the text model, with no more argmax flips past a near tie
// (tieMargin) than they make. Every
// fixture's rows reversed is a violation, and each fixture's layout or mask
// has its own: Gemma 4's rows causal (held to transformers' causal run) and
// bidirectional on every layer, Llama 4's global tile first, Pixtral's rows
// without their breaks.
func TestStatedTowerMatchesTransformersEndToEnd(t *testing.T) {
	gemma := func(g *towerE2EGolden) ([]int32, []int32) {
		return append(slices.Clone(g.Pre), g.BOI), append([]int32{g.EOI}, g.Post...)
	}
	asIs := func(g *towerE2EGolden) ([]int32, []int32) { return g.Pre, g.Post }
	same := func(_ *testing.T, _ *model.Model, _ *towerE2EGolden, rows []float32, _ []int) []float32 { return rows }
	for _, c := range []towerE2E{
		{name: "synth-gemma4v", dir: "gemma4v", script: "gemma4visiongold.py", around: gemma, lay: same, violations: gemma4MaskViolations},
		{name: "synth-gemma3nv", dir: "gemma3nv", script: "gemma3nvisiongold.py", lay: gemma3nLay,
			around: func(g *towerE2EGolden) ([]int32, []int32) { return append(slices.Clone(g.Pre), g.BOI), g.Post }},
		{name: "synth-llama4v", dir: "llama4v", script: "llama4visiongold.py", lay: llama4Lay, violations: llama4Violations,
			around: func(g *towerE2EGolden) ([]int32, []int32) {
				return append(slices.Clone(g.Pre), g.Rows[0]), append([]int32{g.Rows[len(g.Rows)-1]}, g.Post...)
			}},
		{name: "synth-phi4v", dir: "phi4v", script: "phi4visiongold.py", around: asIs, lay: same},
		{name: "synth-pixtral", dir: "pixtral", script: "pixtralgold.py", around: asIs, lay: pixtralLay, violations: pixtralViolations},
		{name: "synth-lfm2vl", script: "lfm2visiongold.py --fixture", load: lfm2E2EGolden, around: asIs, lay: lfm2Lay},
	} {
		t.Run(c.name, func(t *testing.T) { statedTowerEndToEnd(t, c) })
	}
}

// e2eArm is one device's arm of the end-to-end gate.
type e2eArm struct {
	t         *testing.T
	m         *model.Model
	path      string
	g         *towerE2EGolden
	c         towerE2E
	dev       *gpu.GPU
	pre, post []int32
	picture   model.Span
	a8        []float32
	bound     float64
}

func statedTowerEndToEnd(t *testing.T, c towerE2E) {
	g := &towerE2EGolden{}
	if c.load != nil {
		g = c.load(t)
	} else {
		raw, err := os.ReadFile(filepath.Join("..", "testdata", "golden", c.dir, c.name+".json"))
		if err != nil {
			t.Fatalf("GOLDEN MISSING: %v -- run scripts/%s", err, c.script)
		}
		if err := json.Unmarshal(raw, g); err != nil {
			t.Fatal(err)
		}
	}
	if len(g.Logits) < 2 || len(g.Logits) > len(g.Cont)+1 || len(g.Image.EmbdA8) != len(g.Image.Embd) {
		t.Fatalf("the golden has %d logit rows for %d decoded ids and rows of %d and %d values: it holds no end-to-end record",
			len(g.Logits), len(g.Cont), len(g.Image.Embd), len(g.Image.EmbdA8))
	}
	text, proj := testmodels.Path(c.name+".gguf"), testmodels.Path("mmproj-"+c.name+".gguf")
	if !sourcePresent(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (set JITLLM_MODELS) -- run scripts/%s (RULE 11)", text, c.script)
	}
	path := jlmOfPair(t, text, proj)
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	img := rgbPicture(g.Image.RGB, g.Image.W, g.Image.H)
	picture, err := m.PictureSpan(img)
	if err != nil {
		t.Fatal(err)
	}
	mk := m.Container.Vision().Tower.Span
	picture.Bidir = mk.Bidir
	pre, post := c.around(g)
	// The ids that open and close the picture's run are the ones the
	// container states for it.
	if n := len(mk.Before); n > 0 && (len(pre) == 0 || pre[len(pre)-1] != mk.Before[n-1]) {
		t.Fatalf("the picture opens with %v, the reference's prompt before it ends %v", mk.Before, pre)
	}
	if len(mk.After) > 0 && (len(post) == 0 || post[0] != mk.After[0]) {
		t.Fatalf("the picture closes with %v, the reference's prompt after it starts %v", mk.After, post)
	}
	if len(g.LogitsCausal) > 0 && !mk.Bidir {
		t.Fatal("the reference's picture attends both ways and the container states its rows causal")
	}
	s := testInstance(t, m, 64, WithKVF16(false))
	rows, err := s.EncodePicture(img)
	if err != nil {
		t.Fatal(err)
	}
	rows = slices.Clone(rows)
	s.Close()
	a8 := c.lay(t, m, g, g.Image.EmbdA8, nil)
	if len(rows) != len(a8) || len(rows) != picture.N(m.Cfg.NEmbd)*m.Cfg.NEmbd {
		t.Fatalf("the tower writes %d values, the reference's rows laid out %d, the span %d rows of %d",
			len(rows), len(a8), picture.N(m.Cfg.NEmbd), m.Cfg.NEmbd)
	}
	e, _ := nmseOf(rows, c.lay(t, m, g, g.Image.Embd, nil))
	t.Logf("the tower's %d rows: NMSE %.3e against transformers' f32 rows", len(rows)/m.Cfg.NEmbd, e)
	arm := func(t *testing.T, dev *gpu.GPU) {
		x := &e2eArm{t: t, m: m, path: path, g: g, c: c, dev: dev, pre: pre, post: post, picture: picture, a8: a8}
		x.check(e)
	}
	t.Run("cpu", func(t *testing.T) { arm(t, nil) })
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) {
			dev, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff))
			if err != nil || dev == nil {
				noDevice(t, spec, err)
			}
			defer dev.Close()
			arm(t, dev)
		})
	}
}

// rows is a span of the picture's rows given as values, laid and attending
// as the picture's.
func (x *e2eArm) rows(embd []float32) model.Span {
	return model.Span{Embd: embd, Grid: x.picture.Grid, Bidir: x.picture.Bidir}
}

// run prefills the reference's prompt around sp on m, every block of the text
// model and the tower on the arm's card, and decodes the continuation: the
// logits at each decoded position.
func (x *e2eArm) run(m *model.Model, sp model.Span) [][]float32 {
	t := x.t
	t.Helper()
	s := testInstance(t, m, len(x.pre)+sp.N(m.Cfg.NEmbd)+len(x.post)+len(x.g.Logits), WithKVF16(false))
	defer s.Close()
	if x.dev != nil {
		s.SetDeviceFallback(false)
		if err := s.SetDevice(x.dev); err != nil {
			t.Fatal(err)
		}
		if n := s.GPULayers(); n != m.Cfg.NLayer {
			t.Fatalf("the card took %d of %d text blocks: %s", n, m.Cfg.NLayer, x.dev.Err())
		}
		if n := towerOnCard(s); n != len(m.Tower.Blocks) {
			t.Fatalf("the card took %d of the tower's %d blocks: %s", n, len(m.Tower.Blocks), x.dev.Err())
		}
	}
	lg, err := s.PrefillMixed(model.Span{Tokens: x.pre}, sp, model.Span{Tokens: x.post})
	if err != nil {
		t.Fatal(err)
	}
	out := [][]float32{slices.Clone(lg)}
	for _, id := range x.g.Cont[:len(x.g.Logits)-1] {
		if lg, err = s.Forward(id); err != nil {
			t.Fatal(err)
		}
		out = append(out, slices.Clone(lg))
	}
	if x.dev != nil && s.GPULayers() != m.Cfg.NLayer {
		t.Fatalf("after the prompt the card holds %d of %d text blocks: %s", s.GPULayers(), m.Cfg.NLayer, x.dev.Err())
	}
	return out
}

// towerOnCard is how many of s's tower blocks run on a card: the tower's
// placed blocks, or a chain's blocks each a part of the device serving
// picture rows (placeChain).
func towerOnCard(s *Instance) int {
	if !s.m.Tower.Chain {
		return s.tower.s.GPULayers()
	}
	n := 0
	for i := range s.m.Tower.Blocks {
		if s.tower.chainDev(i) != nil {
			n++
		}
	}
	return n
}

// worstAgainst is the worst row's NMSE against ref's heads and the argmaxes
// that part from ref's by more than a near tie: the row's top logit more
// than tieMargin past its logit for the reference's token.
func worstAgainst(ref []e2eLogit, rows [][]float32) (float64, int) {
	w, flips := 0.0, 0
	for i, lg := range rows {
		n := llama4Cmp(ref[i].Head, lg)
		if math.IsNaN(n) {
			return math.Inf(1), len(rows)
		}
		w = math.Max(w, n)
		if a, r := Greedy(lg), ref[i].Argmax; a != r && float64(lg[a]-lg[r]) > tieMargin {
			flips++
		}
	}
	return w, flips
}

func (x *e2eArm) check(e float64) {
	t, g := x.t, x.g
	// What a tower error of the engine's size moves the logits by, and the
	// argmaxes it flips, at its worst over the reference's int8 rows and
	// eight draws.
	arith, aflips := worstAgainst(g.Logits, x.run(x.m, x.rows(x.a8)))
	a8 := arith
	for k := range 8 {
		rng := rand.New(rand.NewPCG(uint64(k), 0x9e3779b97f4a7c15))
		n, f := worstAgainst(g.Logits, x.run(x.m, x.rows(x.c.lay(t, x.m, g, perturbed(rng, g.Image.Embd, e), nil))))
		arith, aflips = math.Max(arith, n), max(aflips, f)
	}
	got := x.run(x.m, x.picture)
	clean, flips := worstAgainst(g.Logits, got)
	for i, lg := range got {
		if a, r := Greedy(lg), g.Logits[i].Argmax; a != r {
			t.Logf("row %d: the argmax %d at %.4f, the reference's %d at %.4f here", i, a, lg[a], r, lg[r])
		}
	}
	t.Logf("%d rows from the pixels: worst NMSE %.3e against transformers, %d argmax flips; a tower error of the engine's size moves them up to %.3e, flipping up to %d (the reference's int8 rows %.3e)",
		len(g.Logits), clean, flips, arith, aflips, a8)
	if !(arith > 0) || !(clean < 2*arith) || flips > aflips {
		t.Fatalf("worst NMSE %.3e against transformers (bound twice the worst draw's %.3e), %d argmax flips where the draws flip up to %d",
			clean, arith, flips, aflips)
	}
	x.bound = 2 * arith
	rev := slices.Clone(x.a8)
	slices.Reverse(rev)
	x.violation("the picture's rows reversed", x.m, x.rows(rev))
	if x.c.violations != nil {
		x.c.violations(x)
	}
}

// violation fails the arm unless sp on m reads five times past the bound:
// the draws' noise is isotropic, and moves a fixture whose text model it
// moves most (Pixtral's) further than the tower's own error does.
func (x *e2eArm) violation(what string, m *model.Model, sp model.Span) [][]float32 {
	x.t.Helper()
	got := x.run(m, sp)
	bad, _ := worstAgainst(x.g.Logits, got)
	if !(bad > 5*x.bound) {
		x.t.Fatalf("VIOLATION: %s reads %.3e against the bound %.3e", what, bad, x.bound)
	}
	x.t.Logf("violation (%s): %.3e, %.0fx the bound", what, bad, bad/x.bound)
	return got
}

// gemma4MaskViolations: the picture's rows causal, which is transformers'
// own causal run and not the bidirectional one; and the container restated
// with the rows bidirectional on every layer, Gemma 3's mask, where Gemma
// 4's global layers keep them causal.
func gemma4MaskViolations(x *e2eArm) {
	t := x.t
	if len(x.g.LogitsEvery) != len(x.g.Logits) {
		t.Fatalf("the golden holds %d every-layer rows for %d: scripts/gemma4visiongold.py --every writes them", len(x.g.LogitsEvery), len(x.g.Logits))
	}
	causal := x.picture
	causal.Bidir = false
	got := x.violation("the picture's rows causal", x.m, causal)
	ref, _ := worstAgainst(x.g.LogitsCausal, got)
	bad, _ := worstAgainst(x.g.Logits, got)
	t.Logf("the causal run against transformers' causal run: %.3e", ref)
	if !(ref < bad/10) {
		t.Fatalf("the causal run reads %.3e against transformers' causal run and %.3e against its bidirectional one", ref, bad)
	}
	every := restatedEveryLayerBidir(t, x.path)
	defer every.Close()
	pic, err := every.PictureSpan(rgbPicture(x.g.Image.RGB, x.g.Image.W, x.g.Image.H))
	if err != nil {
		t.Fatal(err)
	}
	pic.Bidir = true
	// A mask on the two global layers of six moves the logits less than the
	// rows reversed do, so the violation is held both ways: past the bound
	// from transformers' run, and within it of transformers' own run with
	// the rows bidirectional on every layer.
	all := x.run(every, pic)
	ref, _ = worstAgainst(x.g.LogitsEvery, all)
	bad, _ = worstAgainst(x.g.Logits, all)
	t.Logf("violation (the rows bidirectional on every layer): %.3e, %.1fx the bound; %.3e against transformers' every-layer run",
		bad, bad/x.bound, ref)
	if !(bad > x.bound) || !(ref < x.bound) {
		t.Fatalf("VIOLATION: the rows bidirectional on every layer read %.3e against transformers' run and %.3e against its every-layer run (bound %.3e)",
			bad, ref, x.bound)
	}
}

// restatedEveryLayerBidir is the container at src copied and restated with
// a picture's rows attending both ways on every layer, opened.
func restatedEveryLayerBidir(t *testing.T, src string) *model.Model {
	t.Helper()
	b, err := os.ReadFile(src)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "every"+model.Ext)
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	if changed, err := jlm.Restate(path, func(_ *jlm.File, c *jlm.Config) (bool, error) {
		if !slices.Contains(c.BidirLayers, false) {
			t.Fatalf("the container states the rows bidirectional on every layer already (%v): the violation is the clean arm", c.BidirLayers)
		}
		for i := range c.BidirLayers {
			c.BidirLayers[i] = true
		}
		return true, nil
	}); err != nil || !changed {
		t.Fatalf("restating the mask: changed %v, %v", changed, err)
	}
	every, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	return every
}

// llama4Lay is Llama 4's tiles' rows laid out as the picture's span: each
// tile's marker row ahead of it and its trailing one after, the marker rows
// the container's table's; tiles[i] is the tile in place i.
func llama4Lay(t *testing.T, m *model.Model, g *towerE2EGolden, rows []float32, tiles []int) []float32 {
	t.Helper()
	tg := m.Tower
	D := m.Cfg.NEmbd
	marks := markerTable(t, m, D)
	_, n := tg.Square(g.Image.W, g.Image.H, 0)
	if n == 0 || len(rows)%(n*D) != 0 {
		t.Fatalf("%d values for %d tiles of %d-wide rows", len(rows), n, D)
	}
	per := len(rows) / n
	var out []float32
	for i := range n {
		sq, _ := tg.Square(g.Image.W, g.Image.H, i)
		if sq.Mark > 0 {
			out = append(out, marks[(sq.Mark-1)*D:sq.Mark*D]...)
		}
		k := i
		if tiles != nil {
			k = tiles[i]
		}
		out = append(out, rows[k*per:(k+1)*per]...)
		if sq.Trail > 0 {
			out = append(out, marks[(sq.Trail-1)*D:sq.Trail*D]...)
		}
	}
	return out
}

// llama4Violations: the global tile first, as a processor that put the
// overview ahead (MiniCPM-V's) would lay it out.
func llama4Violations(x *e2eArm) {
	_, n := x.m.Tower.Square(x.g.Image.W, x.g.Image.H, 0)
	order := []int{n - 1}
	for i := range n - 1 {
		order = append(order, i)
	}
	x.violation("the global tile first", x.m, x.rows(llama4Lay(x.t, x.m, x.g, x.g.Image.EmbdA8, order)))
}

// gemma3nLay is Gemma 3n's soft rows with the end-of-image row after them:
// its token embedded by the tower's own table of hard tokens
// (embed_vision), which the head's last op states.
func gemma3nLay(t *testing.T, m *model.Model, _ *towerE2EGolden, rows []float32, _ []int) []float32 {
	t.Helper()
	D := m.Cfg.NEmbd
	last := m.Container.Vision().Tower.Head.Ops[len(m.Container.Vision().Tower.Head.Ops)-1]
	for _, x := range m.TowerDense.Vecs {
		if x.Role == jlm.RoleVMNHardEmbd {
			return append(slices.Clone(rows), x.V[int(last.Offset)*D:(int(last.Offset)+1)*D]...)
		}
	}
	t.Fatal("the tower states no table of hard tokens")
	return nil
}

// pixtralLay is Pixtral's merged grid's rows with the [IMG_BREAK] row after
// each merged row but the last.
func pixtralLay(t *testing.T, m *model.Model, g *towerE2EGolden, rows []float32, _ []int) []float32 {
	t.Helper()
	tg := m.Tower
	sq, _ := tg.Square(g.Image.W, g.Image.H, 0)
	D, W, H := len(g.Image.Break), sq.Gw/tg.Frame, sq.Gh/tg.Frame
	if D == 0 || W*H*D != len(rows) {
		t.Fatalf("a %dx%d merged grid of %d, the reference's %d values", W, H, D, len(rows))
	}
	var out []float32
	for y := range H {
		out = append(out, rows[y*W*D:(y+1)*W*D]...)
		if y+1 < H {
			out = append(out, g.Image.Break...)
		}
	}
	return out
}

// pixtralViolations: the merged grid's rows without their breaks.
func pixtralViolations(x *e2eArm) {
	x.violation("the rows without their breaks", x.m, x.rows(x.g.Image.EmbdA8))
}
