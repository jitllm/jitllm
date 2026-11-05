package engine

import (
	"image"
	"image/color"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// openQVLTower opens a Qwen2-VL family fixture -- the text GGUF and its
// mmproj, one container -- and its golden.
func openQVLTower(t *testing.T, name string) (*model.Model, *qvlGolden, []Option) {
	t.Helper()
	g := qvlGoldenOf(t, name)
	text, proj := testmodels.Path(name+".gguf"), testmodels.Path("mmproj-"+name+".gguf")
	if !sourcePresent(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (set JITLLM_MODELS) -- run scripts/qwenvlgold.py %s (RULE 11)",
			text, name)
	}
	instOpts := []Option{WithKVF16(false)}
	m, err := model.Open(jlmOfPair(t, text, proj))
	if err != nil {
		t.Fatal(err)
	}
	return m, g, instOpts
}

// rgbPicture is h x w samples of red, green and blue as a picture.
func rgbPicture(rgb []byte, w, h int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for i := range w * h {
		img.Pix[4*i], img.Pix[4*i+1], img.Pix[4*i+2], img.Pix[4*i+3] = rgb[3*i], rgb[3*i+1], rgb[3*i+2], 255
	}
	return img
}

// towerPixels is the picture op of the class img is encoded in, alone on the
// instance's tower on the CPU device, at the request the encode writes: the
// resized and normalised picture, [y][x][channel], and its square.
func towerPixels(t *testing.T, s *Instance, img image.Image) ([]float32, model.Square) {
	t.Helper()
	if err := s.towerReady(); err != nil {
		t.Fatal(err)
	}
	tr, m := s.tower, s.m
	tg, ts := m.Tower, tr.s
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
	P := tg.Patch
	b := graph.NewBlock(-1)
	b.Out(b.Emit(graph.OpPicture, nil, []graph.Result{{Name: "pixels", Width: 3 * P * P}}, nil, a, nil, nil)[0])
	f, err := ts.formCPUGraph(b.G, cl.Rows, nil)
	if err != nil {
		t.Fatal(err)
	}
	n := sq.Gw * sq.Gh
	tr.size(tg.Lead+n, int(m.Container.Vision().NEmbd))
	out := make([]float32, cl.Rows*3*P*P)
	bd := &ts.cpu.bound
	*bd = cpuBound{li: -1, l: &m.TowerDense, own: ts.cpu.dev.Own(f), n: tg.Lead + n, req: tr.req, pos: tr.posTab, pic: words}
	bd.io[0] = unsafe.Pointer(&out[0])
	ts.cpuRunForm(bd, f)
	return out[:3*sq.Pw*sq.Ph], sq
}

// TestQwen2VLTowerMatchesTransformers holds Qwen2-VL's stated tower to
// transformers' own classes on synth-qwen2vl (scripts/qwenvlgold.py): the
// picture op on a picture the processor resizes -- its own smart_resize
// under the golden's bounds and Pillow's bicubic filter, normalised -- to
// every sample, and the tower's rows for the prompt's picture, the patches
// in the merge windows' order and turned by the grid's rotary, to the
// reference's -- the reference run in the tower's own arithmetic, its
// matrices at the converter's Q8_0 and every matrix's input at int8 per
// block (embd_a8), at the bound the Qwen tower was held to. Each part has its
// violation: a picture resized to the rounded size without the area's
// bound, and the patches in raster order.
func TestQwen2VLTowerMatchesTransformers(t *testing.T) {
	for _, name := range qvlTowers() {
		t.Run(name, func(t *testing.T) { qwen2VLTowerMatchesTransformers(t, name) })
	}
}

// qvlTowers are the Qwen2-VL family fixtures whose towers the converter
// states: Qwen2-VL's, and Qwen2.5-VL's with its windows.
func qvlTowers() []string {
	return []string{"synth-qwen2vl", "synth-qwen25vl", "synth-qwen35vl", "synth-qwen3vl", "synth-qwen3vlmoe"}
}

func qwen2VLTowerMatchesTransformers(t *testing.T, name string) {
	m, g, openOpts := openQVLTower(t, name)
	defer m.Close()
	tg := m.Tower
	if tg == nil || tg.GridDiv != 2 || tg.Fit.Factor != 2*tg.Patch || tg.Table == nil || len(tg.Classes) < 2 {
		t.Fatalf("the tower is %+v: want 2x2 merge windows, a size of its own in multiples of one, a rotary", tg)
	}
	s := testInstance(t, m, 64, openOpts...)
	defer s.Close()

	tg.Fit.MinPix, tg.Fit.MaxPix = g.Prep.Min, g.Prep.Max
	px, sq := towerPixels(t, s, rgbPicture(g.Prep.RGB, g.Prep.W, g.Prep.H))
	if sq.Pw != g.Prep.OutW || sq.Ph != g.Prep.OutH {
		t.Fatalf("a %dx%d picture resized to %dx%d, the processor's %dx%d", g.Prep.W, g.Prep.H, sq.Pw, sq.Ph, g.Prep.OutW, g.Prep.OutH)
	}
	if len(px) != len(g.Prep.PX) {
		t.Fatalf("%d samples, the processor's %d", len(px), len(g.Prep.PX))
	}
	// The golden is rounded to five places; one step of an 8-bit sample is
	// about 0.0145 after normalising, so a sample off by one is far past it.
	differ := 0
	for i, v := range px {
		if !(math.Abs(float64(v-g.Prep.PX[i])) < 2e-4) {
			differ++
		}
	}
	if differ > 0 {
		t.Fatalf("%d of %d samples differ from the processor's", differ, len(px))
	}
	t.Logf("a %dx%d picture resized to %dx%d: every one of %d samples the processor's", g.Prep.W, g.Prep.H, sq.Pw, sq.Ph, len(px))
	tg.Fit.MaxPix = 1 << 30
	if _, bad := towerPixels(t, s, rgbPicture(g.Prep.RGB, g.Prep.W, g.Prep.H)); bad.Pw == g.Prep.OutW && bad.Ph == g.Prep.OutH {
		t.Fatal("VIOLATION: a resize with no bound on the area reaches the processor's size")
	}
	// The golden encodes its picture at its own size (do_resize=False).
	tg.Fit.MinPix, tg.Fit.MaxPix = g.Image.W*g.Image.H, g.Image.W*g.Image.H

	img := rgbPicture(g.Image.RGB, g.Image.W, g.Image.H)
	if gr := tg.SpanGrid(g.Image.W, g.Image.H); gr != (model.ImageGrid{T: g.Image.Grid[0], H: g.Image.Grid[1], W: g.Image.Grid[2]}) {
		t.Fatalf("the picture's grid is %+v, transformers' %v", gr, g.Image.Grid)
	}
	rows, err := s.EncodePicture(img)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != len(g.Image.EmbdA8) {
		t.Fatalf("%d values, transformers' tower %d", len(rows), len(g.Image.EmbdA8))
	}
	// Its taps' rows (Qwen3-VL's deepstack) are held as its head's are.
	if deep := s.EncodedTaps(); len(deep) != len(g.Image.DeepA8) {
		t.Fatalf("%d tap values, transformers' deepstack %d", len(deep), len(g.Image.DeepA8))
	} else if len(deep) > 0 {
		dn, dw := nmseOf(deep, g.Image.DeepA8)
		t.Logf("the tower's %d taps: NMSE %.3e, max|d| %.3g at the engine's arithmetic", len(m.Tower.Taps), dn, dw)
		if math.IsNaN(dn) || dn > 2e-3 {
			t.Fatalf("the taps' rows sit at NMSE %.3e from transformers' deepstack at the engine's arithmetic", dn)
		}
	}
	nmse, worst := nmseOf(rows, g.Image.EmbdA8)
	f32, _ := nmseOf(rows, g.Image.Embd)
	arith, _ := nmseOf(g.Image.EmbdA8, g.Image.Embd)
	t.Logf("the tower's %d rows: NMSE %.3e, max|d| %.3g at the engine's arithmetic (%.3e against the f32 tower, "+
		"where the arithmetic alone moves transformers by %.3e)", len(rows)/int(m.Container.Vision().ProjDim), nmse, worst, f32, arith)
	if !(nmse < 2e-3) {
		t.Fatalf("the tower's rows are NMSE %.3e from transformers' (bound 2e-3)", nmse)
	}
	clean := append([]float32(nil), rows...)
	// The violation: a tower whose patches and rotary lie in raster order.
	rec := m.Container.Vision().Tower
	was := rec.Entry.Ops[0].Width
	rec.Entry.Ops[0].Width = 0
	raster, err := m.BuildTowerGraphs()
	rec.Entry.Ops[0].Width = was
	if err != nil {
		t.Fatal(err)
	}
	m.Tower = raster
	rs := testInstance(t, m, 64, openOpts...)
	bad, err := rs.EncodePicture(img)
	rs.Close()
	m.Tower = tg
	if err != nil {
		t.Fatal(err)
	}
	nb, _ := nmseOf(bad, clean)
	if !(nb > 100*math.Max(nmse, 1e-12)) {
		t.Fatalf("VIOLATION: the patches in raster order read NMSE %.3e against a clean %.3e", nb, nmse)
	}
	t.Logf("violation (raster order): NMSE %.3e", nb)
}

// TestQwen2VLMatchesTransformers runs the golden's prompt -- its text, the
// picture through the stated tower, the text after it -- and its
// continuation, and holds every row's logits to transformers'
// Qwen2VLForConditionalGeneration: the tower's rows, the M-RoPE grid they
// stand on and the text model's turn of them. The reference's tower is f32
// and the engine's runs Q8_0 matrices on int8 activations; the text model is
// f32 on both sides (the M-RoPE gates hold it at 1e-12), so what the logits
// may move is the text model's response to a tower error of the engine's
// size. One draw of that error does not bound a text model that answers
// discontinuously -- a mixture's routing, a hybrid's near-tie argmax answer
// two perturbations of one size very differently -- so the bound is twice
// the worst of nine: transformers' own tower at the engine's arithmetic
// (embd_a8), and the f32 tower's rows perturbed eight times by noise the
// size of the engine tower's error against them; no more argmaxes flip than
// the worst of those flips. The tower's rows themselves are held to the
// reference by TestQwen2VLTowerMatchesTransformers. The violations are the
// picture's rows laid out as text and, with taps, the taps zeroed.
func TestQwen2VLMatchesTransformers(t *testing.T) {
	for _, name := range qvlTowers() {
		t.Run(name, func(t *testing.T) { qwen2VLMatchesTransformers(t, name) })
	}
}

func qwen2VLMatchesTransformers(t *testing.T, name string) {
	m, g, openOpts := openQVLTower(t, name)
	defer m.Close()
	img := rgbPicture(g.Image.RGB, g.Image.W, g.Image.H)
	// The golden encodes the picture at its own size (do_resize=False), which
	// the published bounds need not admit (Qwen3-VL's floor is 65536 pixels).
	m.Tower.Fit.MinPix, m.Tower.Fit.MaxPix = g.Image.W*g.Image.H, g.Image.W*g.Image.H
	picture, err := m.PictureSpan(img)
	if err != nil {
		t.Fatal(err)
	}
	run := func(sp model.Span) [][]float32 {
		t.Helper()
		s := testInstance(t, m, len(g.Pre)+sp.N(m.Cfg.NEmbd)+len(g.Post)+len(g.Cont)+1, openOpts...)
		defer s.Close()
		lg, err := s.PrefillMixed(model.Span{Tokens: g.Pre}, sp, model.Span{Tokens: g.Post})
		if err != nil {
			t.Fatal(err)
		}
		out := [][]float32{append([]float32(nil), lg...)}
		for _, id := range g.Cont[:len(g.Cont)-1] {
			if lg, err = s.Forward(id); err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
		}
		return out
	}
	worst := func(rows [][]float32) (float64, int) {
		w, flips := 0.0, 0
		for i, lg := range rows {
			n := llama4Cmp(g.Logits[i].Head, lg)
			if math.IsNaN(n) {
				return math.Inf(1), len(rows)
			}
			w = math.Max(w, n)
			if Greedy(lg) != g.Logits[i].Argmax {
				flips++
			}
		}
		return w, flips
	}
	// The engine tower's error against the f32 tower's rows and taps.
	s := testInstance(t, m, 64, openOpts...)
	rows, err := s.EncodePicture(img)
	if err != nil {
		t.Fatal(err)
	}
	rows, deep := slices.Clone(rows), slices.Clone(s.EncodedTaps())
	s.Close()
	e, _ := nmseOf(rows, g.Image.Embd)
	ed := 0.0
	if len(g.Image.Deep) > 0 {
		ed, _ = nmseOf(deep, g.Image.Deep)
	}
	// What a tower error of that size moves the logits by, and the argmaxes
	// it flips, at its worst over nine draws.
	arith, aflips := worst(run(model.Span{Embd: g.Image.EmbdA8, Deep: g.Image.DeepA8, Grid: picture.Grid}))
	a8 := arith
	for k := range 8 {
		rng := rand.New(rand.NewPCG(uint64(k), 0x9e3779b97f4a7c15))
		n, f := worst(run(model.Span{Embd: perturbed(rng, g.Image.Embd, e), Deep: perturbed(rng, g.Image.Deep, ed), Grid: picture.Grid}))
		arith, aflips = math.Max(arith, n), max(aflips, f)
	}
	clean, flips := worst(run(picture))
	t.Logf("%d rows: worst NMSE %.3e against transformers, %d argmax flips; a tower error of the engine's size (%.3e) moves them up to %.3e, flipping up to %d (the reference's int8 draw %.3e)",
		len(g.Logits), clean, flips, e, arith, aflips, a8)
	if !(arith > 0) || !(clean < 2*arith) || flips > aflips {
		t.Fatalf("worst NMSE %.3e against transformers (bound twice the worst draw's %.3e), %d argmax flips where the draws flip up to %d",
			clean, arith, flips, aflips)
	}
	// Each violation fails the gate by a margin: ten times past its bound.
	bound := 2 * arith
	text := picture
	text.Grid = model.ImageGrid{}
	bad, _ := worst(run(text))
	if !(bad > 10*bound) {
		t.Fatalf("VIOLATION: the picture laid out as text reads %.3e against the bound %.3e", bad, bound)
	}
	t.Logf("violation (the picture as text): %.3e, %.0fx the bound", bad, bad/bound)
	if len(g.Image.DeepA8) > 0 {
		// The taps are load-bearing: the arithmetic arm with them zeroed.
		none, _ := worst(run(model.Span{Embd: g.Image.EmbdA8, Deep: make([]float32, len(g.Image.DeepA8)), Grid: picture.Grid}))
		if !(none > 10*bound) {
			t.Fatalf("VIOLATION: the taps zeroed read %.3e against the bound %.3e", none, bound)
		}
		t.Logf("violation (the taps zeroed): %.3e, %.0fx the bound", none, none/bound)
	}
}

// TestQwen2VLSeesThePicture is the real checkpoints: Qwen2-VL-2B-Instruct
// and Qwen2.5-VL-3B-Instruct, its windowed tower, name the colours of the
// quadrant picture and fewer in a blank one.
func TestQwen2VLSeesThePicture(t *testing.T) {
	for _, c := range []struct{ name, text, mm string }{
		{"qwen2vl-2b", "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", "mmproj-Qwen2-VL-2B-Instruct-f16.gguf"},
		{"qwen25vl-3b", "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"},
		{"qwen35vl-0.8b", "qwen35/Qwen3.5-0.8B-Q4_K_M.gguf", "mmproj-Qwen3.5-0.8B-f16.gguf"},
		{"qwen3vl-2b", "Qwen3-VL-2B-Instruct-f16.gguf", "mmproj-Qwen3-VL-2B-Instruct-f16.gguf"},
	} {
		t.Run(c.name, func(t *testing.T) { qwen2VLSeesThePicture(t, testmodels.Path(c.text), testmodels.Path(c.mm)) })
	}
}

func qwen2VLSeesThePicture(t *testing.T, text, mm string) {
	if !sourcePresent(text, mm) {
		testmodels.Missing(t, "MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", text, mm)
	}
	m, err := model.Open(jlmOfPair(t, text, mm))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	img := image.NewRGBA(image.Rect(0, 0, 448, 336))
	for y := range 336 {
		for x := range 448 {
			c := color.RGBA{255, 0, 0, 255}
			switch {
			case x >= 224 && y < 168:
				c = color.RGBA{0, 255, 0, 255}
			case x < 224 && y >= 168:
				c = color.RGBA{0, 0, 255, 255}
			case x >= 224 && y >= 168:
				c = color.RGBA{255, 255, 0, 255}
			}
			img.Set(x, y, c)
		}
	}
	seesThePicture(t, m, img)
}

// perturbed is x plus Gaussian noise whose expected NMSE against x is e; nil
// for an empty x.
func perturbed(rng *rand.Rand, x []float32, e float64) []float32 {
	if len(x) == 0 {
		return nil
	}
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	sigma := math.Sqrt(e * ss / float64(len(x)))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v + float32(sigma*rng.NormFloat64())
	}
	return out
}
