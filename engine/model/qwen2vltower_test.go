package model

import (
	"image"
	"image/color"
	"math"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// openQVLTower opens a Qwen2-VL family fixture -- the text GGUF and its
// mmproj, one container -- and its golden.
func openQVLTower(t *testing.T, name string) (*Model, *qvlGolden) {
	t.Helper()
	g := qvlGoldenOf(t, name)
	text, proj := testmodels.Path(name+".gguf"), testmodels.Path("mmproj-"+name+".gguf")
	if !sourcePresent(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (set JITLLM_MODELS) -- run scripts/qwenvlgold.py %s (RULE 11)",
			text, name)
	}
	m, err := Open(jlmOfPair(t, text, proj), WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	return m, g
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
func towerPixels(t *testing.T, s *Instance, img image.Image) ([]float32, square) {
	t.Helper()
	if err := s.towerReady(); err != nil {
		t.Fatal(err)
	}
	tr, m := s.tower, s.m
	tg, ts := m.tower, tr.s
	words, pl, err := pictureWords(img, &tr.pad, &tr.ent)
	if err != nil {
		t.Fatal(err)
	}
	pl.request(tr.req)
	sq, squares := tg.square(pl.w, pl.h, 0)
	cl := tg.classes[sq.class]
	a := cl.pic
	a.Pixels = pl.px
	tr.squareRequest(pl, sq, a, squares)
	P := tg.patch
	b := graph.NewBlock(-1)
	b.Out(b.Emit(graph.OpPicture, nil, []graph.Result{{Name: "pixels", Width: 3 * P * P}}, nil, a, nil, nil)[0])
	f, err := m.formCPUGraph(ts, b.G, cl.rows, nil)
	if err != nil {
		t.Fatal(err)
	}
	n := sq.gw * sq.gh
	tr.size(tg.lead+n, int(m.container.Vision().NEmbd))
	tr.order(tg.lead, sq.gw, sq.gh, tg.merge)
	out := make([]float32, cl.rows*3*P*P)
	bd := &ts.cpu.bound
	*bd = cpuBound{li: -1, l: &m.towerDense, own: ts.cpu.dev.Own(f), n: tg.lead + n, req: tr.req, pos: tr.posTab, pic: words}
	bd.io[0] = unsafe.Pointer(&out[0])
	ts.cpuRunForm(bd, f)
	return out[:3*sq.pw*sq.ph], sq
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
	m, g := openQVLTower(t, "synth-qwen2vl")
	defer m.Close()
	tg := m.tower
	if tg == nil || tg.fit.factor != 28 || tg.merge != 2 || tg.gridDiv != 2 || tg.table == nil || len(tg.classes) < 2 {
		t.Fatalf("the tower is %+v: want a size of its own in multiples of 28, 2x2 merge windows, a rotary", tg)
	}
	s := m.NewInstance(64)
	defer s.Close()

	minPix, maxPix := tg.fit.minPix, tg.fit.maxPix
	tg.fit.minPix, tg.fit.maxPix = g.Prep.Min, g.Prep.Max
	px, sq := towerPixels(t, s, rgbPicture(g.Prep.RGB, g.Prep.W, g.Prep.H))
	if sq.pw != g.Prep.OutW || sq.ph != g.Prep.OutH {
		t.Fatalf("a %dx%d picture resized to %dx%d, the processor's %dx%d", g.Prep.W, g.Prep.H, sq.pw, sq.ph, g.Prep.OutW, g.Prep.OutH)
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
	t.Logf("a %dx%d picture resized to %dx%d: every one of %d samples the processor's", g.Prep.W, g.Prep.H, sq.pw, sq.ph, len(px))
	tg.fit.maxPix = 1 << 30
	if _, bad := towerPixels(t, s, rgbPicture(g.Prep.RGB, g.Prep.W, g.Prep.H)); bad.pw == g.Prep.OutW && bad.ph == g.Prep.OutH {
		t.Fatal("VIOLATION: a resize with no bound on the area reaches the processor's size")
	}
	tg.fit.minPix, tg.fit.maxPix = minPix, maxPix

	img := rgbPicture(g.Image.RGB, g.Image.W, g.Image.H)
	if gr := tg.spanGrid(g.Image.W, g.Image.H); gr != (ImageGrid{T: g.Image.Grid[0], H: g.Image.Grid[1], W: g.Image.Grid[2]}) {
		t.Fatalf("the picture's grid is %+v, transformers' %v", gr, g.Image.Grid)
	}
	rows, err := s.EncodePicture(img)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != len(g.Image.EmbdA8) {
		t.Fatalf("%d values, transformers' tower %d", len(rows), len(g.Image.EmbdA8))
	}
	nmse, worst := nmseOf(rows, g.Image.EmbdA8)
	f32, _ := nmseOf(rows, g.Image.Embd)
	arith, _ := nmseOf(g.Image.EmbdA8, g.Image.Embd)
	t.Logf("the tower's %d rows: NMSE %.3e, max|d| %.3g at the engine's arithmetic (%.3e against the f32 tower, "+
		"where the arithmetic alone moves transformers by %.3e)", len(rows)/int(m.container.Vision().ProjDim), nmse, worst, f32, arith)
	if !(nmse < 2e-3) {
		t.Fatalf("the tower's rows are NMSE %.3e from transformers' (bound 2e-3)", nmse)
	}
	clean := append([]float32(nil), rows...)
	tg.merge = 0
	bad, err := s.EncodePicture(img)
	tg.merge = 2
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
// and the engine's runs Q8_0 matrices on int8 activations, which moves the
// rows by what TestQwen2VLTowerMatchesTransformers logs; the text model is
// f32 on both sides (the M-RoPE gates hold it at 1e-12), so the bound is
// that movement's, with no argmax flipped. The violation is the picture's
// rows laid out as text.
func TestQwen2VLMatchesTransformers(t *testing.T) {
	m, g := openQVLTower(t, "synth-qwen2vl")
	defer m.Close()
	img := rgbPicture(g.Image.RGB, g.Image.W, g.Image.H)
	run := func(grid bool) [][]float32 {
		t.Helper()
		sp, err := m.PictureSpan(img)
		if err != nil {
			t.Fatal(err)
		}
		if !grid {
			sp.Grid = ImageGrid{}
		}
		s := m.NewInstance(len(g.Pre) + sp.PicRows + len(g.Post) + len(g.Cont) + 1)
		defer s.Close()
		lg, err := s.PrefillMixed(Span{Tokens: g.Pre}, sp, Span{Tokens: g.Post})
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
	clean, flips := worst(run(true))
	t.Logf("%d rows: worst NMSE %.3e against transformers, %d argmax flips", len(g.Logits), clean, flips)
	if !(clean < 5e-3) || flips > 0 {
		t.Fatalf("worst NMSE %.3e against transformers (bound 5e-3), %d argmax flips", clean, flips)
	}
	bad, _ := worst(run(false))
	if !(bad > 100*math.Max(clean, 1e-12)) {
		t.Fatalf("VIOLATION: the picture laid out as text reads %.3e against a clean %.3e", bad, clean)
	}
	t.Logf("violation (the picture as text): %.3e", bad)
}

// TestQwen2VLSeesThePicture is the real checkpoint: Qwen2-VL-2B-Instruct
// names the colours of the quadrant picture and none in the blank one.
func TestQwen2VLSeesThePicture(t *testing.T) {
	text := testmodels.Path("Qwen2-VL-2B-Instruct-Q4_K_M.gguf")
	mm := testmodels.Path("mmproj-Qwen2-VL-2B-Instruct-f16.gguf")
	if !sourcePresent(text, mm) {
		t.Skipf("MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", text, mm)
	}
	m, err := Open(jlmOfPair(t, text, mm))
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
