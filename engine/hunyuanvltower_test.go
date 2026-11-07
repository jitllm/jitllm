package engine

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"
)

// TestHunyuanVLTowerMatchesTransformers holds HunyuanVL's stated tower to
// transformers' own classes on synth-hunyuanvl (scripts/hunyuanvlgold.py):
// the picture op on a picture the processor resizes, to every sample; the
// picture's grid, the newline column and the two end rows with it; and the
// tower's rows -- the merged grid's with a newline after each row, between
// the begin and end rows -- to the reference's at the engine's arithmetic
// (embd_a8). Each part has its violation: a resize with no bound on the
// area, and the begin and end rows swapped.
func TestHunyuanVLTowerMatchesTransformers(t *testing.T) {
	m, g, openOpts := openQVLTower(t, "synth-hunyuanvl")
	defer m.Close()
	tg := m.Tower
	if tg == nil || tg.GridDiv != 2 || tg.Line != 1 || tg.Ends != 1 || tg.Frame != 2 || tg.Fit.Factor != 2*tg.Patch {
		t.Fatalf("the tower is %+v: want a grid over 2x2 windows with a newline column and two end rows", tg)
	}
	s := testInstance(t, m, 64, openOpts...)
	defer s.Close()

	tg.Fit.MinPix, tg.Fit.MaxPix = g.Prep.Min, g.Prep.Max
	px, sq := towerPixels(t, s, rgbPicture(g.Prep.RGB, g.Prep.W, g.Prep.H))
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
	t.Logf("a %dx%d picture resized to %dx%d: every one of %d samples the processor's", g.Prep.W, g.Prep.H, sq.Pw, sq.Ph, len(px))
	tg.Fit.MaxPix = 1 << 30
	if _, bad := towerPixels(t, s, rgbPicture(g.Prep.RGB, g.Prep.W, g.Prep.H)); bad.Pw == g.Prep.OutW && bad.Ph == g.Prep.OutH {
		t.Fatal("VIOLATION: a resize with no bound on the area reaches the processor's size")
	}
	// The golden encodes its picture at its own size (do_resize=False).
	tg.Fit.MinPix, tg.Fit.MaxPix = g.Image.W*g.Image.H, g.Image.W*g.Image.H

	want := model.ImageGrid{T: g.Image.Grid[0], H: g.Image.Grid[1], W: g.Image.Grid[2], Line: 1, Ends: 1}
	if gr := tg.SpanGrid(g.Image.W, g.Image.H); gr != want {
		t.Fatalf("the picture's grid is %+v, the reference's %+v", gr, want)
	}
	img := rgbPicture(g.Image.RGB, g.Image.W, g.Image.H)
	rows, err := s.EncodePicture(img)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != len(g.Image.EmbdA8) || len(rows) != want.Rows()*int(m.Container.Vision().ProjDim) {
		t.Fatalf("%d values, transformers' tower %d (%d rows)", len(rows), len(g.Image.EmbdA8), want.Rows())
	}
	nmse, worst := nmseOf(rows, g.Image.EmbdA8)
	f32, _ := nmseOf(rows, g.Image.Embd)
	t.Logf("the tower's %d rows: NMSE %.3e, max|d| %.3g at the engine's arithmetic (%.3e against the f32 tower)",
		want.Rows(), nmse, worst, f32)
	if !(nmse < 2e-3) {
		t.Fatalf("the tower's rows are NMSE %.3e from transformers' (bound 2e-3)", nmse)
	}
	clean := append([]float32(nil), rows...)
	// The violation: the begin and end rows swapped.
	rec := m.Container.Vision().Tower
	var wrap *jlm.TowerOp
	for i := range rec.Head.Ops {
		if rec.Head.Ops[i].Code == jlm.TOpWrapRows {
			wrap = &rec.Head.Ops[i]
		}
	}
	wrap.Bind[0], wrap.Bind[1] = wrap.Bind[1], wrap.Bind[0]
	swapped, err := m.BuildTowerGraphs()
	wrap.Bind[0], wrap.Bind[1] = wrap.Bind[1], wrap.Bind[0]
	if err != nil {
		t.Fatal(err)
	}
	m.Tower = swapped
	rs := testInstance(t, m, 64, openOpts...)
	bad, err := rs.EncodePicture(img)
	rs.Close()
	m.Tower = tg
	if err != nil {
		t.Fatal(err)
	}
	nb, _ := nmseOf(bad, clean)
	if !(nb > 100*math.Max(nmse, 1e-12)) {
		t.Fatalf("VIOLATION: the begin and end rows swapped read NMSE %.3e against a clean %.3e", nb, nmse)
	}
	t.Logf("violation (begin and end rows swapped): NMSE %.3e", nb)
}

// TestHunyuanVLAndKimiVLMatchTransformers runs each fixture's golden prompt
// with its picture through the stated tower and holds the logits to the
// reference's, as TestQwen2VLMatchesTransformers does the Qwen family's:
// HunyuanVL's rows on the XD-RoPE grid with its newline column and end
// rows, Kimi-VL's advancing as text over DeepSeek-V3's model.
func TestHunyuanVLAndKimiVLMatchTransformers(t *testing.T) {
	for _, name := range []string{"synth-hunyuanvl", "synth-kimivl"} {
		t.Run(name, func(t *testing.T) { qwen2VLMatchesTransformers(t, name) })
	}
}
