package engine

import (
	"math"
	"testing"
)

// TestKimiVLTowerMatchesTransformers holds Kimi-VL's stated tower to
// Moonshot's own code on synth-kimivl (scripts/kimivlgold.py): the picture
// op on a picture past a lowered patch limit -- scaled down by Pillow's
// bicubic filter, padded black to whole 2x2 windows and normalised -- to
// every sample, and the tower's rows for the prompt's picture to the
// reference's at the engine's arithmetic (embd_a8). Each part has its
// violation: the limit doubled, which keeps the picture's own size, and the
// rotary's column and row pairs swapped.
func TestKimiVLTowerMatchesTransformers(t *testing.T) {
	m, g, openOpts := openQVLTower(t, "synth-kimivl")
	defer m.Close()
	tg := m.Tower
	if tg == nil || tg.GridDiv != 0 || tg.Fit.Limit == 0 || tg.Fit.Pad != 2*tg.Patch || len(tg.Classes) < 2 {
		t.Fatalf("the tower is %+v: want rows that advance as text and a picture under a patch limit, padded to 2x2 windows", tg)
	}
	s := testInstance(t, m, 64, openOpts...)
	defer s.Close()

	limit := tg.Fit.Limit
	tg.Fit.Limit = g.Prep.Max
	px, sq := towerPixels(t, s, rgbPicture(g.Prep.RGB, g.Prep.W, g.Prep.H))
	if sq.Gw*tg.Patch != g.Prep.OutW || sq.Gh*tg.Patch != g.Prep.OutH {
		t.Fatalf("a %dx%d picture padded to %dx%d, the processor's %dx%d", g.Prep.W, g.Prep.H,
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
	t.Logf("a %dx%d picture resized to %dx%d and padded to %dx%d: every one of %d samples the processor's",
		g.Prep.W, g.Prep.H, sq.Ow, sq.Oh, sq.Gw*tg.Patch, sq.Gh*tg.Patch, len(px))
	tg.Fit.Limit = 2 * g.Prep.Max
	if _, bad := towerPixels(t, s, rgbPicture(g.Prep.RGB, g.Prep.W, g.Prep.H)); bad.Gw*tg.Patch == g.Prep.OutW &&
		bad.Gh*tg.Patch == g.Prep.OutH {
		t.Fatal("VIOLATION: a limit doubled reaches the processor's size")
	}
	tg.Fit.Limit = limit

	img := rgbPicture(g.Image.RGB, g.Image.W, g.Image.H)
	rows, err := s.EncodePicture(img)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != len(g.Image.EmbdA8) {
		t.Fatalf("%d values, the reference's tower %d", len(rows), len(g.Image.EmbdA8))
	}
	nmse, worst := nmseOf(rows, g.Image.EmbdA8)
	f32, _ := nmseOf(rows, g.Image.Embd)
	t.Logf("the tower's %d rows: NMSE %.3e, max|d| %.3g at the engine's arithmetic (%.3e against the f32 tower)",
		len(rows)/int(m.Container.Vision().ProjDim), nmse, worst, f32)
	if !(nmse < 2e-3) {
		t.Fatalf("the tower's rows are NMSE %.3e from the reference's (bound 2e-3)", nmse)
	}
	clean := append([]float32(nil), rows...)
	// The violation: each column pair turned by the row and each row pair by
	// the column.
	rec := m.Container.Vision().Tower
	runs := append(rec.Block.Rope.Runs[:0:0], rec.Block.Rope.Runs...)
	for i := range rec.Block.Rope.Runs {
		rec.Block.Rope.Runs[i].Axis = 1 - rec.Block.Rope.Runs[i].Axis
	}
	swapped, err := m.BuildTowerGraphs()
	copy(rec.Block.Rope.Runs, runs)
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
		t.Fatalf("VIOLATION: the rotary's axes swapped read NMSE %.3e against a clean %.3e", nb, nmse)
	}
	t.Logf("violation (axes swapped): NMSE %.3e", nb)
}
