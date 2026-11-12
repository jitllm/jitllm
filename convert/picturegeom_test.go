package convert

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/model"
)

// TestTileGeometryFollowsTheReference holds the tile geometry to InternVL's
// find_closest_aspect_ratio (transformers' get_optimal_tiled_canvas), each
// grid worked by hand from the reference's rule over 448-pixel tiles, 1 to
// 12 of them: the closest aspect, a tie to the larger grid only while the
// picture covers more than half of it; then the tiles cut row-major and the
// thumbnail after a grid of more than one.
func TestTileGeometryFollowsTheReference(t *testing.T) {
	g := tileGeometry(1, 12, true, 448, 32)
	for _, c := range []struct{ w, h, cols, rows int }{
		{384, 384, 1, 1},  // (2,2) ties at 0 and is more than twice the picture
		{800, 600, 4, 3},  // 4:3 exactly, after (2,2) took the tie with (1,1)
		{1000, 500, 2, 1}, // (4,2) ties at 0 and is more than twice the picture
		{2000, 1000, 4, 2},
		{448, 896, 1, 2},
		{640, 480, 4, 3},
		{100, 1000, 1, 10}, // 1:10 exactly, before (1,11) and (1,12)
	} {
		sq, n, err := model.GeometrySquare(g, c.w, c.h, 0)
		want := c.cols * c.rows
		if want > 1 {
			want++
		}
		if err != nil || sq.Ow != c.cols*448 || sq.Oh != c.rows*448 || n != want {
			t.Errorf("%dx%d: a grid of %dx%d pixels in %d squares (%v), want %dx%d tiles and %d", c.w, c.h, sq.Ow, sq.Oh, n,
				err, c.cols, c.rows, want)
		}
	}
	if sq, _, err := model.GeometrySquare(g, 1000, 500, 1); err != nil || sq.Ow != 896 || sq.Oh != 448 || sq.X0 != 448 || sq.Y0 != 0 {
		t.Errorf("the second tile of 1000x500 is (%d, %d) of %dx%d (%v), want (448, 0) of 896x448", sq.X0, sq.Y0, sq.Ow, sq.Oh, err)
	}
	if sq, _, err := model.GeometrySquare(g, 1000, 500, 2); err != nil || sq.Ow != 448 || sq.Oh != 448 || sq.X0 != 0 || sq.Y0 != 0 {
		t.Errorf("1000x500's thumbnail is (%d, %d) of %dx%d (%v), want the whole picture at 448x448", sq.X0, sq.Y0, sq.Ow, sq.Oh, err)
	}
	if _, n, err := model.GeometrySquare(tileGeometry(1, 12, false, 448, 32), 1000, 500, 0); err != nil || n != 2 {
		t.Errorf("1000x500 without a thumbnail is %d squares (%v), want 2", n, err)
	}
}

// TestNaflexGeometryMatchesProcessor holds the NaFlex geometry (jlm.PreNaFlex)
// to transformers' get_image_size_for_max_num_patches on pictures below the
// least patch count, inside the range and above the most
// (scripts/phi4visiongold.py --model-free); the count unclamped is the
// violation.
func TestNaflexGeometryMatchesProcessor(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "phi4-naflex.json"))
	if err != nil {
		t.Fatalf("GOLDEN MISSING: %v (scripts/phi4visiongold.py --model-free)", err)
	}
	var gold struct {
		Patch, Lo, Hi int
		Sizes         []struct{ W, H, TW, TH int }
	}
	if err := json.Unmarshal(b, &gold); err != nil {
		t.Fatal(err)
	}
	if len(gold.Sizes) == 0 {
		t.Fatal("the golden holds no sizes")
	}
	p := gold.Patch
	g := naflexGeometry(p, gold.Lo, gold.Hi)
	clamped := 0
	for _, s := range gold.Sizes {
		sq, _, err := model.GeometrySquare(g, s.W, s.H, 0)
		if err != nil || sq.Ow != s.TW || sq.Oh != s.TH {
			t.Errorf("%dx%d resizes to %dx%d (%v), the processor's %dx%d", s.W, s.H, sq.Ow, sq.Oh, err, s.TW, s.TH)
		}
		if n := (s.W / p) * (s.H / p); n < gold.Lo || n > gold.Hi {
			clamped++
		}
	}
	if clamped == 0 {
		t.Fatal("no golden picture is outside the patch range: the clamp was never exercised")
	}
	free := naflexGeometry(p, 1, (1<<30)/(p*p))
	same := true
	for _, s := range gold.Sizes {
		sq, _, err := model.GeometrySquare(free, s.W, s.H, 0)
		same = same && err == nil && sq.Ow == s.TW && sq.Oh == s.TH
	}
	if same {
		t.Fatal("VIOLATION: the count unclamped resizes every picture as the processor does")
	}
	t.Logf("%d sizes, every one the processor's (%d clamped); unclamped, they part", len(gold.Sizes), clamped)
}
