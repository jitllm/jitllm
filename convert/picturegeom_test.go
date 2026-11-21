package convert

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"
)

// TestTileGeometryFollowsTheReference holds the tile geometry to InternVL's
// find_closest_aspect_ratio (transformers' get_optimal_tiled_canvas), each
// grid worked by hand from the reference's rule over 448-pixel tiles, 1 to
// 12 of them: the closest aspect, a tie to the larger grid only while the
// picture covers more than half of it; then the tiles cut row-major and the
// thumbnail after a grid of more than one.
func TestTileGeometryFollowsTheReference(t *testing.T) {
	g := ruleGeometry(t, picContext{side: 32}, pictureStep(t, "resize", 448, 448, 0, 0), pictureStep(t, "tile", 1, 12, 1))
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
	if _, n, err := model.GeometrySquare(ruleGeometry(t, picContext{side: 32}, pictureStep(t, "resize", 448, 448, 0, 0),
		pictureStep(t, "tile", 1, 12, 0)), 1000, 500, 0); err != nil || n != 2 {
		t.Errorf("1000x500 without a thumbnail is %d squares (%v), want 2", n, err)
	}
}

// TestNaflexGeometryMatchesProcessor holds the NaFlex geometry (naflex)
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
	g := ruleGeometry(t, picContext{}, pictureStep(t, "naflex", int32(p), int32(gold.Lo), int32(gold.Hi), 0, 0))
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
	free := ruleGeometry(t, picContext{}, pictureStep(t, "naflex", int32(p), 1, int32((1<<30)/(p*p)), 0, 0))
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

// TestPictureRulesAreTheTowers holds the library's picture rules to its
// towers: the rules check, every rule is the one some tower's steps (or its
// video's) take, and a library that breaks the rules' contract is refused
// -- a rule reading an undeclared step, two rules reading the same steps, a
// processor's step of no code, an expression outside the language.
func TestPictureRulesAreTheTowers(t *testing.T) {
	l, err := pictureLibrary()
	if err != nil {
		t.Fatal(err)
	}
	projs, err := projectorLibrary()
	if err != nil {
		t.Fatal(err)
	}
	read := map[string]bool{}
	for _, r := range l.Rules {
		for _, k := range r.Reads {
			read[k] = true
		}
	}
	taken := map[string]bool{}
	take := func(name string, pre []mjPreStep) {
		var kinds []string
		for _, s := range pre {
			if read[s.Kind] {
				kinds = append(kinds, s.Kind)
			}
		}
		slices.Sort(kinds)
		for rn, r := range l.Rules {
			if slices.Equal(slices.Sorted(slices.Values(r.Reads)), kinds) {
				taken[rn] = true
				return
			}
		}
		t.Errorf("%s's steps %v are no rule's", name, kinds)
	}
	for name, p := range projs {
		if p.Tower == nil {
			continue
		}
		take(name, p.Tower.Pre)
		if v := p.Tower.Video; v != nil && len(v.Pre) > 0 {
			take(name+"'s video", v.Pre)
		}
	}
	for rn := range l.Rules {
		if !taken[rn] {
			t.Errorf("rule %s is no tower's", rn)
		}
	}
	for name, edit := range map[string]func(l *mjPictures){
		"an undeclared step": func(l *mjPictures) { l.Rules["fixed"].Reads = []string{"resize", "squash"} },
		"two rules of one read": func(l *mjPictures) {
			again := *l.Rules["fixed"]
			l.Rules["fixed_again"] = &again
		},
		"a step of no code":    func(l *mjPictures) { l.Steps["tile"].Code = 0 },
		"no expression":        func(l *mjPictures) { l.Rules["fixed"].Count = "1 +" },
		"no field of a square": func(l *mjPictures) { l.Rules["fixed"].Square["width"] = "1" },
	} {
		v, err := pictureLibrary()
		if err != nil {
			t.Fatal(err)
		}
		edit(v)
		if v.check() == nil {
			t.Errorf("VIOLATION: a library of %s checks", name)
		}
	}
	t.Logf("%d rules, each a tower's", len(taken))
}

// TestANewProcessorIsALibraryEntry states a processor no family has -- the
// picture halved, each side whole units, at least one -- by a library entry
// alone: its step and its rule. Its squares are what the rule says, and its
// steps lower to a live resize before the normalisation; the violation, the
// entry's rule taken away, refuses the step.
func TestANewProcessorIsALibraryEntry(t *testing.T) {
	l, err := pictureLibrary()
	if err != nil {
		t.Fatal(err)
	}
	l.Steps["halve"] = &mjPicStep{Code: 200, Ints: []string{"unit", "filter"}}
	l.Rules["halve"] = &mjPicRule{
		Doc:   "The picture halved, each side floored to whole units, at least one, read in the entry's patches.",
		Reads: []string{"halve"},
		Bind: []mjPicBind{{Let: "ow", Value: "max(floor(w / 2 / halve.unit), 1) * halve.unit"},
			{Let: "oh", Value: "max(floor(h / 2 / halve.unit), 1) * halve.unit"}},
		Count:  "1",
		Square: map[string]string{"ow": "ow", "oh": "oh", "gw": "floor(ow / patch)", "gh": "floor(oh / patch)"},
		Bounds: mjPicBounds{Unit: "halve.unit", MaxPix: "widest * widest / 4", MaxPatches: "floor(widest * widest / 4 / (patch * patch))"},
		Lower:  []mjPreStep{{Kind: "resize_live", Ints: []string{"halve.filter", "prec.pil"}}},
	}
	if err := l.check(); err != nil {
		t.Fatal(err)
	}
	steps := []jlm.PreStep{{Kind: 200, Ints: []int32{28, int32(jlm.FilterPILBicubic)}},
		{Kind: jlm.PreNormalize, Floats: []float32{0.5, 0.5, 0.5, 0.5, 0.5, 0.5}}}
	g, pre, err := l.picture(steps, picContext{patch: 14})
	if err != nil {
		t.Fatal(err)
	}
	sq, n, err := model.GeometrySquare(g, 1000, 600, 0)
	if err != nil || n != 1 || sq.Ow != 476 || sq.Oh != 280 || sq.Gw != 34 || sq.Gh != 20 || sq.Pw != 476 || sq.Cw != 476 {
		t.Errorf("1000x600 halved is %d squares, the first %+v (%v), want one of 476x280 read as 34x20", n, sq, err)
	}
	if sq, _, err := model.GeometrySquare(g, 20, 20, 0); err != nil || sq.Ow != 28 || sq.Oh != 28 {
		t.Errorf("20x20 halved is %dx%d (%v), want one unit a side", sq.Ow, sq.Oh, err)
	}
	if len(pre) != 2 || pre[0].Kind != jlm.PreResizeLive || !slices.Equal(pre[0].Ints, []int32{int32(jlm.FilterPILBicubic), int32(jlm.PrecPIL)}) ||
		pre[1].Kind != jlm.PreNormalize {
		t.Errorf("the steps lower to %+v, want a live resize then the normalisation", pre)
	}
	delete(l.Rules, "halve")
	if _, _, err := l.picture(steps, picContext{patch: 14}); err == nil {
		t.Error("VIOLATION: a step no rule reads states a picture")
	}
}

// pictureStep is the picture step named name of ints.
func pictureStep(t *testing.T, name string, ints ...int32) jlm.PreStep {
	t.Helper()
	return jlm.PreStep{Kind: kindOf(t, name), Ints: ints}
}

// kindOf is the picture step named name's kind (PictureStep).
func kindOf(t *testing.T, name string) jlm.PreKind {
	t.Helper()
	k, err := PictureStep(name)
	if err != nil {
		t.Fatal(err)
	}
	return k
}

// ruleGeometry is the geometry the library's rule states for steps under c.
func ruleGeometry(t *testing.T, c picContext, steps ...jlm.PreStep) *jlm.PictureGeometry {
	t.Helper()
	l, err := pictureLibrary()
	if err != nil {
		t.Fatal(err)
	}
	g, _, err := l.picture(steps, c)
	if err != nil {
		t.Fatal(err)
	}
	return g
}
