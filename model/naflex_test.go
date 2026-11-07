package model

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestNaflexSizeMatchesProcessor holds the NaFlex resize (jlm.PreNaFlex) to
// transformers' get_image_size_for_max_num_patches on pictures below the
// least patch count, inside the range and above the most
// (scripts/phi4visiongold.py --model-free); the count unclamped is the
// violation.
func TestNaflexSizeMatchesProcessor(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "phi4-naflex.json"))
	if err != nil {
		t.Fatalf("GOLDEN MISSING: %v (scripts/phi4visiongold.py --model-free)", err)
	}
	var g struct {
		Patch, Lo, Hi int
		Sizes         []struct{ W, H, TW, TH int }
	}
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	if len(g.Sizes) == 0 {
		t.Fatal("the golden holds no sizes")
	}
	p := g.Patch
	fit := PictureFit{Factor: p, MinPix: g.Lo * p * p, MaxPix: g.Hi * p * p, naflex: true}
	clamped := 0
	for _, s := range g.Sizes {
		tw, th := fit.smartSize(s.W, s.H)
		if tw != s.TW || th != s.TH {
			t.Errorf("%dx%d resizes to %dx%d, the processor's %dx%d", s.W, s.H, tw, th, s.TW, s.TH)
		}
		if n := (s.W / p) * (s.H / p); n < g.Lo || n > g.Hi {
			clamped++
		}
	}
	if clamped == 0 {
		t.Fatal("no golden picture is outside the patch range: the clamp was never exercised")
	}
	free := PictureFit{Factor: p, MinPix: p * p, MaxPix: 1 << 30, naflex: true}
	same := true
	for _, s := range g.Sizes {
		tw, th := free.smartSize(s.W, s.H)
		same = same && tw == s.TW && th == s.TH
	}
	if same {
		t.Fatal("VIOLATION: the count unclamped resizes every picture as the processor does")
	}
	t.Logf("%d sizes, every one the processor's (%d clamped); unclamped, they part", len(g.Sizes), clamped)
}
