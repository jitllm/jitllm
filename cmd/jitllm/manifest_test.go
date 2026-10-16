package main

import (
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// A manifest round-trips, the same variant shows no difference, and a changed
// recipe, a missing site and a moved placement each show one (RULE 10: the
// comparison has to fire).
func TestManifestNamesTheVariant(t *testing.T) {
	a := &runManifest{Version: 1, Args: []string{"-p=16", "m.jlm"}, Env: map[string]string{"JITLLM_GPU_FFN": "gate2"}}
	a.Model.HeadSHA256, a.Model.Size = "abc", 10
	a.Placed.DecodeBlocks = 16
	a.Recipes = []tier.RecipeChoice{
		{Device: "0:V100 [ptx]", Site: "ffn", Key: "k1", Recipe: "gate2in", Source: "timed"},
		{Device: "0:V100 [ptx]", Site: "actin-o", Key: "k2", Recipe: "two", Source: "cache"},
	}
	path := filepath.Join(t.TempDir(), "m.json")
	if err := a.write(path); err != nil {
		t.Fatal(err)
	}
	b, err := readManifest(path)
	if err != nil {
		t.Fatal(err)
	}
	if d := variantDiff(a, b); len(d) != 0 {
		t.Fatalf("a manifest differs from itself read back: %v", d)
	}
	b.Recipes[0].Recipe = "gate2"
	if d := variantDiff(a, b); len(d) != 1 {
		t.Fatalf("a changed recipe: %v", d)
	}
	b.Recipes = b.Recipes[:1]
	if d := variantDiff(a, b); len(d) != 2 {
		t.Fatalf("a changed recipe and a missing site: %v", d)
	}
	b.Placed.DecodeBlocks = 8
	if d := variantDiff(a, b); len(d) != 3 {
		t.Fatalf("and a moved seam: %v", d)
	}
	if _, err := readManifest(filepath.Join(t.TempDir(), "none.json")); err == nil {
		t.Fatal("a missing manifest read")
	}
}
