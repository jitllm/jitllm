package model

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// InternVL's vision half: InternViT-300M (24 blocks, 1024 wide, a class token,
// layer scales the converter folds into the matrices they follow), a 2x2
// pixel shuffle, then LayerNorm, linear, GELU, linear into the text model's
// width; images are cut into up to twelve 448 tiles plus a thumbnail.

func internvlText() string { return testmodels.Path("vlm/InternVL3-1B-Instruct-Q8_0.gguf") }
func internvlProj() string { return testmodels.Path("vlm/mmproj-InternVL3-1B-Instruct-Q8_0.gguf") }

func openInternVL(t testing.TB, opts ...Option) (*Model, *Tower) {
	t.Helper()
	if !vlmAvailable(internvlText(), internvlProj()) {
		testmodels.Missing(t, "%s and its mmproj %s (or the merged container) are missing. RULE 11: fetch "+
			"ggml-org/InternVL3-1B-Instruct-GGUF's Q8_0 text model and mmproj into %s and re-run",
			internvlText(), internvlProj(), testmodels.Dir())
	}
	m, err := Open(jlmOfPair(t, internvlText(), internvlProj()), opts...)
	if err != nil {
		t.Fatal(err)
	}
	tw := m.Tower()
	if tw == nil {
		m.Close()
		t.Fatal("the merged container carries no tower, so this gate proves nothing")
	}
	return m, tw
}

func TestInternVLTowerLoads(t *testing.T) {
	m, tw := openInternVL(t)
	defer m.Close()
	c := tw.Cfg
	for _, k := range []struct {
		name      string
		got, want int
	}{
		{"NLayer", c.NLayer, 24}, {"NEmbd", c.NEmbd, 1024}, {"NHead", c.NHead, 16},
		{"HeadDim", c.HeadDim, 64}, {"NFFN", c.NFFN, 4096},
		{"ImageSz", c.ImageSz, 448}, {"PatchSz", c.PatchSz, 14}, {"Scale", c.Scale, 2},
		{"Patches", c.Patches(), 1024}, {"Seq", c.Seq(), 1025}, {"Tokens", c.Tokens(), 256},
		{"ProjDim", c.ProjDim, 896}, {"projK", c.projK(), 4096},
		{"MinTiles", c.MinTiles, 1}, {"MaxTiles", c.MaxTiles, 12},
	} {
		if k.got != k.want {
			t.Errorf("%s = %d, want %d", k.name, k.got, k.want)
		}
	}
	if c.Projector != "internvl" || !c.CLS || c.PreLN || c.Rope {
		t.Errorf("projector %q CLS %v PreLN %v Rope %v; InternViT has a class token, "+
			"no pre-norm and a learned table", c.Projector, c.CLS, c.PreLN, c.Rope)
	}
	if tw.postLnW != nil || tw.projNorm == nil || tw.projNormB == nil || tw.projW2.e == nil {
		t.Errorf("post-norm %v, projector norm %v/%v, second matrix %v: InternVL's projector is "+
			"a biased LayerNorm and two matrices, after no post-norm",
			tw.postLnW != nil, tw.projNorm != nil, tw.projNormB != nil, tw.projW2.e != nil)
	}
}

// internvlPairs is b10825's InternVL dump: block 0's input, every block's
// output (after its folded layer scale), the projector's LayerNorm and its
// output.
func internvlPairs(nLayer int) []towerNodePair {
	ps := []towerNodePair{{dump: "pos_embed", ours: "pos_embed"}}
	for i := 0; i < nLayer; i++ {
		n := fmt.Sprintf("layer_out-%d", i)
		ps = append(ps, towerNodePair{dump: n, ours: n})
	}
	return append(ps, towerNodePair{"norm_b", "proj_in", true}, towerNodePair{dump: "node_822", ours: "out"})
}

// TestInternVLTowerMatchesLlamaCpp holds the tower to llama-mtmd-debug node by
// node on the rainbow picture. llama.cpp puts the class token LAST and adds the
// position table in row order, which gives the class token's position row to
// patch 0 -- a departure from the reference, which puts it first. The gate
// runs jitllm in llama.cpp's layout (towerCLSLast) so every node is the same
// computation; the engine's own layout is held to transformers instead
// (TestVisionMatchesTransformers), and the two layouts' distance is logged.
func TestInternVLTowerMatchesLlamaCpp(t *testing.T) {
	m, tw := openInternVL(t)
	defer m.Close()
	dump := parseMtmdDump(t, testmodels.Path("vlm/oracle/internvl-rainbow.txt"))
	px := rainbow(tw.Cfg.ImageSz)
	ours := func(clsLast bool) map[string][]float32 {
		defer func(was bool) { tw.opt.towerCLSLast = was }(tw.opt.towerCLSLast)
		tw.opt.towerCLSLast = clsLast
		s := tw.testState()
		defer s.Close()
		return towerTrace(t, s, px)
	}
	got := ours(true)
	holdTowerToDump(t, got, dump, internvlPairs(tw.Cfg.NLayer), 0.1, 0.05)
	tw.opt.towerCLSLast = true
	towerDumpViolation(t, tw, dump, "node_822", "shuffle transposed", towerFaultShuffleSwap, 0.1)
	tw.opt.towerCLSLast = false

	// The reference's layout against llama.cpp's, on the projector output.
	ref := ours(false)
	rms, _, err := dump["node_822"].against(ref["out"])
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("class token first (the reference's layout) against llama.cpp's: projector output rel RMS %.4f", rms)
}

// InternVL 2.5's 1B: the same InternViT-300M and projector over Qwen2.5-0.5B,
// converted from ggml-org's own GGUF pair.
func internvl25Text() string { return testmodels.Path("vlm/InternVL2_5-1B-Q8_0.gguf") }
func internvl25Proj() string { return testmodels.Path("vlm/mmproj-InternVL2_5-1B-Q8_0.gguf") }

func openInternVL25(t testing.TB) *Model {
	t.Helper()
	if !vlmAvailable(internvl25Text(), internvl25Proj()) {
		testmodels.Missing(t, "%s and its mmproj %s (or the merged container) are missing. RULE 11: fetch "+
			"ggml-org/InternVL2_5-1B-GGUF's Q8_0 text model and mmproj into %s and re-run",
			internvl25Text(), internvl25Proj(), testmodels.Dir())
	}
	m, err := Open(jlmOfPair(t, internvl25Text(), internvl25Proj()))
	if err != nil {
		t.Fatal(err)
	}
	if m.Tower() == nil {
		m.Close()
		t.Fatal("the merged container carries no tower, so this gate proves nothing")
	}
	return m
}
