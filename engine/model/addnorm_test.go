package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// addNormPrefill prefills ids on a fresh tier with Config.NoAddNormFold set
// to noFold, then decodes toks, and returns the prompt's logits followed by
// every step's, the tier's stats and the blocks placed. layers < 0 places
// every block.
func addNormPrefill(t *testing.T, m *Model, ids, toks []int32, noFold, noGraph bool, layers int, cache string, extra ...tier.Option) ([][]float32, tier.Stats, int) {
	t.Helper()
	g, err := openTier(t, append([]tier.Option{tier.WithTuneCache(cache), tier.WithConfig(func(c *tier.Config) {
		c.NoAddNormFold, c.NoGraph = noFold, noGraph
	})}, extra...)...)
	if err != nil || g == nil {
		t.Skipf("no GPU: %v", err)
	}
	defer g.Close()
	s := m.NewState(len(ids) + len(toks) + 4)
	defer s.Close()
	if err := s.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if layers >= 0 {
		s.SetGPULayers(layers)
	}
	placed := s.GPULayers()
	l, err := s.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	out := [][]float32{append([]float32(nil), l...)}
	for _, tok := range toks {
		if l, err = s.Forward(tok); err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), l...))
	}
	s.SetGPULayers(0)
	return out, g.Stats(), placed
}

// TestAddNormFoldMatchesTheAdd prefills a real dense model with each block's
// residual add folded into the next block's attention norm
// (Config.NoAddNormFold false) and with the add alone, and holds the two to
// the same logits bit for bit, the prompt's and a few decoded steps' after
// it: the norm reads the sum it stores. Every block placed, with and without
// the graph, and a partial seam. Stats().AddNormFolded is the selection
// check; on a card with no binary16 GEMM path nothing folds, which is a
// failure here, not a pass.
func TestAddNormFoldMatchesTheAdd(t *testing.T) {
	m := gateUp2Model(t, "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if m == nil {
		testmodels.Missing(t, "MODEL MISSING: Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	}
	defer m.Close()
	ids := make([]int32, 192)
	for i := range ids {
		ids[i] = int32(1 + (i*7919)%(m.Cfg.NVocab-1))
	}
	toks := []int32{100, 137, 174, 211}
	cache := t.TempDir()
	for _, c := range []struct {
		seam    int
		noGraph bool
	}{{-1, false}, {-1, true}, {m.Cfg.NLayer / 2, false}} {
		off, so, _ := addNormPrefill(t, m, ids, toks, true, c.noGraph, c.seam, cache)
		on, st, placed := addNormPrefill(t, m, ids, toks, false, c.noGraph, c.seam, cache)
		want := m.Cfg.NLayer
		if c.seam >= 0 {
			want = c.seam
		}
		if placed != want {
			t.Fatalf("seam %d: the device took %d blocks, want %d", c.seam, placed, want)
		}
		if st.AddNormFolded == 0 {
			t.Fatalf("seam %d: no block-end add was folded (%s)", c.seam, st.LastErr)
		}
		if so.AddNormFolded != 0 {
			t.Fatalf("seam %d: the unfolded arm folded %d adds", c.seam, so.AddNormFolded)
		}
		for i := range on {
			for j := range on[i] {
				if math.Float32bits(on[i][j]) != math.Float32bits(off[i][j]) {
					t.Fatalf("seam %d graph %v row %d logit %d: folded %v, alone %v",
						c.seam, !c.noGraph, i, j, on[i][j], off[i][j])
				}
			}
		}
		t.Logf("seam %d graph %v: %d adds folded, %d rows bit-identical", c.seam, !c.noGraph, st.AddNormFolded, len(on))
	}
}
