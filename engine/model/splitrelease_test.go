//go:build jitllmtest

package model

import (
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestReleasedSplitLeavesTheNextModelWhole runs a model whose global and
// sliding layers attend at two geometries (Gemma 4's, nn.LayerPlan.GeomSplit)
// on a tier, releases it, then runs a one-base model with sliding layers
// (OLMo 3's) on the same tier: every block must stay on the device. The
// GPU's split mark outliving the blocks that set it cut the second model's
// runs as a split model's, which drop the local rotary table (run.tables);
// the device refused the chunk and the prompt quietly ran on the host.
//
// VIOLATION SIGNATURE. Leave GPU.split set in releaseLayers and the second
// model ends its prompt with no device blocks.
func TestReleasedSplitLeavesTheNextModelWhole(t *testing.T) {
	g, err := openTier(t)
	if err != nil || g == nil {
		noDevice(t, "cuda", err)
	}
	defer g.Close()
	for _, n := range [...]string{"synth-gemma4-e", "synth-olmo3"} {
		m, err := Open(filepath.Join(testmodels.Dir(), n+".jlm"), noTune)
		if err != nil {
			t.Fatalf("%s: %v", n, err)
		}
		s := m.NewState(32)
		if err := s.SetDevice(g); err != nil || s.GPULayers() != m.Cfg.NLayer {
			t.Fatalf("%s: the device took %d of %d blocks: %v", n, s.GPULayers(), m.Cfg.NLayer, err)
		}
		if _, err := s.Prefill(m.Vocab.Encode("Hello", true)); err != nil {
			t.Fatalf("%s: %v", n, err)
		}
		if got := s.GPULayers(); got != m.Cfg.NLayer {
			t.Errorf("%s: after the prompt the device holds %d of %d blocks: %s", n, got, m.Cfg.NLayer, s.devWhy())
		}
		s.Close()
		m.Close()
		g.ReleaseLayers(0, 1<<20)
	}
}
