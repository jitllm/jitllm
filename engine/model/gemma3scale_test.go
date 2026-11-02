package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestGemma3GlobalLayersAreLinearlyScaled: gemma-3-4b ships rope.scaling.type
// linear with factor 8, and llama.cpp divides the global layers' rotary angles
// by it while leaving the sliding ones alone (rope_freq_scale_train_swa = 1).
//
// The oracle is llama-eval-callback's rotated K for "The capital" (ids
// 2 818 5279), summed over the three positions -- position 0 is an identity
// rotation, so only a multi-token prompt can see a rotary scale at all:
//
//	llama-eval-callback -m gemma-3-4b-it-Q4_K_M.gguf -p "The capital" -n 1 -ngl 0 -c 64
//
// Layer 5 is global and reads -18.774827 here against -18.902901 (0.7%);
// without the scale it read -21.585079 (14%). Layer 4 is local and reads the
// same either way, which is the other half of the claim.
func TestGemma3GlobalLayersAreLinearlyScaled(t *testing.T) {
	path, ok := existingModel(testmodels.Path("gemma-3-4b-it-Q4_K_M.gguf"))
	if !ok {
		testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path("gemma-3-4b-it-Q4_K_M.gguf")+" (set JITLLM_MODELS to the model directory) -- RULE 11, fetch it; this gate proves nothing without it")
	}
	m, err := Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.Cfg.RopeLinear != 8 {
		t.Fatalf("RopeLinear %v, want the file's 8", m.Cfg.RopeLinear)
	}
	sums := map[int]float64{}
	m.Trace(func(l int, name string, v []float32) {
		if name == "k" {
			for _, x := range v {
				sums[l] += float64(x)
			}
		}
	})
	ids := m.Vocab.Encode("The capital", true)
	st := m.NewInstance(8)
	defer st.Close()
	for _, id := range ids {
		if _, err := st.Forward(id); err != nil {
			t.Fatal(err)
		}
	}
	const want5, want4 = -18.902901, -45.731182
	if rel := math.Abs(sums[5]-want5) / math.Abs(want5); !(rel < 0.03) {
		t.Errorf("global layer 5 rotated K sums to %.6f against llama.cpp's %.6f (%.1f%%)",
			sums[5], want5, 100*rel)
	}
	t.Logf("ids %v; global layer 5 %.6f (llama.cpp %.6f), local layer 4 %.6f (llama.cpp %.6f)",
		ids, sums[5], want5, sums[4], want4)
}
