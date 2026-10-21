//go:build jitllmtest

package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestAddNormFoldSeesABrokenAdd is TestAddNormFoldMatchesTheAdd's violation
// (RULE 10): with the folded add summing the wrong row
// (tier.WithViolation("addnorm")), the prompt's logits must leave the unfolded
// arm's.
func TestAddNormFoldSeesABrokenAdd(t *testing.T) {
	m := gateUp2Model(t, "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if m == nil {
		testmodels.Missing(t, "MODEL MISSING: Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	}
	defer m.Close()
	ids := make([]int32, 192)
	for i := range ids {
		ids[i] = int32(1 + (i*7919)%(m.Cfg.NVocab-1))
	}
	cache := t.TempDir()
	off, _, _ := addNormPrefill(t, m, ids, nil, true, true, -1, cache)
	on, st, _ := addNormPrefill(t, m, ids, nil, false, true, -1, cache, tier.WithViolation("addnorm"))
	if st.AddNormFolded == 0 {
		t.Fatal("no add folded: the violation reached nothing")
	}
	diff := 0
	for j := range on[0] {
		if math.Float32bits(on[0][j]) != math.Float32bits(off[0][j]) {
			diff++
		}
	}
	if diff == 0 {
		t.Fatal("the folded add summed the wrong row and the logits did not move")
	}
	t.Logf("seen: %d of %d logits differ", diff, len(on[0]))
}
