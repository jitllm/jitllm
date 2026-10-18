//go:build jitllmtest

package model

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestQKVGemmSeesASwap is TestQKVGemmMatchesThreeGemms' violation (RULE 10):
// with the grid's k and v segments reading each other's weights
// (tier.WithViolation("qkvgemm")), the unsplit arms' logits must part.
func TestQKVGemmSeesASwap(t *testing.T) {
	m := gateUp2Model(t, "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if m == nil {
		testmodels.Missing(t, "MODEL MISSING: Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	}
	defer m.Close()
	ids := make([]int32, 192)
	for i := range ids {
		ids[i] = int32(1 + (i*7919)%(m.Cfg.NVocab-1))
	}
	off, on, st, _ := qkvGemmPrefill(t, m, ids, nil, true, true, -1, tier.WithViolation("qkvgemm"))
	if off == nil {
		t.Skip("no CUDA device")
	}
	if st.QKVGemmRuns == 0 {
		t.Fatal("no grid ran: the violation reached nothing")
	}
	nmse, diff := gateUp2NMSE(on, off)
	if diff == 0 {
		t.Fatal("k and v swapped in the grid and the logits did not move")
	}
	t.Logf("seen: NMSE %.3g, %d of %d logits differ", nmse, diff, len(on[0]))
}
