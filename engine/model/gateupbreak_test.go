//go:build jitllmtest

package model

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestGateUpFuseSeesASwap is TestGateUpFuseMatchesThreeLaunches' violation
// (RULE 10): with down's prologue handed up's row as the gate
// (tier.SwapGateUpForTest), the fused arm's logits must leave the
// three-launch arm's. If they do not, the bit-identity says nothing about
// which rows the fused down reads.
func TestGateUpFuseSeesASwap(t *testing.T) {
	name := gateUpModels[0]
	p := testmodels.Path(name)
	if _, err := os.Stat(p); err != nil {
		if _, err := os.Stat(strings.TrimSuffix(p, ".gguf") + ".jlm"); err != nil {
			testmodels.Missing(t, "MODEL MISSING: %s", name)
		}
	}
	m, err := Open(jlmOf(t, p), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.Vocab.Encode("The river rose every spring until the bridges were islands, and", true)
	toks := []int32{100, 137, 174, 211}
	cache := t.TempDir()
	off, _, _ := gateUpDecode(t, m, ids, toks, false, true, -1, cache)
	defer tier.SwapGateUpForTest(tier.SwapGateUpForTest(true))
	on, st, _ := gateUpDecode(t, m, ids, toks, true, true, -1, cache)
	if st.GateUpFused == 0 {
		t.Fatal("no FFN fused: the violation reached nothing")
	}
	diff := 0
	for i := range on {
		for j := range on[i] {
			if math.Float32bits(on[i][j]) != math.Float32bits(off[i][j]) {
				diff++
			}
		}
	}
	if diff == 0 {
		t.Fatal("gate and up swapped in down's prologue and the logits did not move")
	}
	t.Logf("seen: %d of %d logits differ with gate and up swapped", diff, len(on)*len(on[0]))
}
