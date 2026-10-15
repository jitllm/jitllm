//go:build jitllmtest

package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestGateUp2SeesASwap is TestGateUp2MatchesTwoLaunches' violation (RULE
// 10): with the Gate2 launch handed up's weight as the gate's
// (tier.SwapGateUp2ForTest), its logits must leave the two-launch arm's. If
// they do not, the bit-identity says nothing about which weight is which.
func TestGateUp2SeesASwap(t *testing.T) {
	m := gateUp2Model(t, gateUp2Models[0])
	if m == nil {
		testmodels.Missing(t, "MODEL MISSING: %s", gateUp2Models[0])
	}
	defer m.Close()
	ids := m.Vocab.Encode("The river rose every spring until the bridges were islands, and", true)
	toks := []int32{100, 137, 174, 211}
	cache := t.TempDir()
	off, _, _ := gateUp2Decode(t, m, ids, toks, false, true, -1, cache)
	defer tier.SwapGateUp2ForTest(tier.SwapGateUp2ForTest(true))
	on, st, _ := gateUp2Decode(t, m, ids, toks, true, true, -1, cache)
	if st.GateUp2Runs == 0 {
		t.Fatal("no FFN ran Gate2: the violation reached nothing")
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
		t.Fatal("gate and up swapped in the Gate2 launch and the logits did not move")
	}
	t.Logf("seen: %d of %d logits differ with gate and up swapped", diff, len(on)*len(on[0]))
}
