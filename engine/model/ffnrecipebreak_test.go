//go:build jitllmtest

package model

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestGate2InSeesASwap is TestFFNRecipesMatchTwoLaunches' violation for the
// gate2in recipe (RULE 10): with the launch handed up's weight as the gate's
// (tier.WithViolation("gateup2")), its logits must leave the two-launch arm's.
func TestGate2InSeesASwap(t *testing.T) {
	m := gateUp2Model(t, gateUp2Models[0])
	if m == nil {
		testmodels.Missing(t, "MODEL MISSING: %s", gateUp2Models[0])
	}
	defer m.Close()
	ids := m.Vocab.Encode("The river rose every spring until the bridges were islands, and", true)
	toks := []int32{100, 137, 174, 211}
	outs, sts := ffnRecipeDecode(t, m, ids, toks, true, -1, t.TempDir(), []tier.FFNForm{tier.FFNTwo, tier.FFNGate2In}, tier.WithViolation("gateup2"))
	if outs == nil {
		t.Skip("no CUDA device here")
	}
	if sts[1].Gate2InRuns == sts[0].Gate2InRuns {
		t.Fatal("no FFN ran gate2in: the violation reached nothing")
	}
	n, diff := gateUp2NMSE(outs[1], outs[0])
	if diff == 0 {
		t.Fatal("gate and up swapped in the gate2in launch and the logits did not move")
	}
	t.Logf("seen: NMSE %.3g, %d of %d logits differ with gate and up swapped", n, diff, len(outs[1])*len(outs[1][0]))
}
