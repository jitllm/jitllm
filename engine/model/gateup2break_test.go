//go:build jitllmtest

package model

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestGateUp2SeesASwap is TestGateUp2MatchesTwoLaunches' violation (RULE
// 10), on every backend this host has: with the Gate2 launch handed up's
// weight as the gate's (tier.WithViolation("gateup2")), its logits must leave the
// two-launch arm's by more than the bound the gate allows off PTX
// (the bound on gateUp2Devices).
func TestGateUp2SeesASwap(t *testing.T) {
	m := gateUp2Model(t, gateUp2Models[0])
	if m == nil {
		testmodels.Missing(t, "MODEL MISSING: %s", gateUp2Models[0])
	}
	defer m.Close()
	ids := m.Vocab.Encode("The river rose every spring until the bridges were islands, and", true)
	toks := []int32{100, 137, 174, 211}
	ran := 0
	for _, dev := range gateUp2Devices {
		cache := t.TempDir()
		off, on, st, _ := gateUp2Decode(t, m, dev, ids, toks, true, -1, cache, tier.WithViolation("gateup2"))
		if off == nil {
			t.Logf("%s: no such device here", dev)
			continue
		}
		if st.GateUp2Runs == 0 {
			t.Fatalf("%s: no FFN ran Gate2: the violation reached nothing", dev)
		}
		nmse, diff := gateUp2NMSE(on, off)
		bound := denseStepBand * orderControl(t, m, dev)
		if w := gateUp2Worst(on, off); w <= bound {
			t.Fatalf("%s: gate and up swapped and the worst step stayed inside the bound (%.3g <= %.3g, %d differ)", dev, w, bound, diff)
		}
		t.Logf("%s seen: NMSE %.3g with gate and up swapped, %d of %d logits differ", dev, nmse, diff, len(on)*len(on[0]))
		ran++
	}
	if ran == 0 {
		testmodels.Missing(t, "no device ran")
	}
}
