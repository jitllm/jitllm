//go:build jitllmfault && (amd64 || arm64)

package model

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestExpertCardGateDiscriminates arms the violation (a sheet copied from the
// expert card is the next expert's) and demands the logits part from the
// host-served run's: TestExpertCardServesCacheMisses would be green on an
// expert card whose sheets were wrong if it could not see this.
func TestExpertCardGateDiscriminates(t *testing.T) {
	specs := testdevices.Take(t, 2, "cuda:0,cuda:1")
	path, ok := existingModel(testmodels.Path("kimik3/Kimi-K3-0.40B.Q8_0.gguf"))
	if !ok {
		t.Skipf("MODEL MISSING: kimik3/Kimi-K3-0.40B.Q8_0.gguf -- this gate proved nothing")
	}
	m, err := Open(jlmOf(t, path), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := prefillWavesIDs(m)
	want, _ := peerRun(t, m, ids, specs[:1], "")
	got, st := peerRun(t, m, ids, specs, specs[1], tier.WithFault("peer", "wrong"))
	worst := 0.0
	for i := range want {
		nm, _ := nmse32(want[i], got[i])
		worst = max(worst, nm)
	}
	t.Logf("wrong sheets from the expert card: worst NMSE %.3e over %d sheets", worst, st.PeerSheets)
	if st.PeerSheets == 0 || worst <= 1e-8 {
		t.Fatalf("with wrong sheets (%d taken) the logits read NMSE %.3e: the gate cannot see it", st.PeerSheets, worst)
	}
}
