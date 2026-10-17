//go:build jitllmfault && (amd64 || arm64)

package model

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestSplitRowsGateDiscriminates runs TestSplitRowsMatchTheHost's split arm
// with the host's share of the step dropped (tier's "splitrowshost"): the
// rows must part from the host past the gate's band.
func TestSplitRowsGateDiscriminates(t *testing.T) {
	for _, md := range splitRowsModels {
		t.Run(md.name, func(t *testing.T) {
			path, ok := existingModel(testmodels.Path(md.path))
			if !ok {
				t.Skipf("MODEL MISSING: %s -- this gate proved nothing", md.path)
			}
			m, err := Open(jlmOf(t, path), noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			ids := m.Vocab.Encode("The capital of France is Paris, and the capital of Italy is Rome; Spain's", true)
			tier.SetMoEFault("splitrowshost")
			defer tier.SetMoEFault("")
			worst, st, _ := splitRowsCase(t, m, ids, md.blocks, false, false)
			t.Logf("splitrowshost: worst NMSE %.2e over %d split row steps", worst, st.SplitRowSteps)
			if st.SplitRowSteps == 0 || worst <= md.band {
				t.Fatalf("with the host's share dropped the rows read NMSE %.2e over %d split row steps: the gate cannot see it",
					worst, st.SplitRowSteps)
			}
		})
	}
}
