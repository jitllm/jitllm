//go:build jitllmfault

package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestBatchedMixtureFloatBanksDiscriminate runs the float-bank gate
// (TestBatchedMixturePrefillFloatBanks) against tier's "floatsrc" violation --
// a float bank's down reading the gate's output in place of the activated
// product -- and demands the arms part past the gate's bound, finitely: a NaN
// would say the violation broke something else.
func TestBatchedMixtureFloatBanksDiscriminate(t *testing.T) {
	for _, name := range floatMoEModels {
		t.Run(name, func(t *testing.T) {
			nmse := batchedMixtureCase(t, name, true, true, tier.WithFault("moe", "floatsrc"))
			t.Logf("floatsrc: last-row logit NMSE %.3e between batched and row-by-row", nmse)
			if math.IsNaN(nmse) || nmse < 1e-2 {
				t.Fatalf("floatsrc reads NMSE %.3e, inside the gate's 1e-2 or not finite: the gate cannot see "+
					"a down reading the wrong buffer", nmse)
			}
		})
	}
}
