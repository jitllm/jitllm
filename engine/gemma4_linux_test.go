//go:build linux

package engine

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestBatchSeamMovesCarryEveryRowGemma4 is the seam gate on Gemma 4: three
// rows in flight while the seam moves, each block's history moving at its own
// geometry. The fixtures are F32 and take the families' bound.
func TestBatchSeamMovesCarryEveryRowGemma4(t *testing.T) {
	w := fusedRunsCheck(t, false)
	defer w.done()
	for _, name := range []string{"synth-gemma4", "synth-gemma4-kv", "synth-gemma4-moe", "synth-gemma4-e", "synth-gemma4-e-ragged"} {
		t.Run(name, func(t *testing.T) {
			p, ok := existingModel(testmodels.Path(name + ".gguf"))
			if !ok {
				testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path(name+".gguf")+" -- run "+gemma4GoldScript)
			}
			batchSeamMoves(t, p, 1e-3, true, w)
		})
	}
}
