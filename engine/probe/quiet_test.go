package probe

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// quiet skips a gate that holds timings to a band in CI's model-free run: a
// shared runner is not a quiet box, the reason dev/bench is left out of that
// run (RULE 4). Everywhere else the gate runs.
func quiet(t *testing.T) {
	t.Helper()
	if testmodels.Free() {
		t.Skip("holds timings to a band, and a shared CI runner is not a quiet box (RULE 4)")
	}
}
