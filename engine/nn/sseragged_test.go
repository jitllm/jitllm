//go:build amd64 && jitllmtest

package nn

import (
	"testing"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestRaggedProductsRunSSE is TestRaggedProductsMatchThePaddedRows on the
// SSE tier: the quantizer and products that tier generates, the ragged tail
// staged the same way.
func TestRaggedProductsRunSSE(t *testing.T) {
	old := cpu.ForceTierForTest(cpu.TierSSE)
	defer cpu.ForceTierForTest(old)
	if cpu.HostTier() != cpu.TierSSE {
		t.Skipf("the tier could not be forced to SSE on this build (host tier %v)", cpu.HostTier())
	}
	raggedProducts(t)
}
