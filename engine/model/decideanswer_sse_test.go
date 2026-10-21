//go:build amd64 && jitllmtest

package model

import (
	"testing"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestDecisionAnswerMatchesTheOracleSSE is the answer sweep on the SSE tier,
// forced before the code generator is built (RULE 11d: the SSE and AVX2 tiers
// are two transcriptions).
func TestDecisionAnswerMatchesTheOracleSSE(t *testing.T) {
	old := cpu.ForceTierForTest(cpu.TierSSE)
	defer cpu.ForceTierForTest(old)
	answerGate(t, "")
}
