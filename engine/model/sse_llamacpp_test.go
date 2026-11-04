//go:build amd64 && jitllmtest

package model_test

import (
	"testing"

	"github.com/jitllm/jitllm/engine/model"
)

// TestGreedyMatchesLlamaCppSSE is the llama.cpp golden gate over every model
// on the CPU device at the SSE tier (cpu:sse) -- the same gate, the same tie
// rule (a divergence on a knife-edge margin is reduction order, not a bug).
// Every model it opens maps onto a Target at the SSE tier, whose gate refuses
// an AVX2 kernel by name: none can be mapped silently. It is the
// external-package twin of TestSSEModelGates, which cannot call this
// package's TestGreedyMatchesLlamaCpp.
//
// Like the other model-level SSE gates it needs every SSE family, and until
// then it fails naming the ops still pending rather than skip.
func TestGreedyMatchesLlamaCppSSE(t *testing.T) {
	greedyMatchesLlamaCpp(t, model.WithDevices("cpu:sse"))
}
