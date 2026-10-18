//go:build jitllmfault

package model

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestKVEvictionGateDiscriminates runs the eviction gate's squeezed arm with
// the streamed pass's upload skipped, so each pass reads whatever the free ids
// held, and demands the comparison fail.
func TestKVEvictionGateDiscriminates(t *testing.T) {
	eg := newEvictGate(t)
	free := eg.run(t, false)
	eg.extra = []tier.Option{tier.WithFault("paged", "stream")}
	got := eg.run(t, true)
	if got.st.KVStreamPasses == 0 {
		t.Fatal("no streamed pass ran: the violation was never exercised")
	}
	dev, at := worstStep(got.logits, free.logits)
	t.Logf("violation: worst logit NMSE %.3e at step %d", dev, at)
	if dev < evictNMSE {
		t.Fatalf("the gate passed a stream that uploaded nothing (NMSE %.3e)", dev)
	}
}

// TestStagedPassGateDiscriminates runs the staged-pass gate's narrow arm
// with every pass after the first left unfolded, and demands it fail.
func TestStagedPassGateDiscriminates(t *testing.T) {
	eg := newEvictGate(t)
	wide, _ := eg.stagedPassRun(t, testdevices.Like(t, "cuda:0"), 0)
	eg.extra = []tier.Option{tier.WithFault("paged", "passfold")}
	narrow, ns := eg.stagedPassRun(t, testdevices.Like(t, "cuda:0"), 128)
	if ns.StagedPasses == 0 {
		t.Fatal("no staged pass ran: the violation was never exercised")
	}
	dev, at := worstStep(narrow, wide)
	t.Logf("violation: worst logit NMSE %.3e at step %d", dev, at)
	if dev < evictNMSE {
		t.Fatalf("the gate passed passes that were never folded (NMSE %.3e)", dev)
	}
}
