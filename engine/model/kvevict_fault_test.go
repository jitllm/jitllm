//go:build jitllmfault

package model

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
)

// TestKVEvictionGateDiscriminates runs the eviction gate's squeezed arm with
// the streamed pass's upload skipped, so each pass reads whatever the free ids
// held, and demands the comparison fail.
func TestKVEvictionGateDiscriminates(t *testing.T) {
	eg := newEvictGate(t)
	free := eg.run(t, false, nil)
	var f gpu.Faults
	f.SetPagedFault("stream")
	got := eg.run(t, true, nil, gpu.WithFaults(&f))
	if got.st.KVStreamPasses == 0 {
		t.Fatal("no streamed pass ran: the violation was never exercised")
	}
	dev, at := worstStep(got.logits, free.logits)
	t.Logf("violation: worst logit NMSE %.3e at step %d", dev, at)
	if dev < evictNMSE {
		t.Fatalf("the gate passed a stream that uploaded nothing (NMSE %.3e)", dev)
	}
}

// TestComposedPassFoldDiscriminates runs the eviction gate's squeezed arm
// with each composed pass folding into the empty partial, so only the last
// pass's keys count, and demands the comparison fail.
func TestComposedPassFoldDiscriminates(t *testing.T) {
	eg := newEvictGate(t)
	free := eg.run(t, false, nil)
	var f gpu.Faults
	f.SetPagedFault("passfold")
	got := eg.run(t, true, nil, gpu.WithFaults(&f))
	if got.st.ComposedStreamPasses == 0 {
		t.Fatal("no composed pass ran: the violation was never exercised")
	}
	dev, at := worstStep(got.logits, free.logits)
	t.Logf("violation: worst logit NMSE %.3e at step %d", dev, at)
	if dev < evictNMSE {
		t.Fatalf("the gate passed composed passes that were never folded (NMSE %.3e)", dev)
	}
}
