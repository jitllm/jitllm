package jit_test

import (
	"testing"
	"time"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
)

// TestCoarsenLeavesALongChainUnformed forms the sampler's candidates over
// vocabularies whose reductions Coarsen once unrolled into millions of nodes
// (65536: 7,976,810 nodes, 3 s; 131072: no end in five minutes and more than
// 3 GB) on a device that states no exact division, and holds each to a
// second of forming: the chain past maxChainNodes is left as its members.
func TestCoarsenLeavesALongChainUnformed(t *testing.T) {
	dev := jit.Device{Kind: jit.DeviceGPU, Slots: 30720, Lanes: 32, Units: 20, Regs: 42}
	for _, v := range []int{65536, 131072} {
		s := graph.SampleTopK{NVocab: v, K: 256, Hist: 256}
		g := s.Build()
		done := make(chan error, 1)
		go func() {
			_, _, err := jit.Default().Form(jit.Request{Graph: g, Part: jit.PartSample, Formats: map[jlm.Role]jlm.Type{},
				Step: jit.Step{Rows: 1, Page: 64, Keys: 1}, Device: dev})
			done <- err
		}()
		start := time.Now()
		select {
		case err := <-done:
			if err != nil {
				t.Fatalf("vocabulary %d: %v", v, err)
			}
			if d := time.Since(start); d > 2*time.Second {
				t.Errorf("vocabulary %d formed in %v", v, d)
			}
		case <-time.After(30 * time.Second):
			t.Fatalf("vocabulary %d: still forming after 30 s", v)
		}
	}
}
