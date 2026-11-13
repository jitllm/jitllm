package cpudev

import (
	"testing"
	"time"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/sched"
)

// TestAPromptClosesTheFusedRun swaps the device's workers to a prefill pool
// wider than the decode pool and back, each time with a fused run open on
// the pool being left. The two pools' crews take turns on one lock, which a
// fused run holds until it closes, so a region on the other pool waited on
// it forever -- the M4's composed CPU arm, where the prefill pool is the P
// and E cores by default, hung in its first paged prompt.
func TestAPromptClosesTheFusedRun(t *testing.T) {
	all := append(sched.SMTSiblings(), sched.ECores()...)
	if len(all) <= len(sched.DecodeCores()) {
		t.Skipf("no core beyond the %d decode cores to prefill on", len(sched.DecodeCores()))
	}
	p := cpu.NewPool(cpu.WithPrefillCPUs(all))
	defer p.Close()
	e := NewExec(p, nil)
	region := func(what string) {
		t.Helper()
		done := make(chan struct{})
		go func() {
			p.Do(64, 1, func(int, int, int) {})
			close(done)
		}()
		select {
		case <-done:
		case <-time.After(30 * time.Second):
			t.Fatalf("%s waited 30 s on the turn a fused run holds", what)
		}
	}
	e.Fuse()
	if !e.fused {
		t.Fatal("the decode pool opened no fused run: nothing is open across the swap")
	}
	e.BeginPrefill(false)
	region("the prompt's first region")
	e.Fuse()
	if !e.fused {
		t.Fatal("the prefill pool opened no fused run: nothing is open across the swap back")
	}
	e.EndPrefill()
	region("the token after the prompt")
	e.Unfuse()
}
