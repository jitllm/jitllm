//go:build linux

package tier

import (
	"testing"
	"time"
)

// TestDoubleCloseReturns: closing a tier twice must return, not wedge. Each
// free routes through the CUDA owner goroutine over c.reqs, and cudaDev.Close
// sets c.reqs = nil, so a second Close once blocked forever on a nil channel.
// A `defer g.Close()` beside an explicit Close is all it takes. The gate is
// time-bounded because the violation is a hang, not a failure.
func TestDoubleCloseReturns(t *testing.T) {
	g, err := OpenWith(WithDevices("auto"), WithDeviceTune(TuneOff))
	if err != nil || g == nil {
		t.Skipf("no device: %v", err)
	}
	// Give it real resident state to tear down, so Close has frees to route.
	p := qkPlan()
	if !g.PrepLayer(0, p, qkWeights(p)) {
		t.Logf("PrepLayer declined (%s); closing an empty tier still has to work", g.Err())
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		g.Close()
		g.Close()
		g.Close()
	}()
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatal("a second Close did not return: it is sending on the nil channel " +
			"the first Close left behind, which blocks forever")
	}
}
