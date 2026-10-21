//go:build darwin

// darwin-only, like mem_test.go: these need a real Metal device, and elsewhere
// they would only be skips. An arm64 Mac runs them from a cross-compiled test binary.

package metal

import "testing"

// TestCounterSamplingCapability asks the device where it can sample GPU
// counters. It is a probe rather than an assertion: the answer decides which
// per-dispatch instrument is possible, and both answers are legitimate
// hardware.
func TestCounterSamplingCapability(t *testing.T) {
	c, err := Open()
	if err != nil {
		t.Skipf("no Metal device here: %v", err)
	}
	defer c.Close()

	stage, dispatch := c.CounterSampling()
	t.Logf("counter sampling: atStageBoundary=%v atDispatchBoundary=%v", stage, dispatch)
	for _, n := range c.CounterSets() {
		t.Logf("counter set: %q", n)
	}
	if !stage && !dispatch {
		t.Log("NEITHER: this device cannot sample counters at all, so per-dispatch " +
			"timing is not available and the 6.3% unattributed term stays " +
			"unattributed by this route.")
	}
	if dispatch {
		t.Log("AT DISPATCH BOUNDARY: the shipping encoder can time each launch " +
			"in situ -- this is the instrument worth building.")
	} else if stage {
		t.Log("STAGE ONLY: per-dispatch timing would need one ENCODER per launch. " +
			"A compute encoder is MTLDispatchTypeSerial, so that serialises and " +
			"adds an encoder's cost per launch -- the instrument would be " +
			"measuring itself. Do NOT build it on this route.")
	}
}
