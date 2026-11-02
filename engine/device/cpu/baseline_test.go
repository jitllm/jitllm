//go:build amd64

// amd64 only: every assertion here is about CPUID.

package cpu

import "testing"

// TestMapRefusesWhatTheHostCannotRun: Map must refuse an AVX2 kernel on a host
// without AVX2 (otherwise it SIGILLs), and must map it on a host that can run
// it, since a gate that only refuses would turn the engine off.
func TestMapRefusesWhatTheHostCannotRun(t *testing.T) {
	if !BaselineForceable() {
		t.Skip("this build cannot force the baseline absent")
	}
	// The host tier's kernel: the one this host maps, on either tier.
	code := hostBytes(t)(hostTable().RMSNorm(64))
	if len(code) == 0 {
		t.Fatal("the RMSNorm emitter produced nothing; this test would prove nothing")
	}

	c := onHost(t)(code, nil)
	c.Close()

	// The violation: the exact condition a pre-AVX2 host is in.
	forceNoBaseline = true
	defer func() { forceNoBaseline = false }()

	if _, err := Map(code); err == nil {
		t.Fatal("Map accepted a kernel with the baseline absent -- the gate is " +
			"not wired, and this is the SIGILL")
	}
	if _, err := MapNamed(code, "rmsnorm"); err == nil {
		t.Fatal("MapNamed accepted a kernel with the baseline absent -- the gate " +
			"is on Map alone, and engine/nn/norm.go and engine/nn/delta.go call MapNamed")
	}

	// The probes must still work with the baseline absent: they go through
	// mapExec, since routing them through the gate would recurse. If this line
	// faults or hangs, that recursion is back.
	if f := CPU(); f == (Features{}) {
		t.Fatal("CPU() returned a zero feature set with the baseline forced " +
			"absent -- the CPUID probe is going through the gate")
	}
}

// TestBaselineIsAProbeNotAKnob asserts no environment variable turns the
// baseline off: forceNoBaseline is assigned from this file and nowhere else.
func TestBaselineIsAProbeNotAKnob(t *testing.T) {
	if forceNoBaseline {
		t.Fatal("forceNoBaseline is set before any test touched it")
	}
	if err := Baseline(); err != nil {
		t.Skipf("this host is genuinely below the baseline: %v", err)
	}
}
