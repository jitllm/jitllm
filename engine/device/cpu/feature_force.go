//go:build amd64 || arm64

package cpu

// forceNoVNNI and forceNoDotProd make the CPU feature probes answer "absent".
// They are test instruments, assigned only from a _test.go file in this
// package, so a shipped binary cannot pretend a CPU lacks an instruction it
// has. They are declared here rather than beside their probes so the test that
// uses both compiles on either architecture.
var (
	forceNoVNNI    bool
	forceNoDotProd bool

	// forceTier makes the amd64 probe report the capabilities of a lower tier
	// (tier.go) when tierForced is set, so the SSE tier's kernels can be
	// selected and run on an AVX2 machine. Also a test instrument: written only
	// by ForceTierForTest (jitllmtest).
	forceTier  Tier
	tierForced bool
)
