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

	// forceDotEncoding makes the A64 emitters encode SDOT whatever the chip,
	// for a test that reads a kernel's instructions and never runs it (the
	// golden shapes) on a host without FEAT_DotProd. Test-only, like the rest.
	forceDotEncoding bool

	// forceTier makes the amd64 probe report the capabilities of a lower tier
	// (tier.go) when tierForced is set, so the SSE tier's kernels can be
	// selected and run on an AVX2 machine. Also a test instrument: written only
	// by retier, which only ForceTierForTest (jitllmtest) and this package's
	// tests call.
	forceTier  Tier
	tierForced bool
)
