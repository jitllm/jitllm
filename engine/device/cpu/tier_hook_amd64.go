//go:build amd64 && jitllmtest

package cpu

// ForceTierForTest makes this package's capability probe report the host as
// tier t, re-probing immediately, and returns the tier it reported before so a
// test can restore it:
//
//	old := cpu.ForceTierForTest(cpu.TierSSE)
//	defer cpu.ForceTierForTest(old)
//	if cpu.HostTier() != cpu.TierSSE { t.Fatal("the force reached nothing") }
//
// Forcing can only take capabilities away: TierSSE clears every VEX-encoded
// unit (AVX2, FMA, F16C, AVX-512, both VNNIs and the OS YMM bits), TierNone
// also clears SSSE3/SSE4.1/SSE4.2 so the refuse-by-name path can run, and
// TierAVX2 removes the force. It resets Baseline's cached verdict too, so every
// question derived from the probe -- HostTier, Baseline, Map's ISA check,
// SupportedNative, SupportedPackedNative, HostDotKind -- answers for the forced
// host. nn keys its kernel caches by tier, so a forced run cannot reuse an
// AVX2 kernel an earlier test cached; force BEFORE model.Open/NewState, since a
// JIT records its tier when it is built.
//
// It is behind jitllmtest and is not an option: a release binary has no path
// to pretend a CPU lacks instructions it has. Legacy SSE executes on every
// AVX2 part, so this is all an AVX2 host needs to run the SSE tier end to
// end.
//
// It is not safe against concurrent emitters.
func ForceTierForTest(t Tier) Tier { return retier(t) }
