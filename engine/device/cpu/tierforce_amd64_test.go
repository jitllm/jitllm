//go:build amd64

package cpu

import "testing"

// withTier makes the probe report tier for the rest of the test and restores
// the previous answer afterwards. It is retier -- the one implementation
// ForceTierForTest (jitllmtest) also calls -- so this package's tests need no
// build tag to exercise the force.
//
// It asserts the force landed, so a test that "forces SSE" cannot silently run
// on the AVX2 tier.
func withTier(t testing.TB, tier Tier) {
	t.Helper()
	old := retier(tier)
	t.Cleanup(func() { retier(old) })
	if tier < TierAVX2 && HostTier() != tier {
		t.Fatalf("forced %v and the probe reports %v -- the force reached nothing", tier, HostTier())
	}
}
