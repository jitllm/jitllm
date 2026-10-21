//go:build amd64

package cpu

import "testing"

// tierArmRuns reports whether this host can execute the bytes of one tier's
// arm in a gate that runs both x86 tiers against the same oracle. The gates
// were written on an AVX2 host, but they also run natively on an SSE-only one
// (a Goldmont Atom), where mapping the AVX2 arm is refused. There the AVX2 arm
// is left out by name and the SSE arm still runs against the oracle; every
// caller maps its SSE arm unconditionally, so a gate never runs empty.
func tierArmRuns(t testing.TB, arm string, code []byte) bool {
	t.Helper()
	if err := mapGate(code); err != nil {
		t.Logf("LOUD: the %s arm is not run on this host: %v -- the SSE arm runs "+
			"against the same oracle here, and a host with those extensions runs this one", arm, err)
		return false
	}
	return true
}
