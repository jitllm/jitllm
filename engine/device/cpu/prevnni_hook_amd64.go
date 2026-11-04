//go:build amd64 && jitllmtest

package cpu

import "sync"

// ForceNoVNNIForTest makes this package's capability probe report no VNNI (or
// stop doing so), re-probing immediately, and returns the previous setting so a
// test can restore it: the composer forms its pre-VNNI byte dots for a CPU
// stating no AVX-VNNI. forceNoVNNI (feature_force.go) is the one forcing
// variable; resetting featuresOnce makes the probe read it late. probeMu lives
// beside the probe in features_amd64.go.
//
// It is not safe against concurrent composition: it rewrites a package-wide
// capability set, which a test wants global and a running engine must never
// see move.
func ForceNoVNNIForTest(v bool) bool {
	probeMu.Lock()
	defer probeMu.Unlock()
	old := forceNoVNNI
	forceNoVNNI = v
	featuresOnce = sync.Once{}
	features = Features{}
	CPU()
	return old
}
