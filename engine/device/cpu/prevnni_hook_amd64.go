//go:build amd64 && jitllmtest

package cpu

// ForceNoVNNIForTest makes this package's capability probe report no VNNI (or
// stop doing so), and returns the previous setting so a test can restore it:
// the composer forms its pre-VNNI byte dots for a CPU stating no AVX-VNNI. A
// Host made after the call states it; force before model.Open.
//
// It is not safe against concurrent composition: it rewrites a package-wide
// capability set, which a test wants global and a running engine must never
// see move.
func ForceNoVNNIForTest(v bool) bool {
	old := forceNoVNNI
	forceNoVNNI = v
	return old
}
