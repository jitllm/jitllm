//go:build amd64 && jitllmtest

package cpu

import "sync"

// The two test instruments the pre-VNNI path needs, both behind the jitllmtest
// build tag and neither reachable from a release binary.
//
// forceNoVNNI (feature_force.go) is the one forcing variable, and resetting
// featuresOnce makes the probe read it late, so every caller (SupportedNative,
// HostDotKind, SupportedPackedNative) sees the forced answer through the one
// probe they share. probeMu, the lock both instruments use, lives beside the
// probe in features_amd64.go.

// ForceNoVNNIForTest makes this package's capability probe report no VNNI (or
// stop doing so), re-probing immediately, and returns the previous setting so a
// test can restore it.
//
// It is not safe against concurrent emitters: it rewrites a package-wide
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

// naiveSignedQ8 makes the pre-VNNI Q8_0 payload keep the +128 centring that
// overflows VPMADDUBSW.
//
// It is the violation the gate is run against: a mechanical port of the VNNI
// kernel that keeps the centring. It runs and returns saturated dot products,
// so only a bit-equality gate can see it.
var naiveSignedQ8 bool

// SetNaiveSignedQ8ForTest arms that violation and returns the previous setting.
func SetNaiveSignedQ8ForTest(v bool) bool {
	old := naiveSignedQ8
	naiveSignedQ8 = v
	return old
}

func (d dotEmitter) naiveActive() bool { return naiveSignedQ8 }

// naiveSigned emits the centred-and-saturating sequence when the violation is
// armed, and reports whether it did.
func (d dotEmitter) naiveSigned(a *Buf, acc, pay, act Reg) bool {
	if !naiveSignedQ8 {
		return false
	}
	a.VPXOR(pay, pay, d.flip)
	a.VPMADDUBSW(pay, pay, act)
	a.VPMADDWD(pay, pay, d.ones)
	a.VPADDD(acc, acc, pay)
	return true
}
