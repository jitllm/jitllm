//go:build amd64 && !jitllmtest

package cpu

// The SSE violation hook does not exist in a release build, for the reason
// prevnni_release.go gives: the naive sequence itself is what the tag
// carries, so a shipped engine has no path to a knowingly-saturating kernel.

func (d sseDot) naiveActive() bool { return false }

func (d sseDot) naiveSigned(*Buf, Reg, Reg, Reg, Reg) bool { return false }
