//go:build amd64 && !jitllmtest

package cpu

// The pre-VNNI violation hook does not exist in a release build.
//
// It is absent rather than false, like nn/disable.go and tier/fault.go: the
// naive sequence itself is what the jitllmtest tag carries, so a shipped
// engine has no path to a knowingly-saturating kernel.
func (d dotEmitter) naiveSigned(*Buf, Reg, Reg, Reg) bool { return false }

func (d dotEmitter) naiveActive() bool { return false }
