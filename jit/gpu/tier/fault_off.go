//go:build !jitllmfault

package tier

// The release build: no fault injection. Both of these inline to a constant, so
// the check in Layers costs nothing and JITLLM_DEVICE_FAIL_AT does nothing at all.
// See fault.go for why this is a build tag rather than a runtime flag. The
// named violations are per tier and live in hooks_off.go.

func faultAt() []int { return nil }

func (g *devTier) injectedFail() bool { return false }
