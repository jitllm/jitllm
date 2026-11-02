//go:build !jitllmfault

package gpu

import "github.com/jitllm/jitllm/composer"

// The release build: no fault injection. Both of these inline to a constant, so
// the check in Layers costs nothing and JITLLM_DEVICE_FAIL_AT does nothing at all.
// See fault.go for why this is a build tag rather than a runtime flag.

func faultAt() []int { return nil }

func (g *devTier) injectedFail() bool { return false }

func faultPanicAt() []int { return nil }

func (g *devTier) injectedPanic() {}

func pagedFaulted(string) bool { return false }

func recFaulted(string) bool { return false }

func rowsFaulted(string) bool { return false }

func roomFaulted(string) bool { return false }

func (g *devTier) rewriteComposer(*composer.Registry) {}

func formFaultedAsDevice() bool { return false }
