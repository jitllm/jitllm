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

type tierFaults struct{}

func (*Config) pagedFaulted(string) bool { return false }

func (*Config) recFaulted(string) bool { return false }

func (*Config) rowsFaulted(string) bool { return false }

func (*Config) roomFaulted(string) bool { return false }

func (g *devTier) rewriteComposer(*composer.Registry) {}

func (*Config) formFaultedAsDevice() bool { return false }
