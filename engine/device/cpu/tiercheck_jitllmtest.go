//go:build jitllmtest

package cpu

import "fmt"

// name is the kernel's perf symbol where the platform's Code carries one; the
// darwin mapping does not, so the message falls back to its size.

// checkTier refuses to run a kernel generated for another tier.
//
// It catches a cache that loses its tier key: Map gates a kernel when it is
// mapped, so a *Code cached under a shape alone and handed out after the
// forced tier moved never passes the gate again. The kernel carries the tier
// it was generated for, so the call can see it. It is in the test build only,
// where the forced tier lives (tier_hook_amd64.go).
//
// The check is one-way: legacy SSE runs on an AVX2 host, and every SSE kernel
// gate relies on that. What is never right is a kernel that needs
// instructions the host reports it lacks.
func checkTier(c *Code) {
	if c.tier != TierAVX2 || HostTier() == TierAVX2 || HostTier() == TierNEON {
		return
	}
	panic(fmt.Sprintf("jit: an %s kernel (%d bytes) was called on a host reporting tier %s: "+
		"a kernel cache is missing its tier key, so this run is not testing the tier it says",
		c.tier, c.Size, HostTier()))
}
