//go:build amd64

package cpu

import "testing"

// xieluVEXGate runs the SSE tier's xIELU through the VEX-leak gate: a VEX
// prefix faults on a host below AVX at any vector length.
func xieluVEXGate(t *testing.T, tier string, code []byte) {
	if KernelTier(code) == TierSSE {
		requireSSEKernel(t, "xielu-"+tier, code)
	}
}
