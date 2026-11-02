//go:build amd64

package cpu

import "testing"

// TestHCKernelsSSEAreLegacy is the VEX-leak gate on DeepSeek V4's two SSE
// kernels: a VEX prefix faults on an SSE host at any vector length, and the
// tier declaration alone does not prove the bytes carry none.
func TestHCKernelsSSEAreLegacy(t *testing.T) {
	for _, it := range []int{0, 1, 20} {
		for _, head := range []bool{false, true} {
			code, err := EmitHCMixSSE(it, head)
			if err != nil {
				t.Fatal(err)
			}
			if KernelTier(code) != TierSSE {
				t.Fatalf("hcmix rounds %d head %v: not declared SSE-tier", it, head)
			}
			requireSSEKernel(t, "sse_hcmix", code)
		}
	}
	code, err := EmitColPoolSSE()
	if err != nil {
		t.Fatal(err)
	}
	if KernelTier(code) != TierSSE {
		t.Fatal("colpool: not declared SSE-tier")
	}
	requireSSEKernel(t, "sse_colpool", code)
}
