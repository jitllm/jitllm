package cpu

import "testing"

// sseKernelCheck is the VEX-leak gate on an SSE-tier kernel (requireSSEKernel).
func sseKernelCheck(t testing.TB, name string, code []byte) { requireSSEKernel(t, name, code) }
