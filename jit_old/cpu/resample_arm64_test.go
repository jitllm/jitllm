package cpu

import "testing"

// sseKernelCheck has nothing to check on arm64, which has no SSE tier.
func sseKernelCheck(testing.TB, string, []byte) {}
