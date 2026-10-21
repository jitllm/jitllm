//go:build amd64

package cpu

import "testing"

// TestAVX2NormEmittersAreByteStable and its SSE twin pin the machine code of
// the row-reduction family (see emitgolden_test.go). The family has no VNNI
// choice: none of these kernels emit a dot product.
func TestAVX2NormEmittersAreByteStable(t *testing.T) {
	checkGolden(t, "avx2_norm", normGoldenCases(t,
		func(n int) ([]byte, error) { return EmitRMSNorm(n), nil },
		func(n int, bias bool, mode RowMode) ([]byte, error) { return emitRowStat(n, bias, mode), nil }))
}

func TestSSENormEmittersAreByteStable(t *testing.T) {
	checkGolden(t, "sse_norm", normGoldenCases(t, EmitRMSNormSSE, emitRowStatSSE))
}
