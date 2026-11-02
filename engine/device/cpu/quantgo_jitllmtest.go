//go:build jitllmtest

package cpu

import "github.com/jitllm/jitllm/internal/oracle"

// quantGoLoop is the Go arithmetic the generated quantizers replaced, kept
// under the test tag as the other arm of the measurement that justifies them:
// nn.WithQuantActGo builds no kernel so every quantize lands here, and
// model.TestQuantActDose alternates the two arms in one process. A release
// build gets quantgo.go, which panics.
func quantGoLoop(t quantType, dst []int8, pairs, half []float32, x []float32, blo, bhi, window int) {
	qaGo.Add(1)
	bias := BiasC(t)
	oracle.QuantizeQ8Window(dst, pairs, x, bias, blo, bhi, window)
	if len(half) > 0 {
		oracle.QuantizeHalfSums(half[2*blo:2*bhi], dst[blo*Q8Block:bhi*Q8Block], bias)
	}
}
