//go:build jitllmtest

package model

import (
	"os"
	"strconv"

	"github.com/jitllm/jitllm/engine/device/gpu"
)

// JITLLM_TEST_GEMM_DEEP=1 runs every model gate with the deep m16n8 GEMM
// tiles forced on (gpu.ForceGemmDeepForTest), -1 forced off.
func init() {
	if n, err := strconv.Atoi(os.Getenv("JITLLM_TEST_GEMM_DEEP")); err == nil {
		gpu.ForceGemmDeepForTest(n)
	}
}
