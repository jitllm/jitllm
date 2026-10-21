package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/format/quant"
)

// The two packed kernels the SSE tier does not have, by decision rather than
// by absence.
//
// The wide kernel has no caller on any tier (MatVecPacked prefers the fused
// kernel whenever a wide one could run), so a port would be code nothing runs.
//
// The token-tiled kernel is refused for the pre-VNNI reason with fewer
// registers: a 16-row group is four XMM accumulators per token, so even tok=2
// does not fit beside the unpack. MatMulPacked then runs the fused kernel per
// token, the same answer. A narrower SSE tile is owed, with its own
// measurement.

func EmitPackedMatVecWideSSE(t quant.Type) ([]byte, error) {
	return nil, fmt.Errorf("jit: %s: the SSE tier has no wide packed kernel "+
		"(it has no caller on any tier; the fused kernel serves every nrows%%32 == 0)", t)
}

func EmitPackedMatMulTiledSSE(t quant.Type, k, nrows, tok int) ([]byte, error) {
	return nil, fmt.Errorf("jit: %s: the SSE tier has no token-tiled packed kernel; "+
		"MatMulPacked runs the fused kernel per token", t)
}

// EmitPackedMatVecFusedAheadSSE refuses, so nn does not duel a prefetch
// distance on this tier: whether a low-power core's prefetcher wants help is
// unmeasured.
func EmitPackedMatVecFusedAheadSSE(t quant.Type, ahead int) ([]byte, error) {
	return nil, fmt.Errorf("jit: %s: the SSE tier's fused kernel has no prefetch form", t)
}

// EmitPackedGEMMSSE refuses: the weight-stationary GEMM holds eight YMM of
// decoded weight, which is sixteen XMM on this tier and the whole register
// file. A narrower SSE form is a kernel with its own measurement, owed.
func EmitPackedGEMMSSE(t quant.Type, k, nrows, win int) ([]byte, error) {
	return nil, fmt.Errorf("jit: %s: the SSE tier has no weight-stationary packed GEMM; "+
		"MatMulPacked runs the fused kernel per token", t)
}

// EmitPackedMatVecFusedWinSSE refuses: the integer super-block fold is an
// arm64 form (MLA scales an int32 lane in one instruction); nn keeps the float
// fused kernel, which is this tier's only one.
func EmitPackedMatVecFusedWinSSE(t quant.Type, win int) ([]byte, error) {
	return nil, fmt.Errorf("jit: %s: the SSE tier's fused kernel has no integer-fold form", t)
}

// MaxTiledTokensSSE is 0 for every format: see above.
func MaxTiledTokensSSE(quant.Type) int { return 0 }
