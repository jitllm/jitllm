//go:build !amd64

package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/format/quant"
)

// The SSE tier is an amd64 tier, and its packed kernels read the amd64
// PackedScratch layout (the nibble mask, the -8/+8/+4 corrections, the scale
// offsets), which the arm64 build does not write. These names exist so the
// SSE table compiles on every architecture that has one (emitters.go); on
// arm64 HostTier is NEON and nothing selects this tier.

func SupportedPackedSSE(quant.Type) bool { return false }

func EmitPackedMatVecSSE(t quant.Type, rows int) ([]byte, error) {
	return nil, fmt.Errorf("jit: %s: the SSE tier's packed kernels are amd64 only", t)
}

func EmitPackedMatVecFusedSSE(t quant.Type) ([]byte, error) {
	return nil, fmt.Errorf("jit: %s: the SSE tier's packed kernels are amd64 only", t)
}
