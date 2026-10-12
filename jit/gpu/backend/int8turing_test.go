package backend_test

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestInt8MMAOnTuringLowering runs MatVecMMA and GemmInt8 lowered for sm_75
// (each m16n8 integer product as m8n8k16s) on whichever CUDA card is here,
// held to the dp4a arm as TestMatVecMMAMatchesDot4 holds the m16n8 lowering:
// bit for bit. A card below sm_75 cannot run the PTX and is skipped by name.
func TestInt8MMAOnTuringLowering(t *testing.T) {
	gpuLock(t)
	ran := 0
	for _, d := range backend.Open() {
		defer d.Close()
		if d.API() != "ptx" || !backend.ForcePTXTarget(d, "sm_75") {
			continue
		}
		ran++
		for _, q := range []kernels.Quant{kernels.Q4_0, kernels.Q8_0, kernels.Q4_K, kernels.Q3_K, kernels.Q5_K, kernels.Q6_K} {
			for _, sp := range []struct{ rows, k, ntok, mt, nt int }{
				{16, 256, 8, 1, 1}, {64, 256, 16, 2, 2}, {256, 2048, 128, 4, 4}, {128, 512, 8, 4, 1},
			} {
				t.Run(fmt.Sprintf("%s/%dx%d/ntok%d", q, sp.rows, sp.k, sp.ntok), func(t *testing.T) {
					mmaMVCase(t, d, q, sp.rows, sp.k, sp.ntok, sp.mt, sp.nt)
				})
			}
		}
	}
	if ran == 0 {
		t.Skip("no CUDA device on this host")
	}
}
