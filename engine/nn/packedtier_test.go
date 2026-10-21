//go:build amd64 || arm64

package nn

import (
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestEveryPackedFormatHasAGeneratedKernel: every quantized type a jlm
// container can store must have a generated packed matvec on the host
// architecture, because that is the inference path.
func TestEveryPackedFormatHasAGeneratedKernel(t *testing.T) {
	// jlm.packerOf's list, which is what a container can hold.
	packed := quant.PackedTypes
	var missing []quant.Type
	for _, q := range packed {
		if !cpu.PackedSupported(q) {
			missing = append(missing, q)
		}
	}
	if len(missing) != 0 {
		t.Errorf("no generated packed matvec for %v on this architecture -- every one of "+
			"those runs kernels.MatVecPacked in Go, and MatVecPacked returns true so "+
			"nothing reports it", missing)
	}
}

// TestEveryPackedFormatHasATokenTiledKernel is the same question for the batch
// path (prefill and the vision tower). It asks the emitter, not the host, so
// the inventory does not depend on which machine runs the suite; the gates that
// execute a tile ask cpu.MaxTiledTokensNative. On a pre-VNNI host no format is
// runnable (the tiled emitter has no DotVEX form), which is logged, not failed.
func TestEveryPackedFormatHasATokenTiledKernel(t *testing.T) {
	packed := quant.PackedTypes
	var missing []quant.Type
	for _, q := range packed {
		if cpu.MaxTiledTokens(q) < 2 {
			missing = append(missing, q)
		}
	}
	if len(missing) != 0 {
		t.Errorf("no token-tiled packed kernel for %v -- prefill and the vision tower "+
			"read the weight once PER TOKEN for those", missing)
	}
	// The host half, reported rather than asserted: it is a property of the CPU
	// running the suite.
	var hostless []quant.Type
	for _, q := range packed {
		if cpu.MaxTiledTokensNative(q) < 2 {
			hostless = append(hostless, q)
		}
	}
	if len(hostless) != 0 {
		t.Logf("NOT RUNNABLE HERE: %v have a tiled kernel this CPU cannot execute "+
			"(no AVX-VNNI), so prefill on this host reads the weight once per "+
			"token. The tiled emitter has no pre-VNNI form; that is the open work.",
			hostless)
	}
}
