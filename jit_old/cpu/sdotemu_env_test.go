//go:build amd64 || arm64

package cpu

import "os"

// JITLLM_FORCE_NO_DOTPROD=1 runs this package's whole suite with the arm64
// probe answering "no FEAT_DotProd", so every kernel gate here -- the ragged
// sweeps, the guards and the oracle comparisons -- runs the widened kernels
// (sdotemu.go) on a chip that has SDOT: an M4, or qemu's default cpu. On a
// chip that lacks it (qemu -cpu cortex-a72) the probe says so without this.
// The test binary reads it; the library reads no environment.
func init() {
	if os.Getenv("JITLLM_FORCE_NO_DOTPROD") == "1" {
		forceNoDotProd = true
	}
}
