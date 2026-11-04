//go:build arm64

package cpu

// FEAT_DotProd detection, so a chip without SDOT runs kernels that widen it
// instead of dying on an illegal instruction. SDOT is ARMv8.2-A:
// Cortex-A53/A57/A72/A73 (a Raspberry Pi 3 or 4) lack it.

// HasDotProd reports FEAT_DotProd for a hardware report. It decides nothing:
// without it the quantized forms widen SDOT and still run.
func HasDotProd() bool { return dotProd() }

// dotProd is FEAT_DotProd as Probe states it.
func dotProd() bool { return probeDotProd() }
