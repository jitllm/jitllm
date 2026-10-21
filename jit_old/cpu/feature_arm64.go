//go:build arm64

package cpu

import (
	"sync"
)

// FEAT_DotProd detection, so a chip without SDOT runs kernels that widen it
// (sdotemu.go) instead of dying on an illegal instruction. SDOT is ARMv8.2-A:
// Cortex-A53/A57/A72/A73 (a Raspberry Pi 3 or 4) lack it.
var (
	dotOnce sync.Once
	hasDot  bool
)

// hasDotProd reports FEAT_DotProd. forceNoDotProd forces it false, which is how
// the fallback is tested on a machine that has the feature (assigned only from
// a _test.go file here).
func hasDotProd() bool {
	// forceNoDotProd is read on every call, outside the Once, so a test can
	// turn it on after the probe fired.
	dotOnce.Do(func() { hasDot = probeDotProd() })
	return hasDot && !forceNoDotProd
}

// a64EmulateDot is whether the A64 emitters widen SDOT into the ARMv8.0
// sequence (sdotemu.go): exactly when this chip lacks FEAT_DotProd.
func a64EmulateDot() bool { return !forceDotEncoding && !hasDotProd() }

// HasDotProd reports FEAT_DotProd for a hardware report. It decides nothing:
// without it the quantized kernels widen SDOT (sdotemu.go) and still run.
func HasDotProd() bool { return hasDotProd() }
