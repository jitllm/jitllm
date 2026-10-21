//go:build arm64 && windows

package cpu

import "syscall"

// pfARMv82DPInstructionsAvailable is PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE,
// the processor feature Windows reports for FEAT_DotProd.
const pfARMv82DPInstructionsAvailable = 43

// probeDotProd asks kernel32's IsProcessorFeaturePresent, which every Windows
// on Arm release answers. Without it the probe said absent, and every packed
// kernel declined on a chip (each Snapdragon X, the CI runner) that has SDOT.
func probeDotProd() bool {
	p := syscall.NewLazyDLL("kernel32.dll").NewProc("IsProcessorFeaturePresent")
	if p.Find() != nil {
		return false // absent, the direction that cannot SIGILL
	}
	r, _, _ := p.Call(pfARMv82DPInstructionsAvailable)
	return r != 0
}
