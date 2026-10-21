//go:build !jitllmtest

package cpu

import "fmt"

// quantGoLoop is what a release build does when a tier has no
// activation-quantize kernel for a window: nothing, loudly.
//
// Every tier generates both window shapes, so reaching this is a contract
// violation. The Go loop exists only under the jitllmtest tag, as the A/B
// arm these kernels are priced against.
func quantGoLoop(t quantType, dst []int8, pairs, half []float32, x []float32, blo, bhi, window int) {
	panic(fmt.Sprintf("jit: no activation-quantize kernel for a %d-element amax window "+
		"on tier %v -- every tier generates both shapes, so this is a wiring bug "+
		"and not a fallback", window, HostTier()))
}
