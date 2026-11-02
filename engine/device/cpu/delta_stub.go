//go:build !amd64 && !arm64

package cpu

import "fmt"

// EmitGatedDelta has no implementation on this architecture; the stubs keep
// callers building.
func EmitGatedDeltaChan(n int) ([]byte, error) {
	return nil, fmt.Errorf("jit: EmitGatedDeltaChan: no implementation on this architecture")
}

func EmitGatedDelta(n int) ([]byte, error) {
	return nil, fmt.Errorf("jit: no gated delta kernel on this architecture")
}

func EmitConv1d(taps, chans int) ([]byte, error) {
	return nil, fmt.Errorf("jit: no causal convolution kernel on this architecture")
}

func EmitDWConv(s DWShape) ([]byte, error) {
	return nil, fmt.Errorf("jit: no depthwise convolution kernel on this architecture")
}

func EmitDeltaGate() []byte { return nil }

func EmitDeltaDecayBound() []byte { return nil }

func EmitGatedSSD(n int) ([]byte, error) {
	return nil, fmt.Errorf("jit: no selective state update kernel on this architecture")
}

func EmitSSDGate() []byte { return nil }

func EmitSelScan(n int) ([]byte, error) {
	return nil, fmt.Errorf("jit: no selective scan kernel on this architecture")
}
