//go:build !amd64 && !arm64

package cpu

import "fmt"

// The image kernels on a host with no emitter: they refuse, as EmitResample
// does.
func EmitResampleH(int) []byte       { return nil }
func EmitPixLUT() []byte             { return nil }
func EmitCopy32() []byte             { return nil }
func EmitAxisTaps(AxisFilter) []byte { return nil }
func EmitLerpGrid() []byte           { return nil }
func EmitSinCosTab() []byte          { return nil }

func EmitPixelRow(PixFmt, int, PixOut) ([]byte, error) {
	return nil, fmt.Errorf("jit: PixelRow: no emitter on this architecture")
}
