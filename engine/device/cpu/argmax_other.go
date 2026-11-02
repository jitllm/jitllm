//go:build !amd64 && !arm64

package cpu

import "math"

// ArgmaxConsts is the amd64 block's twin; see argmax.go for the layout.
func ArgmaxConsts() []float32 {
	c := make([]float32, 11)
	for i := 0; i < 8; i++ {
		c[i] = math.Float32frombits(uint32(i))
	}
	c[8] = math.Float32frombits(8)
	c[9] = math.Float32frombits(0x7FFFFFFF)
	c[10] = float32(math.Inf(-1))
	return c
}

// ArgmaxLanes is the vector width the kernel consumes per iteration.
const ArgmaxLanes = 8

// EmitArgmax has no kernel on this architecture.
func EmitArgmax() []byte { return nil }
