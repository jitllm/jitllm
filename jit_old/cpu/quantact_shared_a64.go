//go:build arm64

package cpu

import "math"

// The narrow-window quantizer's constants on arm64. They are the amd64 file's
// values and must stay so: cpu.QuantActKernels, engine/nn/jit.go and engine/nn/packed.go size
// their scratch from them, and the NEON kernel honours both -- eight blocks a
// group, and sixteen of the twenty-four scratch floats (eight d, eight inv).
const (
	QuantActNarrowBlocks  = 8
	QuantActNarrowScratch = 24
)

// EmitQuantActNarrow is the NEON narrow-window quantizer; quantact_a64.go has
// the body.
func EmitQuantActNarrow(half bool) []byte { return emitQuantActNarrowA64(half) }

// ArgmaxConsts is the block both argmax kernels read; see argmax.go for the
// layout, which one constant block serves on every architecture.
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

// ArgmaxLanes is the vector width the kernel consumes per iteration. It is NOT
// a divisor a caller must respect: the kernel takes its own ragged tail.
const ArgmaxLanes = 8

// EmitArgmax is the NEON greedy argmax; argmax_a64.go has the body.
func EmitArgmax() []byte { return emitArgmaxA64() }
