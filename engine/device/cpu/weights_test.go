package cpu

import (
	"encoding/binary"
	"math/rand"

	"github.com/jitllm/jitllm/format/quant"
)

// Shared by the amd64 and arm64 kernel tests, and deliberately untagged: both
// architectures are held against the same reference weights.

// buildWeights makes a packed weight matrix of the given format and returns it
// alongside the exact float64 values it encodes.
//
// The exact values come from quant.Dequant, which is verified against libggml,
// so the kernel is checked against llama.cpp's definition of the format.
func buildWeights(rng *rand.Rand, t quant.Type, rows, k int) (packed []byte, exact [][]float64) {
	be, bb := int(t.BlockElems()), int(t.BlockBytes())
	nb := k / be
	packed = make([]byte, rows*nb*bb)
	for i := range packed {
		packed[i] = byte(rng.Intn(256))
	}
	// Fix up every f16 scale so it is normal and modest. Random bits land on
	// infinities and NaNs, which is not what a kernel test is about.
	fixF16 := func(off int) {
		v := uint16((rng.Intn(15)+8)<<10) | uint16(rng.Intn(1024))
		binary.LittleEndian.PutUint16(packed[off:], v)
	}
	for blk := 0; blk < rows*nb; blk++ {
		base := blk * bb
		switch t {
		// Every format needs a case: a missed one leaves ~3% of random blocks
		// with an f16 Inf/NaN scale, a non-finite reference, and a gate that
		// can read that as NMSE 0.
		case quant.Q4_0, quant.Q5_0, quant.Q8_0:
			fixF16(base)
		case quant.Q4_K, quant.Q5_K:
			fixF16(base)     // d
			fixF16(base + 2) // dmin
		case quant.Q6_K:
			fixF16(base + 208) // d sits at the END of a Q6_K block
		// Q3_K's d sits at offset 108.
		case quant.Q3_K:
			fixF16(base + 108)
		}
	}
	exact = make([][]float64, rows)
	for r := 0; r < rows; r++ {
		exact[r] = make([]float64, k)
		lo := r * nb * bb
		if err := quant.Dequant(t, packed[lo:lo+nb*bb], exact[r]); err != nil {
			panic(err)
		}
	}
	return packed, exact
}
