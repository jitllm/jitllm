package oracle

import (
	"fmt"
	"math"
)

// q8Block is jit_old/cpu.Q8Block, which this package cannot import (the cpu tests
// import it).
const q8Block = 32

// The activation quantizers in Go: the reference the generated quantize
// kernels (EmitQuantAct and its SSE/arm64 twins) are gated against.

// QuantizeQ8 converts activations to int8 and fills the per-block scale/bias
// pairs the kernel reads.
//
// pairs is 2 float32 per block, interleaved as {scale, negSum}, so the kernel
// walks one cursor at +8 bytes per block: general-purpose registers are the
// scarce resource in this loop.
//
// The second element of each pair is the bias correction, and biasC is what
// makes one kernel shape serve both formats.
//
// VPDPBUSD multiplies unsigned by signed, but quantized weights are signed, so
// something must be biased into unsigned range and then corrected:
//
//	Q4_0: weights are (nibble-8)*d, nibble already unsigned 0..15
//	      sum((n-8)*q) = sum(n*q) - 8*sum(q)          -> biasC = 8
//	Q8_0: weights are signed int8, biased to unsigned by XOR 0x80
//	      sum((u-128)*q) = sum(u*q) - 128*sum(q)      -> biasC = 128
//
// In both cases the correction is weight-independent, so it costs one FMA and
// no memory traffic. The kernel broadcasts it across all 8 lanes and the row
// ends in a horizontal sum, which multiplies by 8 — so storing -sum(q)*biasC/8
// makes that factor fall out of the reduction for free.
func QuantizeQ8(dst []int8, pairs []float32, x []float32, biasC float64) error {
	if len(x)%q8Block != 0 {
		return fmt.Errorf("jit: QuantizeQ8: %d activations is not a multiple of %d", len(x), q8Block)
	}
	nb := len(x) / q8Block
	if len(dst) < len(x) || len(pairs) < 2*nb {
		return fmt.Errorf("jit: QuantizeQ8: need %d int8 and %d float32, have %d and %d",
			len(x), 2*nb, len(dst), len(pairs))
	}
	QuantizeQ8Range(dst, pairs, x, biasC, 0, nb)
	return nil
}

// QuantizeHalfSums fills per-16-element sums, which the 16-wide k-quants need
// for their bias correction. Stored pre-divided by 4, because each half occupies
// four of VPDPBUSD's eight lanes and the row ends in a horizontal sum.
func QuantizeHalfSums(half []float32, q []int8, biasC float64) {
	for h := 0; h*16+16 <= len(q); h++ {
		s := 0
		for _, v := range q[h*16 : h*16+16] {
			s += int(v)
		}
		half[h] = float32(-float64(s) * biasC / 4)
	}
}

// QuantizeQ8Range quantizes blocks [lo, hi). Split out so the pool can spread it
// across cores: quantization is a quarter of decode time.
func QuantizeQ8Range(dst []int8, pairs []float32, x []float32, biasC float64, lo, hi int) {
	QuantizeQ8Window(dst, pairs, x, biasC, lo, hi, q8Block)
}

// QuantizeQ8Window is QuantizeQ8Range with the amax taken over `window`
// elements instead of one block, so every block inside a window shares a scale.
//
// It changes no layout, only a value: the pair array stays (d, -sum*biasC/8)
// at stride 8 per 32 elements, and a wider window writes the same d into every
// pair it covers. A kernel that knows can hoist the scale out of its sub-block
// loop. 256 matches a k-quant super-block (llama.cpp's block_q8_K), so a whole
// super-block accumulates in int32 with one float multiply.
//
// It is a per-model decision: a wider amax is a coarser scale, so it is taken
// only when every matvec type in the model is a k-quant.
func QuantizeQ8Window(dst []int8, pairs []float32, x []float32, biasC float64, lo, hi, window int) {
	per := window / q8Block
	if per < 1 {
		per = 1
	}
	for b := lo; b < hi; b++ {
		off := b * q8Block
		blk := x[off : off+q8Block : off+q8Block]
		// Absolute value by clearing the sign bit, keeping the loop
		// branch-free.
		amaxBits := uint64(0)
		// The window this block belongs to, clamped to the vector.
		w0 := (b / per) * per * q8Block
		w1 := w0 + per*q8Block
		if w1 > len(x) {
			w1 = len(x)
		}
		for _, v := range x[w0:w1] {
			if b := uint64(math.Float32bits(v) &^ (1 << 31)); b > amaxBits {
				amaxBits = b
			}
		}
		amax := math.Float32frombits(uint32(amaxBits))
		d := amax / 127
		inv := float32(0)
		if d != 0 {
			inv = 1 / d
		}
		sum := 0
		out := dst[off : off+q8Block : off+q8Block]
		for i, v := range blk {
			// |v*inv| <= 127 by construction (inv is 127/amax), so no clamp.
			q := int(math.Round(float64(v * inv)))
			out[i] = int8(q)
			sum += q
		}
		pairs[2*b] = d
		pairs[2*b+1] = float32(-float64(sum) * biasC / 8)
	}
}
