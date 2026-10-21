package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/format/quant"
)

// The activation side of the prefill GEMM. It is architecture-neutral: the
// packed layout (eight tokens by four k values per 32-byte lane group) is the
// same shape for AVX-VNNI's VPDPBUSD and for arm64's SDOT.

// GEMMTokens is how many tokens a tile of width nr covers.
func GEMMTokens(nr int) int { return 8 * nr }

// PackGEMMActivationsRange quantizes tokens [lo, hi) of one token-tile and lays
// them out the way EmitGEMMWindow reads them.
//
// x is [tok][k] row-major, one row per token. The output interleaves by k-group
// rather than by token: dst[(g*tok+n)*4+c] is token n's element 4g+c, so eight
// tokens' four k values form one contiguous 32-byte operand.
//
// sum carries BiasC(t)*sum(q) per (block, token) as an int32 -- the correction
// for reading the weights as unsigned, which is 128 for Q8_0's XORed int8 and 8
// for Q4_0's nibbles. It is an integer because the kernel subtracts it from the
// integer accumulator before converting: the bias dominates the accumulator, so
// removing it in float32 would cancel away most of the mantissa.
//
// window must match the decode path's: prefill and decode write the same KV
// cache, so both must quantize activations with the same scale grouping
// (TestMatMulMatchesMatVec).
func PackGEMMActivationsRange(t quant.Type, dst []int8, scale []float32, sum, half []int32, x []float32, tok, k, lo, hi, window int) error {
	return packGEMM(t, dst, scale, sum, half, x, tok, k, lo, hi, -1, -1, window)
}

// packGEMM packs tokens [lo, hi) and k-blocks [blo, bhi); blo < 0 is every
// block.
//
// A block range splits along the other axis than a token range does: the
// layout is k-group major, so for one k-group all tok tokens are contiguous
// (64 bytes, one cache line, at tok=16), and a token split would give every
// worker a slice of every output line. A block split gives each worker whole
// lines and divides evenly. It is correct at every window: the amax window is
// recomputed from x for each block, so a range that cuts a window simply
// rescans it.
func packGEMM(t quant.Type, dst []int8, scale []float32, sum, half []int32, x []float32, tok, k, lo, hi, blo, bhi, window int) error {
	bias := int(BiasC(t))
	wantHalf := half != nil && NeedsHalfSums(t)
	if k%Q8Block != 0 {
		return fmt.Errorf("jit: PackGEMMActivations: k=%d is not a multiple of %d", k, Q8Block)
	}
	nb := k / Q8Block
	per := window / Q8Block
	if per < 1 {
		per = 1
	}
	if len(dst) < tok*k || len(x) < tok*k || len(scale) < nb*tok || len(sum) < nb*tok {
		return fmt.Errorf("jit: PackGEMMActivations: short buffer")
	}
	if blo < 0 {
		blo, bhi = 0, nb
	}
	for n := lo; n < hi; n++ {
		row := x[n*k : (n+1)*k]
		// The amax is per window, so it is cached across the blocks of one
		// window rather than recomputed per block (at window 256 that would be
		// eight scans of the same elements). b ascends, so caching the last
		// window's answer works for whatever block range this worker got.
		lastW0, amax := -1, float32(0)
		for b := blo; b < bhi; b++ {
			blk := row[b*Q8Block : (b+1)*Q8Block]
			// The amax window this block belongs to, clamped to the row.
			w0 := (b / per) * per * Q8Block
			if w0 != lastW0 {
				w1 := w0 + per*Q8Block
				if w1 > k {
					w1 = k
				}
				amax = 0
				for _, v := range row[w0:w1] {
					if v < 0 {
						v = -v
					}
					if v > amax {
						amax = v
					}
				}
				lastW0 = w0
			}
			d := amax / 127
			inv := float32(0)
			if d != 0 {
				inv = 1 / d
			}
			// The half sums are accumulated here from q rather than re-read
			// from dst's scattered layout.
			var acc, lo16 int
			for i, v := range blk {
				q := roundHalfAwayInt(v * inv)
				if q > 127 {
					q = 127
				} else if q < -128 {
					q = -128
				}
				acc += q
				if i < 16 {
					lo16 += q
				}
				ki := b*Q8Block + i
				dst[((ki/4)*tok+n)*4+ki%4] = int8(q)
			}
			scale[b*tok+n] = float32(d)
			sum[b*tok+n] = int32(bias * acc)
			if wantHalf {
				// Two sixteen-element halves of this block, for the formats
				// that scale at that granularity.
				half[2*b*tok+n] = int32(bias * lo16)
				half[(2*b+1)*tok+n] = int32(bias * (acc - lo16))
			}
		}
	}
	return nil
}

// roundHalfAwayInt is roundHalfAway returning an int directly. Doing the add
// in f32 is bit-identical to the f64 form because the value is bounded: inv is
// 127/amax, so |v*inv| <= 127, and f32 represents every x + 0.5 exactly below
// 2^23.
func roundHalfAwayInt(v float32) int {
	if v < 0 {
		return -int(-v + 0.5)
	}
	return int(v + 0.5)
}

// GEMMScratchMax is how many float32 of Args.Scratch the widest tile EmitGEMM
// allows needs, for a caller that sizes one buffer and reuses it for every
// shape. The terms are the accumulator tile (mr*nr of eight floats), the
// per-row scale staging (16 floats), the shuffle staging (8), the unpacked
// super-block (64 floats = 256 bytes per row), and a second accumulator tile
// for the int32 fold (see emitGEMMQ3K). The int tile is sized unconditionally
// rather than per-window: a window-dependent layout would add an invariant
// between nn/ and the emitter. It lives in an untagged file because both
// architectures use it; the arm64 k-quant GEMM's scale staging (256 floats at
// the widest tile) fits within the amd64 figure.
const GEMMScratchMax = 2*8*12 + 8*16 + 8 + 8*64
