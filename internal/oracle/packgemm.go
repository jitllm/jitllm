package oracle

import "fmt"

// PackGEMMActivations is the reference the generated activation packer
// (EmitPackAct, run by jit/cpu.PackActJIT) is gated against: tokens [lo, hi)
// and k-blocks [blo, bhi) of x, [tok][k] row-major, quantized and laid out as
// EmitGEMMWindow reads them; blo < 0 is every block.
//
// dst[(g*tok+n)*4+c] is token n's element 4g+c, so eight tokens' four k
// values form one contiguous 32-byte operand. sum carries bias*sum(q) per
// (block, token), the correction for reading the weights as unsigned
// (jit/cpu.BiasC); half, where wantHalf, the same per sixteen-element half.
// window is the amax window, which must match the decode path's.
func PackGEMMActivations(dst []int8, scale []float32, sum, half []int32, x []float32, tok, k, lo, hi, blo, bhi, window, bias int, wantHalf bool) error {
	wantHalf = wantHalf && half != nil
	if k%q8Block != 0 {
		return fmt.Errorf("oracle: PackGEMMActivations: k=%d is not a multiple of %d", k, q8Block)
	}
	nb := k / q8Block
	per := max(1, window/q8Block)
	if len(dst) < tok*k || len(x) < tok*k || len(scale) < nb*tok || len(sum) < nb*tok {
		return fmt.Errorf("oracle: PackGEMMActivations: short buffer")
	}
	if blo < 0 {
		blo, bhi = 0, nb
	}
	for n := lo; n < hi; n++ {
		row := x[n*k : (n+1)*k]
		lastW0, amax := -1, float32(0)
		for b := blo; b < bhi; b++ {
			blk := row[b*q8Block : (b+1)*q8Block]
			w0 := (b / per) * per * q8Block
			if w0 != lastW0 {
				w1 := min(w0+per*q8Block, k)
				amax = 0
				for _, v := range row[w0:w1] {
					amax = max(amax, abs32(v))
				}
				lastW0 = w0
			}
			d := amax / 127
			inv := float32(0)
			if d != 0 {
				inv = 1 / d
			}
			var acc, lo16 int
			for i, v := range blk {
				q := min(127, max(-128, roundHalfAwayInt(v*inv)))
				acc += q
				if i < 16 {
					lo16 += q
				}
				ki := b*q8Block + i
				dst[((ki/4)*tok+n)*4+ki%4] = int8(q)
			}
			scale[b*tok+n] = d
			sum[b*tok+n] = int32(bias * acc)
			if wantHalf {
				half[2*b*tok+n] = int32(bias * lo16)
				half[(2*b+1)*tok+n] = int32(bias * (acc - lo16))
			}
		}
	}
	return nil
}

// roundHalfAwayInt rounds half away from zero. The add in f32 is exact: the
// callers' |v| is at most 127, and f32 holds every x + 0.5 exactly below 2^23.
func roundHalfAwayInt(v float32) int {
	if v < 0 {
		return -int(-v + 0.5)
	}
	return int(v + 0.5)
}
