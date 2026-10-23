package oracle

import (
	"fmt"
	"math"
)

// The int8 form of an activation row as the device matvecs read it: each
// 32-element block's codes four to a word, its scale (the amax of its window
// over 127) and a sum per sixteen. The reference the tests hold the composed
// quantize to; no production code may quantize in Go (RULE 8).

// PackActivations quantizes activations to int8 and returns the packed bytes,
// a scale per 32 elements, and a sum per sixteen.
//
// The sums are the zero-point correction: nibbles stay unsigned and the true
// weight is q-bias, so sum((q-bias)*a) = dot(q,a) - bias*sum(a), which avoids a
// byte-wise signed subtract the IR cannot express. They are per 16 because
// Q3_K and Q6_K scale per 16; a 32-element format adds two.
func PackActivations(x []float32) (a []uint32, as, sum16 []float32, err error) {
	nb := len(x) / 32
	a = make([]uint32, len(x)/4)
	as = make([]float32, nb)
	sum16 = make([]float32, 2*nb)
	err = PackActivationsInto(a, as, sum16, x, 32)
	return a, as, sum16, err
}

// PackActivationsInto is PackActivations writing into caller buffers, with the
// given amax window; see cpu.QuantizeQ8Window for why the window is a
// per-model decision. window <= 32 is the usual per-block scale.
//
// It must quantize bit-for-bit as the host tier does, or the tiers' dot
// products differ slightly and a deep model's greedy tokens diverge.
func PackActivationsInto(a []uint32, as, sum []float32, x []float32, window int) error {
	if len(x)%32 != 0 {
		return fmt.Errorf("oracle: PackActivations: %d is not a multiple of 32", len(x))
	}
	nb := len(x) / 32
	if len(a) < len(x)/4 || len(as) < nb || len(sum) < 2*nb {
		return fmt.Errorf("oracle: PackActivations: destination too small")
	}
	per := window / 32
	if per < 1 {
		per = 1
	}
	var q [32]int8
	for b := 0; b < nb; b++ {
		blk := x[b*32 : b*32+32]
		// Bit-for-bit the same steps as jit/cpu.QuantizeQ8Window.
		w0 := (b / per) * per * 32
		w1 := w0 + per*32
		if w1 > len(x) {
			w1 = len(x)
		}
		amaxBits := uint64(0)
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
		lo, hi := 0, 0
		for i, v := range blk {
			qi := int(math.Round(float64(v * inv)))
			q[i] = int8(qi)
			if i < 16 {
				lo += qi
			} else {
				hi += qi
			}
		}
		as[b] = float32(d)
		sum[2*b], sum[2*b+1] = float32(lo), float32(hi)
		for w := 0; w < 8; w++ {
			a[b*8+w] = uint32(uint8(q[4*w])) | uint32(uint8(q[4*w+1]))<<8 |
				uint32(uint8(q[4*w+2]))<<16 | uint32(uint8(q[4*w+3]))<<24
		}
	}
	return nil
}
