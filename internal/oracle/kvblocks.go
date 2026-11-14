package oracle

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/jitllm/jitllm/format/quant"
)

// BlockFormat is a llama.cpp block format named as ggml names it (q4_0,
// q4_1, q5_0, q5_1): graph.BlockFmt, which this package cannot import (the
// container's tests import the oracle, and graph imports the container).
type BlockFormat interface{ String() string }

// blockOf is format f's block bytes, whether it holds a minimum and whether
// fifth bits; 0 bytes for a format the oracle does not hold.
func blockOf(f BlockFormat) (bytes int, mins, fifths bool) {
	switch f.String() {
	case "q4_0":
		return 18, false, false
	case "q4_1":
		return 20, true, false
	case "q5_0":
		return 22, false, true
	case "q5_1":
		return 24, true, true
	case "iq4_nl":
		return 18, false, false
	}
	return 0, false, false
}

// QuantizeBlocks writes x, whole blocks of 32 elements, into dst as ggml's
// reference quantizer of format f writes them (quantize_row_q4_0_ref,
// quantize_row_q4_1_ref, quantize_row_q5_0_ref, quantize_row_q5_1_ref): the
// reference the composed OpBlockRows is gated against. Each product is
// rounded before its sum, as the reference's separate statements are.
func QuantizeBlocks(dst []byte, x []float32, f BlockFormat) error {
	bb, mins, fifths := blockOf(f)
	if bb == 0 || len(x)%32 != 0 || len(dst) < len(x)/32*bb {
		return fmt.Errorf("oracle: %d floats into %d bytes of %v blocks", len(x), len(dst), f)
	}
	if f.String() == "iq4_nl" {
		for b := 0; b < len(x)/32; b++ {
			quantizeIQ4NL(dst[18*b:18*b+18], x[32*b:32*b+32])
		}
		return nil
	}
	for b := 0; b < len(x)/32; b++ {
		xb, y := x[32*b:32*b+32], dst[bb*b:bb*b+bb]
		var d, id, lo float32
		levels, bias := 15, float32(8.5)
		if fifths {
			levels, bias = 31, 16.5
		}
		if mins {
			mn, hi := float32(math.MaxFloat32), float32(-math.MaxFloat32)
			for _, v := range xb {
				mn, hi = min(mn, v), max(hi, v)
			}
			lo = mn
			d = (hi - lo) / float32(levels)
			binary.LittleEndian.PutUint16(y[2:], quant.EncodeHalf(lo))
			bias = 0.5
		} else {
			var amax, mx float32
			for _, v := range xb {
				if a := float32(math.Abs(float64(v))); amax < a {
					amax, mx = a, v
				}
			}
			d = mx / -float32((levels+1)/2)
		}
		if d != 0 {
			id = 1 / d
		}
		binary.LittleEndian.PutUint16(y, quant.EncodeHalf(d))
		code := func(v float32) int {
			t := float32(v * id)
			if mins {
				t = float32((v - lo) * id)
			}
			q := int(int8(t + bias))
			switch f.String() {
			case "q5_1":
				return int(uint8(q))
			case "q4_1":
				return min(15, q)
			}
			return min(levels, q)
		}
		qs := 2
		if mins {
			qs = 4
		}
		var qh uint32
		for j := 0; j < 16; j++ {
			q0, q1 := code(xb[j]), code(xb[j+16])
			qh |= uint32(q0>>4&1)<<j | uint32(q1>>4&1)<<(j+16)
			y[qs+4*b2i(fifths)+j] = byte(q0&15 | (q1&15)<<4)
		}
		if fifths {
			binary.LittleEndian.PutUint32(y[qs:], qh)
		}
	}
	return nil
}

// quantizeIQ4NL is quantize_row_iq4_nl_impl at ntry -1 with no importance
// weights (quantize_row_iq4_nl_ref): the indices from max/kvalues[0], the
// scale the weighted least squares over them, each weight the element's
// square, the sums in order. A block whose magnitude is below 1e-15 scales
// by zero; the reference leaves its indices as its last block's, and this
// writes the index of the value nearest zero, as the composed quantizer does.
func quantizeIQ4NL(y []byte, x []float32) {
	kv := [16]int8{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}
	best := func(v float32) int {
		if v <= float32(kv[0]) {
			return 0
		}
		if v >= float32(kv[15]) {
			return 15
		}
		ml, mu := 0, 15
		for mu-ml > 1 {
			mav := (ml + mu) / 2
			if v < float32(kv[mav]) {
				mu = mav
			} else {
				ml = mav
			}
		}
		if v-float32(kv[mu-1]) < float32(kv[mu])-v {
			return mu - 1
		}
		return mu
	}
	var amax, mx float32
	for _, v := range x {
		if a := float32(math.Abs(float64(v))); a > amax {
			amax, mx = a, v
		}
	}
	var l [32]int
	var d float32
	if amax < 1e-15 {
		for j := range l {
			l[j] = 8
		}
	} else {
		d0 := mx / float32(kv[0])
		id := 1 / d0
		var sqx, sq2 float32
		for j, v := range x {
			l[j] = best(float32(id * v))
			q := float32(kv[l[j]])
			wq := float32(float32(v*v) * q)
			sqx += float32(wq * v)
			sq2 += float32(wq * q)
		}
		d = sqx / sq2
	}
	binary.LittleEndian.PutUint16(y, quant.EncodeHalf(d))
	for j := 0; j < 16; j++ {
		y[2+j] = byte(l[j] | l[j+16]<<4)
	}
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

// DequantBlocks reads format f's blocks b into dst as ggml's dequantizer
// does: the container's decoder where it reads the format, and q4_1's
// q*d+m (dequantize_row_q4_1) where it does not.
func DequantBlocks(dst []float32, b []byte, f BlockFormat) error {
	switch f.String() {
	case "q4_0":
		return quant.Dequant32(quant.Q4_0, b, dst)
	case "q5_0":
		return quant.Dequant32(quant.Q5_0, b, dst)
	case "q5_1":
		return quant.Dequant32(quant.Q5_1, b, dst)
	case "iq4_nl":
		return quant.Dequant32(quant.IQ4_NL, b, dst)
	case "q4_1":
		if len(b)%20 != 0 || len(dst) < len(b)/20*32 {
			return fmt.Errorf("oracle: %d bytes of q4_1 into %d floats", len(b), len(dst))
		}
		for i := 0; i < len(b)/20; i++ {
			y := b[20*i:]
			d, m := half(binary.LittleEndian.Uint16(y)), half(binary.LittleEndian.Uint16(y[2:]))
			for j := 0; j < 16; j++ {
				dst[32*i+j] = float32(y[4+j]&15)*d + m
				dst[32*i+j+16] = float32(y[4+j]>>4)*d + m
			}
		}
		return nil
	}
	return fmt.Errorf("oracle: no dequantizer for %v", f)
}

// BlockRow is ggml's blocks of a row of w floats in format f, as
// QuantizeBlocks writes them, laid out by kind as graph.BlockRegions states
// it: the scales (a word a block, d low and m high, where the format has a
// minimum, else a word two blocks), the fifth bits (a word a block), the
// nibbles (four words a block): the row a composed history holds.
func BlockRow(b []byte, f BlockFormat, w int) []uint32 {
	nb := (w + 31) / 32
	bb, mins, hasFifths := blockOf(f)
	scales, fifths := 0, (nb+1)/2
	if mins {
		fifths = nb
	}
	nibbles := fifths
	if hasFifths {
		nibbles += nb
	}
	out := make([]uint32, nibbles+4*nb)
	le := binary.LittleEndian
	for i := 0; i < nb; i++ {
		y := b[i*bb:]
		d, off := uint32(le.Uint16(y)), 2
		if mins {
			out[scales+i] = d | uint32(le.Uint16(y[2:]))<<16
			off = 4
		} else {
			out[scales+i/2] |= d << (16 * (i % 2))
		}
		if hasFifths {
			out[fifths+i] = le.Uint32(y[off:])
			off += 4
		}
		for k := 0; k < 4; k++ {
			out[nibbles+4*i+k] = le.Uint32(y[off+4*k:])
		}
	}
	return out
}
