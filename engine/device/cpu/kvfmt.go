package cpu

import "fmt"

// KVFmt is the element format of a KV cache, which every attention kernel
// bakes: it decides how a kernel widens what it reads and how far apart two
// elements and two positions are in bytes.
//
//	KVF32  four bytes an element
//	KVF16  binary16, two bytes an element
//	KVQ8   q8_0 per head row: a row of hd elements is nb = ceil(hd/32) blocks,
//	       stored as nb*32 int8 (the last block zero-padded) followed by nb
//	       pairs {d, s} of float32; element e is d[e/32] * q[e]
//
// A q8 row is laid out as the activation quantizer writes it
// (QuantActKernels.Run over Q8_0 at a 32-element window: the int8 run, then
// one pair a block), so the append is that generated kernel and nothing else,
// and its rounding is llama.cpp's quantize_row_q8_0 (amax/127, round half
// away from zero). The pair's second float is the quantizer's block-sum term,
// which no attention kernel reads: the row costs 40 bytes a block where
// llama.cpp's block_q8_0 costs 34. A store kernel writing the scale alone (36)
// or as binary16 (34) is the upgrade.
//
// A q8 row is the quantization unit, so a q8 cache is addressed in whole rows:
// kvStride must be a multiple of hd, and a kernel's hd is the row's width (an
// MLA value that is a prefix of a wider row is not expressible, and its cache
// is refused at the model).
type KVFmt uint8

const (
	KVF32 KVFmt = iota
	KVF16
	KVQ8
)

// KVQ8Block is the elements one q8 scale covers.
const KVQ8Block = 32

func (f KVFmt) String() string {
	switch f {
	case KVF32:
		return "f32"
	case KVF16:
		return "f16"
	case KVQ8:
		return "q8_0"
	}
	return fmt.Sprintf("KVFmt(%d)", uint8(f))
}

// KVOf is the format a legacy width flag names.
func KVOf(f16 bool) KVFmt {
	if f16 {
		return KVF16
	}
	return KVF32
}

// KVRowBytes is the bytes one head row of hd elements occupies in format f.
// Every format's row is a multiple of four bytes for the hd the caches use
// (f16 needs hd even), so a row starts on a float32 slot.
func KVRowBytes(f KVFmt, hd int) int {
	switch f {
	case KVF16:
		return 2 * hd
	case KVQ8:
		nb := (hd + KVQ8Block - 1) / KVQ8Block
		return nb * (KVQ8Block + 8)
	}
	return 4 * hd
}

// kvAddr is how a kernel addresses one head row run of format f at head width
// hd: the byte offset of element e, the byte offset of the scale covering it
// (q8 only) and the byte stride between positions.
type kvAddr struct {
	f  KVFmt
	hd int
}

func newKVAddr(f KVFmt, hd, kvStride int) (kvAddr, error) {
	if hd <= 0 {
		return kvAddr{}, fmt.Errorf("jit: attention kernel: hd=%d must be positive", hd)
	}
	if f > KVQ8 {
		return kvAddr{}, fmt.Errorf("jit: unknown KV format %d", f)
	}
	if f == KVQ8 && (kvStride <= 0 || kvStride%hd != 0) {
		return kvAddr{}, fmt.Errorf("jit: a q8_0 KV cache is addressed in whole rows: kvStride=%d is not a multiple of hd=%d", kvStride, hd)
	}
	return kvAddr{f, hd}, nil
}

// off is element e's byte offset from the row's start.
func (k kvAddr) off(e int) int32 {
	switch k.f {
	case KVF16:
		return int32(2 * e)
	case KVQ8:
		return int32(e)
	}
	return int32(4 * e)
}

// scale is the byte offset of the scale covering element e (q8 only).
func (k kvAddr) scale(e int) int32 {
	nb := (k.hd + KVQ8Block - 1) / KVQ8Block
	return int32(nb*KVQ8Block + (e/KVQ8Block)*8)
}

// stride is the byte distance between consecutive positions, kvStride being
// in elements.
func (k kvAddr) stride(kvStride int) int32 {
	if k.f == KVQ8 {
		return int32(kvStride / k.hd * KVRowBytes(KVQ8, k.hd))
	}
	return k.off(kvStride)
}
