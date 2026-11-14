package nn

import (
	"fmt"

	"github.com/jitllm/jitllm/graph"
)

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
//	KVQ4_0, KVQ4_1, KVQ5_0, KVQ5_1, KVIQ4NL  llama.cpp's blocks per head row
//	       (graph.BlockFmt): a row of hd elements is ceil(hd/32) blocks in
//	       ggml's layout, the last zero-padded, quantized as ggml's
//	       reference quantizer does and rounded up to a float32 slot
//
// A q8 or block row is the quantization unit, so such a cache is addressed in whole rows:
// kvStride must be a multiple of hd, and a kernel's hd is the row's width (an
// MLA value that is a prefix of a wider row is not expressible, and its cache
// is refused at the model).
type KVFmt uint8

const (
	KVF32 KVFmt = iota
	KVF16
	KVQ8
	KVQ4_0
	KVQ4_1
	KVQ5_0
	KVQ5_1
	KVIQ4NL
)

// KVFormats is how many KV formats there are, what a table by format holds.
const KVFormats = int(KVIQ4NL) + 1

// Block is f's llama.cpp block format, 0 for one that is not blocks.
func (f KVFmt) Block() graph.BlockFmt {
	switch f {
	case KVQ4_0:
		return graph.BlockQ4_0
	case KVQ4_1:
		return graph.BlockQ4_1
	case KVQ5_0:
		return graph.BlockQ5_0
	case KVQ5_1:
		return graph.BlockQ5_1
	case KVIQ4NL:
		return graph.BlockIQ4NL
	}
	return 0
}

// Quantized is whether f holds head rows quantized whole: q8 rows or blocks.
func (f KVFmt) Quantized() bool { return f == KVQ8 || f.Block() != 0 }

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
	if b := f.Block(); b != 0 {
		return b.String()
	}
	return fmt.Sprintf("KVFmt(%d)", uint8(f))
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
	if b := f.Block(); b != 0 {
		return 4 * graph.BlockRowWords(b, hd)
	}
	return 4 * hd
}
