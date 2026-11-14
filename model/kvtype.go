package model

import (
	"fmt"
)

// KVType is the KV cache's element format, forced per model with WithKVType
// (or SetKVType). Unforced, the cache is f32 or f16 by nn.KVWidthPaysOff, and
// a quantized cache is never chosen for the caller: it changes the answer.
//
//	KVF32   4 bytes an element
//	KVF16   binary16, 2 bytes an element
//	KVQ8_0  q8_0 per head row (32-element blocks, llama.cpp's rounding):
//	        1.25 bytes an element as the cache lays it out
//	KVQ4_0, KVQ4_1, KVQ5_0, KVQ5_1  llama.cpp's blocks of those formats per
//	        head row, quantized as ggml's reference quantizer does: 0.5625,
//	        0.625, 0.6875 and 0.75 bytes an element
//
// A q8 history's row is its heads' q8 rows (composer.Pool.Q8): the append
// quantizes each head's row whole, and every read widens its element. MLA's
// one latent row is a head row whose value is read as its prefix, and an
// indexer's key as a span of the row. A forced q8 cache rounds an indexer's
// key too, so its selection is the rounded key's: the caller's choice. A
// block history's row is its heads' rows of blocks (composer.Pool.Block),
// held as a q8 history's are.
type KVType uint8

const (
	KVF32 KVType = iota
	KVF16
	KVQ8_0
	KVQ4_0
	KVQ4_1
	KVQ5_0
	KVQ5_1
)

func (t KVType) String() string {
	switch t {
	case KVF16:
		return "f16"
	case KVQ8_0:
		return "q8_0"
	case KVQ4_0:
		return "q4_0"
	case KVQ4_1:
		return "q4_1"
	case KVQ5_0:
		return "q5_0"
	case KVQ5_1:
		return "q5_1"
	}
	return "f32"
}

// ParseKVType reads a format's name as KVType.String spells it.
func ParseKVType(s string) (KVType, error) {
	for _, t := range []KVType{KVF32, KVF16, KVQ8_0, KVQ4_0, KVQ4_1, KVQ5_0, KVQ5_1} {
		if t.String() == s {
			return t, nil
		}
	}
	return 0, fmt.Errorf("model: KV type %q is not one of f32, f16, q8_0, q4_0, q4_1, q5_0, q5_1", s)
}
