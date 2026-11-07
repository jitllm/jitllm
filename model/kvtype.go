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
//
// A q8 history's row is its heads' q8 rows (composer.Pool.Q8): the append
// quantizes each head's row whole, and every read widens its element. MLA's
// one latent row is a head row whose value is read as its prefix, and an
// indexer's key as a span of the row. A forced q8 cache rounds an indexer's
// key too, so its selection is the rounded key's: the caller's choice.
//
// q4_0 is not offered: its nibble pairs need a second widening in every
// attention kernel and a quantizer the activation path does not have.
type KVType uint8

const (
	KVF32 KVType = iota
	KVF16
	KVQ8_0
)

func (t KVType) String() string {
	switch t {
	case KVF16:
		return "f16"
	case KVQ8_0:
		return "q8_0"
	}
	return "f32"
}

// ParseKVType reads a format's name as KVType.String spells it.
func ParseKVType(s string) (KVType, error) {
	for _, t := range []KVType{KVF32, KVF16, KVQ8_0} {
		if t.String() == s {
			return t, nil
		}
	}
	return 0, fmt.Errorf("model: KV type %q is not one of f32, f16, q8_0", s)
}
