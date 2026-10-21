package model

import (
	"fmt"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// KVType is the KV cache's element format, forced per model with WithKVType
// (or SetKVType). Unforced, the cache is f32 or f16 by nn.KVWidthPaysOff, and
// a quantized cache is never chosen for the caller: it changes the answer.
//
//	KVF32   4 bytes an element
//	KVF16   binary16, 2 bytes an element
//	KVQ8_0  q8_0 per head row (32-element blocks, llama.cpp's rounding):
//	        1.25 bytes an element as the cache lays it out (cpu.KVFmt)
//
// The q8 cache runs on every host tier (AVX2, SSE, NEON): the append is the
// generated activation quantizer, the attention the f32 kernels with a
// widening load. Refused by name (KVTypeRefusal):
//
//   - MLA's latent row (deepseek2, kimi-k3, kimilinear, glm4moe-lite): the
//     value is a prefix of the key's row, which a q8 row quantized as a whole
//     cannot hand a kernel at its own width.
//   - The lightning indexer (deepseek32) and MiniMax Sparse Attention: both
//     read the indexer's key as a tail of the cached row and select blocks
//     discretely, so a rounded key is a different attention (the f16 default
//     is refused for the MSA key for the same reason).
//   - DeepSeek V4: its cached row carries the compressor's pending inputs,
//     which the references hold at f32 (the f16 cache is refused there too).
//
// q4_0 is not offered: its nibble pairs need a second widening in every
// attention kernel and a quantizer the activation path does not have.
type KVType uint8

const (
	KVF32 KVType = iota
	KVF16
	KVQ8_0
)

func (t KVType) String() string { return t.fmt().String() }

func (t KVType) fmt() cpu.KVFmt {
	switch t {
	case KVF16:
		return cpu.KVF16
	case KVQ8_0:
		return cpu.KVQ8
	}
	return cpu.KVF32
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

// KVTypeRefusal is why this model cannot run its cache at t, or nil.
func (c *Config) KVTypeRefusal(t KVType) error {
	if t != KVQ8_0 {
		return nil
	}
	switch {
	case c.Indexer():
		return fmt.Errorf("model: %s reads its lightning indexer's key from the cached row: "+
			"a q8_0 KV cache is not implemented for it", c.Arch)
	case c.MSA():
		return fmt.Errorf("model: %s selects blocks by its indexer's cached key (MiniMax Sparse Attention): "+
			"a q8_0 KV cache is not implemented for it", c.Arch)
	case c.QSARatio != 0:
		return fmt.Errorf("model: %s pools its QSA indexer's raw cached keys, which its references hold "+
			"at f32: a q8_0 KV cache is not implemented for it", c.Arch)
	case c.DSV4():
		return fmt.Errorf("model: %s caches its compressor's pending inputs beside the key: "+
			"a q8_0 KV cache is not implemented for it", c.Arch)
	case c.MLA():
		return fmt.Errorf("model: %s caches MLA's latent row, whose value is a prefix of the key's row: "+
			"a q8_0 KV cache is not implemented for it", c.Arch)
	}
	return nil
}

// SetKVType forces the cache's format for States this model creates
// afterwards; ClearKVF16 hands the choice back to the tuner.
func (m *Model) SetKVType(t KVType) error {
	if err := m.Cfg.KVTypeRefusal(t); err != nil {
		return err
	}
	m.opt.kvType, m.opt.kvTypeSet = t, true
	return nil
}

// KVType is the format this State's cache holds.
func (s *State) KVType() KVType {
	switch s.kvFmt {
	case cpu.KVF16:
		return KVF16
	case cpu.KVQ8:
		return KVQ8_0
	}
	return KVF32
}

// KVBytesPerToken is what one position of history costs in this State's
// cache, summed over every layer that keeps one: the format's row bytes times
// the layer's kv heads, K and V (an MLA row once). A recurrent layer costs
// nothing a position.
func (s *State) KVBytesPerToken() int {
	n := 0
	for li := s.lo; li < s.hi && li < len(s.kv.layers); li++ {
		if s.kv.layers[li].p == 0 || s.c.KVShared(li) {
			continue
		}
		l := s.kvlAt(li)
		b := l.slots(l.kvDim()) * 4
		if s.kv.layers[li].latent {
			n += b
		} else {
			n += 2 * b
		}
	}
	return n
}
