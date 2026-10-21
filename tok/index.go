package tok

import (
	"encoding/binary"
	"math/bits"

	"github.com/jitllm/jitllm/format/jlm"
)

// index finds a token's id by its text: open addressing over the vocabulary's
// string table. A map[string]int32 of a vocabulary is a pointer per token the
// collector scans on every cycle; this is pointer-free. A slot holds the upper
// half of its string's hash and its id plus one (zero for empty), and where
// the string lies in the table, so a probe past another string costs an
// integer compare and a hit reads the slot and the string's bytes. The first
// id of a duplicated string wins.
type index struct {
	all  string
	slot []slot
	mask uint64
	// join is find2's buffer for the two halves of a merge.
	join []byte
}

type slot struct {
	key    uint64 // hash &^ 0xffffffff | id+1
	off, n uint32
}

func newIndex(t jlm.Strings) index {
	n := 1
	for n < 2*t.Len() {
		n <<= 1
	}
	x := index{all: t.All(), slot: make([]slot, n), mask: uint64(n - 1)}
	for i := range t.Len() {
		s := t.At(i)
		if _, dup := x.find(s); dup {
			continue
		}
		h := hashString(s)
		j := h & x.mask
		for x.slot[j].key != 0 {
			j = (j + 1) & x.mask
		}
		lo, hi := t.Span(i)
		x.slot[j] = slot{key: h&^0xffffffff | uint64(i+1), off: lo, n: hi - lo}
	}
	return x
}

func (x *index) find(s string) (int32, bool) {
	if len(x.slot) == 0 {
		return 0, false
	}
	h := hashString(s)
	tag := h &^ 0xffffffff
	for j := h & x.mask; ; j = (j + 1) & x.mask {
		e := &x.slot[j]
		if e.key == 0 {
			return 0, false
		}
		if e.key&^0xffffffff == tag && int(e.n) == len(s) && x.all[e.off:e.off+e.n] == s {
			return int32(e.key&0xffffffff) - 1, true
		}
	}
}

func (x *index) findBytes(s []byte) (int32, bool) {
	if len(x.slot) == 0 {
		return 0, false
	}
	h := hashBytes(s)
	tag := h &^ 0xffffffff
	for j := h & x.mask; ; j = (j + 1) & x.mask {
		e := &x.slot[j]
		if e.key == 0 {
			return 0, false
		}
		if e.key&^0xffffffff == tag && int(e.n) == len(s) && x.all[e.off:e.off+e.n] == string(s) {
			return int32(e.key&0xffffffff) - 1, true
		}
	}
}

// find2 is the id of a+b, joined in a buffer the index keeps. It is the
// tokenizer's load, a merge at a time; an encode does not call it.
func (x *index) find2(a, b string) (int32, bool) {
	x.join = append(append(x.join[:0], a...), b...)
	return x.findBytes(x.join)
}

// hashString is a hash for short strings: eight bytes at a time, each word
// mixed in, and MurmurHash3's finalizer. A token is a few bytes, where a
// seeded runtime hash costs more than the probe.
func hashString(s string) uint64 {
	h := uint64(len(s)) * 0x9e3779b97f4a7c15
	for len(s) >= 8 {
		h = bits.RotateLeft64(h^binary.LittleEndian.Uint64([]byte(s[:8]))*0xff51afd7ed558ccd, 31) * 0xc4ceb9fe1a85ec53
		s = s[8:]
	}
	var w uint64
	for i := range len(s) {
		w |= uint64(s[i]) << (8 * i)
	}
	return fmix(h ^ w*0xff51afd7ed558ccd)
}

func hashBytes(b []byte) uint64 { return hashString(string(b)) }

func fmix(h uint64) uint64 {
	h ^= h >> 33
	h *= 0xff51afd7ed558ccd
	h ^= h >> 33
	h *= 0xc4ceb9fe1a85ec53
	return h ^ h>>33
}
