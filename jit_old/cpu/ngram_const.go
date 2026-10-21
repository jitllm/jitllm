package cpu

import (
	"encoding/binary"
	"fmt"
)

// The n-gram hash: Qwen3.8-Flash-Next's n-gram embedding rows for one
// position (ngram.go on amd64, ngram_a64.go on arm64), its whole arithmetic.
// It is integer code, so the AVX2 and SSE tiers share one emitter.
//
//	W        the window, n uint32: the position's token, then its n-1
//	         predecessors nearest first, each stored as id+1 with zero for
//	         none
//	Out      the table rows, (n-1)*per uint64, written
//	Out2     the next position's predecessors, n-1 uint32 (id+1), written:
//	         this token, then the window's first n-2 predecessors
//	Scr      NGramConsts: eos, the n multipliers, then per head its offset
//	         and its vocabulary, each a uint64
//	Scratch  n uint64 of the kernel's own
//
// A predecessor that is none, or at or before an end-of-sequence id (with
// reset), reads as eos; the token's own end of sequence does not cut its
// window. Head h = (g-2)*per + k of the g-grams reads row
//
//	(c0*m0 ^ c1*m1 ^ ... ^ c(g-1)*m(g-1)) mod voc[h] + off[h]
//
// unsigned 64-bit, wrapping, over the window's ids c.

// ngramConstEOS and friends are byte offsets into NGramConsts.
const ngramConstEOS = 0

func ngramConstMul(j int) int32    { return int32(8 + 8*j) }
func ngramConstOff(n, h int) int32 { return int32(8 + 8*n + 16*h) }
func ngramConstVoc(n, h int) int32 { return ngramConstOff(n, h) + 8 }
func ngramHeads(n, per int) int    { return (n - 1) * per }
func ngramShape(n, per int) error {
	if n < 2 || per < 1 || ngramHeads(n, per) > 64 {
		return fmt.Errorf("jit: ngram: %d-grams with %d heads each", n, per)
	}
	return nil
}

// NGramConsts is the constant block for an n-gram hash of len(mul)-grams.
func NGramConsts(eos uint64, mul, off, voc []uint64) []byte {
	b := make([]byte, 8+8*len(mul)+16*len(off))
	binary.LittleEndian.PutUint64(b, eos)
	for j, m := range mul {
		binary.LittleEndian.PutUint64(b[ngramConstMul(j):], m)
	}
	for h := range off {
		binary.LittleEndian.PutUint64(b[ngramConstOff(len(mul), h):], off[h])
		binary.LittleEndian.PutUint64(b[ngramConstVoc(len(mul), h):], voc[h])
	}
	return b
}
