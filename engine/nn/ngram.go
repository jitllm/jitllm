//go:build amd64 || arm64

package nn

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// The n-gram hash (jit_old/cpu/ngram_const.go has the contract): Qwen3.8-Flash-
// Next's table rows for one position, on generated integer code.

type ngramKey struct {
	t      cpu.Tier
	n, per int
	reset  bool
}

var ngramHashes sync.Map

// NGramConsts is the hash's constant block (cpu.NGramConsts).
func NGramConsts(eos uint64, mul, off, voc []uint64) []byte {
	return cpu.NGramConsts(eos, mul, off, voc)
}

// NGramHashJIT writes the table rows of one position into rows, (n-1)*per of
// them, and the next position's predecessors into next (n-1), from its window
// win (n: the token, then its predecessors as id+1, zero for none), with the
// constant block konst and ids, n words of scratch.
func NGramHashJIT(rows []uint64, next, win []uint32, konst []byte, ids []uint64, per int, reset bool) {
	n := len(win)
	if n < 2 || per < 1 || len(rows) < (n-1)*per || len(next) < n-1 || len(ids) < n ||
		len(konst) < 8+8*n+16*(n-1)*per {
		panic(fmt.Sprintf("jit: ngram hash over a window of %d, %d heads each, into %d rows and %d "+
			"predecessors, %d bytes of constants", n, per, len(rows), len(next), len(konst)))
	}
	k := ngramKey{cpu.HostTier(), n, per, reset}
	c, ok := ngramHashes.Load(k)
	if !ok {
		c, _ = ngramHashes.LoadOrStore(k, mustEmit("ngramhash")(cpu.EmittersFor(k.t).NGramHash(n, per, reset)))
	}
	c.(*cpu.Code).Call(&cpu.Args{
		Out:     (*float32)(unsafe.Pointer(&rows[0])),
		W:       (*byte)(unsafe.Pointer(&win[0])),
		Scr:     &konst[0],
		Scratch: (*byte)(unsafe.Pointer(&ids[0])),
		Out2:    (*float32)(unsafe.Pointer(&next[0])),
	})
}
