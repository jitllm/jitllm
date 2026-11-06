package cpudev

import (
	"unsafe"

	"github.com/jitllm/jitllm/nn"
)

// HistoryBinding is a block's history as a call reads and writes it: its
// pages' keys and values, the page table, where the row's keys start and the
// positions a page holds.
type HistoryBinding struct {
	Keys, Values      unsafe.Pointer
	Pages             []uint32
	KeyStart, PagePos int
}

// History is a session's history as the CPU device asks for it: the policy
// that holds its pages (residency, sharing, the store) is the engine's.
type History interface {
	// Prepare makes block li's history ready for a row at pos, its page
	// table nseq sequences wide: the page pos is written into resident and
	// its own, the pages the row reads resident. A decline says why it is not.
	Prepare(li, pos, nseq int) (HistoryBinding, nn.Decline, error)
	// Commit records that block li wrote the row at pos.
	Commit(li, pos int)
}
