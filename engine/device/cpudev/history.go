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
	// Slots is the pages a sequence's arena holds, at least 1: where a row
	// of sequence s finds its pages in the table.
	Slots int
}

// History is a session's history as the CPU device asks for it: the policy
// that holds its pages (residency, sharing, the store) is the engine's.
type History interface {
	// View is the session's view of the device the history's blocks read
	// their pages and recurrent state through; nil reads as an empty one.
	View() *View
	// Own makes the page block li writes pos into resident and its own,
	// where its history is paged: a row of a step across sessions writes
	// its own session's page before any session's rows read.
	Own(li, pos int)
	// Prepare makes block li's history ready for a row at pos, its page
	// table nseq sequences wide: the page pos is written into resident and
	// its own, the pages the row reads resident. A decline says why it is not.
	Prepare(li, pos, nseq int) (HistoryBinding, nn.Decline, error)
	// Commit records that block li wrote the row at pos.
	Commit(li, pos int)
	// PrepareRows makes block li's history ready for rows at pos, in any
	// order: every row's page written before any is read, the keys every
	// row reads resident; keys is the most the call's form reads. A decline
	// says why not.
	PrepareRows(li int, pos []int, keys int) nn.Decline
	// Rows is how a call of rows binds block li's history, its page table
	// nseq sequences wide.
	Rows(li, nseq int) (HistoryBinding, error)
	// KeyStart is where block li's keys the row at pos reads start.
	KeyStart(li, pos int) int
}
