package nn

import (
	"unsafe"

	"github.com/jitllm/jitllm/nn/history"
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
	// Entries and EntryPages are the block's entries plane (its history 1)
	// and its page table, where the block keeps one.
	Entries    unsafe.Pointer
	EntryPages []uint32
}

// History is a session's history as the CPU device asks for it: the policy
// that holds its pages (residency, sharing, the store) is the engine's.
type History interface {
	// View is the session's view of the device the history's blocks read
	// their pages and recurrent state through; nil reads as an empty one.
	View() *history.View
	// MigrateKV takes block li's history of pos positions from the floats k
	// and v (in), or gives it into them, its rows converted from or to the
	// history's own format through conv where they are not floats.
	MigrateKV(li int, k, v []float32, pos int, in bool, conv KVConvDevice) bool
	// Own makes the page block li writes pos into resident and its own,
	// where its history is paged: a row of a step across sessions writes
	// its own session's page before any session's rows read.
	Own(li, pos int)
	// Prepare makes block li's history ready for a row at pos, its page
	// table nseq sequences wide: the page pos is written into resident and
	// its own, the pages the row reads resident. A decline says why it is not.
	Prepare(li, pos, nseq int) (HistoryBinding, Decline, error)
	// Commit records that block li wrote the row at pos.
	Commit(li, pos int)
	// PrepareRows makes block li's history ready for rows at pos, in any
	// order: every row's page written before any is read, the keys every
	// row reads resident; keys is the most the call's form reads. A decline
	// says why the form cannot take them, an error why a page could not be
	// had.
	PrepareRows(li int, pos []int, keys int) (Decline, error)
	// Rows is how a call of rows binds block li's history, its page table
	// nseq sequences wide.
	Rows(li, nseq int) (HistoryBinding, error)
	// KeyStart is where block li's keys the row at pos reads start.
	KeyStart(li, pos int) int
}

// HistoryOf is what a session's calls read their history from: the
// session's history as it is at the call, which a relayout replaces.
type HistoryOf interface{ History() History }

// Prompt is a session's say over a linear block's summary as its prompt's
// rows run on the CPU device: whether a speculation step keeps each row's
// (Snaps, Snap), and the restore points the prompt seals it at (Points,
// Point, Seal).
type Prompt interface {
	Points() bool
	Point(pos int) bool
	Seal(li, slot, pos int)
}

// SessionBinder is a device that binds a session's view of it to the
// instance it runs for: prev is the session's view from an earlier lease
// (nil the first time), bound again in place, h its history and p its
// prompt. It says false where the device runs no blocks for the instance.
type SessionBinder interface {
	BindSession(prev LayerRunner, h HistoryOf, p Prompt) (LayerRunner, bool)
}
