//go:build !windows

package offheap

// procs is nothing here: a commit and a drop are CommitOffHeap's and
// DropOffHeap's, which resolve nothing.
type procs struct{}

func newProcs() procs { return procs{} }

func (procs) commit(b []byte) error { return CommitOffHeap(b) }

func (procs) drop(b []byte) { DropOffHeap(b) }
