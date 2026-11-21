package offheap

// A Committer commits and drops ranges of a ReserveOffHeap reservation for
// the owner that holds it, the platform's entry points resolved once when it
// is made: a warm step's commits and drops allocate nothing.
type Committer struct{ procs }

// NewCommitter is a Committer for this platform.
func NewCommitter() Committer { return Committer{newProcs()} }

// Commit is CommitOffHeap through c's entry points.
func (c Committer) Commit(b []byte) error { return c.commit(b) }

// Drop is DropOffHeap through c's entry points.
func (c Committer) Drop(b []byte) { c.drop(b) }
