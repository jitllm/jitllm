package offheap

import "testing"

// TestAMappingPastTheHostIsNoMemory commits more than any host's commit
// limit, inside the 128 TiB user address space (past it VirtualAlloc states
// an invalid parameter, not a want of memory): a mapping, and a commit of a
// reservation, the arena's path here. Each refusal is ErrNoMemory, so a caller
// answers it as a limit rather than a fault. With hostErr returning err
// unmarked it fails.
func TestAMappingPastTheHostIsNoMemory(t *testing.T) {
	const n = 1 << 46
	wantNoMemory(t, "mapping", n, func() error {
		b, _, err := MapOffHeap(n)
		if err == nil {
			UnmapOffHeap(b)
		}
		return err
	})
	r, _, err := ReserveOffHeap(n)
	if err != nil {
		t.Fatalf("a %d-byte reservation: %v", n, err)
	}
	defer UnmapOffHeap(r)
	wantNoMemory(t, "commit", n, func() error { return CommitOffHeap(r) })
}
