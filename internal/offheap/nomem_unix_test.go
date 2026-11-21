//go:build linux || darwin

package offheap

import "testing"

// TestAMappingPastTheHostIsNoMemory asks for more than any host's address
// space holds: mmap refuses it with ENOMEM, and the refusal is ErrNoMemory, so
// a caller answers it as a limit rather than a fault. With hostErr returning
// err unmarked it fails.
func TestAMappingPastTheHostIsNoMemory(t *testing.T) {
	const n = 1 << 50
	wantNoMemory(t, "mapping", n, func() error {
		b, _, err := MapOffHeap(n)
		if err == nil {
			UnmapOffHeap(b)
		}
		return err
	})
	wantNoMemory(t, "reservation", n, func() error {
		b, mapped, err := ReserveOffHeap(n)
		if mapped {
			UnmapOffHeap(b)
		}
		return err
	})
}
