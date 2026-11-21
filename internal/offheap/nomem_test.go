//go:build linux || darwin || windows

package offheap

import (
	"errors"
	"testing"
)

// TestAMappingPastTheHostIsNoMemory asks for more than any host's address
// space holds: the refusal is ErrNoMemory, so a caller answers it as a limit
// rather than a fault. With hostErr returning err unmarked it fails.
func TestAMappingPastTheHostIsNoMemory(t *testing.T) {
	const n = 1 << 50
	if b, _, err := MapOffHeap(n); !errors.Is(err, ErrNoMemory) {
		if err == nil {
			UnmapOffHeap(b)
		}
		t.Fatalf("a %d-byte mapping: %v, want ErrNoMemory", n, err)
	}
	if b, mapped, err := ReserveOffHeap(n); !errors.Is(err, ErrNoMemory) {
		if mapped {
			UnmapOffHeap(b)
		}
		t.Fatalf("a %d-byte reservation: %v, want ErrNoMemory", n, err)
	}
}
