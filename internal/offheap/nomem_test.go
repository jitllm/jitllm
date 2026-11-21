//go:build linux || darwin || windows

package offheap

import (
	"errors"
	"testing"
)

// wantNoMemory fails unless asking for n bytes the way ask does is refused
// as ErrNoMemory.
func wantNoMemory(t *testing.T, what string, n int, ask func() error) {
	t.Helper()
	if err := ask(); !errors.Is(err, ErrNoMemory) {
		t.Fatalf("a %d-byte %s: %v, want ErrNoMemory", n, what, err)
	}
}
