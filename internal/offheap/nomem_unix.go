//go:build linux || darwin

package offheap

import (
	"errors"
	"fmt"
	"syscall"
)

// hostErr is err, marked ErrNoMemory where mmap refused for memory (ENOMEM).
func hostErr(err error) error {
	if errors.Is(err, syscall.ENOMEM) {
		return fmt.Errorf("%w: %w", ErrNoMemory, err)
	}
	return err
}
