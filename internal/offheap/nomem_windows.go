package offheap

import (
	"errors"
	"fmt"
	"syscall"
)

// The system error codes VirtualAlloc states for want of memory.
const (
	errorNotEnoughMemory syscall.Errno = 8
	errorOutOfMemory     syscall.Errno = 14
	errorCommitmentLimit syscall.Errno = 1455
)

// hostErr is err, marked ErrNoMemory where VirtualAlloc refused for memory or
// for the commit limit.
func hostErr(err error) error {
	if errors.Is(err, errorNotEnoughMemory) || errors.Is(err, errorOutOfMemory) || errors.Is(err, errorCommitmentLimit) {
		return fmt.Errorf("%w: %w", ErrNoMemory, err)
	}
	return err
}
