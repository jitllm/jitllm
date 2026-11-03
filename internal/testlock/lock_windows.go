//go:build windows

package testlock

import (
	"os"
	"syscall"
	"unsafe"
)

// LockFileEx and UnlockFileEx are not in Go's syscall package on Windows, and
// the root module takes no dependency for a test lock, so they are called
// through kernel32 directly.
func kernel32(proc string) *syscall.LazyProc {
	return syscall.NewLazyDLL("kernel32.dll").NewProc(proc)
}

const lockfileExclusiveLock = 0x2

// lock takes an exclusive lock on the file's first byte, blocking until it is
// free: the wait flock(LOCK_EX) gives on Unix.
func lock(f *os.File) error {
	var ol syscall.Overlapped
	r, _, err := kernel32("LockFileEx").Call(f.Fd(), lockfileExclusiveLock, 0, 1, 0, uintptr(unsafe.Pointer(&ol)))
	if r == 0 {
		return err
	}
	return nil
}

// unlock's result is moot: closing the handle releases the lock anyway.
func unlock(f *os.File) {
	var ol syscall.Overlapped
	kernel32("UnlockFileEx").Call(f.Fd(), 0, 1, 0, uintptr(unsafe.Pointer(&ol)))
}
