//go:build !linux && !darwin

package kernels

import "unsafe"

// mapAligned is the fallback where Go's syscall package has no anonymous mmap:
// over-allocate by one alignment and re-slice to the first aligned byte.
//
// The address is aligned but it is a Go heap address, so it must not be
// handed to a driver that keeps it, and releasing it returns memory to the
// runtime rather than the OS. A host on this path can page but cannot import
// (see arena_unix.go). The re-slice keeps the whole backing array alive.
func mapAligned(n int) ([]byte, bool, error) {
	a := uintptr(HostAlign())
	b := make([]byte, uintptr(n)+a)
	off := int(-uintptr(unsafe.Pointer(&b[0])) & (a - 1))
	return b[off : off+n : off+n], false, nil
}

func unmapAligned([]byte) {}

// MapOffHeap is the heap fallback here; see arena_unix.go.
func MapOffHeap(n int) (b []byte, mapped bool, err error) { return mapAligned(n) }

// UnmapOffHeap does nothing to a heap fallback.
func UnmapOffHeap(b []byte) {}
