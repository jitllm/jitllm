package offheap

import (
	"syscall"
	"unsafe"
)

// The mappings on Windows are VirtualAlloc's, for the reasons map_unix.go
// gives for mmap: a stable address, memory that goes back when released, and
// nothing for the collector to scan. A reservation is address space alone,
// and unlike MAP_NORESERVE a page of it is not committed by writing it, so a
// caller commits each range before filling it (CommitOffHeap).

// kernel32 is the named kernel32.dll entry point, resolved when called: a
// map, a commit, a drop and an unmap are the callers, none on a token's path.
func kernel32(name string) *syscall.LazyProc {
	return syscall.NewLazyDLL("kernel32.dll").NewProc(name)
}

const (
	memCommit     = 0x1000
	memReserve    = 0x2000
	memDecommit   = 0x4000
	memRelease    = 0x8000
	pageNoAccess  = 0x01
	pageReadWrite = 0x04
)

func virtualAlloc(at unsafe.Pointer, n int, kind, protect uintptr) ([]byte, error) {
	addr, _, err := kernel32("VirtualAlloc").Call(uintptr(at), uintptr(n), kind, protect)
	if addr == 0 {
		return nil, err
	}
	return unsafe.Slice((*byte)(unsafe.Add(nil, addr)), n), nil
}

// mapAligned is n bytes of zeroed memory, reserved and committed at an
// allocation-granularity boundary, which is page aligned.
func mapAligned(n int) ([]byte, bool, error) {
	b, err := virtualAlloc(nil, n, memReserve|memCommit, pageReadWrite)
	if err != nil {
		return nil, false, err
	}
	return b, true, nil
}

func unmapAligned(b []byte) {
	if len(b) > 0 {
		kernel32("VirtualFree").Call(uintptr(unsafe.Pointer(&b[0])), 0, memRelease)
	}
}

// MapOffHeap is n bytes of zeroed, page-aligned memory outside the Go heap;
// see map_unix.go.
func MapOffHeap(n int) (b []byte, mapped bool, err error) { return mapAligned(n) }

// UnmapOffHeap returns memory MapOffHeap mapped or ReserveOffHeap reserved,
// b starting where the mapping does.
func UnmapOffHeap(b []byte) { unmapAligned(b) }

// ReserveOffHeap is n bytes of address space outside the Go heap that holds
// no memory until a range of it is committed (CommitOffHeap).
func ReserveOffHeap(n int) ([]byte, bool, error) {
	b, err := virtualAlloc(nil, n, memReserve, pageNoAccess)
	if err != nil {
		return nil, false, err
	}
	return b, true, nil
}

// CommitOffHeap makes b, a range of a ReserveOffHeap reservation, memory the
// caller may write: zeros where it held nothing, what it held where it did.
func CommitOffHeap(b []byte) error {
	if len(b) == 0 {
		return nil
	}
	_, err := virtualAlloc(unsafe.Pointer(&b[0]), len(b), memCommit, pageReadWrite)
	return err
}

// DropOffHeap hands the memory behind b, a page-aligned range of a
// ReserveOffHeap reservation, back to the system: what it held is gone, and
// the range is committed again before it is written.
func DropOffHeap(b []byte) {
	if len(b) > 0 {
		kernel32("VirtualFree").Call(uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)), memDecommit)
	}
}
