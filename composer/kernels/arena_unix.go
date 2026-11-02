//go:build linux || darwin

package kernels

import (
	"syscall"
	"unsafe"
)

// mapAligned returns n bytes of zeroed host memory whose base address is page
// aligned, and therefore at least HostAlign() aligned.
//
// mmap rather than a Go slice, for three reasons besides alignment:
//
//   - A stable address to hand out. The arena hands host pointers to drivers
//     that keep them across calls (VK_EXT_external_memory_host,
//     newBufferWithBytesNoCopy:); Go does not promise heap objects never move.
//   - Memory that comes back when released: munmap is immediate, where a
//     dropped []byte returns to the OS whenever the runtime decides.
//   - Nothing for the GC to scan or account against its heap target.
//
// It is zeroed by the kernel, which makes a partially filled region
// deterministic.
func mapAligned(n int) ([]byte, bool, error) {
	b, err := syscall.Mmap(-1, 0, n, syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		return nil, false, err
	}
	return b, true, nil
}

// MapOffHeap is n bytes of zeroed, page-aligned memory outside the Go heap, and
// whether it is a mapping: format/jlm's page frames and dense region come from
// here as the arena does, for the reasons above. Where there is no anonymous
// mmap it is the heap fallback, and mapped is false.
func MapOffHeap(n int) (b []byte, mapped bool, err error) { return mapAligned(n) }

// UnmapOffHeap returns memory MapOffHeap mapped.
func UnmapOffHeap(b []byte) { unmapAligned(b) }

func unmapAligned(b []byte) {
	if len(b) > 0 {
		syscall.Munmap(b)
	}
}

// ReserveOffHeap is n bytes of address space outside the Go heap that commits
// no memory until a byte of it is written, and whether it is a mapping: a
// range the caller fills in pieces, each piece costing memory only while it
// holds something (DropOffHeap).
func ReserveOffHeap(n int) ([]byte, bool, error) {
	b, err := syscall.Mmap(-1, 0, n, syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_PRIVATE|syscall.MAP_ANON|syscall.MAP_NORESERVE)
	if err != nil {
		return nil, false, err
	}
	return b, true, nil
}

// DropOffHeap hands the memory behind b, a range of a ReserveOffHeap
// reservation, back to the system: what it held is gone, and the range is
// written again before it is read.
func DropOffHeap(b []byte) {
	if len(b) > 0 {
		syscall.Syscall(syscall.SYS_MADVISE, uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)), syscall.MADV_DONTNEED)
	}
}
