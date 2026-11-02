package offheap

import (
	"syscall"
	"unsafe"
)

// ReleaseOffHeap hands the memory of b, a span of a MapOffHeap mapping, back
// to the OS: it reads as zeros when next touched. A span the page size does
// not align is cleared in place.
func ReleaseOffHeap(b []byte) {
	if len(b) == 0 {
		return
	}
	if !pageSpan(b) || syscall.Madvise(b, syscall.MADV_DONTNEED) != nil {
		clear(b)
	}
}

// pageSpan says b starts and ends on a page boundary.
func pageSpan(b []byte) bool {
	ps := uintptr(syscall.Getpagesize())
	a := uintptr(unsafe.Pointer(&b[0]))
	return a%ps == 0 && uintptr(len(b))%ps == 0
}
