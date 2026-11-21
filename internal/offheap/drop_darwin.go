package offheap

import (
	"syscall"
	"unsafe"
)

// DropOffHeap hands the memory behind b, a range of a ReserveOffHeap
// reservation, back to the system: what it held is gone, and the range reads
// zeros when it is next touched. Darwin's MADV_DONTNEED may leave an
// anonymous page's bytes in place, so the range is cleared before the advice.
func DropOffHeap(b []byte) {
	if len(b) > 0 {
		clear(b)
		syscall.Syscall(syscall.SYS_MADVISE, uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)), syscall.MADV_DONTNEED)
	}
}
