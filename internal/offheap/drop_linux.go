package offheap

import (
	"syscall"
	"unsafe"
)

// DropOffHeap hands the memory behind b, a range of a ReserveOffHeap
// reservation, back to the system: what it held is gone, and the range reads
// zeros when it is next touched.
func DropOffHeap(b []byte) {
	if len(b) > 0 {
		syscall.Syscall(syscall.SYS_MADVISE, uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)), syscall.MADV_DONTNEED)
	}
}
