package offheap

import (
	"syscall"
	"unsafe"
)

// procs is kernel32's VirtualAlloc and VirtualFree, resolved once.
type procs struct {
	alloc, free *syscall.LazyProc
}

func newProcs() procs {
	k := syscall.NewLazyDLL("kernel32.dll")
	return procs{alloc: k.NewProc("VirtualAlloc"), free: k.NewProc("VirtualFree")}
}

func (p procs) commit(b []byte) error {
	if len(b) == 0 {
		return nil
	}
	if addr, _, err := p.alloc.Call(uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)), memCommit, pageReadWrite); addr == 0 {
		return err
	}
	return nil
}

func (p procs) drop(b []byte) {
	if len(b) > 0 {
		p.free.Call(uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)), memDecommit)
	}
}
