package offheap

import (
	"fmt"
	"syscall"
	"unsafe"
)

// procs is kernel32's VirtualAlloc and VirtualFree, resolved once and called
// through syscall.SyscallN: a LazyProc's Call takes its arguments as a slice
// and returns its error boxed, and both allocate on every call.
type procs struct {
	alloc, free uintptr
	err         error
}

func newProcs() procs {
	k := syscall.NewLazyDLL("kernel32.dll")
	a, f := k.NewProc("VirtualAlloc"), k.NewProc("VirtualFree")
	if err := a.Find(); err != nil {
		return procs{err: fmt.Errorf("offheap: %w", err)}
	}
	if err := f.Find(); err != nil {
		return procs{err: fmt.Errorf("offheap: %w", err)}
	}
	return procs{alloc: a.Addr(), free: f.Addr()}
}

func (p procs) commit(b []byte) error {
	if len(b) == 0 {
		return nil
	}
	if p.err != nil {
		return p.err
	}
	if addr, _, e := syscall.SyscallN(p.alloc, uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)), memCommit, pageReadWrite); addr == 0 {
		return e
	}
	return nil
}

func (p procs) drop(b []byte) {
	if len(b) > 0 && p.err == nil {
		syscall.SyscallN(p.free, uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)), memDecommit)
	}
}
