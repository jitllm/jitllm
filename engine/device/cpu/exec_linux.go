//go:build linux

package cpu

import (
	"fmt"
	"runtime"
	"syscall"
)

// Code is mapped, executable machine code.
type Code struct {
	page  []byte
	entry *byte
	Size  int
	Name  string // symbol reported to perf (Target.MapNamed)
}

// Map makes code executable using the W^X sequence: map read+write, copy, then
// flip the whole page to read+execute. The page is never simultaneously
// writable and executable, so a stray write cannot become arbitrary execution.
// See exec_darwin.go for the macOS hardened-runtime caveat.

func mapExec(code []byte) (*Code, error) {
	if len(code) == 0 {
		return nil, fmt.Errorf("jit: refusing to map empty code")
	}
	page, err := syscall.Mmap(-1, 0, len(code),
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		return nil, fmt.Errorf("jit: mmap %d bytes: %w", len(code), err)
	}
	copy(page, code)
	if err := syscall.Mprotect(page, syscall.PROT_READ|syscall.PROT_EXEC); err != nil {
		syscall.Munmap(page)
		return nil, fmt.Errorf("jit: mprotect: %w", err)
	}
	c := &Code{page: page, entry: &page[0], Size: len(code)}
	// A leaked RX mapping is not reclaimed by the GC on its own.
	runtime.SetFinalizer(c, func(c *Code) { c.Close() })
	return c, nil
}

// Close unmaps the code. Calling it while a kernel is running is a segfault, so
// the caller owns that ordering.
func (c *Code) Close() error {
	if c.page == nil {
		return nil
	}
	p := c.page
	c.page, c.entry = nil, nil
	runtime.SetFinalizer(c, nil)
	return syscall.Munmap(p)
}

// Call enters the kernel with args in RDI.
//
// args must stay reachable for the whole call. It does, because it is an
// argument of this function and the assembly stub declares it as a pointer, so
// the GC sees it — but every pointer the kernel follows must be reachable from
// args itself, never smuggled in as a uintptr.
func (c *Code) Call(args *Args) {
	if c.entry == nil {
		panic("jit: Call on closed code")
	}
	callKernel(c.entry, args)
	runtime.KeepAlive(c)
	runtime.KeepAlive(args)
}
