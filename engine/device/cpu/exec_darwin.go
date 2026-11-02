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
	// tier is the tier this kernel was generated for, read from its own
	// declaration when it was mapped; see checkTier.
	tier Tier
}

// Map makes code executable using the W^X sequence: map read+write, copy, then
// flip the whole page to read+execute. The page is never simultaneously
// writable and executable.
//
// These are the same three syscalls as Linux. MAP_JIT and
// pthread_jit_write_protect_np are not needed (the latter would need cgo): they
// exist for a page that is writable and executable at once, and jitllm uses two
// disjoint phases.
//
// Apple silicon does SIGKILL a process silently (exit 137, no Go error) on its
// first instruction fetch from an anonymous RX page, but only when the binary
// carries the hardened runtime flag (codesign -o runtime) without
// com.apple.security.cs.allow-unsigned-executable-memory. Linker ad-hoc and
// plain ad-hoc signatures run fine; the arm64 test runs sign with the
// entitlement anyway, as insurance against someone adding -o runtime.
//
// MapNamed is Map plus a symbol name for the JIT profiler map. Darwin's
// samplers take symbols from a different mechanism, so the name is accepted
// and ignored here rather than making every caller branch on GOOS.
func mapExecNamed(code []byte, name string) (*Code, error) { return mapExec(code) }

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

// Call enters the kernel with args in x0, AAPCS64's first integer argument and
// the analogue of amd64's RDI.
//
// args must stay reachable for the whole call. It does, because it is an
// argument of this function and the assembly stub declares it as a pointer, so
// the GC sees it — but every pointer the kernel follows must be reachable from
// args itself, never smuggled in as a uintptr.
//
// No explicit instruction-cache flush is needed: mprotect to PROT_EXEC is the
// synchronization point on arm64, the kernel doing the I-cache maintenance.
// This is why W^X is not merely a hygiene choice here.
func (c *Code) Call(args *Args) {
	if c.entry == nil {
		panic("jit: Call on closed code")
	}
	checkTier(c)
	callKernel(c.entry, args)
	runtime.KeepAlive(c)
	runtime.KeepAlive(args)
}
