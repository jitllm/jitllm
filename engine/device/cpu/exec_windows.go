package cpu

import (
	"fmt"
	"runtime"
	"syscall"
	"unsafe"
)

// W^X on Windows: allocate read/write, copy, then flip to execute+read. The
// page is never simultaneously writable and executable. VirtualAlloc and
// VirtualProtect are reached through kernel32 with syscall.NewLazyDLL, to stay
// inside the standard library.
//
// FlushInstructionCache is required: on x86 it is nearly a no-op, but on arm64
// the caches are not coherent and stale instruction bytes fault at a plausible
// address. Windows on ARM is a real target, so it is always called.
var (
	kernel32              = syscall.NewLazyDLL("kernel32.dll")
	procVirtualAlloc      = kernel32.NewProc("VirtualAlloc")
	procVirtualFree       = kernel32.NewProc("VirtualFree")
	procVirtualProtect    = kernel32.NewProc("VirtualProtect")
	procFlushInstrCache   = kernel32.NewProc("FlushInstructionCache")
	procGetCurrentProcess = kernel32.NewProc("GetCurrentProcess")
)

const (
	memCommit       = 0x1000
	memReserve      = 0x2000
	memRelease      = 0x8000
	pageReadWrite   = 0x04
	pageExecuteRead = 0x20
)

// Code is mapped, executable machine code.
type Code struct {
	addr  uintptr
	entry *byte
	Size  int
	Name  string
	// tier is the tier this kernel was generated for, read from its own
	// declaration when it was mapped; see checkTier.
	tier Tier
}

// MapNamed is Map with a symbol name. Windows samplers do not read a perf-style
// JIT map, so the name is recorded and otherwise unused.
func mapExecNamed(code []byte, name string) (*Code, error) {
	c, err := mapExec(code)
	if err == nil {
		c.Name = name
	}
	return c, err
}

func mapExec(code []byte) (*Code, error) {
	if len(code) == 0 {
		return nil, fmt.Errorf("jit: refusing to map empty code")
	}
	addr, _, err := procVirtualAlloc.Call(0, uintptr(len(code)),
		memCommit|memReserve, pageReadWrite)
	if addr == 0 {
		return nil, fmt.Errorf("jit: VirtualAlloc %d bytes: %w", len(code), err)
	}
	copy(unsafe.Slice((*byte)(unsafe.Add(nil, addr)), len(code)), code)

	var old uint32
	ok, _, err := procVirtualProtect.Call(addr, uintptr(len(code)),
		pageExecuteRead, uintptr(unsafe.Pointer(&old)))
	if ok == 0 {
		procVirtualFree.Call(addr, 0, memRelease)
		return nil, fmt.Errorf("jit: VirtualProtect: %w", err)
	}
	proc, _, _ := procGetCurrentProcess.Call()
	procFlushInstrCache.Call(proc, addr, uintptr(len(code)))

	c := &Code{addr: addr, entry: (*byte)(unsafe.Add(nil, addr)), Size: len(code)}
	// A leaked RX mapping is not reclaimed by the GC on its own.
	runtime.SetFinalizer(c, func(c *Code) { c.Close() })
	return c, nil
}

// Close releases the code. Calling it while a kernel is running is a crash, so
// the caller owns that ordering.
func (c *Code) Close() error {
	if c.addr == 0 {
		return nil
	}
	addr := c.addr
	c.addr, c.entry = 0, nil
	runtime.SetFinalizer(c, nil)
	if ok, _, err := procVirtualFree.Call(addr, 0, memRelease); ok == 0 {
		return fmt.Errorf("jit: VirtualFree: %w", err)
	}
	return nil
}

// Call enters the kernel.
//
// args must stay reachable for the whole call. It does, because it is an
// argument of this function and the assembly stub declares it as a pointer, so
// the GC sees it — but every pointer the kernel follows must be reachable from
// args itself, never smuggled in as a uintptr.
func (c *Code) Call(args *Args) {
	if c.entry == nil {
		panic("jit: Call on closed code")
	}
	checkTier(c)
	callKernel(c.entry, args)
	runtime.KeepAlive(c)
	runtime.KeepAlive(args)
}
