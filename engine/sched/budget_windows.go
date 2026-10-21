//go:build windows

package sched

import (
	"syscall"
	"unsafe"
)

// MemBudget is how many bytes of weights may be resident, when the caller does
// not say: eight tenths of what Windows reports available, as on Linux and
// macOS.
//
// Zero is not an answer here either. The pager and the server read a budget of
// zero as "unlimited", so a Windows host that returned it never paged by
// budget and the server's default page budget was 0 B
// (server TestAPageBudgetBelowTheDenseRegionStillPages).
func MemBudget() uint64 {
	avail := windowsAvailable()
	if avail == 0 {
		return 0 // the call failed: the caller decides what unknown means
	}
	b := avail / 10 * 8
	if cap := gcBudgetCap(); cap > 0 && cap < b {
		b = cap
	}
	return b
}

// MemLimit has no cgroup to read on this platform. A job object's limit would
// be the counterpart; nothing here sets one.
func MemLimit() uint64 { return 0 }

// memoryStatusEx is MEMORYSTATUSEX.
type memoryStatusEx struct {
	length               uint32
	memoryLoad           uint32
	totalPhys            uint64
	availPhys            uint64
	totalPageFile        uint64
	availPageFile        uint64
	totalVirtual         uint64
	availVirtual         uint64
	availExtendedVirtual uint64
}

// GlobalMemoryStatusEx is not in Go's syscall package, and the engine takes no
// dependency for one call, so it is reached through kernel32 as jit_old/cpu
// reaches VirtualAlloc.
var procGlobalMemoryStatusEx = syscall.NewLazyDLL("kernel32.dll").NewProc("GlobalMemoryStatusEx")

// windowsAvailable is the physical memory available now: free, standby and
// zeroed pages, the counterpart of Linux's MemAvailable. Zero when the call
// fails.
func windowsAvailable() uint64 {
	total, avail := windowsMemory()
	if total == 0 || avail > total {
		return 0
	}
	return avail
}

// windowsMemory is the physical memory installed and available.
func windowsMemory() (total, avail uint64) {
	ms := memoryStatusEx{length: uint32(unsafe.Sizeof(memoryStatusEx{}))}
	if r, _, _ := procGlobalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&ms))); r == 0 {
		return 0, 0
	}
	return ms.totalPhys, ms.availPhys
}
