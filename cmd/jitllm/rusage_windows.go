package main

import (
	"syscall"
	"unsafe"
)

// processMemoryCounters is PROCESS_MEMORY_COUNTERS.
type processMemoryCounters struct {
	cb                         uint32
	pageFaultCount             uint32
	peakWorkingSetSize         uintptr
	workingSetSize             uintptr
	quotaPeakPagedPoolUsage    uintptr
	quotaPagedPoolUsage        uintptr
	quotaPeakNonPagedPoolUsage uintptr
	quotaNonPagedPoolUsage     uintptr
	pagefileUsage              uintptr
	peakPagefileUsage          uintptr
}

// processUsage is the process's user+system CPU seconds (GetProcessTimes) and
// its peak working set in bytes (K32GetProcessMemoryInfo), zero where Windows
// refuses either.
func processUsage() (cpu float64, maxRSS uint64) {
	h, err := syscall.GetCurrentProcess()
	if err != nil {
		return 0, 0
	}
	var created, exited, kernel, user syscall.Filetime
	if syscall.GetProcessTimes(h, &created, &exited, &kernel, &user) == nil {
		// A FILETIME counts 100 ns intervals.
		ticks := func(f syscall.Filetime) uint64 { return uint64(f.HighDateTime)<<32 | uint64(f.LowDateTime) }
		cpu = float64(ticks(kernel)+ticks(user)) / 1e7
	}
	var pmc processMemoryCounters
	pmc.cb = uint32(unsafe.Sizeof(pmc))
	if ok, _, _ := syscall.NewLazyDLL("kernel32.dll").NewProc("K32GetProcessMemoryInfo").Call(uintptr(h), uintptr(unsafe.Pointer(&pmc)), uintptr(pmc.cb)); ok != 0 {
		maxRSS = uint64(pmc.peakWorkingSetSize)
	}
	return cpu, maxRSS
}
