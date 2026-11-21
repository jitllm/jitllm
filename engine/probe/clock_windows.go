//go:build windows

package probe

import (
	"syscall"
	"unsafe"
)

// Windows advances time's monotonic reading only at the timer interrupt,
// every 0.5 to 15.6 ms, so a form that runs for microseconds reads zero
// through it. The probe reads the performance counter instead.
var (
	perfCounter = syscall.NewLazyDLL("kernel32.dll").NewProc("QueryPerformanceCounter")
	perfHz      = perfFrequency()
)

func perfFrequency() float64 {
	var f int64
	syscall.SyscallN(syscall.NewLazyDLL("kernel32.dll").NewProc("QueryPerformanceFrequency").Addr(), uintptr(unsafe.Pointer(&f)))
	return float64(f)
}

// mark reads the probe's clock: the performance counter.
func mark() int64 {
	var c int64
	syscall.SyscallN(perfCounter.Addr(), uintptr(unsafe.Pointer(&c)))
	return c
}

// since is the seconds from mark m to now.
func since(m int64) float64 { return float64(mark()-m) / perfHz }
