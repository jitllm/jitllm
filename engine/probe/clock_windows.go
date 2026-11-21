//go:build windows

package probe

import (
	"fmt"
	"syscall"
	"unsafe"
)

// clock is the probe's clock: the performance counter. Windows advances
// time's monotonic reading only at the timer interrupt, every 0.5 to 15.6
// ms, so a form that runs for microseconds reads zero through it.
type clock struct {
	counter uintptr
	hz      float64
}

func newClock() (clock, error) {
	k := syscall.NewLazyDLL("kernel32.dll")
	qc, qf := k.NewProc("QueryPerformanceCounter"), k.NewProc("QueryPerformanceFrequency")
	if err := qc.Find(); err != nil {
		return clock{}, fmt.Errorf("probe: %w", err)
	}
	if err := qf.Find(); err != nil {
		return clock{}, fmt.Errorf("probe: %w", err)
	}
	var f int64
	syscall.SyscallN(qf.Addr(), uintptr(unsafe.Pointer(&f)))
	if f <= 0 {
		return clock{}, fmt.Errorf("probe: the performance counter states %d ticks a second", f)
	}
	return clock{counter: qc.Addr(), hz: float64(f)}, nil
}

// mark reads the clock.
func (c clock) mark() int64 {
	var n int64
	syscall.SyscallN(c.counter, uintptr(unsafe.Pointer(&n)))
	return n
}

// since is the seconds from mark m to now.
func (c clock) since(m int64) float64 { return float64(c.mark()-m) / c.hz }
