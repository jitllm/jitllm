//go:build darwin

package sched

import (
	"encoding/binary"
	"syscall"
)

// Budget is how many bytes of weights may be resident under g, when the
// caller does not say: eight tenths of what macOS reports available, as on
// Linux.
//
// Zero is not an answer here. The pager and the server read a budget of zero
// as "unlimited", so a Mac that returned it never paged: a model larger than
// its memory was read whole, and the server's default page budget reported
// 0 B. Apple Silicon's memory is shared with the GPU, which is one more reason
// the host budget has to be a number.
func (g GCPlan) Budget() uint64 {
	avail := darwinAvailable()
	if avail == 0 {
		return 0 // the sysctls failed: the caller decides what unknown means
	}
	b := avail / 10 * 8
	if cap := g.Cap(); cap > 0 && cap < b {
		b = cap
	}
	return b
}

// MemLimit has no cgroup to read on this platform.
func MemLimit() uint64 { return 0 }

// darwinAvailable is the physical memory times the kernel's memorystatus
// level, the percentage of memory available before pressure (what
// memory_pressure prints as the system-wide free percentage): free,
// inactive and purgeable pages, the counterpart of Linux's MemAvailable.
func darwinAvailable() uint64 {
	total := darwinMemsize()
	level, err := syscall.SysctlUint32("kern.memorystatus_level")
	if total == 0 || err != nil || level == 0 || level > 100 {
		return 0
	}
	return total / 100 * uint64(level)
}

// darwinMemsize is hw.memsize, a 64-bit value. syscall.Sysctl returns it as a
// string with a trailing NUL byte removed, which for a power-of-two size is a
// zero byte of the number, so it is padded back to eight.
func darwinMemsize() uint64 {
	s, err := syscall.Sysctl("hw.memsize")
	if err != nil || len(s) > 8 {
		return 0
	}
	var b [8]byte
	copy(b[:], s)
	return binary.LittleEndian.Uint64(b[:])
}
