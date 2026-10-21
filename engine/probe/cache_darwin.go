//go:build darwin

package probe

import (
	"fmt"
	"syscall"
)

// cacheSizes is the performance cores' data caches, nearest first, as
// Darwin states them: L1 a core's own, L2 shared by a cluster's cores.
// Darwin states no last-level cache it puts in front of memory.
func cacheSizes() ([]cache, error) {
	l1, err := syscall.SysctlUint32("hw.perflevel0.l1dcachesize")
	if err != nil {
		return nil, fmt.Errorf("probe: hw.perflevel0.l1dcachesize: %w", err)
	}
	l2, err := syscall.SysctlUint32("hw.perflevel0.l2cachesize")
	if err != nil {
		return nil, fmt.Errorf("probe: hw.perflevel0.l2cachesize: %w", err)
	}
	per, err := syscall.SysctlUint32("hw.perflevel0.cpusperl2")
	if err != nil {
		return nil, fmt.Errorf("probe: hw.perflevel0.cpusperl2: %w", err)
	}
	return []cache{{bytes: int64(l1)}, {bytes: int64(l2), shared: per > 1}}, nil
}
