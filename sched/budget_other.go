//go:build !linux && !darwin && !windows

package sched

// Budget returns 0 -- unknown -- on a platform with nothing to read it from
// (Linux reads /proc and the cgroup, darwin its sysctls, Windows
// GlobalMemoryStatusEx). The caller treats that as "do not manage residency".
func (GCPlan) Budget() uint64 { return 0 }

// MemLimit has no cgroup to read on this platform.
func MemLimit() uint64 { return 0 }

// GCPlan.Cap is not redeclared here: gc.go's is portable, and with MemLimit 0
// it returns 0.
