//go:build !linux && !darwin

package sched

import "runtime"

// Everything else: no topology, no pinning, no assumptions.
//
// runtime.NumCPU counts logical processors, so on an SMT machine this pool is
// half hyperthreads, a small decode loss, but the honest answer without
// OS-specific topology. Add a topo file for the OS to do better.
func (m machine) pCores() []int {
	n := runtime.NumCPU()
	if n < 1 {
		n = 1
	}
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}

func (m machine) eCores() []int      { return nil }
func (m machine) smtSiblings() []int { return m.pCores() }

// coreSource says where the pool's CPUs came from, for a report.
func (m machine) coreSource() string { return "runtime.NumCPU" }
