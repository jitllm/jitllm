//go:build darwin

package sched

import (
	"runtime"
	"syscall"
)

// Apple Silicon reports its clusters through perflevel sysctls: level 0 is the
// performance cluster, level 1 the efficiency one.
//
// The indices are placeholders, not CPU ids: macOS exposes no processor
// affinity, and QoS schedules busy default-class threads on performance cores.
// Dynamic chunk claiming lets a thread that lands on an E-core simply claim
// fewer chunks.
func (m machine) pCores() []int { return seq(perfLevel(0, runtime.NumCPU())) }

func (m machine) eCores() []int {
	n := perfLevel(1, 0)
	p := perfLevel(0, runtime.NumCPU())
	out := make([]int, n)
	for i := range out {
		out[i] = p + i
	}
	return out
}

// smtSiblings has nothing to add: no Apple Silicon core is SMT.
func (m machine) smtSiblings() []int { return m.pCores() }

func perfLevel(level, dflt int) int {
	name := "hw.perflevel0.physicalcpu"
	if level == 1 {
		name = "hw.perflevel1.physicalcpu"
	}
	if v, err := syscall.SysctlUint32(name); err == nil && v > 0 {
		return int(v)
	}
	return dflt
}

// coreSource says where the pool's CPUs came from, for a report.
func (m machine) coreSource() string { return "perflevel sysctls" }
