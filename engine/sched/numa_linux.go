//go:build linux

package sched

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"unsafe"
)

// numaNodes is the memory nodes whose CPUs this process may run on, ascending.
//
// A two-socket system is two memories and the decode pool spans both, so half of
// every weight read can cross the socket link. A node none of the process's
// CPUs are on is left out, so a taskset onto one socket means one node.
//
// Empty or one node means there is nothing to decide.
func (m machine) numaNodes() []int {
	ok := m.allowed()
	dirs, _ := filepath.Glob(filepath.Join(m.sys, "devices/system/node/node[0-9]*"))
	var nodes []int
	for _, d := range dirs {
		n, err := strconv.Atoi(strings.TrimPrefix(filepath.Base(d), "node"))
		if err != nil {
			continue
		}
		b, err := os.ReadFile(filepath.Join(d, "cpulist"))
		if err != nil {
			continue
		}
		for cpu := range parseCPUList(strings.TrimSpace(string(b))) {
			if len(ok) == 0 || ok[cpu] {
				nodes = append(nodes, n)
				break
			}
		}
	}
	sort.Ints(nodes)
	return nodes
}

// MemPolicyDefault reports whether this process runs under the kernel's
// default memory policy -- that nobody (numactl, a parent) chose one. A policy
// the caller set is theirs, and jitllm does not override it.
func MemPolicyDefault() bool {
	var mode int32
	_, _, errno := syscall.RawSyscall6(syscall.SYS_GET_MEMPOLICY,
		uintptr(unsafe.Pointer(&mode)), 0, 0, 0, 0, 0)
	return errno == 0 && mode == 0 // MPOL_DEFAULT
}

// NodeOf is the NUMA node holding the page at p, or -1 when the kernel will
// not say.
func NodeOf(p unsafe.Pointer) int {
	var node int32
	const mpolFNode, mpolFAddr = 1 << 0, 1 << 1
	if _, _, errno := syscall.Syscall6(syscall.SYS_GET_MEMPOLICY, uintptr(unsafe.Pointer(&node)),
		0, 0, uintptr(p), mpolFNode|mpolFAddr, 0); errno != 0 {
		return -1
	}
	return int(node)
}

// nodeCPUs is each node of NUMANodes() with the CPUs of it this process may
// use, in the same order.
func (m machine) nodeCPUs() (nodes []int, cpus [][]int) {
	ok := m.allowed()
	for _, n := range m.numaNodes() {
		b, err := os.ReadFile(filepath.Join(m.sys, fmt.Sprintf("devices/system/node/node%d/cpulist", n)))
		if err != nil {
			return nil, nil
		}
		var set []int
		for c := range parseCPUList(strings.TrimSpace(string(b))) {
			if len(ok) == 0 || ok[c] {
				set = append(set, c)
			}
		}
		sort.Ints(set)
		nodes, cpus = append(nodes, n), append(cpus, set)
	}
	return nodes, cpus
}

// currentCPU is the CPU the calling thread is on, or -1.
func currentCPU() int {
	if sysGetcpu == 0 {
		return -1
	}
	var cpu uint32
	if _, _, errno := syscall.RawSyscall(sysGetcpu, uintptr(unsafe.Pointer(&cpu)), 0, 0); errno != 0 {
		return -1
	}
	return int(cpu)
}
