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

// Budget is how many bytes of weights may be resident under g, when the
// caller does not say.
//
// It is eight tenths of the smallest of three limits, each read from the OS
// when it is asked:
//
//   - what the machine has available (MemAvailable), so memory other
//     processes hold is not counted as ours;
//   - the cgroup's ceiling: under scripts/cap a process with plenty of
//     MemAvailable can be limited far lower, and cgroup v2 charges page cache
//     to the cgroup;
//   - the memory of the NUMA nodes the process may allocate from: under
//     `numactl --membind=0` the kernel kills the process when node 0 is full,
//     whatever the other socket has free (an OOM with
//     constraint=CONSTRAINT_MEMORY_POLICY).
//
// The rest is left for everything that is not weights -- the KV cache, the
// scratch, the Go heap and the runtime itself. Reading the OS's limits is not
// reading the environment (AGENTS.md Scope): every caller that sizes a budget
// takes an explicit one too (engine.WithPageBudget, jitllmd -maxmem and
// -host-mem).
func (g GCPlan) Budget() uint64 { return host().mem().Budget(g) }

// MemLimit is the hard ceiling this process must stay under, or 0 when there is
// none: the cgroup's, or the bound NUMA nodes' total, whichever is lower -- not
// a share of it.
//
// It exists for Go's garbage collector, which does not know about cgroups or
// memory policies: at GOGC=100 a heap holding 18 GiB of page frames plans for
// ~36 GiB, which under memory.high means permanent direct reclaim. The caller
// passes it to debug.SetMemoryLimit, which is process-wide and so not a
// library's decision. MemBudget is a share of this, for weights.
func MemLimit() uint64 { return host().memLimit() }

func (m machine) memLimit() uint64 {
	h := m.mem()
	return minKnown(h.Cgroup, h.NodeTotal)
}

// HostMemory is the memory this process may use, as the OS states it now. Zero in
// a field is "no such limit" or "not reported".
type HostMemory struct {
	// Available is /proc/meminfo's MemAvailable.
	Available uint64
	// Cgroup is the cgroup-v2 ceiling (memory.high when lower than
	// memory.max).
	Cgroup uint64
	// Nodes is the NUMA nodes the process may allocate from when that is not
	// every node of the machine: a bind policy (numactl --membind) intersected
	// with the cpuset's Mems_allowed. Empty when every node is allowed.
	Nodes []int
	// NodeTotal and NodeAvailable are those nodes' MemTotal and what they can
	// hand out (free plus reclaimable file pages and slab).
	NodeTotal, NodeAvailable uint64
}

// Limit is the smallest of the three: what the process can actually be given.
func (h HostMemory) Limit() uint64 {
	return minKnown(minKnown(h.Available, h.Cgroup), h.NodeAvailable)
}

// Budget is g's Budget from h: eight tenths of Limit, under g's Cap when
// the weights are on the Go heap.
func (h HostMemory) Budget(g GCPlan) uint64 {
	avail := h.Limit()
	if avail == 0 {
		return 0 // unknown: the caller decides what that means
	}
	// Not all of it: the KV cache is unreclaimable, the pool has scratch, and a
	// cgroup at MemoryHigh is throttled (which is what systemd-oomd reacts to).
	// Eight tenths leaves room and keeps the budget dominated by weights.
	b := avail / 10 * 8
	// With the weights on the Go heap, eight tenths can leave the permanently
	// live page frames on the collector's goal, so the plan's Cap applies; off
	// the heap (the default) it is no cap. See GCPlan.
	if cap := g.Cap(); cap > 0 && cap < b {
		b = cap
	}
	return b
}

func (h HostMemory) String() string {
	return fmt.Sprintf("available %d, cgroup %d, nodes %v total %d available %d",
		h.Available, h.Cgroup, h.Nodes, h.NodeTotal, h.NodeAvailable)
}

// minKnown is the smaller of a and b, where 0 is "no limit".
func minKnown(a, b uint64) uint64 {
	switch {
	case a == 0:
		return b
	case b == 0:
		return a
	}
	return min(a, b)
}

func (m machine) mem() HostMemory {
	h := HostMemory{Available: m.meminfoAvailable(), Cgroup: m.cgroupLimit()}
	h.Nodes = m.boundNodes()
	for _, n := range h.Nodes {
		total, avail := m.nodeMeminfo(n)
		h.NodeTotal += total
		h.NodeAvailable += avail
	}
	return h
}

// boundNodes is the nodes this process may allocate from, or nil when that is
// every node of the machine (or the machine has one).
func (m machine) boundNodes() []int {
	all := m.machineNodes()
	if len(all) < 2 {
		return nil
	}
	allowed := map[int]bool{}
	for _, n := range all {
		allowed[n] = true
	}
	if ma := m.memsAllowed(); ma != nil {
		for n := range allowed {
			if !ma[n] {
				delete(allowed, n)
			}
		}
	}
	if b := m.bind(); b != nil {
		in := map[int]bool{}
		for _, n := range b {
			in[n] = true
		}
		for n := range allowed {
			if !in[n] {
				delete(allowed, n)
			}
		}
	}
	if len(allowed) == len(all) || len(allowed) == 0 {
		return nil
	}
	out := make([]int, 0, len(allowed))
	for n := range allowed {
		out = append(out, n)
	}
	sort.Ints(out)
	return out
}

// machineNodes is every memory node sysfs lists.
func (m machine) machineNodes() []int {
	dirs, _ := filepath.Glob(filepath.Join(m.sys, "devices/system/node/node[0-9]*"))
	var nodes []int
	for _, d := range dirs {
		if n, err := strconv.Atoi(strings.TrimPrefix(filepath.Base(d), "node")); err == nil {
			nodes = append(nodes, n)
		}
	}
	sort.Ints(nodes)
	return nodes
}

// memsAllowed is the cpuset's memory nodes (/proc/self/status
// Mems_allowed_list), or nil when it is not stated.
func (m machine) memsAllowed() map[int]bool {
	b, err := os.ReadFile(filepath.Join(m.proc, "self/status"))
	if err != nil {
		return nil
	}
	for _, ln := range strings.Split(string(b), "\n") {
		if v, ok := strings.CutPrefix(ln, "Mems_allowed_list:"); ok {
			return parseCPUList(strings.TrimSpace(v))
		}
	}
	return nil
}

// kernelBindNodes asks get_mempolicy(2) for the process's policy and returns
// its node mask when the mode is MPOL_BIND. A preferred or interleaved policy
// is not a limit: the kernel falls back to other nodes before it kills.
func kernelBindNodes() []int {
	const mpolBind = 2
	var mode int32
	var mask [16]uint64 // 1024 nodes, past any kernel's MAX_NUMNODES default
	if _, _, errno := syscall.RawSyscall6(syscall.SYS_GET_MEMPOLICY,
		uintptr(unsafe.Pointer(&mode)), uintptr(unsafe.Pointer(&mask[0])), uintptr(len(mask)*64), 0, 0, 0); errno != 0 {
		return nil
	}
	if mode&0xffff != mpolBind { // the high bits are mode flags
		return nil
	}
	var nodes []int
	for w, m := range mask {
		for bit := 0; bit < 64; bit++ {
			if m&(1<<bit) != 0 {
				nodes = append(nodes, w*64+bit)
			}
		}
	}
	return nodes
}

// nodeMeminfo is node n's MemTotal and what it can hand out: MemFree plus its
// file pages and reclaimable slab, the per-node counterpart of MemAvailable
// (which the kernel does not publish per node).
func (m machine) nodeMeminfo(n int) (total, avail uint64) {
	b, err := os.ReadFile(filepath.Join(m.sys, fmt.Sprintf("devices/system/node/node%d/meminfo", n)))
	if err != nil {
		return 0, 0
	}
	for _, ln := range strings.Split(string(b), "\n") {
		// "Node 0 MemFree:         6467300 kB"
		f := strings.Fields(ln)
		if len(f) < 4 {
			continue
		}
		kb, err := strconv.ParseUint(f[3], 10, 64)
		if err != nil {
			continue
		}
		switch f[2] {
		case "MemTotal:":
			total = kb * 1024
		case "MemFree:", "Active(file):", "Inactive(file):", "SReclaimable:":
			avail += kb * 1024
		}
	}
	return total, min(avail, total)
}

// cgroupLimit is this process's cgroup-v2 memory ceiling, or 0 if there is none.
// memory.high is used when it is lower than memory.max, because crossing HIGH is
// what throttles -- reaching MAX only kills, and by then it is too late.
func (m machine) cgroupLimit() uint64 {
	rel := ""
	if b, err := os.ReadFile(filepath.Join(m.proc, "self/cgroup")); err == nil {
		for _, ln := range strings.Split(strings.TrimSpace(string(b)), "\n") {
			// cgroup v2 is the single "0::<path>" line.
			if strings.HasPrefix(ln, "0::") {
				rel = strings.TrimPrefix(ln, "0::")
			}
		}
	}
	if rel == "" {
		return 0
	}
	var lim uint64
	for _, f := range []string{"memory.high", "memory.max"} {
		v := readLimit(filepath.Join(m.sys, "fs/cgroup", rel, f))
		if v > 0 && (lim == 0 || v < lim) {
			lim = v
		}
	}
	return lim
}

// readLimit reads a cgroup limit file. "max" means unlimited, which is 0 here.
func readLimit(path string) uint64 {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	s := strings.TrimSpace(string(b))
	if s == "max" {
		return 0
	}
	v, err := strconv.ParseUint(s, 10, 64)
	if err != nil {
		return 0
	}
	return v
}

func (m machine) meminfoAvailable() uint64 {
	b, err := os.ReadFile(filepath.Join(m.proc, "meminfo"))
	if err != nil {
		return 0
	}
	for _, ln := range strings.Split(string(b), "\n") {
		if !strings.HasPrefix(ln, "MemAvailable:") {
			continue
		}
		f := strings.Fields(ln)
		if len(f) < 2 {
			return 0
		}
		kb, err := strconv.ParseUint(f[1], 10, 64)
		if err != nil {
			return 0
		}
		return kb * 1024
	}
	return 0
}
