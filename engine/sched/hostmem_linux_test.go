//go:build linux

package sched

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"testing"
)

// fakeHost writes a /proc and /sys and is the machine that reads them: two
// nodes of 128 GiB a node, avail on the whole machine, a cgroup ceiling cg (0
// for none), the cpuset's memory nodes mems, each node's free memory, and no
// memory-policy bind.
func fakeHost(t *testing.T, avail, cg uint64, mems string, nodeFree [2]uint64) machine {
	t.Helper()
	proc, sys := t.TempDir(), t.TempDir()
	write := func(p, s string) {
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(s), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write(filepath.Join(proc, "meminfo"), fmt.Sprintf("MemTotal: %d kB\nMemAvailable: %d kB\n", 256<<20, avail>>10))
	write(filepath.Join(proc, "self/status"), "Name:\tjitllmd\nMems_allowed_list:\t"+mems+"\n")
	write(filepath.Join(proc, "self/cgroup"), "0::/jitllm.scope\n")
	lim := "max"
	if cg > 0 {
		lim = strconv.FormatUint(cg, 10)
	}
	write(filepath.Join(sys, "fs/cgroup/jitllm.scope/memory.max"), lim+"\n")
	write(filepath.Join(sys, "fs/cgroup/jitllm.scope/memory.high"), "max\n")
	for n := 0; n < 2; n++ {
		d := filepath.Join(sys, fmt.Sprintf("devices/system/node/node%d", n))
		write(filepath.Join(d, "cpulist"), fmt.Sprintf("%d-%d\n", n*14, n*14+13))
		// Free memory split between MemFree and the reclaimable file pages,
		// so a reader that takes MemFree alone reads short.
		free := nodeFree[n]
		write(filepath.Join(d, "meminfo"), fmt.Sprintf(
			"Node %d MemTotal:       %d kB\nNode %d MemFree:        %d kB\nNode %d Active(file):   %d kB\nNode %d Inactive(file): 0 kB\nNode %d SReclaimable:   0 kB\n",
			n, 128<<20, n, free/2>>10, n, (free-free/2)>>10, n, n))
	}
	return machine{proc: proc, sys: sys, bind: func() []int { return nil }, affinity: readAffinity}
}

// TestMemBudgetHonoursEveryLimit: the budget is eight tenths of the smallest of
// what is available, the cgroup's ceiling and the bound NUMA nodes' memory, and
// MemLimit the smaller of the cgroup and the bound nodes' total. The first
// membind case is the one that OOM-killed jitllmd: a 251 GB two-socket box,
// the process under numactl --membind=0, the budget sized from the whole
// machine.
func TestMemBudgetHonoursEveryLimit(t *testing.T) {
	const gib = uint64(1) << 30
	cases := []struct {
		name         string
		avail, cg    uint64
		mems         string
		bind         []int
		nodeFree     [2]uint64
		want         uint64 // the limit Budget takes eight tenths of
		wantNodes    []int
		wantMemLimit uint64
	}{
		{"unbound", 240 * gib, 0, "0-1", nil, [2]uint64{120 * gib, 120 * gib}, 240 * gib, nil, 0},
		{"cgroup", 240 * gib, 8 * gib, "0-1", nil, [2]uint64{120 * gib, 120 * gib}, 8 * gib, nil, 8 * gib},
		{"membind", 240 * gib, 0, "0-1", []int{0}, [2]uint64{100 * gib, 120 * gib}, 100 * gib, []int{0}, 128 * gib},
		{"cpuset", 240 * gib, 0, "1", nil, [2]uint64{100 * gib, 60 * gib}, 60 * gib, []int{1}, 128 * gib},
		{"membind under a cgroup", 240 * gib, 16 * gib, "0-1", []int{0}, [2]uint64{100 * gib, 120 * gib}, 16 * gib, []int{0}, 16 * gib},
		{"others hold memory", 30 * gib, 0, "0-1", []int{0}, [2]uint64{100 * gib, 120 * gib}, 30 * gib, []int{0}, 128 * gib},
		{"bound to every node", 240 * gib, 0, "0-1", []int{0, 1}, [2]uint64{10 * gib, 10 * gib}, 240 * gib, nil, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := fakeHost(t, c.avail, c.cg, c.mems, c.nodeFree)
			bind := c.bind
			m.bind = func() []int { return bind }
			h := m.mem()
			if !slices.Equal(h.Nodes, c.wantNodes) {
				t.Fatalf("nodes %v, want %v (%v)", h.Nodes, c.wantNodes, h)
			}
			if got := h.Limit(); got != c.want {
				t.Fatalf("limit %.1f GiB, want %.1f GiB (%v)", float64(got)/float64(gib), float64(c.want)/float64(gib), h)
			}
			if got, want := m.mem().Budget(GCPlan{}), c.want/10*8; !(GCPlan{}).onHeap() && got != want {
				t.Fatalf("MemBudget %d, want %d", got, want)
			}
			if got := m.memLimit(); got != c.wantMemLimit {
				t.Fatalf("MemLimit %d, want %d", got, c.wantMemLimit)
			}
		})
	}
}

// TestKernelBindNodesReadsThisProcess: get_mempolicy on a process nobody bound
// is not a bind, so the kernel reader says nil. Under `numactl --membind=0`
// (run the compiled test binary under it) it must name node 0.
func TestKernelBindNodesReadsThisProcess(t *testing.T) {
	if !MemPolicyDefault() {
		n := kernelBindNodes()
		t.Logf("bound to %v", n)
		if len(n) == 0 {
			t.Fatal("a process under a memory policy reads unbound")
		}
		return
	}
	if n := kernelBindNodes(); n != nil {
		t.Fatalf("an unbound process reads bound to %v", n)
	}
}
