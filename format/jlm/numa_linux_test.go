//go:build linux

package jlm

import (
	"fmt"
	"os"
	"strings"

	"github.com/jitllm/jitllm/sched"

	"syscall"
	"testing"
	"unsafe"
)

// TestPlacementIsPerFile is the server's case: two models in one process, one
// interleaved and on 4 KB pages, one on the defaults. Each frame must carry its
// own File's placement, asked of the kernel per address -- the memory policy,
// and the "hg" VmFlag MADV_HUGEPAGE sets on the mapping. The node list is
// {0, 0}, which passes the two-node floor on a one-node host, so the gate runs
// on every Linux host. Placement is per File, not a process-wide switch where
// the last setter wins for every model.
func TestPlacementIsPerFile(t *testing.T) {
	const size = 8 << 20
	spread := newFake(4, size, 0, Align)
	spread.SetPlacement(Placement{NoHugePages: true, Nodes: []int{0, 0}})
	plain := newFake(4, size, 0, Align)
	plain.SetPlacement(Placement{})
	if !spread.Interleaved() || plain.Interleaved() {
		t.Fatalf("Interleaved: spread %v, plain %v", spread.Interleaved(), plain.Interleaved())
	}
	for i := 0; i < 2; i++ { // alternate, so a shared setting would leak across
		spread.fakeIn(i, 1)
		plain.fakeIn(i, 1)
	}
	for i := 0; i < 2; i++ {
		if got := policyAt(t, spread.pages[i]); got != mpolInterleave {
			t.Errorf("spread frame %d: policy %d, want MPOL_INTERLEAVE (%d)", i, got, mpolInterleave)
		}
		if got := policyAt(t, plain.pages[i]); got != 0 {
			t.Errorf("plain frame %d: policy %d, want the default (0)", i, got)
		}
		if hugeAdvised(t, spread.pages[i]) {
			t.Errorf("spread frame %d is advised onto huge pages; its File said NoHugePages", i)
		}
		if !hugeAdvised(t, plain.pages[i]) {
			t.Errorf("plain frame %d is not advised onto huge pages", i)
		}
	}
}

func policyAt(t *testing.T, b []byte) int32 {
	t.Helper()
	var mode int32
	const mpolFAddr = 1 << 1
	if _, _, errno := syscall.Syscall6(syscall.SYS_GET_MEMPOLICY, uintptr(unsafe.Pointer(&mode)),
		0, 0, uintptr(unsafe.Pointer(&b[len(b)/2])), mpolFAddr, 0); errno != 0 {
		t.Fatalf("get_mempolicy: %v", errno)
	}
	return mode
}

// hugeAdvised reports whether the mapping holding b's middle byte carries
// MADV_HUGEPAGE ("hg" in /proc/self/smaps' VmFlags).
func hugeAdvised(t *testing.T, b []byte) bool {
	t.Helper()
	addr := uint64(uintptr(unsafe.Pointer(&b[len(b)/2])))
	raw, err := os.ReadFile("/proc/self/smaps")
	if err != nil {
		t.Fatal(err)
	}
	in := false
	for _, line := range strings.Split(string(raw), "\n") {
		var lo, hi uint64
		if n, _ := fmt.Sscanf(line, "%x-%x", &lo, &hi); n == 2 && strings.Contains(line, " ") {
			in = addr >= lo && addr < hi
			continue
		}
		if in && strings.HasPrefix(line, "VmFlags:") {
			return strings.Contains(line, " hg")
		}
	}
	t.Fatalf("no mapping holds %#x", addr)
	return false
}

// TestInterleaveFollowsThePageAddress is what NUMA stage 2 stands on: an
// interleaved frame puts page i on node (i + c) mod nodes, so a 1024-row block
// of a page-aligned word-line has one node for every word of a tensor whose row
// count is a multiple of 2048. Asked of the kernel, page by page, on a real
// two-node host; a one-node host has no second node to alternate with.
func TestInterleaveFollowsThePageAddress(t *testing.T) {
	if _, err := os.Stat("/sys/devices/system/node/node1"); err != nil {
		t.Skip("one NUMA node: nothing to interleave over")
	}
	const pages = 4096
	b := alignedBuf(pages * 4096)
	placeMemory(b, true, []uint64{3})
	for i := 0; i < len(b); i += 4096 {
		b[i] = 1 // fault every page in under the policy
	}
	base := uintptr(unsafe.Pointer(&b[0])) >> 12
	c := -1
	for i := 0; i < pages; i++ {
		n := sched.NodeOf(unsafe.Pointer(&b[i*4096]))
		if n < 0 {
			t.Fatalf("page %d: the kernel reports no node", i)
		}
		want := (int(base) + i) % 2
		if c < 0 {
			c = n ^ want
		}
		if n != want^c {
			t.Fatalf("page %d is on node %d; pages %d..%d alternated as (addr>>12 + %d) mod 2",
				i, n, 0, i-1, c)
		}
	}
	t.Logf("%d pages, node = (addr>>12 + %d) mod 2 on every one", pages, c)
}
