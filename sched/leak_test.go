//go:build linux

package sched

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

// osThreads reads the process's live OS thread count from /proc.
//
// A goroutine leak and a thread leak are different failures, so both are
// asserted; a locked OS thread is invisible to runtime.NumGoroutine.
func osThreads(t *testing.T) int {
	t.Helper()
	b, err := os.ReadFile("/proc/self/status")
	if err != nil {
		t.Skipf("no /proc/self/status: %v", err)
	}
	for _, ln := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(ln, "Threads:") {
			n, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(ln, "Threads:")))
			if err != nil {
				t.Fatalf("Threads: %v", err)
			}
			return n
		}
	}
	t.Fatal("no Threads: line in /proc/self/status")
	return 0
}

// settle waits for goroutines that are on their way out to actually leave.
// Close signals rather than joins, deliberately, so "has it leaked" is a
// question about the steady state and not about the instant after Close.
func settle(d time.Duration) {
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		runtime.Gosched()
		time.Sleep(2 * time.Millisecond)
	}
	runtime.GC()
}

// TestPoolsDoNotLeakGoroutinesOrThreads creates and closes many pools and
// requires the process to come back to where it started.
//
// A thread count that returns to baseline is not proof no masked thread
// leaked, so the affinity assertion below is separate.
func TestPoolsDoNotLeakGoroutinesOrThreads(t *testing.T) {
	settle(200 * time.Millisecond)
	g0, th0 := runtime.NumGoroutine(), osThreads(t)

	const rounds = 40
	for i := 0; i < rounds; i++ {
		p := New(DecodeCores())
		n := 4096
		got := make([]int32, n)
		p.Do(n, 64, func(_, lo, hi int) {
			for j := lo; j < hi; j++ {
				got[j]++
			}
		})
		for j, v := range got {
			if v != 1 {
				t.Fatalf("round %d: element %d ran %d times, want 1", i, j, v)
			}
		}
		p.Close()
	}
	settle(2 * time.Second)

	g1, th1 := runtime.NumGoroutine(), osThreads(t)
	// A couple of goroutines of slack: the runtime's own bookkeeping moves.
	if g1 > g0+2 {
		t.Errorf("goroutines %d -> %d after %d pools: workers are not exiting on Close",
			g0, g1, rounds)
	}
	// Threads are allowed to grow to the pool width (the runtime keeps idle Ms)
	// but must not grow with the number of pools, which is what a per-pool
	// thread leak looks like.
	if th1 > th0+len(DecodeCores())+4 {
		t.Errorf("OS threads %d -> %d after %d pools of %d workers: that scales with "+
			"the pool COUNT, which is a thread leak", th0, th1, rounds, len(DecodeCores()))
	}
	t.Logf("%d pools: goroutines %d -> %d, OS threads %d -> %d", rounds, g0, g1, th0, th1)
}

// TestPoolLeavesNoThreadMaskedToOneCPU is the direct gate on the affinity leak.
//
// It reads every live thread's Cpus_allowed_list and requires none to be
// narrower than the process mask: a leaked sched_setaffinity mask would pass
// the count test above.
func TestPoolLeavesNoThreadMaskedToOneCPU(t *testing.T) {
	want := len(host().allowed())
	if want == 0 {
		t.Skip("cannot read the process affinity mask")
	}
	for i := 0; i < 8; i++ {
		p := New(DecodeCores())
		p.Do(8192, 64, func(_, lo, hi int) {})
		p.Close()
	}
	settle(1500 * time.Millisecond)

	ents, err := os.ReadDir("/proc/self/task")
	if err != nil {
		t.Skipf("no /proc/self/task: %v", err)
	}
	var narrow []string
	for _, e := range ents {
		b, err := os.ReadFile(fmt.Sprintf("/proc/self/task/%s/status", e.Name()))
		if err != nil {
			continue // the thread left while we were looking
		}
		for _, ln := range strings.Split(string(b), "\n") {
			if !strings.HasPrefix(ln, "Cpus_allowed_list:") {
				continue
			}
			list := strings.TrimSpace(strings.TrimPrefix(ln, "Cpus_allowed_list:"))
			if n := countCPUList(list); n > 0 && n < want {
				narrow = append(narrow, fmt.Sprintf("tid %s -> %s (%d of %d CPUs)",
					e.Name(), list, n, want))
			}
		}
	}
	if len(narrow) > 0 {
		t.Fatalf("%d live thread(s) carry an affinity mask NARROWER than the process's "+
			"%d CPUs. A pool worker set that and it outlived the worker; the runtime "+
			"will hand such a thread to any goroutine:\n  %s",
			len(narrow), want, strings.Join(narrow, "\n  "))
	}
	t.Logf("%d live threads, none masked below the process's %d CPUs", len(ents), want)
}

// countCPUList counts the CPUs in a "0-3,8,12-13" list.
func countCPUList(s string) int {
	n := 0
	for _, f := range strings.Split(s, ",") {
		f = strings.TrimSpace(f)
		if f == "" {
			continue
		}
		if lo, hi, ok := strings.Cut(f, "-"); ok {
			a, e1 := strconv.Atoi(lo)
			b, e2 := strconv.Atoi(hi)
			if e1 == nil && e2 == nil && b >= a {
				n += b - a + 1
			}
			continue
		}
		if _, err := strconv.Atoi(f); err == nil {
			n++
		}
	}
	return n
}
