//go:build linux

package tier

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

// A CUDA context is per OS thread, so jit_old/gpu/cuda keeps one owner goroutine per
// device on a locked thread, which makes the GPU path the one place a thread
// leak can hide from sched's gates. One locked thread per open device is the
// design; the failure is a count that grows with open/close cycles.

func procThreads(t *testing.T) int {
	t.Helper()
	b, err := os.ReadFile("/proc/self/status")
	if err != nil {
		t.Skipf("no /proc/self/status: %v", err)
	}
	for _, ln := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(ln, "Threads:") {
			n, _ := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(ln, "Threads:")))
			return n
		}
	}
	return 0
}

func quiesce(d time.Duration) {
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		runtime.Gosched()
		time.Sleep(3 * time.Millisecond)
	}
	runtime.GC()
}

// TestDeviceOpenCloseDoesNotLeakThreads opens and closes the tier repeatedly.
//
// A device that is open owns a locked thread; a device that has been CLOSED must
// give it back. If the count tracks the cycle count rather than the live device
// count, every model load in a long-running process costs a thread permanently.
func TestDeviceOpenCloseDoesNotLeakThreads(t *testing.T) {
	g, err := OpenWith(WithDevices("auto"), WithDeviceTune(TuneOff))
	if err != nil || g == nil {
		t.Skipf("no device: %v", err)
	}
	name := g.Name()
	g.Close()
	quiesce(700 * time.Millisecond)

	g0, th0 := runtime.NumGoroutine(), procThreads(t)
	const rounds = 8
	for i := 0; i < rounds; i++ {
		h, err := OpenWith(WithDevices("auto"), WithDeviceTune(TuneOff))
		if err != nil || h == nil {
			t.Fatalf("round %d: %v", i, err)
		}
		h.Close()
	}
	quiesce(2 * time.Second)
	g1, th1 := runtime.NumGoroutine(), procThreads(t)

	t.Logf("%s: %d open/close cycles, goroutines %d -> %d, OS threads %d -> %d",
		name, rounds, g0, g1, th0, th1)
	if g1 > g0+2 {
		t.Errorf("goroutines %d -> %d over %d cycles: a device owner is not exiting",
			g0, g1, rounds)
	}
	// Slack for the runtime's own idle Ms, but not one per cycle.
	if th1 > th0+4 {
		t.Errorf("OS threads %d -> %d over %d cycles: that tracks the CYCLE count, "+
			"which is a per-open thread leak", th0, th1, rounds)
	}
}

// TestDeviceOpenLeavesNoThreadMaskedToOneCPU is sched's affinity gate applied to
// the GPU path.
//
// Nothing in jit_old/gpu calls sched_setaffinity; this keeps it so, catching
// a driver thread, an FFI trampoline or a future pinning optimisation.
func TestDeviceOpenLeavesNoThreadMaskedToOneCPU(t *testing.T) {
	g, err := OpenWith(WithDevices("auto"), WithDeviceTune(TuneOff))
	if err != nil || g == nil {
		t.Skipf("no device: %v", err)
	}
	defer g.Close()

	self, err := os.ReadFile("/proc/self/status")
	if err != nil {
		t.Skip("no /proc")
	}
	want := 0
	for _, ln := range strings.Split(string(self), "\n") {
		if strings.HasPrefix(ln, "Cpus_allowed_list:") {
			want = countCPUs(strings.TrimSpace(strings.TrimPrefix(ln, "Cpus_allowed_list:")))
		}
	}
	if want == 0 {
		t.Skip("cannot read the process affinity mask")
	}

	ents, err := os.ReadDir("/proc/self/task")
	if err != nil {
		t.Skip("no /proc/self/task")
	}
	var narrow []string
	for _, e := range ents {
		b, err := os.ReadFile(fmt.Sprintf("/proc/self/task/%s/status", e.Name()))
		if err != nil {
			continue
		}
		for _, ln := range strings.Split(string(b), "\n") {
			if !strings.HasPrefix(ln, "Cpus_allowed_list:") {
				continue
			}
			list := strings.TrimSpace(strings.TrimPrefix(ln, "Cpus_allowed_list:"))
			if n := countCPUs(list); n > 0 && n < want {
				narrow = append(narrow, fmt.Sprintf("tid %s -> %s (%d of %d)", e.Name(), list, n, want))
			}
		}
	}
	if len(narrow) > 0 {
		t.Fatalf("with a device open, %d thread(s) carry a mask narrower than the "+
			"process's %d CPUs:\n  %s", len(narrow), want, strings.Join(narrow, "\n  "))
	}
	t.Logf("device open: %d live threads, none masked below %d CPUs", len(ents), want)
}

func countCPUs(s string) int {
	n := 0
	for _, f := range strings.Split(s, ",") {
		if f = strings.TrimSpace(f); f == "" {
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
