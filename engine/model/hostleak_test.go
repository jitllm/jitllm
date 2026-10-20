//go:build linux

package model

import (
	"os"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/internal/testmodels"
)

func rssKB(t *testing.T) int {
	t.Helper()
	b, err := os.ReadFile("/proc/self/status")
	if err != nil {
		t.Skipf("no /proc/self/status: %v", err)
	}
	for _, ln := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(ln, "VmRSS:") {
			f := strings.Fields(ln)
			n, _ := strconv.Atoi(f[1])
			return n
		}
	}
	return 0
}

func openFDs(t *testing.T) int {
	t.Helper()
	e, err := os.ReadDir("/proc/self/fd")
	if err != nil {
		t.Skipf("no /proc/self/fd: %v", err)
	}
	return len(e)
}

// TestModelOpenCloseDoesNotLeak drives the whole host path -- container, pager,
// JIT, worker pool, State -- through repeated open/close and requires the
// process to come back.
//
// Between sched's pool gate and the tier's device-buffer gate sit the page
// frames, the dense region, the tokenizer, the code mappings and the file
// handle: the parts a server cycles most. RSS, file descriptors and goroutines
// are checked because they fail independently.
func TestModelOpenCloseDoesNotLeak(t *testing.T) {
	path := jlmOf(t, testmodels.Path("stories15M-q8_0.gguf"))

	// Warm: the first open maps code, builds the tokenizer tables and grows the
	// heap to its working size. Measuring from zero would measure the warm-up.
	for i := 0; i < 2; i++ {
		m, err := Open(path)
		if err != nil {
			t.Skip(err)
		}
		s := m.NewState(16)
		if _, err := s.Forward(1); err != nil {
			t.Fatal(err)
		}
		s.Close()
		m.Close()
	}
	runtime.GC()
	runtime.GC()
	time.Sleep(300 * time.Millisecond)
	rss0, fd0, g0 := rssKB(t), openFDs(t), runtime.NumGoroutine()

	const rounds = 12
	for i := 0; i < rounds; i++ {
		m, err := Open(path)
		if err != nil {
			t.Fatalf("round %d: %v", i, err)
		}
		s := m.NewState(32)
		for k := 0; k < 4; k++ {
			if _, err := s.Forward(int32(1 + k)); err != nil {
				t.Fatalf("round %d: %v", i, err)
			}
		}
		s.Close()
		m.Close()
	}
	runtime.GC()
	runtime.GC()
	time.Sleep(700 * time.Millisecond)
	rss1, fd1, g1 := rssKB(t), openFDs(t), runtime.NumGoroutine()

	t.Logf("%d open/close cycles: RSS %d -> %d KiB (%+d), fds %d -> %d, goroutines %d -> %d",
		rounds, rss0, rss1, rss1-rss0, fd0, fd1, g0, g1)

	if fd1 > fd0 {
		t.Errorf("file descriptors %d -> %d over %d cycles: a container handle is not closed",
			fd0, fd1, rounds)
	}
	if g1 > g0+2 {
		t.Errorf("goroutines %d -> %d over %d cycles", g0, g1, rounds)
	}
	// RSS is the loose one on purpose: Go returns freed spans to the OS lazily,
	// so a fixed slack is the honest bound. What it MUST catch is growth that
	// scales with the cycle count -- one model's resident bytes per round.
	if grow := rss1 - rss0; grow > 24*1024 {
		t.Errorf("RSS grew %d KiB over %d cycles (%d KiB per cycle): that tracks the "+
			"cycle count, which is the model's own bytes never coming back",
			grow, rounds, grow/rounds)
	}
}

// TestPagedModelOpenCloseDoesNotLeak is the same question where the pager is
// actually doing work: a real model under a budget too small to hold it, so
// every round allocates frames, evicts, re-reads and must give all of it back.
//
// The tiny-model arm cannot see a frame leak: stories15M pages nothing.
func TestPagedModelOpenCloseDoesNotLeak(t *testing.T) {
	if testing.Short() {
		t.Skip("opens a real model repeatedly")
	}
	// RSS is not a frame measurement under -race: the detector's shadow memory
	// grows with the bytes a process touches, so after a memory-heavy sibling
	// it reports the tool. The gate runs in every ordinary build.
	if raceEnabled {
		t.Skip("process RSS is not a frame measurement under -race; the detector's shadow " +
			"memory grows with the bytes touched")
	}
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))

	open := func() (*Model, error) {
		m, err := Open(path)
		if err != nil {
			return nil, err
		}
		// A budget below the model forces frames, eviction and re-reads.
		m.SetPageBudget(4 * m.PageSize())
		return m, nil
	}
	for i := 0; i < 2; i++ {
		m, err := open()
		if err != nil {
			t.Skip(err)
		}
		s := m.NewState(16)
		if _, err := s.Forward(1); err != nil {
			t.Fatal(err)
		}
		s.Close()
		m.Close()
	}
	runtime.GC()
	runtime.GC()
	time.Sleep(400 * time.Millisecond)
	rss0, fd0, g0 := rssKB(t), openFDs(t), runtime.NumGoroutine()

	const rounds = 6
	for i := 0; i < rounds; i++ {
		m, err := open()
		if err != nil {
			t.Fatalf("round %d: %v", i, err)
		}
		s := m.NewState(24)
		for k := 0; k < 3; k++ {
			if _, err := s.Forward(int32(1 + k)); err != nil {
				t.Fatalf("round %d: %v", i, err)
			}
		}
		s.Close()
		m.Close()
	}
	runtime.GC()
	runtime.GC()
	time.Sleep(900 * time.Millisecond)
	rss1, fd1, g1 := rssKB(t), openFDs(t), runtime.NumGoroutine()

	t.Logf("%d paged open/close cycles: RSS %d -> %d KiB (%+d), fds %d -> %d, goroutines %d -> %d",
		rounds, rss0, rss1, rss1-rss0, fd0, fd1, g0, g1)
	if fd1 > fd0 {
		t.Errorf("file descriptors %d -> %d over %d cycles", fd0, fd1, rounds)
	}
	if g1 > g0+2 {
		t.Errorf("goroutines %d -> %d over %d cycles", g0, g1, rounds)
	}
	// One frame of this model is the unit a leak would come in; the slack is
	// below that, so a single leaked frame per cycle fails.
	if grow := rss1 - rss0; grow > 64*1024 {
		t.Errorf("RSS grew %d KiB over %d cycles (%d KiB per cycle): frames are not "+
			"coming back", grow, rounds, grow/rounds)
	}
}
