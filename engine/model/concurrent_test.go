package model

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"sync"
	"testing"
	"time"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestConcurrentMultiDeviceDoesNotLeak runs several States at once, with the
// blocks spread over every device the host has, and requires the process to
// come back after each cycle: several States sharing a tier, devices sharing a
// host budget, pools sharing cores. Concurrency is the point: sched.Pool.Do is
// single-caller, which is safe only because each State builds its own JIT, and
// a shared tier.GPU tests devTier's locking.
//
// It checks the live heap, the device byte ledger, the driver objects the
// engine owns, file descriptors, threads and goroutines, because they fail
// independently.
func TestConcurrentMultiDeviceDoesNotLeak(t *testing.T) {
	if testing.Short() {
		t.Skip("opens a real model on every device")
	}
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))

	const workers = 4
	cycle := func() {
		m, err := Open(path)
		if err != nil {
			t.Fatalf("open: %v", err)
		}
		defer m.Close()

		// Every device the host reports, not just the fastest.
		g, err := tier.OpenWith(tier.WithDevices("all"), tier.WithDeviceTune(tier.TuneOff))
		if err != nil || g == nil {
			noDevice(t, "devices", err)
		}
		defer g.Close()

		// States are built and attached serially, forwards run concurrently:
		// sessions are admitted one at a time and then served together, as a
		// server does. (Concurrent construction is gated by
		// nn.TestConcurrentNewJITKeepsEachJITsOptions and
		// TestConcurrentOpenKeepsEachModelsOptions.)
		ss := make([]*State, workers)
		// Closed explicitly, not by defer, so the device ledger can be read
		// while the tier is still open.
		closed := false
		defer func() {
			if !closed {
				for _, s := range ss {
					if s != nil {
						s.Close()
					}
				}
			}
		}()
		for w := range ss {
			ss[w] = m.NewState(48)
			ss[w].SetDevice(g)
		}
		st0 := g.Stats()
		var wg sync.WaitGroup
		errs := make([]error, workers)
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func(w int) {
				defer wg.Done()
				s := ss[w]
				for k := 0; k < 6; k++ {
					if _, err := s.Forward(int32(1 + w*7 + k)); err != nil {
						errs[w] = fmt.Errorf("worker %d step %d: %w", w, k, err)
						return
					}
				}
			}(w)
		}
		wg.Wait()
		for _, e := range errs {
			if e != nil {
				t.Fatal(e)
			}
		}
		// Closing every session must hand back every byte of history. The KV
		// pool keeps pages a closed session released, as free room for the
		// next one (kvCompact gives them back when something needs it), so
		// the device may hold more than at placement -- but only as pool that
		// holds nothing. Checked while the tier is open: after g.Close() a
		// driver reclaims the context and a leak becomes invisible.
		//
		// Two counts, because the budget is not one number: what the device's
		// buffers hold by the backend's own count (Allocated), and what the
		// tier charged for them (BudgetUsed less RoundingBytes). Each may grow
		// by the free pool's bytes and by nothing else. The driver's page
		// rounding, the rest of BudgetUsed, is left out on purpose: it is a
		// function of which buffers are live, so a buffer kept is already in
		// Allocated, and the pool's own rounding moves with its layout (a
		// buffer past 1 MiB rounds to 2 MiB on CUDA) where KVPoolBytes does not.
		for _, s := range ss {
			s.Close()
		}
		closed = true
		st := g.Stats()
		if st.KVBytes != 0 {
			t.Errorf("%d bytes of attention history held after every State closed", st.KVBytes)
		}
		grew := st.KVPoolBytes - min(st.KVPoolBytes, st0.KVPoolBytes)
		if st.Allocated > st0.Allocated+grew {
			t.Errorf("the device's buffers hold %d bytes after every State closed (%d at placement, "+
				"%d more of free KV pool): a session's buffers are not returned",
				st.Allocated, st0.Allocated, grew)
		}
		charged := func(s tier.Stats) uint64 { return s.BudgetUsed - s.RoundingBytes }
		if charged(st) > charged(st0)+grew {
			t.Errorf("the device's budget charges %d bytes after every State closed (%d at placement, "+
				"%d more of free KV pool), the driver's rounding apart: a session's charge is not refunded",
				charged(st), charged(st0), grew)
		}
	}

	// The assertion is the live heap and the device ledger, not RSS: RSS after a
	// scavenge is the Go scavenger's opinion about returning pages and varies
	// run to run by as much as the signal (see
	// docs/engineering-history/scheduling-and-measurement.md). RSS stays in the
	// log, where a divergence from the heap is worth seeing.
	cycle() // warm: contexts, code mappings, tokenizer tables
	runtime.GC()
	debug.FreeOSMemory()
	time.Sleep(700 * time.Millisecond)
	p0, g0 := procNow(t), runtime.NumGoroutine()
	own0 := backend.OwnedNow()
	heap0 := liveHeap()

	const rounds = 4
	for i := 0; i < rounds; i++ {
		cycle()
	}
	runtime.GC()
	debug.FreeOSMemory()
	time.Sleep(1500 * time.Millisecond)
	p1, g1 := procNow(t), runtime.NumGoroutine()
	own1 := backend.OwnedNow()
	heap1 := liveHeap()

	t.Logf("%d cycles x %d concurrent States on every device: live heap %d -> %d KiB (%+d), "+
		"goroutines %d -> %d, driver objects owned %+v -> %+v",
		rounds, workers, heap0/1024, heap1/1024, (int64(heap1)-int64(heap0))/1024,
		g0, g1, own0, own1)
	if p0.ok {
		t.Logf("RSS %d -> %d KiB (%+d), fds %d -> %d, Go Ms %d -> %d, driver threads %d -> %d",
			p0.rss, p1.rss, p1.rss-p0.rss, p0.fds, p1.fds,
			p0.th.goMs, p1.th.goMs, p0.th.foreign, p1.th.foreign)
	}

	if p0.ok && p1.fds > p0.fds {
		t.Errorf("file descriptors %d -> %d: a container or a driver handle is not closed", p0.fds, p1.fds)
	}
	if g1 > g0 {
		t.Errorf("goroutines %d -> %d: a device owner or a pool worker is not exiting", g0, g1)
	}
	if own1 != own0 {
		t.Errorf("driver objects owned %+v -> %+v: a device or a session's queue is not closed",
			own0, own1)
	}
	// Threads are judged by kind, not by count. A driver's threads belong to
	// its contexts and queues, so more of them after Close is a context or a
	// queue kept. The Go Ms are the runtime's pool: it makes one when every M
	// is busy or blocked in a driver call and destroys one only under a
	// goroutine that exits locked to it, so their count is the high-water of
	// goroutines blocked in the drivers at once. Four States on every device
	// push it up a step at a time over the first twenty-odd cycles (22 -> 27
	// over 24 cycles on the laptop, the steps at no fixed cycle) and then hold
	// it; that was the "OS threads 20 -> 25" this test used to fail on. A
	// leaked M is never idle -- its goroutine is still live, which the
	// goroutine count above catches -- and a goroutine that exits locked
	// takes the process down in this cgo-free build before any count is read
	// (see cuda.Device.Close). So the Go M count is logged and not gated: no
	// bound on it within four cycles separates a leak from the pool.
	if p0.ok && p1.th.foreign > p0.th.foreign {
		t.Errorf("driver threads %d -> %d over %d cycles: a context or a queue is not torn down",
			p0.th.foreign, p1.th.foreign, rounds)
	}
	// HeapAlloc right after runtime.GC() is what a leak is: bytes still
	// reachable. The scavenger cannot move it and it needs no sleep to settle.
	if grow := int64(heap1) - int64(heap0); grow > 24<<20 {
		t.Errorf("live heap grew %d KiB over %d cycles (%d KiB per cycle): a State, "+
			"a pool or a device view is still reachable after Close",
			grow/1024, rounds, grow/1024/int64(rounds))
	}
}

// liveHeap is bytes still reachable after a collection: the exact quantity a
// leak is, unlike RSS.
func liveHeap() uint64 {
	runtime.GC()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	return ms.HeapAlloc
}
