package sched

import (
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
)

// TestTurnGrantsEveryWaiter is the turn's liveness and bound gate: views of
// one crew run regions of random lengths back to back on GOMAXPROCS 1 to 4,
// with and without more busy goroutines than there are Ps, and every view
// must finish -- a lost wake-up leaves a waiter queued forever and the test
// hangs until go test's own deadline names the goroutine. The lock reports
// each queue and each handoff (turn.onQueue, turn.onGrant), and a queued view
// must be granted after at most one grant per view queued ahead of it. A
// count, not a clock. Every region must also cover its range exactly once.
//
// VIOLATION SIGNATURE. With Unlock handing the turn to the newest waiter
// instead of the oldest, a view that queues first is passed by every one that
// queues after it, and this fails with "a queued view waited N grants to
// others, bound 4".
func TestTurnGrantsEveryWaiter(t *testing.T) {
	for _, procs := range []int{1, 2, 3, 4} {
		for _, busy := range []bool{false, true} {
			t.Run(fmt.Sprintf("procs=%d/busy=%v", procs, busy), func(t *testing.T) { turnStress(t, procs, busy) })
		}
	}
}

func turnStress(t *testing.T, procs int, busy bool) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(procs))
	const views, regions = 5, 100
	cpus := []int{0, 1, 2}
	vs := make([]*Pool, views)
	for i := range vs {
		vs[i] = Shared(cpus)
		defer vs[i].Close()
	}

	// Under the turn's own mutex: each queued view's grant count at queue
	// time, and the grants since.
	var (
		grants int64
		at     = map[chan struct{}]int64{}
		worst  int64
		queued int64
	)
	tn := vs[0].w().mu
	tn.onQueue = func(w chan struct{}) {
		at[w] = grants
		queued++
	}
	tn.onGrant = func(w chan struct{}) {
		if d := grants - at[w]; d > worst {
			worst = d
		}
		delete(at, w)
		grants++
	}

	stop := make(chan struct{})
	var spin sync.WaitGroup
	if busy {
		// More runnable goroutines than Ps: the turn's waiters and the crew's
		// workers compete for a P with work that never blocks.
		for range 2 * procs {
			spin.Add(1)
			go func() {
				defer spin.Done()
				for x := 0; ; x++ {
					if x&1023 == 0 {
						select {
						case <-stop:
							return
						default:
						}
						// Always runnable, never blocked: it yields so a
						// held turn's waiters are not each a full slice.
						runtime.Gosched()
					}
				}
			}()
		}
	}

	var wg sync.WaitGroup
	var bad atomic.Int64
	for i, v := range vs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(i)))
			for range regions {
				n := 8 + rng.Intn(120)
				work := rng.Intn(200)
				var got atomic.Int64
				v.Do(n, 1, func(_, lo, hi int) {
					if lo == 0 {
						// The holder gives its P up mid-region, so on one P
						// the other views run and queue behind it.
						runtime.Gosched()
					}
					for k := 0; k < work; k++ {
						spinSink.Add(0)
					}
					got.Add(int64(hi - lo))
				})
				if got.Load() != int64(n) {
					bad.Add(1)
				}
			}
		}()
	}
	wg.Wait()
	close(stop)
	spin.Wait()

	if bad.Load() != 0 {
		t.Fatalf("%d region(s) lost or doubled a chunk", bad.Load())
	}
	if queued == 0 {
		t.Fatal("no view ever queued: the gate exercised no contention")
	}
	// Strict arrival order hands a queued view the turn after at most the
	// views ahead of it.
	if bound := int64(views - 1); worst > bound {
		t.Fatalf("a queued view waited %d grants to others, bound %d", worst, bound)
	}
	t.Logf("%d queue(s), %d handoff(s); a waiter saw at most %d grants to others", queued, grants, worst)
}

var spinSink atomic.Int64
