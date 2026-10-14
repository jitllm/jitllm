package sched

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Two users of one core set share its workers: a second view starts no
// goroutines, regions from both run concurrently without losing or doubling a
// chunk, each view keeps its own participant count, and the workers stop when
// the last view closes.
func TestSharedViewsTakeTurnsOnOneCrew(t *testing.T) {
	cpus := []int{0, 1, 2, 3}
	// Counted by identity, not by runtime.NumGoroutine: a closed pool's
	// workers exit after Close returns (it does not join), so another test's
	// stragglers move the process count while this one reads it.
	g0 := runtime.NumGoroutine()
	a := Shared(cpus)
	b := Shared(cpus)
	// A failed check below must not leave this crew registered for the next
	// run to be handed. Close is idempotent.
	t.Cleanup(func() { a.Close(); b.Close() })
	if a.crew != b.crew {
		t.Fatal("two views of one core set got two crews")
	}
	if a.crew.crew != nil || len(a.crew.start) != len(cpus)-1 || a.start != nil || b.start != nil {
		t.Fatalf("the crew has %d workers and the views %d and %d: want %d workers once, none for a view",
			len(a.crew.start), len(a.start), len(b.start), len(cpus)-1)
	}

	a.SetParticipants(1)
	if a.N() != 1 || b.N() != len(cpus) {
		t.Fatalf("participants a=%d b=%d: one view's count leaked into the other", a.N(), b.N())
	}

	// A region on a capped view must not use workers above its cap.
	var above atomic.Int64
	a.Do(64, 1, func(w, lo, hi int) {
		if w >= 1 {
			above.Add(1)
		}
	})
	if above.Load() != 0 {
		t.Fatalf("a view capped at 1 participant ran %d chunk(s) on other workers", above.Load())
	}
	a.SetParticipants(len(cpus))

	const regions, total = 2000, 257
	var wg sync.WaitGroup
	for _, p := range []*Pool{a, b} {
		wg.Add(1)
		go func(p *Pool) {
			defer wg.Done()
			for r := 0; r < regions; r++ {
				var sum atomic.Int64
				p.Do(total, 3, func(_, lo, hi int) { sum.Add(int64(hi - lo)) })
				if sum.Load() != total {
					t.Errorf("a region covered %d of %d", sum.Load(), total)
					return
				}
			}
		}(p)
	}
	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(60 * time.Second):
		// Without the crew's lock two views overwrite each other's job and the
		// acknowledgement count, and a region never ends.
		t.Fatal("two views' regions did not finish: they are not taking turns")
	}
	if pa, _ := a.Regions(); pa < regions {
		t.Fatalf("view a counted %d parallel regions, want at least %d: its counters are not its own", pa, regions)
	}

	a.Close()
	a.Close() // idempotent: must not release b's hold
	if b.crew.stopped.Load() {
		t.Fatal("closing one view stopped the crew another still uses")
	}
	b.Close()
	deadline := time.Now().Add(5 * time.Second)
	for runtime.NumGoroutine() > g0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if g := runtime.NumGoroutine(); g > g0 {
		t.Fatalf("%d goroutine(s) left after the last view closed", g-g0)
	}
	if c := Shared(cpus); c.crew == b.crew {
		t.Fatal("a closed crew was handed out again")
	} else {
		c.Close()
	}
}

// Crews over overlapping cores share their lock; disjoint ones do not.
func TestOverlappingCrewsShareTheirTurn(t *testing.T) {
	a, b, c := Shared([]int{0, 1}), Shared([]int{1, 2, 3}), Shared([]int{4, 5})
	defer a.Close()
	defer b.Close()
	defer c.Close()
	if a.crew.mu != b.crew.mu {
		t.Fatal("crews over overlapping cores take turns separately")
	}
	if a.crew.mu == c.crew.mu {
		t.Fatal("crews over disjoint cores share a lock they do not need")
	}
}

// The E-cores alone are a set of their own, disjoint from the P-cores, so a
// session on them takes turns apart from one decoding on the P-cores.
func TestECoresAreTheirOwnCrew(t *testing.T) {
	e := CoreSet("e")
	if len(ECores()) == 0 {
		if len(e) != 0 {
			t.Fatalf("no E-cores here, and the \"e\" set is %v", e)
		}
		t.Skip("no E-cores on this host: the \"e\" set is empty")
	}
	if fmt.Sprint(e) != fmt.Sprint(ECores()) {
		t.Fatalf("the \"e\" set is %v, want ECores %v", e, ECores())
	}
	p, ev := Shared(CoreSet("p")), Shared(e)
	defer p.Close()
	defer ev.Close()
	if p.crew.mu == ev.crew.mu {
		t.Fatal("the E-cores' crew takes turns with the P-cores'")
	}
}

// A view waiting for the crew is handed the turn before the holder takes it
// again: on a three-core runner one session once ran a whole reply while
// another's generate produced no token. View a loops regions back to back on
// GOMAXPROCS 1, 2 and 3 under a three-worker crew while b runs regions too, and the
// lock reports each time b queues (onTurnQueue). From b queuing to b's region
// starting, a may finish the one region it already held and start none: a
// count, not a timing, so a runner too loaded to schedule b promptly changes
// how often b queues and never the bound. (Counting a's regions between two
// of b's measured scheduling: with b not yet back in Lock, a rightly runs on.)
//
// VIOLATION SIGNATURE. With the turn's Lock and Unlock a plain sync.Mutex
// (and the hook called when a TryLock fails), a re-takes the turn past the
// waiting b and this fails with "view a started N regions while b was queued".
func TestSharedViewsAlternateUnderContention(t *testing.T) {
	for _, procs := range []int{1, 2, 3} {
		t.Run(fmt.Sprintf("procs=%d", procs), func(t *testing.T) { alternate(t, procs) })
	}
}

func alternate(t *testing.T, procs int) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(procs))
	cpus := []int{0, 1, 2}
	a, b := Shared(cpus), Shared(cpus)
	defer a.Close()
	defer b.Close()

	var bQueued, aIn atomic.Bool
	var queued, worst, cur, aRuns atomic.Int64
	onTurnQueue = func(w chan struct{}) {
		if w == b.wake {
			// The count restarts before the flag goes up, or a region of a
			// reading the new flag adds to the last wait's count.
			cur.Store(0)
			queued.Add(1)
			bQueued.Store(true)
		}
	}
	defer func() { onTurnQueue = nil }()

	stop := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			select {
			case <-stop:
				return
			default:
			}
			// Counted on the first chunk only: the region is one turn.
			a.Do(64, 1, func(_, lo, hi int) {
				if lo == 0 {
					// a holds the turn until b has queued behind it, so every
					// one of b's regions meets a held turn: whether b ever
					// queues no longer depends on how the runner schedules the
					// two goroutines (on a small one a ran back to back and b
					// found the turn free for all 500 regions, which this gate
					// then reported as no contention). bQueued is cleared by
					// b's own region, so a waits again for the next.
					aIn.Store(true)
					for !bQueued.Load() {
						select {
						case <-stop:
							return
						default:
							runtime.Gosched()
						}
					}
				}
				if lo == 0 && bQueued.Load() {
					if n := cur.Add(1); n > worst.Load() {
						worst.Store(n)
					}
				}
			})
			aRuns.Add(1)
		}
	}()
	for !aIn.Load() {
		runtime.Gosched() // a is inside a region, holding the turn, before b asks
	}
	const regions = 500
	for r := 0; r < regions; r++ {
		b.Do(64, 1, func(_, lo, hi int) {
			if lo == 0 {
				bQueued.Store(false)
			}
		})
	}
	close(stop)
	<-done
	if queued.Load() == 0 {
		t.Fatal("b never queued behind a in 500 regions: the gate exercised no contention")
	}
	// The region a held when b queued may count once, if its first chunk ran
	// after b queued; any further one started past the waiting b.
	if worst.Load() > 1 {
		t.Fatalf("view a started %d regions while b was queued: a waiting view is not handed the turn", worst.Load())
	}
	t.Logf("b queued %d time(s); a ran at most %d region(s) while b waited", queued.Load(), worst.Load())
}
