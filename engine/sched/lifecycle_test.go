//go:build linux

package sched

import (
	"testing"
	"time"
)

// TestCloseIsIdempotent: a second Close must not panic.
func TestCloseIsIdempotent(t *testing.T) {
	p := New(DecodeCores())
	p.Close()
	p.Close()
	p.Close()
}

// TestDoAfterCloseRunsInlineRatherThanHanging: a region dispatched on a stopped
// pool must still run, on the caller, and must return: a stopped worker never
// acknowledges, so waiting for it hangs. The gate is time-bounded because the
// violation is a hang, not a wrong answer.
func TestDoAfterCloseRunsInlineRatherThanHanging(t *testing.T) {
	p := New(DecodeCores())
	p.Close()

	const total = 4096
	got := make([]int32, total)
	done := make(chan struct{})
	go func() {
		defer close(done)
		p.Do(total, 64, func(_, lo, hi int) {
			for i := lo; i < hi; i++ {
				got[i]++
			}
		})
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("Do on a closed pool did not return: it is waiting on workers " +
			"that have already left and will never acknowledge")
	}
	for i, v := range got {
		if v != 1 {
			t.Fatalf("element %d ran %d times, want exactly 1: a closed pool must "+
				"still compute the region, once", i, v)
		}
	}
}

// TestClosedPoolCountsTheRegionAsSerial: the inline path above must be visible.
// Regions() is what prices dispatch, and a region that silently changed class
// would misreport the thing the counter exists to answer.
func TestClosedPoolCountsTheRegionAsSerial(t *testing.T) {
	p := New(DecodeCores())
	p.Close()
	p.ResetRegions()
	p.Do(4096, 64, func(_, lo, hi int) {})
	par, ser := p.Regions()
	if par != 0 || ser != 1 {
		t.Fatalf("Regions() = (%d parallel, %d serial), want (0, 1)", par, ser)
	}
}

// TestRestParksEveryWorker: Rest returns with every worker parked, long
// before a spin budget of a second would have run out; workers that ignore
// it read as a Rest that waited out its bound.
func TestRestParksEveryWorker(t *testing.T) {
	p := New(DecodeCores(), WithSpin(time.Second))
	defer p.Close()
	if len(p.parked) == 0 {
		t.Fatal("a pool with no workers proves nothing")
	}
	p.Do(4096, 1, func(int, int, int) {})
	start := time.Now()
	p.Rest()
	took := time.Since(start)
	for i := range p.parked {
		if !p.parked[i].Load() {
			t.Fatalf("worker %d still spins after Rest (%v)", i, took)
		}
	}
	t.Logf("%d workers parked in %v", len(p.parked), took)
	if took > 500*time.Millisecond {
		t.Fatalf("Rest took %v: the workers spun out their budget", took)
	}
}
