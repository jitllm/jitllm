//go:build jitllmtest

package sched

import (
	"sync"
	"sync/atomic"
	"testing"
)

// TestARegionDoesNotWaitForADescheduledWorker: a worker that has seen a
// region but is kept off a CPU -- here held at Pool.onWake, as an
// oversubscribed host holds it for a scheduling slice or many -- must not
// hold the region up: the caller drains the chunks, closes the gate and
// returns, and the held worker, once let go, finds the gate closed and
// touches nothing. On a laptop at load average 31 a pool that waited for
// every worker to acknowledge every region ran each region at the slowest
// worker's scheduling latency, and a host-only server produced no token for
// minutes.
//
// VIOLATION SIGNATURE. A publish that waits for every worker (the old
// acknowledgement count) never returns while the worker is held: the test
// hangs and go test's deadline names publish's wait.
func TestARegionDoesNotWaitForADescheduledWorker(t *testing.T) {
	release := make(chan struct{})
	held := make(chan struct{})
	var once sync.Once
	hook := func(q *Pool, id int) {
		if id == 0 {
			once.Do(func() {
				close(held)
				<-release
			})
		}
	}
	p := New([]int{0, 1, 2}, func(c *config) { c.onWake = hook })
	defer p.Close()

	const regions = 200
	run := func(what string) {
		for r := 0; r < regions; r++ {
			var sum atomic.Int64
			p.Do(64, 1, func(_, lo, hi int) { sum.Add(int64(hi - lo)) })
			if sum.Load() != 64 {
				t.Fatalf("%s: region %d covered %d of 64", what, r, sum.Load())
			}
		}
	}
	// One region wakes worker 0, which then sits in the hook having seen
	// a region, as a descheduled worker sits.
	var sum atomic.Int64
	p.Do(64, 1, func(_, lo, hi int) { sum.Add(int64(hi - lo)) })
	<-held
	run("worker 0 held")
	close(release)
	// The held worker comes back to a later region's gate, or a closed one,
	// and the pool goes on.
	run("after the release")
}
