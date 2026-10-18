package sched

import (
	"sync"
	"sync/atomic"
)

// turn is the lock a crew's views take a region each under, handed to waiters
// in arrival order.
//
// A sync.Mutex is not that: its holder relocks before a woken waiter is
// scheduled, and a waiter only forces a handoff after a millisecond of
// failing. A session whose region loop relocks at once keeps the cores for as
// long as the other's goroutine stays unscheduled -- on a host with fewer
// threads than the crew has spinning workers that was a whole short reply
// while the other session produced no token.
//
// The uncontended path is one CAS to take and one to give, as a mutex's is.
// Once a waiter is queued the state stays contended and no newcomer can take
// the turn ahead of it: Unlock hands the turn to the queue's head, so a view
// waiting runs after at most the region in flight.
type turn struct {
	// state is turnFree, turnHeld, or turnQueued (held, and q is not empty).
	state atomic.Int32
	mu    sync.Mutex // guards q and every move into or out of turnQueued
	q     []chan struct{}
	// users is how many callers hold views on the crews that take turns on
	// this lock, under crews.mu: a view opened beside another of the same
	// caller's (SharedBeside) is not one more.
	users int
	// onQueue, set only by a test before the turn is contended, is told under
	// mu each time a caller queues behind the holder. It is on the contended
	// path alone, so the uncontended turn pays nothing for it.
	onQueue func(wake chan struct{})
	// onGrant, set only by a test, is told under mu each time Unlock hands the
	// turn to a queued caller.
	onGrant func(wake chan struct{})
}

const (
	turnFree int32 = iota
	turnHeld
	turnQueued
)

// Lock takes the turn; wake is the caller's own channel (buffered, one slot),
// on which a handoff arrives. One caller per channel at a time.
func (t *turn) Lock(wake chan struct{}) {
	if t.state.CompareAndSwap(turnFree, turnHeld) {
		return
	}
	t.mu.Lock()
	for {
		switch s := t.state.Load(); s {
		case turnFree:
			if t.state.CompareAndSwap(turnFree, turnHeld) {
				t.mu.Unlock()
				return
			}
		case turnHeld:
			if !t.state.CompareAndSwap(turnHeld, turnQueued) {
				continue // released meanwhile
			}
			fallthrough
		case turnQueued:
			t.q = append(t.q, wake)
			if t.onQueue != nil {
				t.onQueue(wake)
			}
			t.mu.Unlock()
			<-wake // the holder handed the turn over; it is ours
			return
		}
	}
}

// Unlock gives the turn to the longest waiter, or frees it.
func (t *turn) Unlock() {
	if t.state.CompareAndSwap(turnHeld, turnFree) {
		return
	}
	t.mu.Lock()
	next := t.q[0]
	n := copy(t.q, t.q[1:])
	t.q[n] = nil
	t.q = t.q[:n]
	if n == 0 {
		t.state.Store(turnHeld)
	}
	if t.onGrant != nil {
		t.onGrant(next)
	}
	t.mu.Unlock()
	next <- struct{}{}
}
