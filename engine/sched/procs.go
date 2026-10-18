package sched

import (
	"runtime"
	"sync"
)

// GOMAXPROCS is process state, and several models in one process each raise it
// for their own prefill. A save-and-restore per model breaks as soon as two of
// those phases overlap: the first to end restores a width the second still
// needs, and the second then restores the first's raised value. procs holds
// the live requests instead and sets the maximum of them, and puts back the
// value it found once the last one ends.
var procs struct {
	mu   sync.Mutex
	base int         // GOMAXPROCS when the first live request arrived
	live map[int]int // requested width -> live requests at that width
}

// RaiseProcs asks for GOMAXPROCS of at least n until the returned release
// runs. Concurrent requests get their maximum; when the last is released the
// value the first one found is restored. release is safe to call twice.
func RaiseProcs(n int) (release func()) {
	h := new(ProcsHold)
	h.Raise(n)
	return h.Release
}

// ProcsHold is one GOMAXPROCS request held by its owner, RaiseProcs without
// the release closure: a JIT raises and releases one every prefill, so it
// keeps the hold and the pair allocates nothing. The zero value holds
// nothing.
type ProcsHold struct{ n int }

// Raise asks for GOMAXPROCS of at least n until Release; a hold already live
// keeps its request.
func (h *ProcsHold) Raise(n int) {
	if h.n != 0 || n <= 0 {
		return
	}
	procs.mu.Lock()
	defer procs.mu.Unlock()
	if len(procs.live) == 0 {
		procs.base = runtime.GOMAXPROCS(0)
	}
	if procs.live == nil {
		procs.live = map[int]int{}
	}
	procs.live[n]++
	h.n = n
	setProcsLocked()
}

// Release ends the request; it is safe to call on a hold that is not live.
func (h *ProcsHold) Release() {
	if h.n == 0 {
		return
	}
	procs.mu.Lock()
	defer procs.mu.Unlock()
	if procs.live[h.n]--; procs.live[h.n] == 0 {
		delete(procs.live, h.n)
	}
	h.n = 0
	setProcsLocked()
}

// Held reports whether the hold's request is live.
func (h *ProcsHold) Held() bool { return h.n != 0 }

// setProcsLocked sets GOMAXPROCS to the widest live request, or the base when
// none is wider or none is live.
func setProcsLocked() {
	want := procs.base
	for n := range procs.live {
		want = max(want, n)
	}
	if runtime.GOMAXPROCS(0) != want {
		runtime.GOMAXPROCS(want)
	}
}

// seq is the CPU ids 0..n-1, at least one.
func seq(n int) []int {
	if n < 1 {
		n = 1
	}
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}
