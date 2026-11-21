package sched

import (
	"runtime"
	"sync"
)

// Crews is the worker crews a process's pools run on and the GOMAXPROCS their
// requests hold, shared by every pool made with it (WithCrews): a pool on a
// core set a crew already runs is a view of that crew rather than a second
// set of workers behind a second barrier, crews over overlapping cores take
// turns, and several models' prefills raising GOMAXPROCS are held together
// (ProcsHold). A process running several models gives them one; pools made
// with none share nothing with pools made elsewhere.
type Crews struct {
	mu sync.Mutex
	m  map[string]*Pool

	// GOMAXPROCS is process state: a save-and-restore per model breaks as soon
	// as two raised phases overlap, the first to end restoring a width the
	// second still needs. The crews hold the live requests instead, set the
	// maximum of them, and put back the value found once the last one ends.
	procs struct {
		mu   sync.Mutex
		base int         // GOMAXPROCS when the first live request arrived
		live map[int]int // requested width -> live requests at that width
	}
}

// NewCrews is crews running nothing yet.
func NewCrews() *Crews { return &Crews{m: map[string]*Pool{}} }

// WithCrews makes a pool one of c's: on a core set one of them runs, it joins
// that crew (Shared).
func WithCrews(c *Crews) Option { return func(cfg *config) { cfg.crews = c } }

// CrewsOf is the crews opts name, nil for none.
func CrewsOf(opts ...Option) *Crews { return newConfig(opts).crews }

// Hold is a GOMAXPROCS request of c's, holding nothing until Raise.
func (c *Crews) Hold() ProcsHold { return ProcsHold{c: c} }

// RaiseProcs asks for GOMAXPROCS of at least n until the returned release
// runs. Concurrent requests get their maximum; when the last is released the
// value the first one found is restored. release is safe to call twice.
func (c *Crews) RaiseProcs(n int) (release func()) {
	h := c.Hold()
	h.Raise(n)
	return h.Release
}

// ProcsHold is one GOMAXPROCS request of a Crews held by its owner,
// RaiseProcs without the release closure: a pool raises and releases one
// every prefill, so it keeps the hold and the pair allocates nothing.
type ProcsHold struct {
	c *Crews
	n int
}

// Raise asks for GOMAXPROCS of at least n until Release; a hold already live
// keeps its request, and a hold of no Crews asks nothing.
func (h *ProcsHold) Raise(n int) {
	if h.c == nil || h.n != 0 || n <= 0 {
		return
	}
	p := &h.c.procs
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.live) == 0 {
		p.base = runtime.GOMAXPROCS(0)
	}
	if p.live == nil {
		p.live = map[int]int{}
	}
	p.live[n]++
	h.n = n
	h.c.setProcsLocked()
}

// Release ends the request; it is safe to call on a hold that is not live.
func (h *ProcsHold) Release() {
	if h.n == 0 {
		return
	}
	p := &h.c.procs
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.live[h.n]--; p.live[h.n] == 0 {
		delete(p.live, h.n)
	}
	h.n = 0
	h.c.setProcsLocked()
}

// Held reports whether the hold's request is live.
func (h *ProcsHold) Held() bool { return h.n != 0 }

// setProcsLocked sets GOMAXPROCS to the widest live request, or the base when
// none is wider or none is live.
func (c *Crews) setProcsLocked() {
	want := c.procs.base
	for n := range c.procs.live {
		want = max(want, n)
	}
	if runtime.GOMAXPROCS(0) != want {
		runtime.GOMAXPROCS(want)
	}
}
