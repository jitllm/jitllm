package jlm

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
)

// The speculative half of the pager's reads.
//
// A demanded read (EnsureRanges, EnsurePage) runs on its caller at once and
// never queues, so it cannot wait behind a guess. A speculative read is a
// guess about the next block's experts: Speculate queues the pages, a fixed
// set of workers reads them through the same claim, pin and busy bookkeeping,
// and the properties a prefetch needs fall out of that bookkeeping rather
// than out of a second cache:
//
//   - a page being read is joined, never read twice: planRunsBusy marks the
//     chunks it will read, and a demanded caller that wants the same chunks
//     waits for those alone and reads the rest itself;
//   - a demanded page that is still queued is promoted by the caller reading
//     it: the queued record then finds the page filled and does nothing;
//   - a frame being read is pinned for the read, so an eviction cannot hand
//     it to another page, and a guess nobody started is dropped (Speculate
//     replaces the queue, Abandon empties it).
//
// Speculative occupancy is bounded twice: by the queue (the pages waiting)
// and by specHeld, the frames holding a guess nobody has demanded yet, which
// a speculative claim refuses to grow past SetSpeculation's cap. A wrong
// guess stays charged until it is demanded or evicted, so the ceiling is a
// cap on how much of the budget guesses can occupy, not an exact account of
// the useful ones.

const (
	// specRing is how many pages can wait for a worker.
	specRing = 64
	// specWorkers is how many speculative reads run at once. Demanded reads
	// are not counted: they do not use these.
	specWorkers = 4
	// defaultSpecPages is the speculative occupancy cap, in pages, until
	// SetSpeculation says otherwise.
	defaultSpecPages = 32
)

var errSpecFull = errors.New("jlm: speculative occupancy is full")

// specSvc is the queue and its workers. mu is taken before f.mu, never
// after.
type specSvc struct {
	mu      sync.Mutex
	cond    sync.Cond
	ring    []int
	head, n int
	started bool
	closed  bool
	wg      sync.WaitGroup
	// skipped and failed count guesses refused for occupancy and guesses
	// whose read failed; neither fails anything.
	skipped, failed atomic.Int64
}

// SetSpeculation sets the most frames speculative reads may hold at once
// (pages nobody has demanded yet). 0 restores the default.
func (f *File) SetSpeculation(pages int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.specCap = pages
}

// SpeculationStats is how many guesses were refused for occupancy and how
// many failed to read. A failed guess is not a failed token.
func (f *File) SpeculationStats() (skipped, failed int64) {
	return f.spec.skipped.Load(), f.spec.failed.Load()
}

func (f *File) specLimit() int {
	if f.specCap > 0 {
		return f.specCap
	}
	return defaultSpecPages
}

// markSpec charges page b to the speculative occupancy. Callers hold f.mu.
func (f *File) markSpec(b int) {
	if f.specOn == nil {
		f.specOn = make([]bool, len(f.pages))
	}
	if !f.specOn[b] {
		f.specOn[b] = true
		f.specHeld++
	}
}

// unspec takes page b's charge off: it was demanded, or it left. Callers
// hold f.mu.
func (f *File) unspec(b int) {
	if f.specOn != nil && f.specOn[b] {
		f.specOn[b] = false
		f.specHeld--
	}
}

// Speculate replaces the queue of guesses with expert pages sel of bank
// entry e: guesses not yet started are dropped, and the pages that are not
// already resident and filled are queued for the workers. It never blocks and
// allocates nothing once the service has started. A nil sel is Abandon.
func (f *File) Speculate(e *Entry, sel []uint32) {
	q := &f.spec
	q.mu.Lock()
	defer q.mu.Unlock()
	q.n = 0
	if q.closed {
		return
	}
	if q.ring == nil {
		q.ring = make([]int, specRing)
		q.cond.L = &q.mu
	}
	for _, x := range sel {
		p := f.ExpertPage(e, int(x))
		if p < 0 || q.n == len(q.ring) || f.pageWarm(p) || q.queued(p) {
			continue
		}
		q.ring[(q.head+q.n)%len(q.ring)] = p
		q.n++
	}
	if q.n == 0 {
		return
	}
	if !q.started {
		q.started = true
		for range specWorkers {
			q.wg.Add(1)
			go f.specWorker()
		}
	}
	q.cond.Broadcast()
}

// Abandon drops every guess no worker has started. A read in progress
// finishes: its frame is pinned and may be joined by a demanded read.
func (f *File) Abandon() {
	q := &f.spec
	q.mu.Lock()
	q.n = 0
	q.mu.Unlock()
}

// queued reports whether page p is waiting. Callers hold q.mu.
func (q *specSvc) queued(p int) bool {
	for i := range q.n {
		if q.ring[(q.head+i)%len(q.ring)] == p {
			return true
		}
	}
	return false
}

// pageWarm reports whether page p is resident with every chunk filled.
func (f *File) pageWarm(p int) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.warmLocked(p)
}

func (f *File) warmLocked(p int) bool {
	if f.pages[p] == nil || f.filled == nil || f.filled[p] == nil {
		return false
	}
	for _, ok := range f.filled[p] {
		if !ok {
			return false
		}
	}
	return true
}

// demandWarm is pageWarm for a demanded page: a warm hit is a use, so it
// ticks the LRU order claim would have, and takes the page off the
// speculative charge.
func (f *File) demandWarm(p int) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.warmLocked(p) {
		return false
	}
	f.tick++
	if f.used != nil {
		f.used[p] = f.tick
	}
	f.unspec(p)
	return true
}

func (f *File) specWorker() {
	q := &f.spec
	defer q.wg.Done()
	rs := make([]Range, 1)
	q.mu.Lock()
	defer q.mu.Unlock()
	for {
		for q.n == 0 && !q.closed {
			q.cond.Wait()
		}
		if q.closed {
			return
		}
		p := q.ring[q.head]
		q.head = (q.head + 1) % len(q.ring)
		q.n--
		q.mu.Unlock()
		off, size := f.pageAt(p)
		rs[0] = Range{Off: off, N: size}
		switch err := f.ensureRanges(p, rs, true); {
		case err == nil:
		case errors.Is(err, errSpecFull):
			q.skipped.Add(1)
		default:
			q.failed.Add(1)
		}
		q.mu.Lock()
	}
}

// stopSpec ends the workers and waits for the reads in flight, so Close never
// unmaps a frame a worker is filling.
func (f *File) stopSpec() {
	q := &f.spec
	q.mu.Lock()
	q.closed = true
	q.n = 0
	if q.ring != nil {
		q.cond.Broadcast()
	}
	q.mu.Unlock()
	q.wg.Wait()
}

// EnsureExperts makes expert pages sel of bank entry e readable: the demanded
// read of a routed selection. The pages that are not warm are read together,
// one on the caller and the others beside it, so the queue depth is the
// selection's; a page a speculative read is already reading is joined by
// EnsurePage, so the caller waits for that page alone. Nothing here queues
// behind a speculative read, and a selection that is all warm allocates
// nothing and starts nothing.
func (f *File) EnsureExperts(e *Entry, sel []uint32) error {
	cold := false
	for _, x := range sel {
		p := f.ExpertPage(e, int(x))
		if p < 0 {
			return fmt.Errorf("jlm: expert %d has no expert page", x)
		}
		if !f.demandWarm(p) {
			cold = true
		}
	}
	if !cold {
		return nil
	}
	var (
		wg    sync.WaitGroup
		errMu sync.Mutex
		first error
	)
	note := func(err error) {
		if err != nil {
			errMu.Lock()
			if first == nil {
				first = err
			}
			errMu.Unlock()
		}
	}
	inline := -1
	for _, x := range sel {
		p := f.ExpertPage(e, int(x))
		if f.demandWarm(p) {
			continue
		}
		if inline < 0 {
			inline = p
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			note(f.EnsurePage(p))
		}()
	}
	if inline >= 0 {
		note(f.EnsurePage(inline))
	}
	wg.Wait()
	return first
}
