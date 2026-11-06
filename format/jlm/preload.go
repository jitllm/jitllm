package jlm

import (
	"sync"
	"sync/atomic"
)

// Preload reads every page into memory in the background of whatever the
// caller does next, depth pages at a time, and returns once all are in or stop
// is closed.
//
// A cold model otherwise reads its pages where the first token faults them in:
// one block at a time, each read only when the forward pass reaches it, with
// the generated code compiled and run in between. That leaves the drive idle
// while a kernel is emitted and the cores idle while a page lands. Preload
// keeps depth pages in flight (each already split ReadThreads ways by readAt,
// so the queue holds up to depth*ReadThreads requests) in block order, so the
// first token finds its early blocks resident and the JIT runs beside the
// reads rather than between them.
//
// Preload only runs when the budget holds every page: below that the forward
// pass's own order decides residency (MRU over a cyclic scan, see EnsurePage),
// and reading ahead would evict pages the pass is about to use. It is a
// returned no-op there, not an error. It reads exactly what EnsurePage reads,
// through the same claim, pin and fill bookkeeping, so a page the forward pass
// reaches first is not read twice and the answer cannot depend on which got
// there first.
func (f *Frames) Preload(depth int, stop <-chan struct{}) error {
	if f == nil || depth < 1 || f.CanEvict() {
		return nil
	}
	f.preloadStart()
	defer f.preloadEnd()
	n := f.file.NPages()
	var (
		next  atomic.Int64
		wg    sync.WaitGroup
		errMu sync.Mutex
		first error
	)
	for w := 0; w < min(depth, n); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				i := int(next.Add(1) - 1)
				if i >= n {
					return
				}
				// The budget can narrow after Open (a device on the host's
				// memory comes off it); a preload that keeps going then
				// evicts what it just read.
				if f.CanEvict() {
					return
				}
				f.pageStart()
				read := f.EnsurePage
				if f.preloadRead != nil {
					read = f.preloadRead
				}
				err := read(i)
				f.inPages.Add(-1)
				// The budget may have narrowed while the page was in flight:
				// a pinned frame is no victim, so the shrink could not take
				// it, and it would stay resident beyond the budget. It goes
				// now, and the CanEvict check above ends the preload.
				f.fitBudget()
				if err != nil {
					errMu.Lock()
					if first == nil {
						first = err
					}
					errMu.Unlock()
					return
				}
			}
		}()
	}
	wg.Wait()
	return first
}

// settlePreload is SetBudget's half for a budget that no longer holds the
// model: a running preload stops at its next page (its CanEvict check), and
// the pages it has in flight are pinned, which the shrink could not take. So
// the shrink waits for those reads and then evicts down to the budget, and a
// caller reading residency after SetBudget sees the budget it set. The wait
// is at most the preload's depth of page reads. Called without f.mu.
func (f *Frames) settlePreload() {
	if !f.CanEvict() {
		return
	}
	f.preMu.Lock()
	for f.preN > 0 {
		f.preCond.Wait()
	}
	f.preMu.Unlock()
	f.fitBudget()
}

// preloadStart and preloadEnd count running Preloads for settlePreload.
func (f *Frames) preloadStart() {
	f.preMu.Lock()
	if f.preCond.L == nil {
		f.preCond.L = &f.preMu
	}
	f.preN++
	f.preMu.Unlock()
}

func (f *Frames) preloadEnd() {
	f.preMu.Lock()
	f.preN--
	f.preCond.Broadcast()
	f.preMu.Unlock()
}

// fitBudget evicts until what is resident is inside the budget: what SetBudget
// does, for frames that were pinned when it ran.
func (f *Frames) fitBudget() {
	defer f.flushRecycled()
	f.mu.Lock()
	defer f.mu.Unlock()
	for f.budget > 0 && f.resident > f.budget {
		if !f.evictOne() {
			return
		}
	}
}

// ReadConcurrency is the most read requests that were ever outstanding at
// once on this File, and ReadBytes the bytes every request asked for: the
// queue depth a load reached and what it moved, deterministic beside a
// stopwatch that is not.
func (f *File) ReadConcurrency() int64 { return f.peakIO.Load() }
func (f *File) ReadBytes() int64       { return f.readBytes.Load() }

// ioStart and ioEnd bracket one ReadAt for ReadConcurrency.
func (f *File) ioStart(n int) {
	f.readBytes.Add(int64(n))
	c := f.inIO.Add(1)
	for {
		p := f.peakIO.Load()
		if c <= p || f.peakIO.CompareAndSwap(p, c) {
			return
		}
	}
}

func (f *File) ioEnd() { f.inIO.Add(-1) }

// StartPreload runs Preload behind the caller, depth pages in flight.
// WaitPreload returns its error once it has finished; Close stops it and
// waits for it first, so no read lands in a frame Close has given back.
func (f *Frames) StartPreload(depth int) {
	f.preStop, f.preDone = make(chan struct{}), make(chan struct{})
	go func() {
		defer close(f.preDone)
		f.preErr = f.Preload(depth, f.preStop)
	}()
}

// WaitPreload blocks until StartPreload's read has finished and returns its
// error; nil at once when there was none. A failed preload is not fatal: the
// page it could not read is read again, and its error returned, when a token
// faults it in.
func (f *Frames) WaitPreload() error {
	if f.preDone == nil {
		return nil
	}
	<-f.preDone
	return f.preErr
}

func (f *Frames) stopPreload() {
	if f.preStop == nil {
		return
	}
	select {
	case <-f.preStop:
	default:
		close(f.preStop)
	}
	<-f.preDone
}

// PreloadDepth is the most pages Preload ever had in flight at once: the
// page-level queue depth, where ReadConcurrency counts requests (one page-in
// alone issues up to ReadThreads of those).
func (f *Frames) PreloadDepth() int64 { return f.peakPages.Load() }

func (f *Frames) pageStart() {
	c := f.inPages.Add(1)
	for {
		p := f.peakPages.Load()
		if c <= p || f.peakPages.CompareAndSwap(p, c) {
			return
		}
	}
}
