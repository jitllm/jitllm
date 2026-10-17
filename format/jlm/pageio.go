package jlm

import (
	"os"
	"runtime"
	"sync"
)

// pageIO is a File's page reader: a fixed set of goroutines that serve read
// pieces from a queue. A page-in used to start a goroutine per run and readAt
// another ReadThreads per large run, and every go statement with arguments is
// a heap allocation: a paged decode, which faults a block per block it does
// not hold every token, made hundreds of them a token. The queue depth is the
// same -- every piece of every run is in flight at once -- and the workers
// block in pread, so their number bounds the depth and nothing else.
type pageIO struct {
	once sync.Once
	q    chan *ioPiece
	done sync.WaitGroup
	// plans are the idle ioPlans, one per page-in that was ever in flight at
	// once. Guarded by mu, its own: a page-in takes one before f.mu.
	mu    sync.Mutex
	plans []*ioPlan
}

// ioWorkers is the queue depth the pool sustains: two page-ins' worth of
// ReadThreads, so two sessions faulting at once each keep their full depth.
const ioWorkers = 2 * ReadThreads

// ioPiece is one pread: a slice of a run.
type ioPiece struct {
	f   *File
	h   *os.File
	dst []byte
	off int64
	err error
	wg  *sync.WaitGroup
}

// ioPlan is one EnsureRanges' scratch, kept on the File so a warm page-in
// allocates nothing: the runs planRuns appends, a piece per pread, and the
// wait. Not a sync.Pool, which a collection empties: the plans would be
// allocated again after every cycle.
type ioPlan struct {
	runs   []run
	pieces []ioPiece
	errs   []error
	runLen []int // bytes per run, in order: the pieces are attributed back by it
	wg     sync.WaitGroup
	// HoldPages' scratch: one entry per page with reads, and the one range
	// planRuns is asked for.
	held []heldRuns
	rng  [1]Range
}

// heldRuns is one page of a HoldPages plan: its runs are runs[r0:r1] of the
// plan, its errors errs[r0:r1].
type heldRuns struct{ i, b, r0, r1 int }

func (f *File) getPlan() *ioPlan {
	f.io.mu.Lock()
	defer f.io.mu.Unlock()
	if n := len(f.io.plans); n > 0 {
		p := f.io.plans[n-1]
		f.io.plans[n-1] = nil
		f.io.plans = f.io.plans[:n-1]
		return p
	}
	return new(ioPlan)
}

func (f *File) putPlan(p *ioPlan) {
	clear(p.pieces[:cap(p.pieces)])
	clear(p.errs[:cap(p.errs)])
	p.runs, p.pieces, p.errs, p.runLen, p.held = p.runs[:0], p.pieces[:0], p.errs[:0], p.runLen[:0], p.held[:0]
	f.io.mu.Lock()
	defer f.io.mu.Unlock()
	f.io.plans = append(f.io.plans, p)
}

func (f *File) startIO() {
	f.io.q = make(chan *ioPiece, ioWorkers)
	f.io.done.Add(ioWorkers)
	for range ioWorkers {
		go func() {
			defer f.io.done.Done()
			for p := range f.io.q {
				p.f.ioStart(len(p.dst))
				_, p.err = p.h.ReadAt(p.dst, p.off)
				p.f.ioEnd()
				p.wg.Done()
			}
		}()
	}
}

// stopIO ends the workers. Callers hold no page-in in flight (Close).
func (f *File) stopIO() {
	if f.io.q == nil {
		return
	}
	close(f.io.q)
	f.io.done.Wait()
	f.io.q = nil
}

// readRuns reads every run of a plan into page, each split as readAt splits
// one, and returns an error per run.
func (f *File) readRuns(pl *ioPlan, page []byte, base, psize uint64, runs []run) []error {
	f.io.once.Do(f.startIO)
	f.cutRuns(pl, page, base, psize, runs)
	return f.sendRuns(pl)
}

// cutRuns appends one page's runs to the plan as pieces, each run split as
// readAt splits one, with an error slot and its length per run.
func (f *File) cutRuns(pl *ioPlan, page []byte, base, psize uint64, runs []run) {
	cs := f.chunkSize()
	for _, r := range runs {
		cl, ch := uint64(r.lo)*cs, min(uint64(r.hi)*cs, psize)
		dst, off := page[cl:ch], int64(base+cl)
		// A pread is priced per call, so the count is per run, as readAt's.
		f.countRead()
		h := f.f
		if f.direct != nil && directOK(off, len(dst)) && aligned(dst) {
			h = f.direct
		}
		g := 1
		// Below a few MiB a split costs more than the queue depth buys.
		if len(dst) >= 4<<20 && ReadThreads >= 2 {
			g = min(ReadThreads, runtime.GOMAXPROCS(0)*4)
		}
		chunk := (len(dst) + g - 1) / g
		if h == f.direct {
			chunk = (chunk + Align - 1) &^ (Align - 1)
		}
		for lo := 0; lo < len(dst); lo += chunk {
			hi := min(lo+chunk, len(dst))
			pl.pieces = append(pl.pieces, ioPiece{f: f, h: h, dst: dst[lo:hi], off: off + int64(lo), wg: &pl.wg})
		}
		pl.errs = append(pl.errs, nil)
		pl.runLen = append(pl.runLen, len(dst))
	}
}

// sendRuns puts every piece of the plan in flight at once, waits for them,
// and returns an error per run, in the order cutRuns appended them.
func (f *File) sendRuns(pl *ioPlan) []error {
	pl.wg.Add(len(pl.pieces))
	for i := range pl.pieces {
		f.io.q <- &pl.pieces[i]
	}
	pl.wg.Wait()
	// Each piece is attributed back to its run in the order they were cut.
	k := 0
	for ri, rl := range pl.runLen {
		for n := 0; n < rl; k++ {
			if pl.errs[ri] == nil {
				pl.errs[ri] = pl.pieces[k].err
			}
			n += len(pl.pieces[k].dst)
		}
	}
	return pl.errs
}
