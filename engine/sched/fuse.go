package sched

import (
	"runtime"
	"sync/atomic"
	"time"
)

// A fused region is one pool region that runs many regions' work: the workers
// join once (Fuse), and every Do, DoRange, DoLabeled or DoNodes the caller
// issues until Unfuse is a PHASE of that region rather than a region of its
// own. A phase is published by one store and ends when its last chunk is done;
// the workers stay joined between phases, spinning on the phase word, so a
// phase costs a claim and a completion count where a region costs a sequence
// bump, a gate to join and close, and a wait for every joined worker to leave.
//
// Phases are complete before the next is published, so the work, its chunks
// and its order are those of the regions it replaces: the kernels compute the
// same sums and a fused block is bit-identical to the unfused one
// (model.TestFusedBlocksMatchPerOp). What a phase removes is the round trip,
// not a barrier the data flow needs.
//
// Completion is counted in CHUNKS, not in workers, so a worker the OS keeps off
// a CPU holds nothing up unless it has claimed a chunk -- the property the
// region gate exists for. A worker that joins late drains whatever phase is
// current; one that never joins is never waited for.

// Layout of Pool.fword: the phase's tag, its chunk count and the next chunk to
// claim. A claim is a CAS on the whole word, so a worker that read a phase's
// tag can never claim a chunk of the next one, and it reads the job only after
// a claim succeeded -- the caller rewrites the job only once every chunk of
// the phase is done.
const (
	fwBits  = 18
	fwMask  = 1<<fwBits - 1
	fwShift = 2 * fwBits
	fwMax   = fwMask // most chunks a phase holds; Do's chunk is scaled up past it
)

func fwTag(w uint64) uint64 { return w >> fwShift }
func fwTotal(w uint64) int  { return int(w >> fwBits & fwMask) }
func fwNext(w uint64) int   { return int(w & fwMask) }

// fused is the per-crew state of an open fused region. Only the caller that
// opened it writes owner, tag and end; workers read end after seeing a new tag.
type fused struct {
	owner atomic.Pointer[Pool] // the pool or view that opened the region; nil when none is open
	tag   uint64
	word  atomic.Uint64
	done  atomic.Int64
	end   atomic.Bool
	n     atomic.Int64 // phases run, for Phases
	// ahead is what a worker does when it finds nothing left to claim in
	// the current phase (FuseWith); nil does nothing.
	ahead Lookahead
}

// Lookahead is the work a fused region's idle workers do between phases: a
// worker that found nothing left to claim calls Ahead once per phase, and Ahead
// returns as soon as Moved reports the region has gone past that phase. It
// must not write anything a phase reads -- it exists to warm the caches with
// the next phase's operands (nn's weight prefetch), so a phase runs the same
// work whether or not it ran.
type Lookahead interface {
	Ahead(p *Pool, worker, workers int, tag uint64)
}

// Moved reports whether phase tag is over: every chunk of it is done, a new
// phase was published or the region ended. A Lookahead polls it between
// pieces, so the idle work fills the phase's tail and never delays the next
// phase -- the caller runs it too, and publishes the next phase only after.
func (p *Pool) Moved(tag uint64) bool {
	c := p.w()
	w := c.fz.word.Load()
	return fwTag(w) != tag || c.fz.end.Load() || c.fz.done.Load() >= int64(fwTotal(w))
}

// Fuse opens a fused region on p and reports whether it did. It does not when
// p is nil, stopped, NUMA-aware (a phase drains flat, and DoNodes' placement
// is what such a pool is for), already fused, or a view whose cores another
// user may want -- another view on its crew, or another caller's view on a
// crew it takes turns with; the caller's own view beside it (SharedBeside)
// is not one: a fused region holds the crew for its whole length, and a
// caller holding it while waiting on a lock another session's region holds
// would never get it back. Nor with region labels on: a profile attributes
// generated code by the region's pprof label, which a phase does not carry, so
// a profiled run is the per-op one. Every Fuse that returns true is paired
// with Unfuse.
func (p *Pool) Fuse() bool { return p.FuseWith(nil) }

// FuseWith is Fuse with l as the region's idle work (Lookahead): set once the
// region is this caller's, cleared by Unfuse.
func (p *Pool) FuseWith(l Lookahead) bool {
	if p == nil || p.stopped.Load() || regionLabels.Load() {
		return false
	}
	c := p.w()
	if c.cpuNode != nil || c.fz.owner.Load() != nil || len(c.cpus) < 2 {
		return false
	}
	if p.crew != nil {
		// The crew's only user, under the registry's lock: a view opened
		// after this point waits for the crew's turn like any region does.
		crews.mu.Lock()
		alone := c.views-c.idle == 1 && c.mu.users == 1
		crews.mu.Unlock()
		if !alone {
			return false
		}
		c.lockAll(p.wake)
		if c.stopped.Load() {
			c.unlockAll()
			return false
		}
	}
	c.fz.owner.Store(p)
	c.fz.ahead = l
	c.fz.end.Store(false)
	part := int(p.part.Load())
	if part < 2 {
		part = 2
	}
	// The region is published like any other, with a body that never claims a
	// chunk from it: the joined workers go to the phase loop instead.
	p.nParallel.Add(1)
	c.j = job{total: part, chunk: 1, limit: part}
	c.limit.Store(int64(part))
	c.next.Store(int64(part)) // nothing to claim: the caller does not run it
	c.left.Store(0)
	c.gate.Store((c.seq.Load() + 1) << gateEpochShift)
	c.seq.Add(1)
	for i := range min(len(c.parked), part-1) {
		if c.parked[i].Load() {
			select {
			case c.start[i] <- struct{}{}:
			default:
			}
		}
	}
	return true
}

// Unfuse closes the region Fuse opened: the joined workers leave the phase
// loop and the caller waits for them, as the end of any region does.
func (p *Pool) Unfuse() {
	c := p.w()
	if c.fz.owner.Load() != p {
		panic("sched: Unfuse without a matching Fuse")
	}
	c.fz.end.Store(true)
	c.fz.tag++
	c.fz.word.Store(c.fz.tag << fwShift)
	c.wakeParked(len(c.parked))
	joined := int64(c.gate.Or(gateClosed) & gateCount)
	for c.left.Load() != joined {
		runtime.Gosched()
	}
	c.fz.ahead = nil
	c.fz.owner.Store(nil)
	if p.crew != nil {
		c.unlockAll()
	}
}

// Fused reports whether p is inside a fused region it opened.
func (p *Pool) Fused() bool { return p != nil && p.w().fz.owner.Load() == p }

// Phases is how many phases fused regions ran on p's workers since the pool
// was built: the regions fusion replaced, which Regions no longer counts.
func (p *Pool) Phases() int64 {
	if p == nil {
		return 0
	}
	return p.w().fz.n.Load()
}

// wakeParked wakes the first n workers that parked.
func (c *Pool) wakeParked(n int) {
	for i := range min(len(c.parked), n) {
		if c.parked[i].Load() {
			select {
			case c.start[i] <- struct{}{}:
			default:
			}
		}
	}
}

// phase runs j as one phase of the open fused region and returns when it is
// done. The chunking is j's, scaled up only when its chunk count would not fit
// the phase word.
func (c *Pool) phase(j job, part int) {
	phaseHook(j.total, j.chunk)
	total, chunk := j.total, j.chunk
	n := (total + chunk - 1) / chunk
	if n > fwMax {
		chunk *= (n + fwMax - 1) / fwMax
		j.chunk = chunk
		n = (total + chunk - 1) / chunk
	}
	limit := min(part, n)
	j.limit = limit
	j.block = 0 // a phase drains flat; Fuse refuses a NUMA pool
	c.j = j
	c.limit.Store(int64(limit))
	c.fz.done.Store(0)
	c.fz.tag++
	tag := c.fz.tag
	c.fz.word.Store(tag<<fwShift | uint64(n)<<fwBits)
	c.fz.n.Add(1)
	c.wakeParked(limit - 1)
	c.fdrain(0, tag)
	if a := c.fz.ahead; a != nil {
		a.Ahead(c.fz.owner.Load(), 0, limit, tag)
	}
	for spins := 0; c.fz.done.Load() != int64(n); spins++ {
		if spins&63 == 63 {
			runtime.Gosched()
		}
	}
}

// fdrain claims chunks of phase tag until it has none left or has moved on.
func (c *Pool) fdrain(id int, tag uint64) {
	for {
		w := c.fz.word.Load()
		if fwTag(w) != tag {
			return
		}
		k, n := fwNext(w), fwTotal(w)
		if k >= n {
			return
		}
		if !c.fz.word.CompareAndSwap(w, w+1) {
			continue
		}
		// The claim holds the phase open, so the job is this phase's.
		lo := k * c.j.chunk
		c.j.call(id, lo, min(lo+c.j.chunk, c.j.total))
		c.fz.done.Add(1)
	}
}

// phaseLoop is a joined worker's life inside a fused region: wait for a phase,
// drain it if it is under the phase's participant limit, until Unfuse.
func (c *Pool) phaseLoop(id int) {
	var seen uint64
	for {
		w := c.awaitPhase(id, seen)
		if c.fz.end.Load() || !c.alive() {
			return
		}
		seen = fwTag(w)
		if lim := int(c.limit.Load()); id < lim {
			c.fdrain(id, seen)
			if a := c.fz.ahead; a != nil {
				a.Ahead(c.fz.owner.Load(), id, lim, seen)
			}
		}
	}
}

// awaitPhase spins until the phase word carries a tag other than seen, then
// parks as await does: the caller's Go between phases (a page-in, a device
// call) may be long, and a spinning worker would burn a core through it.
func (c *Pool) awaitPhase(id int, seen uint64) uint64 {
	pid := id - 1 // the parking slot of worker id (participant 0 is the caller)
	for {
		var deadline int64
		for n := 0; ; n++ {
			if w := c.fz.word.Load(); fwTag(w) != seen {
				return w
			}
			if n&1023 == 1023 {
				now := time.Now().UnixNano()
				if deadline == 0 {
					deadline = now + c.spin.Load()
				} else if now > deadline {
					break
				}
			}
		}
		c.parked[pid].Store(true)
		if w := c.fz.word.Load(); fwTag(w) != seen {
			c.parked[pid].Store(false)
			select {
			case <-c.start[pid]:
			default:
			}
			return w
		}
		select {
		case <-c.quit:
			c.parked[pid].Store(false)
			return c.fz.word.Load()
		case <-c.start[pid]:
		}
		c.parked[pid].Store(false)
	}
}
