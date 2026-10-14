package sched

import (
	"fmt"
	"sync"
)

// Shared returns a view of the process's pool for cpus: one set of workers per
// core set, however many users run on it. Every model.State builds a JIT with
// its own pool, and two pools on the same cores are two sets of spinning
// workers behind two barriers -- the slowest worker of either is both regions'
// end. Views take turns a region at a time instead, in arrival order (turn),
// so several sessions run on the cores without a lock held across a whole
// generate and without one session's regions starving another's.
//
// A view has its own participant count (SetParticipants) and region counters;
// SetSpinning is the crew's, so the last caller's choice holds for every view.
// The crew stops when its last view is closed.
func Shared(cpus []int, opts ...Option) *Pool { return SharedBeside(cpus, nil, opts...) }

// SharedBeside is Shared for a caller that already holds view of: the view it
// returns is that caller's too, so it is not another user of the cores to a
// fused region on of (Pool.Fuse). A JIT's prefill pool on the P and E cores
// is that beside its decode pool on the P cores: the two overlap and take
// turns on one lock, and one caller drives one of them at a time -- the
// prefill pool is swapped in for a prompt and out before the next decode
// token -- so a fused decode run never waits on its own prefill view. Of nil,
// or a view on cores that do not take turns with the new one's, is Shared.
func SharedBeside(cpus []int, of *Pool, opts ...Option) *Pool {
	cfg := newConfig(opts)
	if len(cpus) == 0 {
		cpus = DecodeCores(opts...)
	}
	key := fmt.Sprint(cpus, cfg.spin, cfg.spinSet, cfg.numa)
	crews.mu.Lock()
	defer crews.mu.Unlock()
	c := crews.m[key]
	if c == nil {
		c = New(cpus, opts...)
		c.key = key
		// Crews over overlapping cores take turns with each other too: a
		// prefill on the wide set and a decode on the P-cores would otherwise
		// both drive the same cores at once.
		c.mu = &turn{}
		// A set that overlaps two crews which do not overlap each other takes
		// the first one's lock; the machine's core sets are nested (P inside
		// P+E), so that case is not met, and a lock per core is the upgrade if
		// it ever is.
		for _, o := range crews.m {
			if overlaps(o.cpus, cpus) {
				c.mu = o.mu
				break
			}
		}
		crews.m[key] = c
	}
	c.views++
	v := &Pool{crew: c, wake: make(chan struct{}, 1)}
	v.beside = of != nil && of.crew != nil && of.crew.mu == c.mu
	if !v.beside {
		c.mu.users++
	}
	v.part.Store(int64(len(c.cpus)))
	return v
}

var crews = struct {
	mu sync.Mutex
	m  map[string]*Pool
}{m: map[string]*Pool{}}

// release closes a view; the last one stops the crew.
func (c *Pool) release(v *Pool) {
	crews.mu.Lock()
	defer crews.mu.Unlock()
	if v.stopped.Swap(true) {
		return // this view was already closed
	}
	c.views--
	if !v.beside {
		c.mu.users--
	}
	if c.views > 0 {
		return
	}
	delete(crews.m, c.key)
	c.Close()
}

func overlaps(a, b []int) bool {
	for _, x := range a {
		for _, y := range b {
			if x == y {
				return true
			}
		}
	}
	return false
}
