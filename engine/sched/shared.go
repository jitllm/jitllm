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
		//
		// A crew wholly on the E-cores is the exception, so a session
		// prefilling there does not wait on one decoding on the P-cores (nor
		// is a user of its cores): it takes turns with other E-core crews, and
		// a crew whose cores include the E-cores (the P and E prefill set)
		// takes its lock too, after its own (lockAll), so the two still never
		// drive a core at once.
		e := allE(cpus)
		adopted := false
		for _, o := range crews.m {
			if !overlaps(o.cpus, cpus) {
				continue
			}
			oe := allE(o.cpus)
			switch {
			case e && oe, !e && !oe:
				if !adopted {
					c.mu, adopted = o.mu, true
				}
			case e:
				o.ex.Store(addLock(o.ex.Load(), c.mu))
			default:
				c.ex.Store(addLock(c.ex.Load(), o.mu))
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
	switch {
	case v.away:
		c.idle--
	case !v.beside:
		c.mu.users--
	}
	if c.views > 0 {
		return
	}
	delete(crews.m, c.key)
	for _, o := range crews.m {
		o.ex.Store(dropLock(o.ex.Load(), c.mu, crews.m))
	}
	c.Close()
}

// lockAll takes the crew's turn and then every E-core crew's it overlaps; in
// that order always, so two crews cannot wait on each other.
func (c *Pool) lockAll(wake chan struct{}) {
	c.mu.Lock(wake)
	var held []*turn
	if l := c.ex.Load(); l != nil {
		held = *l
	}
	for _, t := range held {
		t.Lock(wake)
	}
	c.held = held
}

func (c *Pool) unlockAll() {
	held := c.held
	c.held = nil
	for i := len(held) - 1; i >= 0; i-- {
		held[i].Unlock()
	}
	c.mu.Unlock()
}

func addLock(l *[]*turn, t *turn) *[]*turn {
	var cur []*turn
	if l != nil {
		cur = *l
	}
	for _, x := range cur {
		if x == t {
			return l
		}
	}
	n := append(cur[:len(cur):len(cur)], t)
	return &n
}

// dropLock removes t from l unless a crew still open holds it as its own.
func dropLock(l *[]*turn, t *turn, open map[string]*Pool) *[]*turn {
	if l == nil {
		return nil
	}
	for _, o := range open {
		if o.mu == t {
			return l
		}
	}
	var n []*turn
	for _, x := range *l {
		if x != t {
			n = append(n, x)
		}
	}
	return &n
}

// Park says the caller of view v is not using its cores for now, so a fused
// region (Fuse) on another view of the crew, or of one that takes turns with
// it, is its cores' only user: the view stays open and is still there to run
// a region, and Unpark takes it back. A session that sends its prefill to the
// E-cores parks its decode view on the P-cores for the length of the prefill,
// so a session decoding beside it can fuse. Park and Unpark pair, on one
// caller's view, and are not for a view of a pool that owns its workers.
func (p *Pool) Park() {
	if p == nil || p.crew == nil {
		return
	}
	crews.mu.Lock()
	defer crews.mu.Unlock()
	if p.away || p.stopped.Load() {
		return
	}
	p.away = true
	p.crew.idle++
	if !p.beside {
		p.crew.mu.users--
	}
}

// Unpark is Park's pair: v is a user of its crew's cores again.
func (p *Pool) Unpark() {
	if p == nil || p.crew == nil {
		return
	}
	crews.mu.Lock()
	defer crews.mu.Unlock()
	if !p.away {
		return
	}
	p.away = false
	p.crew.idle--
	if !p.beside {
		p.crew.mu.users++
	}
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
