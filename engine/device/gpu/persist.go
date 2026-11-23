package gpu

import (
	"fmt"
	"slices"
	"sync"
	"unsafe"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer/ir"
)

// The persistent decode block (Config.Persist): a decode block's launches,
// collected as they are issued and composed into persistent kernels
// (ir.Persist) -- one grid that runs the launches' bodies in order, phases
// separated by a grid barrier -- instead of one launch each.
//
// It sits in the launcher, below every choice the tier makes about which
// kernels a block runs, so the per-op path is the reference by construction:
// the same kernels, the same groups, the same buffers, composed. The plans
// are composed when a block is placed (persistPlace), from the launches its
// decode step issues; a step only finds its block's plan, and one it does not
// find runs per-op (Stats.PersistUnplanned). A launch the
// composer cannot take (no IR, a width that does not divide the run's, a
// buffer bound both as an input and an output) cuts the run and goes alone; a
// run of one is a plain launch. Relocation, paging and seam moves change the
// buffers a block's launches bind, never the composition: the plan is keyed by
// the kernels, the grids and the aliasing pattern, and the buffers are bound
// per launch.
//
// It runs only where the driver guarantees the co-residency a grid barrier
// needs, a CUDA cooperative launch (cuda/coop.go). Vulkan and Metal state a
// compute-unit count and no guarantee, and two sessions' persistent kernels
// on two queues could each hold units the other waits for.

const (
	persistMaxPend = 128 // launches one block may issue
	persistMaxArgs = 24  // buffers one launch binds
	persistMaxBufs = 63  // distinct buffers one block binds (one more is the sync buffer)
)

// PersistLaunched is what the tier's devices issued down the persistent path:
// the collected runs of a block's launches (Stats.PersistRuns), the
// persistent kernels launched, those ordered by dependency counters, and the
// runs a step found no plan for (Stats.PersistUnplanned). It reads only the
// counts, so it answers after Close.
func (g *GPU) PersistLaunched() (runs, launched, deps, unplanned int) {
	for _, d := range g.devs {
		d.mu.Lock()
		runs += d.tot.PersistRuns
		launched += d.tot.PersistLaunches
		deps += d.tot.PersistDepLaunches
		unplanned += d.tot.PersistUnplanned
		d.mu.Unlock()
	}
	return runs, launched, deps, unplanned
}

// depWords is a session's dep buffer, in u32 words: every composed kernel's
// counter region is carved from it at plan time (persistCache.depNext), at
// the same offset in every session's buffer. A plan that would pass the end
// is composed with grid barriers instead.
const depWords = 1 << 16

// persistPend is one collected launch.
type persistPend struct {
	k             backend.Kernel
	src           *ir.Kernel
	groups, width int
	nb            int
	bufs          [persistMaxArgs]backend.Buf
}

// persister is a session's collection state, in its launchTo.
type persister struct {
	on   bool
	n    int
	pend [persistMaxPend]persistPend
	// dist is the block's distinct buffers in first-use order and bind each
	// launch's indices into it, rebuilt at every flush.
	dist [persistMaxBufs]backend.Buf
	nd   int
	bind [persistMaxPend][persistMaxArgs]int
	sig  []uint64
	args []backend.Buf
	// plan is a placement's collection, which issues nothing: each run it
	// cuts, where a step would issue it, is planned instead.
	plan bool
	// sync is the session's grid barrier buffer (ir.GridSyncWords words),
	// dep its dependency counters (depWords words, Config.PersistDeps).
	sync, dep backend.Buf
}

// persistSeg is one piece of a plan: launches [lo, hi), composed into k, or
// issued one by one when k is nil.
type persistSeg struct {
	lo, hi int
	k      backend.Kernel
	bufs   []int
	grid   int
	width  int
	phases int
	// dep is a kernel ordered by dependency counters, launched on the
	// session's dep buffer; tiles is its tile-wise edges.
	dep   bool
	tiles int
}

// persistPlan is a block's composition, found again by its signature.
type persistPlan struct {
	sig  []uint64
	segs []persistSeg
}

// persistCache is the device's plans: kernels are stateless (the sync buffer
// is a launch argument), so every session shares them.
type persistCache struct {
	mu    sync.Mutex
	plans map[uint64][]*persistPlan
	why   string // the last composition that fell back, and why
	// depNext is the first dep-buffer word no plan's region holds.
	depNext int
}

// persistOK reports whether this device runs persistent kernels, and why not.
func (g *devTier) persistOK() (bool, string) {
	if g.dev.API() != "ptx" {
		return false, fmt.Sprintf("%s states no co-residency guarantee (only a CUDA cooperative launch does)", g.dev.API())
	}
	return true, ""
}

// ensurePersist readies the session's collection state and sync buffer. It
// runs outside any Session, before the submission: the buffer must be zero
// before a launch can read it, and a write inside a capture would be baked in.
// Callers hold g.mu.
func (g *devTier) ensurePersist() bool {
	if !g.Persist {
		return false
	}
	if g.dev.API() != "ptx" {
		// Said once: a message built every token would be a heap object a
		// token on every device that declines.
		if g.PersistWhy == "" {
			_, g.PersistWhy = g.persistOK()
		}
		return false
	}
	if g.pcache == nil {
		g.pcache = &persistCache{}
	}
	t := &g.launchTo
	if t.ps == nil {
		t.ps = &persister{sig: make([]uint64, 0, 1024), args: make([]backend.Buf, 0, persistMaxBufs+1)}
	}
	if t.ps.sync == nil {
		b, err := g.alloc(4 * ir.GridSyncWords)
		if err != nil {
			g.PersistWhy = "sync buffer: " + err.Error()
			return false
		}
		if err := b.Write(make([]byte, 4*ir.GridSyncWords)); err != nil {
			b.Free()
			g.PersistWhy = "sync buffer: " + err.Error()
			return false
		}
		t.ps.sync = b
	}
	if g.PersistDeps && t.ps.dep == nil {
		b, err := g.alloc(4 * depWords)
		if err != nil {
			g.PersistWhy = "dep buffer: " + err.Error()
			return false
		}
		if err := b.Write(make([]byte, 4*depWords)); err != nil {
			b.Free()
			g.PersistWhy = "dep buffer: " + err.Error()
			return false
		}
		t.ps.dep = b
	}
	return true
}

// closePersist frees the session's sync buffer. Its recordings, which name
// it, are gone first (forgetSess, Close).
func (t *launchTo) closePersist() {
	if t.ps != nil && t.ps.sync != nil {
		t.ps.sync.Free()
		t.ps.sync = nil
	}
	if t.ps != nil && t.ps.dep != nil {
		t.ps.dep.Free()
		t.ps.dep = nil
	}
}

// closePersistCache closes every composed kernel. Callers hold g.mu.
func (g *devTier) closePersistCache() {
	c := g.pcache
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, ps := range c.plans {
		for _, p := range ps {
			for _, s := range p.segs {
				if s.k != nil {
					s.k.Close()
				}
			}
		}
	}
	c.plans = nil
}

// persistBlock starts collecting a block's launches, issuing whatever the
// previous block collected; on false it stops collecting.
func (l *launcher) persistBlock(on bool) {
	ps := l.to.ps
	if ps == nil {
		return
	}
	l.persistFlush()
	ps.on = on
}

// persistFlush issues what has been collected and stops collecting.
func (l *launcher) persistFlush() {
	ps := l.to.ps
	if ps == nil {
		return
	}
	ps.on = false
	if ps.n == 0 {
		return
	}
	n := ps.n
	ps.n = 0
	if *l.err != nil {
		return
	}
	*l.err = l.g.persistIssue(l.to, ps, n)
}

// persistAdd collects one launch, false when it cannot be held (the caller
// then flushes and launches it itself).
func (ps *persister) add(k backend.Kernel, groups, width int, bufs []backend.Buf) bool {
	if ps.n == persistMaxPend || len(bufs) > persistMaxArgs {
		return false
	}
	p := &ps.pend[ps.n]
	p.k, p.src, p.groups, p.width, p.nb = k, backend.IROf(k), groups, width, len(bufs)
	copy(p.bufs[:], bufs)
	ps.n++
	return true
}

// persistSign binds n collected launches to the block's distinct buffers and
// hashes the signature a plan is found by: the mode, then each launch's
// kernel, grid, width and binding pattern. It is false past persistMaxBufs.
func (g *devTier) persistSign(ps *persister, n int) (uint64, bool) {
	ps.nd = 0
	ps.sig = ps.sig[:0]
	// The mode first: a plan composed with grid barriers is not one composed
	// with dependency counters.
	if g.PersistDeps {
		ps.sig = append(ps.sig, 1)
	} else {
		ps.sig = append(ps.sig, 0)
	}
	for i := 0; i < n; i++ {
		p := &ps.pend[i]
		ps.sig = append(ps.sig, uint64(uintptr(unsafe.Pointer(p.src))), uint64(p.groups)<<32|uint64(p.width), uint64(p.nb))
		for j := 0; j < p.nb; j++ {
			b := p.bufs[j]
			x := -1
			for d := 0; d < ps.nd; d++ {
				if ps.dist[d] == b {
					x = d
					break
				}
			}
			if x < 0 {
				if ps.nd == persistMaxBufs {
					return 0, false
				}
				x = ps.nd
				ps.dist[x] = b
				ps.nd++
			}
			ps.bind[i][j] = x
			ps.sig = append(ps.sig, uint64(x))
		}
	}
	h := uint64(14695981039346656037)
	for _, w := range ps.sig {
		h = (h ^ w) * 1099511628211
	}
	return h, true
}

// persistIssue issues n collected launches by the plan their block's
// placement composed, or per-op where it composed none for them. A
// placement's collection (persister.plan) plans them instead.
func (g *devTier) persistIssue(t *launchTo, ps *persister, n int) error {
	if ps.plan {
		if h, ok := g.persistSign(ps, n); ok {
			g.persistPlan(h, ps, n)
		}
		for i := range n {
			clear(ps.pend[i].bufs[:ps.pend[i].nb])
		}
		return nil
	}
	g.PersistRuns++
	h, ok := g.persistSign(ps, n)
	if !ok {
		return g.persistPlain(t, ps, 0, n)
	}
	plan := g.persistFind(h, ps)
	if plan == nil {
		g.PersistUnplanned++
		if g.PersistUnplannedWhy == "" {
			// Once: a message built every token would be a heap object a token.
			name := "?"
			if src := ps.pend[0].src; src != nil {
				name = src.Name
			}
			g.PersistUnplannedWhy = fmt.Sprintf("%d launches from %s", n, name)
		}
		return g.persistPlain(t, ps, 0, n)
	}
	for _, s := range plan.segs {
		if s.k == nil {
			if err := g.persistPlain(t, ps, s.lo, s.hi); err != nil {
				return err
			}
			continue
		}
		ps.args = ps.args[:0]
		for _, x := range s.bufs {
			ps.args = append(ps.args, ps.dist[x])
		}
		if s.dep {
			ps.args = append(ps.args, ps.dep)
		} else {
			ps.args = append(ps.args, ps.sync)
		}
		err := t.s.Launch(s.k, s.grid, s.width, ps.args...)
		for i := s.lo; i < s.hi; i++ {
			clear(ps.pend[i].bufs[:ps.pend[i].nb])
		}
		clear(ps.args)
		if err != nil {
			return err
		}
		g.PersistLaunches++
		g.PersistParts += s.hi - s.lo
		g.PersistPhases += s.phases
		if s.dep {
			g.PersistDepLaunches++
			g.PersistDepTiles += s.tiles
		}
	}
	return nil
}

// persistPlain issues launches [lo, hi) as they were collected.
func (g *devTier) persistPlain(t *launchTo, ps *persister, lo, hi int) error {
	for i := lo; i < hi; i++ {
		p := &ps.pend[i]
		err := t.s.Launch(p.k, p.groups, p.width, p.bufs[:p.nb]...)
		clear(p.bufs[:p.nb])
		if err != nil {
			return err
		}
	}
	return nil
}

// persistFind is the plan composed for the signature persistSign left in
// ps, or nil: a step finds plans and never composes one.
func (g *devTier) persistFind(h uint64, ps *persister) *persistPlan {
	c := g.pcache
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, p := range c.plans[h] {
		if slices.Equal(p.sig, ps.sig) {
			return p
		}
	}
	return nil
}

// persistPlace composes block li's persistent plan at its placement: the
// launches its decode step issues at one row -- the same part, instance and
// buffers (servedStep, blockIO), and the copy after a part that writes apart
// -- collected without a session, and a plan built for each run the step
// will cut them into (a launch the collection cannot hold goes alone). A
// block whose step makes calls of its own between launches (rowsRun,
// pagedRun, streamRun) gets none and runs per-op. Callers hold g.mu and
// prewarming.
func (g *devTier) persistPlace(bs *blockScratch, l *layer, li int) {
	if !g.Persist || !persistable(l, bs) {
		return
	}
	if ok, why := g.persistOK(); !ok {
		g.PersistWhy = why
		return
	}
	sv, inst, cp, err := g.servedStep(bs, l, li, &bs.p, 1, true)
	if err != nil || inst == nil || sv.f.rowSplit > 0 || sv.f.split > 0 || len(sv.f.sweeps) > 0 {
		return
	}
	var inA [2]backend.Buf
	var outA [1]backend.Buf
	in, outs, err := g.blockIO(bs, l, cp != nil, 1, &inA, &outA)
	if err != nil {
		return
	}
	if g.pcache == nil {
		g.pcache = &persistCache{}
	}
	ps := &persister{on: true, plan: true, sig: make([]uint64, 0, 1024)}
	var lerr error
	lc := &launcher{to: &launchTo{ps: ps}, err: &lerr, g: g}
	if _, err := sv.issue(lc, inst, in, outs); err != nil {
		return
	}
	if cp != nil {
		if _, err := cp.issue(lc, cp.inst, []backend.Buf{bs.h}, []backend.Buf{bs.x}); err != nil {
			return
		}
	}
	lc.persistFlush()
}

// persistPlan builds the plan for the launches persistSign bound, where no
// plan of their signature exists. Only placement calls it (persistPlace).
func (g *devTier) persistPlan(h uint64, ps *persister, n int) {
	c := g.pcache
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, p := range c.plans[h] {
		if slices.Equal(p.sig, ps.sig) {
			return
		}
	}
	p := &persistPlan{sig: append([]uint64(nil), ps.sig...)}
	p.segs = g.persistCompose(ps, n)
	if c.plans == nil {
		c.plans = map[uint64][]*persistPlan{}
	}
	c.plans[h] = append(c.plans[h], p)
}

// persistCompose asks the composer for the launches' runs and composes each
// run of two or more. Called once per signature at placement, under the
// cache's lock.
func (g *devTier) persistCompose(ps *persister, n int) []persistSeg {
	parts := make([]ir.Part, n)
	for i := range n {
		p := &ps.pend[i]
		parts[i] = ir.Part{K: p.src, Groups: p.groups, Width: p.width, Bind: slices.Clone(ps.bind[i][:p.nb])}
	}
	var segs []persistSeg
	for _, r := range ir.PersistRuns(parts) {
		if !r.Composed() {
			segs = append(segs, persistSeg{lo: r.Lo, hi: r.Hi})
			if r.Why != nil {
				name := "?"
				if src := parts[r.Lo].K; src != nil {
					name = src.Name
				}
				g.pcache.why = fmt.Sprintf("%s alone: %v", name, r.Why)
			}
			continue
		}
		s, err := g.persistBuild(parts[r.Lo:r.Hi], r)
		if err != nil {
			g.pcache.why = err.Error()
			g.PersistFallbacks++
			segs = append(segs, persistSeg{lo: r.Lo, hi: r.Hi})
		} else {
			segs = append(segs, s)
		}
	}
	return segs
}

// persistBuild composes run r and compiles the result on a grid the device
// holds resident: composed once to learn the limit (the occupancy calculator
// counts the kernel's registers and shared memory, which do not depend on the
// grid), and again at the limit when the composer's grid is past it.
func (g *devTier) persistBuild(parts []ir.Part, r ir.PersistRun) (persistSeg, error) {
	// With dependency counters the kernel's region starts at the first word
	// no plan holds; a plan that does not fit composes with grid barriers.
	var dep ir.DepRegion
	if g.PersistDeps {
		dep = ir.DepRegion{Base: g.pcache.depNext, End: depWords}
	}
	compose := func(grid int) (*ir.Persisted, backend.Kernel, error) {
		pk, err := ir.PersistAt(parts, r.W, grid, dep)
		if err != nil {
			return nil, nil, err
		}
		if !pk.Dep {
			dep = ir.DepRegion{}
		}
		k, err := g.dev.Compile(pk.K)
		if err != nil {
			return nil, nil, err
		}
		return pk, k, nil
	}
	grid := ir.PersistGrid(parts, g.PersistGrid)
	pk, k, err := compose(grid)
	if err != nil {
		return persistSeg{}, err
	}
	limit, err := backend.GridLimit(g.dev, k)
	if err != nil {
		k.Close()
		return persistSeg{}, err
	}
	if grid > limit {
		k.Close()
		if pk, k, err = compose(limit); err != nil {
			return persistSeg{}, err
		}
	}
	g.PersistCompiles++
	if pk.Dep {
		g.pcache.depNext = dep.Base + pk.Words
	}
	return persistSeg{lo: r.Lo, hi: r.Hi, k: k, bufs: pk.Bufs, grid: pk.Grid, width: r.W, phases: pk.Phases, dep: pk.Dep, tiles: pk.Tiles}, nil
}

// persistable reports whether block l's launches are held: a block whose
// emission makes calls of its own on the session between launches (a
// streamed mixture's selection read back, a linear block's own
// submissions) runs per-op.
func persistable(l *layer, bs *blockScratch) bool {
	return l != nil && l.stream == nil && !l.linear && !l.wide
}

// persistDrain issues what is held and keeps collecting.
func (l *launcher) persistDrain() {
	ps := l.to.ps
	if ps == nil {
		return
	}
	on := ps.on
	l.persistFlush()
	ps.on = on
}

// persistSession is the session a collecting submission makes its own calls
// through: each one issues what is held first, so a write, a read, a launch
// that did not come through the launcher or the end of a recording never
// overtakes a held launch.
type persistSession struct {
	backend.Session
	t *launchTo
	g *devTier
}

// drain issues what is held and keeps collecting. It holds no launcher: a
// launcher stored here would move it, and every local it points at, to the
// heap on every submission.
func (p *persistSession) drain() error {
	ps := p.t.ps
	if ps == nil || ps.n == 0 {
		return nil
	}
	n := ps.n
	ps.n = 0
	return p.g.persistIssue(p.t, ps, n)
}

func (p *persistSession) Write(b backend.Buf, d []byte) error {
	if err := p.drain(); err != nil {
		return err
	}
	return p.Session.Write(b, d)
}

func (p *persistSession) WriteAt(b backend.Buf, off int, d []byte) error {
	if err := p.drain(); err != nil {
		return err
	}
	return p.Session.WriteAt(b, off, d)
}

func (p *persistSession) Read(b backend.Buf, d []byte) error {
	if err := p.drain(); err != nil {
		return err
	}
	return p.Session.Read(b, d)
}

func (p *persistSession) Launch(k backend.Kernel, groups, width int, bufs ...backend.Buf) error {
	if err := p.drain(); err != nil {
		return err
	}
	return p.Session.Launch(k, groups, width, bufs...)
}

func (p *persistSession) Sync() error {
	if err := p.drain(); err != nil {
		return err
	}
	return p.Session.Sync()
}

// The recorder: persistOK admits CUDA alone, whose sessions all record.

func (p *persistSession) BeginRecord() error {
	if err := p.drain(); err != nil {
		return err
	}
	return p.Session.(backend.Recorder).BeginRecord()
}

func (p *persistSession) EndRecord() (backend.Recording, error) {
	if err := p.drain(); err != nil {
		return nil, err
	}
	return p.Session.(backend.Recorder).EndRecord()
}

func (p *persistSession) Replay(r backend.Recording) error {
	if err := p.drain(); err != nil {
		return err
	}
	return p.Session.(backend.Recorder).Replay(r)
}
