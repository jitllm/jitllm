package maintier

import (
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// The persistent decode block (Config.Persist): a decode block's launches,
// collected as they are issued and composed into persistent kernels
// (ir.Persist) -- one grid that runs the launches' bodies in order, phases
// separated by a grid barrier -- instead of one launch each.
//
// It sits in the launcher, below every choice the tier makes about which
// kernels a block runs, so the per-op path is the reference by construction:
// the same kernels, the same groups, the same buffers, composed. A launch the
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

// persistDefault is Config.Persist's default, false outside a jitllmtest
// build (PersistByDefaultForTest), and persistTotal every persistent kernel
// launched in the process, which that build's gates read.
var (
	persistDefault, persistDepsDefault bool
	persistTotal, persistDepTotal      atomic.Int64
)

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
		b, err := g.dev.Alloc(4 * ir.GridSyncWords)
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
		b, err := g.dev.Alloc(4 * depWords)
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

// persistIssue binds, plans (once per signature) and issues n collected
// launches.
func (g *devTier) persistIssue(t *launchTo, ps *persister, n int) error {
	// Distinct buffers and bindings, and the signature: each launch's kernel,
	// grid, width and binding pattern.
	ps.nd = 0
	ps.sig = ps.sig[:0]
	// The mode first: a plan composed with grid barriers is not one composed
	// with dependency counters.
	if g.PersistDeps && ps.dep != nil {
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
					return g.persistPlain(t, ps, 0, n)
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
	plan := g.persistPlanFor(h, ps, n)
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
		persistTotal.Add(1)
		g.PersistParts += s.hi - s.lo
		g.PersistPhases += s.phases
		if s.dep {
			g.PersistDepLaunches++
			g.PersistDepTiles += s.tiles
			persistDepTotal.Add(1)
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

// persistPlanFor finds or builds the plan for the collected launches.
func (g *devTier) persistPlanFor(h uint64, ps *persister, n int) *persistPlan {
	c := g.pcache
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, p := range c.plans[h] {
		if equalSig(p.sig, ps.sig) {
			return p
		}
	}
	p := &persistPlan{sig: append([]uint64(nil), ps.sig...)}
	p.segs = g.persistCompose(ps, n)
	if c.plans == nil {
		c.plans = map[uint64][]*persistPlan{}
	}
	c.plans[h] = append(c.plans[h], p)
	return p
}

func equalSig(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// persistJoins reports whether launch i can join a run of width w.
func (ps *persister) joins(i, w int) error {
	p := &ps.pend[i]
	if p.src == nil {
		return fmt.Errorf("no IR")
	}
	if err := ir.PersistJoins(p.src, w); err != nil {
		return err
	}
	// A buffer bound to two parameters, one read and one written, is an
	// in-place launch the composer's phases cannot see inside.
	rs, ws := persistAccess(p.src)
	for _, r := range rs {
		for _, x := range ws {
			if r != x && ps.bind[i][r] == ps.bind[i][x] {
				return fmt.Errorf("%s binds one buffer as an input and an output", p.src.Name)
			}
		}
	}
	return nil
}

// persistAccess lists the parameters k reads and writes.
func persistAccess(k *ir.Kernel) (reads, writes []int) {
	isParam := func(v ir.Value) int {
		if v == 0 || int(v) > len(k.Ops) || k.Ops[v-1].Kind != ir.OpParam {
			return -1
		}
		return int(k.Ops[v-1].Imm)
	}
	for _, o := range k.Ops {
		p := isParam(o.Args[0])
		if p < 0 {
			continue
		}
		switch o.Kind {
		case ir.OpLoad, ir.OpLoadV, ir.OpTileLoad:
			reads = append(reads, p)
		case ir.OpStore, ir.OpStoreV, ir.OpTileStore:
			writes = append(writes, p)
		}
	}
	return reads, writes
}

// persistCompose cuts the launches into runs and composes each run of two or
// more. Called once per signature, under the cache's lock.
func (g *devTier) persistCompose(ps *persister, n int) []persistSeg {
	var segs []persistSeg
	for i := 0; i < n; {
		w := ps.pend[i].width
		j := i
		for j < n && ps.joins(j, w) == nil {
			j++
		}
		if j-i < 2 {
			hi := max(j, i+1)
			segs = append(segs, persistSeg{lo: i, hi: hi})
			if j == i {
				name := "?"
				if src := ps.pend[i].src; src != nil {
					name = src.Name
				}
				g.pcache.why = fmt.Sprintf("%s alone: %v", name, ps.joins(i, w))
			}
			i = hi
			continue
		}
		s, err := g.persistBuild(ps, i, j, w)
		if err != nil {
			g.pcache.why = err.Error()
			g.PersistFallbacks++
			segs = append(segs, persistSeg{lo: i, hi: j})
		} else {
			segs = append(segs, s)
		}
		i = j
	}
	return segs
}

// persistBuild composes launches [lo, hi) of width w and compiles the result
// on a grid the device holds resident: composed once to learn the limit
// (the occupancy calculator counts the kernel's registers and shared memory,
// which do not depend on the grid), and again at the grid it chose when that
// differs.
func (g *devTier) persistBuild(ps *persister, lo, hi, w int) (persistSeg, error) {
	parts := make([]ir.Part, 0, hi-lo)
	most := 0
	for i := lo; i < hi; i++ {
		p := &ps.pend[i]
		bind := make([]int, p.nb)
		copy(bind, ps.bind[i][:p.nb])
		parts = append(parts, ir.Part{K: p.src, Groups: p.groups, Bind: bind})
		most = max(most, p.groups)
	}
	// With dependency counters the kernel's region starts at the first word
	// no plan holds; a plan that does not fit composes with grid barriers.
	dep := g.PersistDeps && ps.dep != nil
	base := g.pcache.depNext
	compose := func(grid int) (*ir.Persisted, backend.Kernel, error) {
		var pk *ir.Persisted
		var err error
		if dep {
			pk, err = ir.PersistDeps(fmt.Sprintf("persistdep%d_%d", hi-lo, grid), parts, w, grid, base)
			if err == nil && base+pk.Words > depWords {
				dep = false
			}
		}
		if !dep {
			pk, err = ir.Persist(fmt.Sprintf("persist%d_%d", hi-lo, grid), parts, w, grid)
		}
		if err != nil {
			return nil, nil, err
		}
		k, err := g.dev.Compile(pk.K)
		if err != nil {
			return nil, nil, err
		}
		return pk, k, nil
	}
	grid := most
	if g.PersistGrid > 0 {
		grid = min(grid, g.PersistGrid)
	}
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
		grid = limit
		if pk, k, err = compose(grid); err != nil {
			return persistSeg{}, err
		}
	}
	g.PersistCompiles++
	tiles := 0
	if dep {
		g.pcache.depNext = base + pk.Words
		for _, o := range pk.K.Ops {
			if o.Kind == ir.OpDepWait && o.Imm&ir.DepTile != 0 {
				tiles++
			}
		}
	}
	return persistSeg{lo: lo, hi: hi, k: k, bufs: pk.Bufs, grid: pk.Grid, width: w, phases: pk.Phases, dep: dep, tiles: tiles}, nil
}

// persistable reports whether block l's launches are held: a block whose
// emission makes calls of its own on the session between launches (a
// streamed mixture's selection read home, a linear block's or DeepSeek V4's
// own submissions) runs per-op.
func persistable(l *layer, bs *blockScratch) bool {
	return l != nil && l.stream == nil && !l.linear && l.ds4 == nil && l.k3 == nil && l.q4e == nil && l.altRouter == nil
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
