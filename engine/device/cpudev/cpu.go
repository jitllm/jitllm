package cpudev

import (
	"fmt"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/sched"
)

// CPU is the CPU as a device the engine places blocks on, shared by every
// instance that runs on its cores, of one model or several: the target its
// forms are mapped for, the crews each instance's workers are views of (two
// instances on the same cores take turns on one set of workers), each model's
// forms and history arenas, and the seats of the instances on it, so a
// prefill can ask whether another instance decodes.
type CPU struct {
	// execs are the CPU as instances open it, one per option set (Execs).
	execs  map[[2]bool]*Execs
	Target *cpu.Target
	pool   []cpu.PoolOption

	mu     sync.Mutex
	models map[uint64]*modelOn
	// holds counts the instances holding each model on c (HoldModel).
	holds map[uint64]int
	seats []*Seat
}

// modelOn is what the CPU holds for one model: its blocks' forms, its
// histories' arenas and the frames its pages are read into.
type modelOn struct {
	forms  Forms
	arenas Arenas
	pages  PagePool
	seq    atomic.Int64
	// attached is the engine's (Attached).
	attached any
	// frames is the model's frames on the CPU (Frames), made under framesMu,
	// quotas the page quota of each instance holding the model on c
	// (NewQuota), and base the quota whose view is frames itself; every other
	// quota pages through a view of its own (Quota.Frames).
	framesMu sync.Mutex
	frames   *jlm.Frames
	quotas   map[*Quota]quota
	base     *Quota
}

// quota is one instance's page quota for a model on c: the bytes it lets the
// model's frames keep, and what the device it attached holds of the host's
// memory, which comes off them.
type quota struct {
	bytes, held uint64
}

// net is the bytes q lets the frames keep: its own less its device's
// holding, one byte when that holds it all, since the frames read zero as
// unlimited; 0 is unlimited.
func (q quota) net() uint64 {
	switch {
	case q.bytes == 0:
		return 0
	case q.held >= q.bytes:
		return 1
	}
	return q.bytes - q.held
}

// budget is the frames' budget the quotas make: the largest net quota, 0 --
// unlimited -- when any is or none is recorded. Not their sum: the frames
// hold a block every instance runs once, and two instances running the same
// blocks under quotas of X keep X. Each instance keeping its own order within
// its own quota, over the blocks it alone runs, is the instance-paging
// design's retention quota, not built yet. Callers hold framesMu.
func (o *modelOn) budget() uint64 {
	var b uint64
	for _, q := range o.quotas {
		n := q.net()
		if n == 0 {
			return 0
		}
		b = max(b, n)
	}
	return b
}

// close lets the model's forms, arenas and frames go.
func (o *modelOn) close() {
	o.forms.Close()
	o.arenas.Close()
	o.framesMu.Lock()
	fr := o.frames
	o.frames = nil
	o.framesMu.Unlock()
	if fr != nil {
		fr.Close()
	}
}

// Open is the CPU device opts state (cpu.SpecOptions), its instances'
// workers made with pool: the caller's crews (cpu.WithSched, sched.WithCrews)
// or one set of its own. A CPU the composer lowers no code for, or one below
// the emitter's baseline, is refused, naming what it lacks.
func Open(opts []cpu.Option, pool ...cpu.PoolOption) (*CPU, error) {
	t := cpu.New(opts...)
	if !composer.LowersCPU(uint64(t.ISA())) {
		return nil, fmt.Errorf("cpudev: the composer lowers no code for %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	if err := t.Baseline(); err != nil {
		return nil, fmt.Errorf("cpudev: %w", err)
	}
	pool = slices.Clone(pool)
	if cpu.CrewsOf(pool...) == nil {
		pool = append(pool, cpu.WithSched(sched.WithCrews(sched.NewCrews())))
	}
	return &CPU{Target: t, pool: pool}, nil
}

// PoolOptions is what an instance's workers on c are made with.
func (c *CPU) PoolOptions() []cpu.PoolOption { return c.pool[:len(c.pool):len(c.pool)] }

func (c *CPU) of(model uint64) *modelOn {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.models == nil {
		c.models = map[uint64]*modelOn{}
	}
	o := c.models[model]
	if o == nil {
		o = &modelOn{forms: Forms{Target: c.Target}}
		o.arenas.HoldWith(func() { c.HoldModel(model) }, func() { c.ReleaseModel(model) })
		c.models[model] = o
	}
	return o
}

// Forms is model's forms on c.
func (c *CPU) Forms(model uint64) *Forms { return &c.of(model).forms }

// Frames is model's frames on c -- the memory the pages of the blocks its
// instances run on c are read into, as a card reads its blocks into its own
// -- made by mk on the first ask, and closed with the model's forms
// (ReleaseModel). Every instance of the model on c reads a block into them
// once.
func (c *CPU) Frames(model uint64, mk func(...jlm.FramesOption) (*jlm.Frames, error)) (*jlm.Frames, error) {
	o := c.of(model)
	o.framesMu.Lock()
	defer o.framesMu.Unlock()
	if o.frames == nil {
		fr, err := mk(jlm.Budget(o.budget()))
		if err != nil {
			return nil, err
		}
		o.frames = fr
	}
	return o.frames, nil
}

// HeldFrames is model's frames on c if c holds them, nil if not: a
// statistic's read, which makes nothing.
func (c *CPU) HeldFrames(model uint64) *jlm.Frames {
	c.mu.Lock()
	o := c.models[model]
	c.mu.Unlock()
	if o == nil {
		return nil
	}
	o.framesMu.Lock()
	defer o.framesMu.Unlock()
	return o.frames
}

// Quota is one instance's page quota for a model on c (NewQuota): the
// budget of its view of the model's frames there (Frames), its own order
// over the pages every instance shares, until Drop.
type Quota struct {
	o    *modelOn
	view *jlm.Frames
}

// Frames is the quota's view of model's frames on c, the frames themselves
// made by mk on the first ask of any quota: the first quota to ask, or the
// first after it dropped, pages through them; every other through a view of
// its own (jlm.Frames.View), each at its own budget.
func (q *Quota) Frames(mk func(...jlm.FramesOption) (*jlm.Frames, error)) (*jlm.Frames, error) {
	o := q.o
	o.framesMu.Lock()
	defer o.framesMu.Unlock()
	if q.view != nil {
		return q.view, nil
	}
	n := o.quotas[q].net()
	switch {
	case o.frames == nil:
		fr, err := mk(jlm.Budget(n))
		if err != nil {
			return nil, err
		}
		o.frames, o.base, q.view = fr, q, fr
	case o.base == nil:
		o.base, q.view = q, o.frames
		o.frames.SetBudget(n)
	default:
		v, err := o.frames.View(n)
		if err != nil {
			return nil, err
		}
		q.view = v
	}
	return q.view, nil
}

// NewQuota records a page quota of bytes, 0 for unlimited, for model on c,
// before the frames it budgets are made: frames made after it are budgeted
// from their first read (Frames).
func (c *CPU) NewQuota(model uint64, bytes uint64) *Quota {
	q := &Quota{o: c.of(model)}
	q.Set(bytes, 0)
	return q
}

// Set is the quota bytes, of which the device its instance attached holds
// held of the host's memory; the frames are re-budgeted by every quota
// recorded. It returns that budget.
func (q *Quota) Set(bytes, held uint64) uint64 {
	o := q.o
	o.framesMu.Lock()
	defer o.framesMu.Unlock()
	if o.quotas == nil {
		o.quotas = map[*Quota]quota{}
	}
	o.quotas[q] = quota{bytes: bytes, held: held}
	b := o.quotas[q].net()
	if q.view != nil && b != q.view.Budget() {
		q.view.SetBudget(b)
	}
	return b
}

// Drop is the quota gone: its instance closed. The frames keep the budget
// the remaining quotas make, or the one they had when none is left.
func (q *Quota) Drop() {
	o := q.o
	o.framesMu.Lock()
	defer o.framesMu.Unlock()
	delete(o.quotas, q)
	if q.view != nil {
		q.view.Release()
	}
	if o.base == q {
		o.base = nil
	}
	q.view = nil
}

// Arenas is model's history arenas on c.
func (c *CPU) Arenas(model uint64) *Arenas { return &c.of(model).arenas }

// Attached is what the engine keeps beside model on c for every instance of
// it there -- the scratch a closing instance hands the next -- made by mk on
// the first ask. The device does not read it; it goes with the model's forms
// (ReleaseModel).
func (c *CPU) Attached(model uint64, mk func() any) any {
	o := c.of(model)
	c.mu.Lock()
	defer c.mu.Unlock()
	if o.attached == nil {
		o.attached = mk()
	}
	return o.attached
}

// KVPool is the history buffers model's closed sessions on c gave back.
func (c *CPU) KVPool(model uint64) *PagePool { return &c.of(model).pages }

// NextSeq numbers model's histories on c, from 1.
func (c *CPU) NextSeq(model uint64) int64 { return c.of(model).seq.Add(1) }

// HoldModel counts an instance of model holding c (nn.ModelOwner).
func (c *CPU) HoldModel(model uint64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.holds == nil {
		c.holds = map[uint64]int{}
	}
	c.holds[model]++
}

// ReleaseModel is one instance's release of model (nn.ModelOwner); the
// last lets model's forms, arenas and frames go.
func (c *CPU) ReleaseModel(model uint64) {
	c.mu.Lock()
	if c.holds[model] > 1 {
		c.holds[model]--
		c.mu.Unlock()
		return
	}
	delete(c.holds, model)
	o := c.models[model]
	delete(c.models, model)
	c.mu.Unlock()
	if o != nil {
		o.close()
	}
}

// Close lets every model's forms and arenas go, then the target.
func (c *CPU) Close() {
	c.mu.Lock()
	ms := c.models
	c.models = nil
	c.mu.Unlock()
	for _, o := range ms {
		o.close()
	}
	c.Target.Close()
}

// Seat is an instance's place on the CPU: when it last ran a token.
type Seat struct{ ranAt atomic.Int64 }

// decodeGrace is how long after its last token a seat still counts as
// decoding: a streaming client's gaps between tokens are short against it, and
// a session that has stopped is not held against the next prefill for long.
const decodeGrace = 250 * time.Millisecond

// Sit is a new seat on c.
func (c *CPU) Sit() *Seat {
	s := &Seat{}
	c.mu.Lock()
	c.seats = append(c.seats, s)
	c.mu.Unlock()
	return s
}

// Leave gives s up.
func (c *CPU) Leave(s *Seat) {
	c.mu.Lock()
	if i := slices.Index(c.seats, s); i >= 0 {
		c.seats = slices.Delete(c.seats, i, i+1)
	}
	c.mu.Unlock()
}

// Touch records that s is decoding now. One clock read and a store, so a warm
// token still allocates nothing.
func (s *Seat) Touch() { s.ranAt.Store(time.Now().UnixNano()) }

// Decoding reports whether s ran a token within the grace.
func (s *Seat) Decoding() bool {
	t := s.ranAt.Load()
	return t != 0 && time.Duration(time.Now().UnixNano()-t) < decodeGrace
}

// OthersDecoding reports whether a seat on c other than s is decoding.
func (c *CPU) OthersDecoding(s *Seat) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, o := range c.seats {
		if o != s && o.Decoding() {
			return true
		}
	}
	return false
}
