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
	"github.com/jitllm/jitllm/sched"
)

// CPU is the CPU as a device the engine places blocks on, shared by every
// instance that runs on its cores, of one model or several: the target its
// forms are mapped for, the crews each instance's workers are views of (two
// instances on the same cores take turns on one set of workers), each model's
// forms and history arenas, and the seats of the instances on it, so a
// prefill can ask whether another instance decodes.
type CPU struct {
	Target *cpu.Target
	pool   []cpu.PoolOption

	mu     sync.Mutex
	models map[uint64]*modelOn
	seats  []*Seat
}

// modelOn is what the CPU holds for one model: its blocks' forms and its
// histories' arenas.
type modelOn struct {
	forms  Forms
	arenas Arenas
	pages  PagePool
	seq    atomic.Int64
	// attached is the engine's (Attached).
	attached any
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
		c.models[model] = o
	}
	return o
}

// Forms is model's forms on c.
func (c *CPU) Forms(model uint64) *Forms { return &c.of(model).forms }

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

// ReleaseModel lets model's forms and arenas go (nn.ModelOwner): every
// instance of the model has closed.
func (c *CPU) ReleaseModel(model uint64) {
	c.mu.Lock()
	o := c.models[model]
	delete(c.models, model)
	c.mu.Unlock()
	if o != nil {
		o.forms.Close()
		o.arenas.Close()
	}
}

// Close lets every model's forms and arenas go, then the target.
func (c *CPU) Close() {
	c.mu.Lock()
	ms := c.models
	c.models = nil
	c.mu.Unlock()
	for _, o := range ms {
		o.forms.Close()
		o.arenas.Close()
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
