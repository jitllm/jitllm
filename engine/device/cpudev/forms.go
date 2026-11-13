package cpudev

import (
	"sync"
	"sync/atomic"
	"time"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/graph"
)

// Forms is a model's forms for the CPU device by composer Key, each built
// and mapped once whatever instance asks, and the registry they are asked
// of.
type Forms struct {
	// Target is the CPU the forms are mapped onto.
	Target *cpu.Target
	mu     sync.Mutex
	reg    *composer.Registry
	// reg32 is reg with histories kept at f32 (composer.Registry.WithoutHalves).
	reg32 *composer.Registry
	m     map[composer.Key]*built
	// asks counts Of's calls, a cached form's among them, and took their
	// time in nanoseconds, summed over calls made side by side.
	asks, took atomic.Int64
}

// built is one Key's form or why it was refused, the builder's to write
// until done is closed.
type built struct {
	f    *Form
	err  error
	done chan struct{}
}

// Asks is how many times a form was asked of c, a cached one among them.
func (c *Forms) Asks() int64 { return c.asks.Load() }

// Took is the time Of's calls took: composing, lowering and mapping what
// was not formed yet, finding what was.
func (c *Forms) Took() time.Duration { return time.Duration(c.took.Load()) }

// PassTimes is where Took went: each composer pass's time over every form
// asked of c, and the lowerings' (composer.Registry.PassTimes).
func (c *Forms) PassTimes() map[string]time.Duration {
	c.mu.Lock()
	reg := c.reg
	c.mu.Unlock()
	if reg == nil {
		return nil
	}
	return reg.PassTimes()
}

// Of is request req's form, its graph gr at rows rows (BuildForm's apart
// and whole): asked of the registry keeping histories at f32 where f32 is
// set, and forming no int8 or split kernels where float is. Calls for
// different forms compose side by side; one for a form another call is
// building waits for it.
func (c *Forms) Of(req composer.Request, gr *graph.Graph, rows int, apart, whole, f32, float bool) (*Form, error) {
	c.asks.Add(1)
	t0 := time.Now()
	defer func() { c.took.Add(int64(time.Since(t0))) }()
	c.mu.Lock()
	if c.reg == nil {
		c.reg = composer.Default()
		c.reg32 = c.reg.WithoutHalves(composer.DeviceCPU)
	}
	if c.m == nil {
		c.m = map[composer.Key]*built{}
	}
	reg := c.reg
	c.mu.Unlock()
	if f32 {
		reg = c.reg32
	}
	if float {
		reg = reg.WithoutInt8(composer.DeviceCPU).WithoutSplits(composer.DeviceCPU)
	}
	key, codes, err := reg.Form(req)
	if err != nil {
		return nil, err
	}
	c.mu.Lock()
	if b, ok := c.m[key]; ok {
		c.mu.Unlock()
		<-b.done
		return b.f, b.err
	}
	b := &built{done: make(chan struct{})}
	c.m[key] = b
	c.mu.Unlock()
	defer close(b.done)
	if b.f, b.err = BuildForm(c.Target, gr, codes, rows, apart, whole); b.err == nil {
		b.f.Keys = req.Step.Keys
	}
	return b.f, b.err
}

// All is every form c holds.
func (c *Forms) All() []*Form {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]*Form, 0, len(c.m))
	for _, b := range c.m {
		<-b.done
		if b.f != nil {
			out = append(out, b.f)
		}
	}
	return out
}

// Count is how many forms c holds, refused ones among them.
func (c *Forms) Count() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.m)
}

// Close unmaps every form c holds.
func (c *Forms) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, b := range c.m {
		<-b.done
		if b.f != nil {
			b.f.Close()
		}
	}
	c.m = nil
}
