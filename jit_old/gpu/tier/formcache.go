package tier

import (
	"slices"
	"sync"

	"github.com/jitllm/jitllm/jit_old/gpu/compose"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

// FormCache is formed kernels shared between tiers on one target
// (WithFormCache): a kernel's IR is a function of its encoded structure and
// schedule, so a tier asking for one another formed takes that one, and
// compiles it on its own device.
type FormCache struct {
	mu sync.Mutex
	m  map[compose.Key][]formed
}

type formed struct {
	enc []uint32
	k   *ir.Kernel
	err error
}

// NewFormCache is an empty FormCache.
func NewFormCache() *FormCache { return &FormCache{m: map[compose.Key][]formed{}} }

// WithFormCache shares c's formed kernels with every tier given it, all on
// one target.
func WithFormCache(c *FormCache) Option { return func(o *openOpts) { o.forms = c } }

// form is enc's kernel, built once.
func (c *FormCache) form(enc []uint32, build func() (*ir.Kernel, error)) (*ir.Kernel, error) {
	key := compose.KeyOf(enc)
	c.mu.Lock()
	for _, f := range c.m[key] {
		if slices.Equal(f.enc, enc) {
			c.mu.Unlock()
			return f.k, f.err
		}
	}
	c.mu.Unlock()
	k, err := build()
	if err == nil {
		k.Comment = composedComment(k.Comment)
	}
	c.mu.Lock()
	c.m[key] = append(c.m[key], formed{enc: enc, k: k, err: err})
	c.mu.Unlock()
	return k, err
}
