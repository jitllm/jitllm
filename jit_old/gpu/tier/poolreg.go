package tier

import devpage "github.com/jitllm/jitllm/engine/page"

// PoolRegistry is one budget per physical card shared by every tier opened
// with it (devpage.PoolRegistry).
type PoolRegistry = devpage.PoolRegistry

// NewPoolRegistry is an empty registry.
func NewPoolRegistry() *PoolRegistry { return devpage.NewPoolRegistry() }

// WithPoolRegistry has the tier take each discrete card's pool from r, keyed
// by the card's identity (backend.Identity), the first tier to open a card
// sizing it. A card with no identity, and the host's pool, stay the tier's
// own.
func WithPoolRegistry(r *PoolRegistry) Option { return func(o *openOpts) { o.pools = r } }
