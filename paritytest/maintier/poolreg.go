package maintier

import "sync"

// PoolRegistry is one weight-and-KV budget per physical card shared by every
// tier opened with it (WithPoolRegistry): a server holding several models,
// each with its own tier, spends one card's memory once. Without it each
// tier sizes its own pool from what was free when it opened, so a card a
// model gives back stays out of reach of the model that needs it -- and a
// server that moves blocks between models to make room (an orchestrator) has
// no room to make.
type PoolRegistry struct {
	mu    sync.Mutex
	cards map[string]*Pool
}

// NewPoolRegistry is an empty registry.
func NewPoolRegistry() *PoolRegistry { return &PoolRegistry{cards: map[string]*Pool{}} }

// WithPoolRegistry has the tier take each discrete card's pool from r, keyed
// by the card's identity (backend.Identity), the first tier to open a card
// sizing it. A card with no identity, and the host's pool, stay the tier's
// own.
func WithPoolRegistry(r *PoolRegistry) Option { return func(o *openOpts) { o.pools = r } }

// poolFor is the card key's pool, p when the card is new.
func (r *PoolRegistry) poolFor(key string, p *Pool) *Pool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if have, ok := r.cards[key]; ok {
		return have
	}
	r.cards[key] = p
	return p
}

// Pools is every card's pool the registry holds, by card key.
func (r *PoolRegistry) Pools() map[string]*Pool {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make(map[string]*Pool, len(r.cards))
	for k, p := range r.cards {
		out[k] = p
	}
	return out
}
