// Package page is the paging policy every runner of a device shares: the
// byte account a heap's devices spend from (Pool), a block's page record and
// the victim a page-in evicts. The mechanism -- how a block's bytes reach a
// device and leave it -- is each runner's own. See docs/design/block-paging.md.
package page

import "sync"

// HostPoolName is what a report calls the pool an integrated GPU spends from.
const HostPoolName = "host RAM"

// Pool is a weight budget that every device spending the same bytes shares.
//
// An integrated GPU's heap is system memory, so a budget of its own beside the
// host's would spend the same bytes twice; a discrete card's VRAM is genuinely
// separate. So a device does not own a budget, it maps to one:
//
//	pool "host"   host RAM   <- the iGPU's weights and KV, and the host's own
//	pool "sm_86"  4 GiB VRAM <- the discrete card's weights and KV
//	(on Apple Silicon there is exactly one pool and everything is in it)
//
// backend.Device.UnifiedMemory() says which pool a device belongs to. The iGPU
// is not refused: its execution units are real, and it is charged to the host
// pool.
//
// A device also keeps its own limit, and both are checked: the pool stops two
// devices double-spending one heap, the per-device limit is what -vram means
// for one card.
type Pool struct {
	mu    sync.Mutex
	name  string
	used  uint64
	limit uint64
	// host marks the pool whose bytes are the host's own memory; its used
	// bytes are what the host's reservation (tier.GPU.HostReserved) reads, and so what the host's weight
	// budget is reduced by.
	host bool
}

// NewPool is a named budget. limit 0 means unbounded, which is what a caller
// that wants the per-device limits to be the only gate passes.
func NewPool(name string, limit uint64) *Pool { return &Pool{name: name, limit: limit} }

// NewHostPool is the budget for devices whose memory is the host's: an
// integrated GPU, a CPU rasteriser, every device on Apple Silicon. limit is the
// share of the host's own budget those devices may hold between them; the host
// takes off its own only what they hold (see the host's reservation (tier.GPU.HostReserved)).
func NewHostPool(limit uint64) *Pool {
	return &Pool{name: HostPoolName, limit: limit, host: true}
}

func (p *Pool) Name() string { return p.name }

// Host reports that this pool's bytes come out of host RAM.
func (p *Pool) Host() bool { return p != nil && p.host }

func (p *Pool) Used() uint64 { p.mu.Lock(); defer p.mu.Unlock(); return p.used }

func (p *Pool) Limit() uint64 { return p.limit }

// Free is what the pool's limit leaves above what it holds, 0 when it is
// at or past it.
func (p *Pool) Free() uint64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.used < p.limit {
		return p.limit - p.used
	}
	return 0
}

// SetLimit retargets the heap this pool stands for, at runtime. It does not
// evict: a pool is an account, not an owner, and only a device knows which of
// its blocks can be given back. A device's SetBudget lowers the limits and then asks
// each device to trim.
func (p *Pool) SetLimit(n uint64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.limit = n
}

// Room reports whether n more bytes would fit, without taking them. PrepLayer
// prices a whole block before uploading any of it (see blockBytes), so the
// check and the charge are separate operations.
func (p *Pool) Room(n uint64) bool {
	if p == nil || p.limit == 0 {
		return true
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.used+n <= p.limit
}

func (p *Pool) Take(n uint64) {
	if p == nil {
		return
	}
	p.mu.Lock()
	p.used += n
	p.mu.Unlock()
}

func (p *Pool) Give(n uint64) {
	if p == nil {
		return
	}
	p.mu.Lock()
	if p.used >= n {
		p.used -= n
	} else {
		p.used = 0
	}
	p.mu.Unlock()
}

// PoolRegistry is one weight-and-KV budget per physical card shared by every
// device user made with it (tier.WithPoolRegistry): a server holding several models,
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

// PoolFor is the card key's pool, p when the card is new.
func (r *PoolRegistry) PoolFor(key string, p *Pool) *Pool {
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
