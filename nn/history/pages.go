package history

import (
	"slices"
	"sync"
)

// Pages is one block's history memory on the CPU device, for one session:
// each page's key and value buffer, the released buffers the next page grows
// into, and, once the block takes its arena, the table its forms read the
// pages through. The session's history decides what a page holds and when it
// goes; Pages is where it is. A latent block's value is its key's memory.
type Pages struct {
	pp     int
	latent bool
	k, v   [][]float32
	spare  [][]float32
	pool   *PagePool
	t      *Table
}

// Pages is block li's history memory in the view, pp floats a page, its
// buffers from pool where the arena has none: the view's for the block from
// now on. Memory the block had before stays with whatever history still holds
// it until that lets it go.
func (v *View) Pages(li, pp int, latent bool, pool *PagePool) *Pages {
	for len(v.pages) <= li {
		v.pages = append(v.pages, nil)
	}
	p := &Pages{pp: pp, latent: latent, pool: pool}
	v.pages[li] = p
	return p
}

// Len is the pages addressable.
func (p *Pages) Len() int { return len(p.k) }

// K is page n's key, nil where it is not resident.
func (p *Pages) K(n int) []float32 {
	if n >= len(p.k) {
		return nil
	}
	return p.k[n]
}

// V is page n's value, nil where it is not resident.
func (p *Pages) V(n int) []float32 {
	if n >= len(p.v) {
		return nil
	}
	return p.v[n]
}

// Resident says page n's key and value are both here.
func (p *Pages) Resident(n int) bool {
	return n < len(p.k) && p.k[n] != nil && p.v[n] != nil
}

// Reserve sizes the page index for n pages and the spares for keep buffers,
// so a decode crossing a page boundary grows neither.
func (p *Pages) Reserve(n, keep int) {
	p.k = slices.Grow(p.k, n)
	p.v = slices.Grow(p.v, n)
	p.spare = slices.Grow(p.spare, keep)
}

// Extend makes page n addressable without making it resident.
func (p *Pages) Extend(n int) {
	for len(p.k) <= n {
		p.k = append(p.k, nil)
		p.v = append(p.v, nil)
	}
}

// Set makes page n's key and value k and v, and tells the table.
func (p *Pages) Set(n int, k, v []float32) {
	p.k[n], p.v[n] = k, v
	if p.t != nil {
		p.t.Set(n, k)
	}
}

// Reset leaves no page addressable; the memory is the caller's to have let go.
func (p *Pages) Reset() {
	p.k, p.v = p.k[:0], p.v[:0]
}

// Grow makes page n resident: a slot of the arena where the page has neither
// buffer and one is free, else a zeroed buffer for whichever it lacks.
func (p *Pages) Grow(n int) {
	p.Extend(n)
	if p.k[n] == nil && p.v[n] == nil {
		if k, v := p.Take(); k != nil {
			p.Set(n, k, v)
			return
		}
	}
	k, v := p.k[n], p.v[n]
	if k == nil {
		k = p.Zeroed()
	}
	switch {
	case p.latent:
		v = k
	case v == nil:
		v = p.Zeroed()
	}
	p.Set(n, k, v)
}

// Zeroed is a zeroed buffer of a page: a spare first, then the pool's.
func (p *Pages) Zeroed() []float32 {
	if n := len(p.spare); n > 0 {
		b := p.spare[n-1]
		p.spare[n-1] = nil
		p.spare = p.spare[:n-1]
		clear(b)
		return b
	}
	return p.pool.get(p.pp)
}

// Keep takes b, a buffer of the block's own it no longer holds as a page, for
// the next page to grow into.
func (p *Pages) Keep(b []float32) { p.spare = append(p.spare, b) }

// Alloc is a fresh buffer of a page, for a read that fills it whole.
func (p *Pages) Alloc() []float32 { return make([]float32, p.pp) }

// Release lets page n go and leaves it not resident: a slot's hold back to
// the arena, and a buffer of the page's own to the spares while they number
// under keep (no limit below zero). A page whose memory is another holder's
// (a shared one) is released with keep 0, after its holder let it go.
func (p *Pages) Release(n, keep int) {
	if n >= len(p.k) {
		return
	}
	for i, b := range [2][]float32{p.k[n], p.v[n]} {
		if i == 1 && p.latent {
			break
		}
		if b != nil && len(b) == p.pp && !p.InArena(b) && (keep < 0 || len(p.spare) < keep) {
			p.spare = append(p.spare, b)
		}
	}
	if p.ArenaPage(n) {
		p.t.Let(p.k[n])
	}
	p.Set(n, nil, nil)
}

// UseArena gives the block arena a's slots for its pages and a table for
// slots pages; the pages it already holds stay where they are.
func (p *Pages) UseArena(a *Arena, slots int) {
	slots = max(slots, 1)
	p.t = &Table{a: a, slots: slots, of: make([]uint32, slots)}
}

// Table is the table the block's forms read its pages through, nil before it
// takes an arena.
func (p *Pages) Table() *Table { return p.t }

// Take is a fresh slot's key and value, held by the block, nil where it has
// no arena or every slot is taken.
func (p *Pages) Take() (k, v []float32) {
	if p.t == nil {
		return nil, nil
	}
	return p.t.Take()
}

// Give lets the block's hold on the slot b starts go, where b is one.
func (p *Pages) Give(b []float32) {
	if p.t != nil {
		p.t.Let(b)
	}
}

// InArena says b is one of the arena's slots.
func (p *Pages) InArena(b []float32) bool { return p.t != nil && p.t.Holds(b) }

// ArenaPage says page n's key and value are both slots of the arena.
func (p *Pages) ArenaPage(n int) bool {
	return n < len(p.k) && p.InArena(p.k[n]) && p.InArena(p.v[n])
}

// Unmap lets the arena go: no page in it is read again. The pages that are
// slots are let go by the caller first (Release), or are another holder's.
func (p *Pages) Unmap() {
	if p.t == nil {
		return
	}
	for n := range p.k {
		if p.InArena(p.k[n]) || p.InArena(p.v[n]) {
			p.Set(n, nil, nil)
		}
	}
	p.t.Clear()
	p.t = nil
}

// PagePool is the history buffers closed sessions gave back, by page size,
// for the next session's history to grow into: a model's, beside its arenas.
//
// A session's history was a fresh make per page, the largest thing a request
// allocates; handed back, the pages cost the collector nothing per request. It
// holds at most one session's worth -- the largest history any single one has
// handed back -- so a burst of concurrent sessions does not leave its peak
// resident after it ends. A reused buffer is cleared: make's zeros are what
// every reader of an unwritten position has always seen.
type PagePool struct {
	mu    sync.Mutex
	free  map[int][][]float32
	bytes uint64
	limit uint64
}

// get is a zeroed buffer of pp elements.
func (p *PagePool) get(pp int) []float32 {
	if p != nil {
		p.mu.Lock()
		if l := p.free[pp]; len(l) > 0 {
			b := l[len(l)-1]
			l[len(l)-1] = nil
			p.free[pp] = l[:len(l)-1]
			p.bytes -= 4 * uint64(pp)
			p.mu.Unlock()
			clear(b)
			return b
		}
		p.mu.Unlock()
	}
	return make([]float32, pp)
}

// Put takes every buffer of every block's pages, which are not read again: a
// shared page's holder and the arena's slots have let theirs go first, so
// what is left is the blocks' own.
func (p *PagePool) Put(pages ...*Pages) {
	if p == nil {
		return
	}
	var n uint64
	for _, s := range pages {
		if s == nil {
			continue
		}
		for i := range s.k {
			if s.k[i] != nil {
				n += 4 * uint64(len(s.k[i]))
			}
			if s.v[i] != nil && !s.latent {
				n += 4 * uint64(len(s.v[i]))
			}
		}
		for _, b := range s.spare {
			n += 4 * uint64(len(b))
		}
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.limit = max(p.limit, n)
	if p.free == nil {
		p.free = map[int][][]float32{}
	}
	for _, s := range pages {
		if s == nil {
			continue
		}
		for i := range s.k {
			p.keep(s.k[i], s.pp)
			if !s.latent {
				p.keep(s.v[i], s.pp)
			}
			s.Set(i, nil, nil)
		}
		for i, b := range s.spare {
			p.keep(b, s.pp)
			s.spare[i] = nil
		}
		s.spare = s.spare[:0]
	}
}

// keep adds one buffer while the pool is under its limit. Callers hold p.mu.
func (p *PagePool) keep(b []float32, pp int) {
	if b == nil || len(b) != pp || p.bytes+4*uint64(pp) > p.limit {
		return
	}
	p.free[pp] = append(p.free[pp], b)
	p.bytes += 4 * uint64(pp)
}

// Held is the bytes the pool holds, and calls f on each buffer while it holds
// them, for a gate that poisons what a new history must not read.
func (p *PagePool) Held(f func(b []float32)) uint64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	if f != nil {
		for _, l := range p.free {
			for _, b := range l {
				f(b)
			}
		}
	}
	return p.bytes
}
