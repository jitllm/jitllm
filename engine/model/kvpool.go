package model

import "sync"

// kvPagePool is the host KV pages closed instances gave back, by page size, for
// the next session's history to grow into.
//
// A request is a fresh instance, and its history was a fresh make per page, the
// largest thing such a request allocates. Handed back, the pages cost the
// collector nothing per request.
//
// It holds at most one instance's worth -- the largest history any single instance
// has handed back -- so a burst of concurrent sessions does not leave its
// peak resident after it ends. A reused page is cleared: make's zeros are what
// every reader of an unwritten position has always seen.
type kvPagePool struct {
	mu    sync.Mutex
	free  map[int][][]float32
	bytes uint64
	limit uint64
}

// get is a zeroed page of pp elements.
func (p *kvPagePool) get(pp int) []float32 {
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

// put takes every resident page of kc, which must not be used again.
func (p *kvPagePool) put(kc *kvCache) {
	if p == nil || kc == nil {
		return
	}
	// A shared page is the store's: its reference goes, never the memory.
	kc.unshareAll()
	var n uint64
	for li := kc.lo; li < len(kc.layers); li++ {
		s := &kc.layers[li]
		s.unmapArena()
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
	for li := kc.lo; li < len(kc.layers); li++ {
		s := &kc.layers[li]
		for i := range s.k {
			p.keep(s.k[i], s.pp)
			// MLA's value is the key's prefix, the same memory; see kvPages.
			if !s.latent {
				p.keep(s.v[i], s.pp)
			}
			s.setPage(i, nil, nil)
		}
		// A windowed layer's released pages are pages all the same.
		for i, b := range s.spare {
			p.keep(b, s.pp)
			s.spare[i] = nil
		}
		s.spare = s.spare[:0]
	}
}

// keep adds one page while the pool is under its limit. Callers hold p.mu.
func (p *kvPagePool) keep(b []float32, pp int) {
	if b == nil || len(b) != pp || p.bytes+4*uint64(pp) > p.limit {
		return
	}
	p.free[pp] = append(p.free[pp], b)
	p.bytes += 4 * uint64(pp)
}

// usePool makes kc's layers grow from p.
func (kc *kvCache) usePool(p *kvPagePool) {
	for li := kc.lo; li < len(kc.layers); li++ {
		kc.layers[li].pool = p
	}
}
