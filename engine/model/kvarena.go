package model

import (
	"unsafe"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// kvArena is one layer's history pages as one span of memory outside the Go
// heap, page n at slot n: what a composed CPU block's attention addresses as
// a pool, through a table of page slots, as a card's pool is (jit.History).
// It is reserved for the State's whole context and committed by the pages
// written: a page let go is handed back to the OS (kernels.ReleaseOffHeap),
// so a windowed layer still holds its window. Every other page source --
// the spare list, the pool of closed States, a shared store -- stays outside
// it, and a page outside it is not one a composed block can address
// (cpuBlockPage).
type kvArena struct {
	kb, vb []byte
	k, v   []float32
	slots  int
}

// useArena reserves the arena for maxSeq positions. A layer with no history
// keeps none, and a layer whose pages are already held keeps them where they
// are.
func (s *kvPages) useArena(maxSeq int) error {
	if s.p == 0 || maxSeq <= 0 || s.arena != nil {
		return nil
	}
	for n := range s.k {
		if s.k[n] != nil {
			return nil
		}
	}
	a, err := mapKVArena(s.pp, (maxSeq+s.p-1)/s.p, s.latent)
	if err != nil {
		return err
	}
	s.arena = a
	return nil
}

// mapKVArena is an arena of slots pages of pp floats each, its keys and its
// values -- one span where latent, MLA's value the key's prefix.
func mapKVArena(pp, slots int, latent bool) (*kvArena, error) {
	n := 4 * pp * slots
	a := &kvArena{slots: slots}
	var err error
	if a.kb, _, err = kernels.MapOffHeap(n); err != nil {
		return nil, err
	}
	a.k = unsafe.Slice((*float32)(unsafe.Pointer(&a.kb[0])), pp*slots)
	if latent {
		a.vb, a.v = a.kb, a.k
		return a, nil
	}
	if a.vb, _, err = kernels.MapOffHeap(n); err != nil {
		kernels.UnmapOffHeap(a.kb)
		return nil, err
	}
	a.v = unsafe.Slice((*float32)(unsafe.Pointer(&a.vb[0])), pp*slots)
	return a, nil
}

// unmap returns the arena's memory, a's two spans once each.
func (a *kvArena) unmap() {
	kernels.UnmapOffHeap(a.kb)
	if &a.vb[0] != &a.kb[0] {
		kernels.UnmapOffHeap(a.vb)
	}
}

// arenaSlot is page n's slots of the arena, the key's and the value's, nil
// past it.
func (s *kvPages) arenaSlot(n int) (k, v []float32) {
	a := s.arena
	if a == nil || n >= a.slots {
		return nil, nil
	}
	lo, hi := n*s.pp, (n+1)*s.pp
	return a.k[lo:hi:hi], a.v[lo:hi:hi]
}

// inArena says b is the arena's slot for page n.
func (s *kvPages) inArena(n int, b []float32) bool {
	k, v := s.arenaSlot(n)
	return len(b) > 0 && k != nil && (&b[0] == &k[0] || &b[0] == &v[0])
}

// arenaPage says page n's key and value are both the arena's own slots.
func (s *kvPages) arenaPage(n int) bool {
	return n < len(s.k) && s.inArena(n, s.k[n]) && s.inArena(n, s.v[n])
}

// releaseSlot hands slot n's memory back, zero when it is next touched.
func (s *kvPages) releaseSlot(n int) {
	k, v := s.arenaSlot(n)
	if k == nil {
		return
	}
	kernels.ReleaseOffHeap(unsafe.Slice((*byte)(unsafe.Pointer(&k[0])), 4*len(k)))
	if !s.latent {
		kernels.ReleaseOffHeap(unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), 4*len(v)))
	}
}

// unmapArena returns the arena; no page of it may be read again.
func (s *kvPages) unmapArena() {
	a := s.arena
	if a == nil {
		return
	}
	for n := range s.k {
		if s.inArena(n, s.k[n]) || s.inArena(n, s.v[n]) {
			s.k[n], s.v[n] = nil, nil
		}
	}
	a.unmap()
	s.arena = nil
}

// useArenas gives every attending layer of kc its arena (kvArena), for a
// State whose CPU blocks run composed. A cache that shares pages with other
// sessions keeps its own: the store may take a session's page as its copy.
func (kc *kvCache) useArenas(maxSeq int) error {
	if kc.share != nil {
		return nil
	}
	for li := kc.lo; li < len(kc.layers); li++ {
		if err := kc.layers[li].useArena(maxSeq); err != nil {
			return err
		}
	}
	return nil
}
