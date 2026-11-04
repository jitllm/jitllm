package model

import "github.com/jitllm/jitllm/engine/device/cpudev"

// kvArenaFor is the model's arena for layer li's pages of pp floats
// (cpudev.Arenas), made on the first ask: reserved for what the host's memory
// budget can hold of that layer and no fewer than minSlots pages.
func (m *Model) kvArenaFor(li, pp int, latent bool, minSlots int) (*cpudev.Arena, error) {
	slots := max(minSlots, int(min(m.opt.gc.Budget()/uint64(4*pp), cpudev.ArenaElems/uint64(pp))))
	return m.eng.kvArenas.For(li, pp, latent, slots)
}

// useArena gives the layer the model's arena for its pages, the instance's
// table sized for maxSeq positions. A layer with no history keeps none, and
// a layer whose pages are already held keeps them where they are.
func (s *kvPages) useArena(li, maxSeq int, arenaFor func(li, pp int, latent bool, minSlots int) (*cpudev.Arena, error)) error {
	if s.p == 0 || maxSeq <= 0 || s.arena != nil {
		return nil
	}
	for n := range s.k {
		if s.k[n] != nil {
			return nil
		}
	}
	s.reserved = (maxSeq + s.p - 1) / s.p
	a, err := arenaFor(li, s.pp, s.latent, s.reserved)
	if err != nil {
		return err
	}
	s.arena = a
	return nil
}

// takeSlot is a fresh slot of the arena for a page this layer writes, held
// by it, nil where the layer has no arena or every slot is taken.
func (s *kvPages) takeSlot() (k, v []float32) {
	if s.arena == nil {
		return nil, nil
	}
	i, ok := s.arena.Take()
	if !ok {
		return nil, nil
	}
	return s.arena.Page(i)
}

// inArena says b is one of the arena's slots.
func (s *kvPages) inArena(_ int, b []float32) bool {
	if s.arena == nil {
		return false
	}
	_, ok := s.arena.SlotOf(b)
	return ok
}

// arenaPage says page n's key and value are both slots of the arena.
func (s *kvPages) arenaPage(n int) bool {
	return n < len(s.k) && s.inArena(n, s.k[n]) && s.inArena(n, s.v[n])
}

// releaseSlot lets page n's slot go, the layer's own hold on it: the caller
// has dropped any shared reference first (unshare).
func (s *kvPages) releaseSlot(n int) {
	if s.arena == nil {
		return
	}
	if i, ok := s.arena.SlotOf(s.k[n]); ok {
		s.arena.Drop(i)
	}
}

// giveSlot lets the layer's hold on the slot b starts go, where b is one.
func (s *kvPages) giveSlot(b []float32) {
	if s.arena == nil {
		return
	}
	if i, ok := s.arena.SlotOf(b); ok {
		s.arena.Drop(i)
	}
}

// unmapArena lets every slot the layer holds go; no page of it may be read
// again. The arena is the model's and stays.
func (s *kvPages) unmapArena() {
	if s.arena == nil {
		return
	}
	for n := range s.k {
		if !s.isShared(n) && s.arenaPage(n) {
			s.releaseSlot(n)
		}
		if s.inArena(n, s.k[n]) || s.inArena(n, s.v[n]) {
			s.k[n], s.v[n] = nil, nil
		}
	}
	s.arena = nil
}

// table is the page table a composed CPU block reads this layer's history
// through: sequence q's page j, at q*slots+j, names the arena slot holding
// it -- slot*nseq+q, a page holding every sequence's positions in order
// (kvLayout) -- and a page not held there names slot 0's, which no row's
// range reaches (cpuReadable).
func (s *kvPages) table(nseq int) []uint32 {
	slots := max(s.reserved, 1)
	n := slots * nseq
	if cap(s.tab) < n {
		s.tab = make([]uint32, n)
	}
	t := s.tab[:n]
	for j := range slots {
		i := 0
		if s.arena != nil && j < len(s.k) {
			if si, ok := s.arena.SlotOf(s.k[j]); ok {
				i = si
			}
		}
		for q := range nseq {
			t[q*slots+j] = uint32(i*nseq + q)
		}
	}
	return t
}

// useArenas gives every attending layer of kc the model's arena for its
// pages (cpudev.Arena), for an instance whose CPU blocks run composed.
func (kc *kvCache) useArenas(maxSeq int, arenaFor func(li, pp int, latent bool, minSlots int) (*cpudev.Arena, error)) error {
	for li := kc.lo; li < len(kc.layers); li++ {
		if err := kc.layers[li].useArena(li, maxSeq, arenaFor); err != nil {
			return err
		}
	}
	return nil
}
