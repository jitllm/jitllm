package model

import "github.com/jitllm/jitllm/engine/device/cpudev"

// kvArenaFor is the model's arena for layer li's pages of pp floats
// (cpudev.Arenas), made on the first ask: reserved for what the host's memory
// budget can hold of that layer and no fewer than minSlots pages.
func (m *Model) kvArenaFor(li, pp int, latent bool, minSlots int) (*cpudev.Arena, error) {
	slots := max(minSlots, int(min(m.opt.gc.Budget()/uint64(4*pp), cpudev.ArenaElems/uint64(pp))))
	return m.eng.kvArenas.For(li, pp, latent, slots)
}

// cpuView is the session's view of the CPU device's history, made on the
// first ask.
func (s *Session) cpuView() *cpudev.View {
	if s.view == nil {
		s.view = cpudev.NewView(len(s.kv.layers))
	}
	return s.view
}

// useArena gives the layer the model's arena for its pages and a table in
// the view for maxSeq positions. A layer with no history keeps none, and a
// layer whose pages are already held keeps them where they are.
func (s *kvPages) useArena(li, maxSeq int, arenaFor func(li, pp int, latent bool, minSlots int) (*cpudev.Arena, error), view *cpudev.View) error {
	if s.p == 0 || maxSeq <= 0 || s.t != nil {
		return nil
	}
	for n := range s.k {
		if s.k[n] != nil {
			return nil
		}
	}
	slots := (maxSeq + s.p - 1) / s.p
	a, err := arenaFor(li, s.pp, s.latent, slots)
	if err != nil {
		return err
	}
	s.t = view.Table(li, a, slots)
	return nil
}

// takeSlot is a fresh slot of the arena for a page this layer writes, held
// by it, nil where the layer has no arena or every slot is taken.
func (s *kvPages) takeSlot() (k, v []float32) {
	if s.t == nil {
		return nil, nil
	}
	return s.t.Take()
}

// inArena says b is one of the arena's slots.
func (s *kvPages) inArena(_ int, b []float32) bool {
	return s.t != nil && s.t.Holds(b)
}

// arenaPage says page n's key and value are both slots of the arena.
func (s *kvPages) arenaPage(n int) bool {
	return n < len(s.k) && s.inArena(n, s.k[n]) && s.inArena(n, s.v[n])
}

// releaseSlot lets page n's slot go, the layer's own hold on it: the caller
// has dropped any shared reference first (unshare).
func (s *kvPages) releaseSlot(n int) {
	if s.t != nil {
		s.t.Let(s.k[n])
	}
}

// giveSlot lets the layer's hold on the slot b starts go, where b is one.
func (s *kvPages) giveSlot(b []float32) {
	if s.t != nil {
		s.t.Let(b)
	}
}

// unmapArena lets every slot the layer holds go; no page of it may be read
// again. The arena is the model's and stays.
func (s *kvPages) unmapArena() {
	if s.t == nil {
		return
	}
	for n := range s.k {
		if !s.isShared(n) && s.arenaPage(n) {
			s.releaseSlot(n)
		}
		if s.inArena(n, s.k[n]) || s.inArena(n, s.v[n]) {
			s.setPage(n, nil, nil)
		}
	}
	s.t.Clear()
	s.t = nil
}

// useArenas gives every attending layer of kc the model's arena for its
// pages (cpudev.Arena) and a table in view, for an instance whose CPU blocks
// run composed.
func (kc *kvCache) useArenas(maxSeq int, arenaFor func(li, pp int, latent bool, minSlots int) (*cpudev.Arena, error), view *cpudev.View) error {
	for li := kc.lo; li < len(kc.layers); li++ {
		if err := kc.layers[li].useArena(li, maxSeq, arenaFor, view); err != nil {
			return err
		}
	}
	return nil
}
