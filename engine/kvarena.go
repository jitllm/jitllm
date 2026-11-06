package engine

import "github.com/jitllm/jitllm/engine/device/cpudev"

// kvArenaFor is the model's arena for layer li's pages of pp floats
// (cpudev.Arenas), made on the first ask: reserved for what the host's memory
// budget can hold of that layer and no fewer than minSlots pages.
func (s *Instance) kvArenaFor(li, pp int, latent bool, minSlots int) (*cpudev.Arena, error) {
	slots := max(minSlots, int(min(s.m.Gc.Budget()/uint64(4*pp), cpudev.ArenaElems/uint64(pp))))
	return s.eng.cpu.Arenas(s.m.Id).For(li, pp, latent, slots)
}

// useArena gives the layer the model's arena for its pages and a table for
// maxSeq positions (cpudev.Pages.UseArena). A layer with no history keeps
// none, and a layer whose pages are already held keeps them where they are.
func (s *kvPages) useArena(li, maxSeq int, arenaFor func(li, pp int, latent bool, minSlots int) (*cpudev.Arena, error)) error {
	if s.p == 0 || maxSeq <= 0 || s.mem.Table() != nil {
		return nil
	}
	for n := range s.mem.Len() {
		if s.mem.K(n) != nil {
			return nil
		}
	}
	slots := (maxSeq + s.p - 1) / s.p
	a, err := arenaFor(li, s.pp, s.latent, slots)
	if err != nil {
		return err
	}
	s.mem.UseArena(a, slots)
	return nil
}

// unmapArena lets every slot the layer holds go; no page of it may be read
// again. The arena is the model's and stays. A shared page's slot is the
// store's, never one the session lets go.
func (s *kvPages) unmapArena() {
	if s.mem.Table() == nil {
		return
	}
	for n := range s.mem.Len() {
		if !s.isShared(n) && s.mem.ArenaPage(n) {
			s.mem.Give(s.mem.K(n))
		}
	}
	s.mem.Unmap()
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
