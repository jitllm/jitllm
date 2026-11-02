package model

import (
	"sync"
	"unsafe"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// kvArena is one layer's history pages as one span of memory outside the Go
// heap: what a composed CPU block's attention addresses as a pool, through a
// table of page slots, as a card's pool is (jit.History). The model holds one
// for each layer and page size (Model.kvArenaFor), so every session's pages and
// the shared store's are slots of it, and a prefix shared between sessions is
// one slot every holder's table names (kvPages.table). It is reserved once
// and committed by the pages written: a slot let go is handed back to the OS
// (kernels.ReleaseOffHeap), so a windowed layer still holds its window. A
// slot is held by the instance that grew it and by each store entry that took
// it (SharedStore.publish), and goes back to the free list when the last lets
// go. The spare list and the pool of closed instances stay outside it; a page
// outside it is not one a composed block can address (cpuReadable).
type kvArena struct {
	kb, vb []byte
	k, v   []float32
	pp     int
	slots  int
	mu     sync.Mutex
	holds  []int32
	free   []int32
	next   int
	// live is the slots held; closed says the model has let the arena go,
	// so the last slot let go after it unmaps it (a store may outlive it).
	live   int
	closed bool
}

// mapKVArena is an arena of slots pages of pp floats each, its keys and its
// values -- one span where latent, MLA's value the key's prefix.
func mapKVArena(pp, slots int, latent bool) (*kvArena, error) {
	n := 4 * pp * slots
	a := &kvArena{pp: pp, slots: slots}
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

// take is a free slot, held once, or false where every slot is taken.
func (a *kvArena) take() (int, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	i := -1
	if n := len(a.free); n > 0 {
		i, a.free = int(a.free[n-1]), a.free[:n-1]
	} else if a.next < a.slots {
		i = a.next
		a.next++
	}
	if i < 0 {
		return 0, false
	}
	if len(a.holds) < a.slots {
		a.holds = append(a.holds, make([]int32, a.slots-len(a.holds))...)
	}
	a.holds[i] = 1
	a.live++
	return i, true
}

// hold adds a holder to slot i.
func (a *kvArena) hold(i int) {
	a.mu.Lock()
	a.holds[i]++
	a.mu.Unlock()
}

// drop lets one holder of slot i go; the last hands its memory back and the
// slot to the free list.
func (a *kvArena) drop(i int) {
	a.mu.Lock()
	a.holds[i]--
	last := a.holds[i] == 0
	if last {
		a.free = append(a.free, int32(i))
		a.live--
	}
	gone := last && a.closed && a.live == 0
	a.mu.Unlock()
	switch {
	case gone:
		a.unmap()
		return
	case !last:
		return
	}
	k, v := a.page(i)
	kernels.ReleaseOffHeap(unsafe.Slice((*byte)(unsafe.Pointer(&k[0])), 4*len(k)))
	if &a.vb[0] != &a.kb[0] {
		kernels.ReleaseOffHeap(unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), 4*len(v)))
	}
}

// page is slot i's key and value.
func (a *kvArena) page(i int) (k, v []float32) {
	lo, hi := i*a.pp, (i+1)*a.pp
	return a.k[lo:hi:hi], a.v[lo:hi:hi]
}

// slotOf is the slot b starts, the key's span or the value's, or false where
// b is not one of a's pages.
func (a *kvArena) slotOf(b []float32) (int, bool) {
	if len(b) == 0 {
		return 0, false
	}
	p := uintptr(unsafe.Pointer(&b[0]))
	for _, base := range [2]uintptr{uintptr(unsafe.Pointer(&a.k[0])), uintptr(unsafe.Pointer(&a.v[0]))} {
		if p >= base && p < base+uintptr(4*len(a.k)) {
			return int(p-base) / (4 * a.pp), true
		}
	}
	return 0, false
}

// kvArenaElems bounds an arena's span in floats: a composed block addresses
// it by an unsigned 32-bit index, kept below 2^31.
const kvArenaElems = 1 << 31

// kvArenaKey names a model's arena: a layer, its page's floats, and whether
// the value is the key's memory.
type kvArenaKey struct {
	li, pp int
	latent bool
}

// kvArenaFor is the model's arena for layer li's pages of pp floats, made
// on the first ask: reserved for what the host's memory budget can hold of
// that layer and no fewer than minSlots pages.
func (m *Model) kvArenaFor(li, pp int, latent bool, minSlots int) (*kvArena, error) {
	m.kvArenaMu.Lock()
	defer m.kvArenaMu.Unlock()
	k := kvArenaKey{li, pp, latent}
	if a := m.kvArenas[k]; a != nil {
		return a, nil
	}
	slots := max(minSlots, int(min(sched.MemBudget()/uint64(4*pp), kvArenaElems/uint64(pp))))
	a, err := mapKVArena(pp, slots, latent)
	if err != nil {
		return nil, err
	}
	if m.kvArenas == nil {
		m.kvArenas = map[kvArenaKey]*kvArena{}
	}
	m.kvArenas[k] = a
	return a, nil
}

// unmapKVArenas lets the model's arenas go, for Close: each is returned now,
// or when the last slot a store still holds is let go.
func (m *Model) unmapKVArenas() {
	m.kvArenaMu.Lock()
	defer m.kvArenaMu.Unlock()
	for k, a := range m.kvArenas {
		a.close()
		delete(m.kvArenas, k)
	}
}

// close marks the arena let go by its model, and returns it where no slot is
// held.
func (a *kvArena) close() {
	a.mu.Lock()
	a.closed = true
	gone := a.live == 0
	a.mu.Unlock()
	if gone {
		a.unmap()
	}
}

// useArena gives the layer the model's arena for its pages, the instance's
// table sized for maxSeq positions. A layer with no history keeps none, and
// a layer whose pages are already held keeps them where they are.
func (s *kvPages) useArena(li, maxSeq int, arenaFor func(li, pp int, latent bool, minSlots int) (*kvArena, error)) error {
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
	i, ok := s.arena.take()
	if !ok {
		return nil, nil
	}
	return s.arena.page(i)
}

// inArena says b is one of the arena's slots.
func (s *kvPages) inArena(_ int, b []float32) bool {
	if s.arena == nil {
		return false
	}
	_, ok := s.arena.slotOf(b)
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
	if i, ok := s.arena.slotOf(s.k[n]); ok {
		s.arena.drop(i)
	}
}

// giveSlot lets the layer's hold on the slot b starts go, where b is one.
func (s *kvPages) giveSlot(b []float32) {
	if s.arena == nil {
		return
	}
	if i, ok := s.arena.slotOf(b); ok {
		s.arena.drop(i)
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
			if si, ok := s.arena.slotOf(s.k[j]); ok {
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
// pages (kvArena), for an instance whose CPU blocks run composed.
func (kc *kvCache) useArenas(maxSeq int, arenaFor func(li, pp int, latent bool, minSlots int) (*kvArena, error)) error {
	for li := kc.lo; li < len(kc.layers); li++ {
		if err := kc.layers[li].useArena(li, maxSeq, arenaFor); err != nil {
			return err
		}
	}
	return nil
}
