package cpudev

import (
	"sync"
	"unsafe"

	"github.com/jitllm/jitllm/internal/offheap"
)

// Arena is one block's history pages as one span of memory outside the Go
// heap: what a composed CPU block's attention addresses as a pool, through a
// table of page slots, as a card's pool is (composer.History). A model holds
// one for each block and page size (Arenas), so every session's pages and the
// shared store's are slots of it, and a prefix shared between sessions is one
// slot every holder's table names. It is reserved once and committed by the
// pages written: a slot let go is handed back to the OS
// (offheap.ReleaseOffHeap), so a windowed block still holds its window. A
// slot is held by the instance that grew it and by each store entry that took
// it, and goes back to the free list when the last lets go.
type Arena struct {
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

// ArenaElems bounds an arena's span in floats: a composed block addresses
// it by an unsigned 32-bit index, kept below 2^31.
const ArenaElems = 1 << 31

// MapArena is an arena of slots pages of pp floats each, its keys and its
// values -- one span where latent, MLA's value the key's prefix.
func MapArena(pp, slots int, latent bool) (*Arena, error) {
	n := 4 * pp * slots
	a := &Arena{pp: pp, slots: slots}
	var err error
	if a.kb, _, err = offheap.MapOffHeap(n); err != nil {
		return nil, err
	}
	a.k = unsafe.Slice((*float32)(unsafe.Pointer(&a.kb[0])), pp*slots)
	if latent {
		a.vb, a.v = a.kb, a.k
		return a, nil
	}
	if a.vb, _, err = offheap.MapOffHeap(n); err != nil {
		offheap.UnmapOffHeap(a.kb)
		return nil, err
	}
	a.v = unsafe.Slice((*float32)(unsafe.Pointer(&a.vb[0])), pp*slots)
	return a, nil
}

// Keys and Values are the arena's two spans, the same one where latent.
func (a *Arena) Keys() []float32   { return a.k }
func (a *Arena) Values() []float32 { return a.v }

// Slots is how many pages the arena holds.
func (a *Arena) Slots() int { return a.slots }

// unmap returns the arena's memory, a's two spans once each.
func (a *Arena) unmap() {
	offheap.UnmapOffHeap(a.kb)
	if &a.vb[0] != &a.kb[0] {
		offheap.UnmapOffHeap(a.vb)
	}
}

// Take is a free slot, held once, or false where every slot is taken.
func (a *Arena) Take() (int, bool) {
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

// Hold adds a holder to slot i.
func (a *Arena) Hold(i int) {
	a.mu.Lock()
	a.holds[i]++
	a.mu.Unlock()
}

// Drop lets one holder of slot i go; the last hands its memory back and the
// slot to the free list.
func (a *Arena) Drop(i int) {
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
	k, v := a.Page(i)
	offheap.ReleaseOffHeap(unsafe.Slice((*byte)(unsafe.Pointer(&k[0])), 4*len(k)))
	if &a.vb[0] != &a.kb[0] {
		offheap.ReleaseOffHeap(unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), 4*len(v)))
	}
}

// Page is slot i's key and value.
func (a *Arena) Page(i int) (k, v []float32) {
	lo, hi := i*a.pp, (i+1)*a.pp
	return a.k[lo:hi:hi], a.v[lo:hi:hi]
}

// SlotOf is the slot b starts, the key's span or the value's, or false where
// b is not one of a's pages.
func (a *Arena) SlotOf(b []float32) (int, bool) {
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

// close marks the arena let go by its model, and returns it where no slot is
// held.
func (a *Arena) close() {
	a.mu.Lock()
	a.closed = true
	gone := a.live == 0
	a.mu.Unlock()
	if gone {
		a.unmap()
	}
}

// Arenas is a model's arenas: a block's for each page size, and whether its
// value is the key's memory.
type Arenas struct {
	mu sync.Mutex
	m  map[arenaKey]*Arena
}

type arenaKey struct {
	li, pp int
	latent bool
}

// For is block li's arena for pages of pp floats, made on the first ask with
// slots pages.
func (as *Arenas) For(li, pp int, latent bool, slots int) (*Arena, error) {
	as.mu.Lock()
	defer as.mu.Unlock()
	k := arenaKey{li, pp, latent}
	if a := as.m[k]; a != nil {
		return a, nil
	}
	a, err := MapArena(pp, slots, latent)
	if err != nil {
		return nil, err
	}
	if as.m == nil {
		as.m = map[arenaKey]*Arena{}
	}
	as.m[k] = a
	return a, nil
}

// Close lets every arena go, for the model's Close: each is returned now, or
// when the last slot a store still holds is let go.
func (as *Arenas) Close() {
	as.mu.Lock()
	defer as.mu.Unlock()
	for k, a := range as.m {
		a.close()
		delete(as.m, k)
	}
}
