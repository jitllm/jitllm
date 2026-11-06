package engine

import (
	"github.com/jitllm/jitllm/nn/history"
)

// giveBack hands every page of kc to pool (history.PagePool), for the next
// session's history to grow into; kc is not used again. A shared page is the
// store's: its reference goes, never the memory, and the arena's slots go
// back to the arena.
func (kc *kvCache) giveBack() {
	pool := kc.pool
	if pool == nil || kc == nil {
		return
	}
	kc.unshareAll()
	pages := make([]*history.Pages, 0, len(kc.layers)-kc.lo)
	for li := kc.lo; li < len(kc.layers); li++ {
		s := &kc.layers[li]
		s.unmapArena()
		pages = append(pages, s.mem)
	}
	pool.Put(pages...)
}
