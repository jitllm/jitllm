package gpu

import "fmt"

// Prefix sharing on the card: the engine's prefix cache (engine.SharedStore,
// kvCache) names a sealed page by every token up to its end, and two
// sequences whose prompts agree that far hold the same bytes. A layer's page
// the cache names (PublishKV) is held by every sequence that maps it
// (MapKV) through the sequences' own page tables -- the same id in each, a
// count of holders beside it -- so a warm prompt restores its prefix without a
// byte moving. Nothing else about the pool changes, and the kernels read a
// page wherever its id says.
//
//   - A named page is never written in place: a write into one copies that
//     page on the card first (cowPages), and the sequence's table takes the
//     copy.
//   - A sequence letting a named page go drops its hold (releaseID); the page
//     stays named, for the next prompt, until its room is wanted: then the
//     unheld named pages go first (reclaimNamed), before the layer grows.
//     Where the unheld named pages took the free pages the pool was grown
//     ahead by (PrewarmKV), a restore mapping a page and a session leaving a
//     layer grow it back, where the card has the room (keepAhead): a pool
//     sized to its sequences' pages otherwise gave the cache none, and a
//     restore's copy of the page it ends in took the page the next prompt
//     restores from. The growth is there, between steps, and never in one,
//     which takes a free page or reclaims and allocates nothing.
//   - Eviction sends a sequence's view of a named page to host memory and drops its
//     hold; compaction gives the unheld named pages back first and leaves a
//     layer with held ones as it is.

// sharedIDs is a layer's named pages: holders per id, and the id each name
// answers with.
type sharedIDs struct {
	holds map[uint32]int
	byKey map[string]uint32
}

// isNamed says id is a page the prefix cache names.
func (l *kvLayerPool) isNamed(id uint32) bool {
	_, ok := l.sh.holds[id]
	return ok
}

// releaseID drops one hold on id: a named page stays (unheld, until its room
// is wanted); any other is fenced as before.
func (l *kvLayerPool) releaseID(id uint32) {
	if id == 0 {
		return
	}
	if h, ok := l.sh.holds[id]; ok {
		if h > 0 {
			l.sh.holds[id] = h - 1
		}
		return
	}
	l.fenceID(id)
}

// forgetNamed takes id out of the names; it is the caller's to free.
func (l *kvLayerPool) forgetNamed(id uint32) {
	for k, v := range l.sh.byKey {
		if v == id {
			delete(l.sh.byKey, k)
		}
	}
	delete(l.sh.holds, id)
}

// reclaimNamed frees up to want unheld named pages (fenced, as a released
// page is), oldest name first being no better known than any: what a
// growth or an allocation needs before it pages anything out. It reports how
// many went.
func (l *kvLayerPool) reclaimNamed(want int) int {
	n := 0
	for id, h := range l.sh.holds {
		if n >= want {
			break
		}
		if h == 0 {
			l.forgetNamed(id)
			l.fenceID(id)
			n++
		}
	}
	if n > 0 {
		l.fence()
	}
	return n
}

// unheldNamed counts the layer's named pages no sequence holds: the prefix
// cache's, room a sequence may take back.
func (l *kvLayerPool) unheldNamed() int {
	n := 0
	for _, h := range l.sh.holds {
		if h == 0 {
			n++
		}
	}
	return n
}

// keepNamed grows layer l to n free pages where the card has the room
// without reclaiming anything for it -- the grown pair and a scratch as large
// as every one built so far fit the budget, and a streaming device keeps its
// slot (prewarmKV's test) -- so the prefix cache keeps the unheld pages a
// growth would otherwise reclaim. It reports whether l has n free pages.
// Callers hold g.mu, outside any submission.
func (g *devTier) keepNamed(kp *kvPool, l *kvLayerPool, n int) bool {
	if len(l.free) >= n {
		return true
	}
	if l.unheldNamed() == 0 {
		return false
	}
	need := l.n + n - len(l.free)
	for _, want := range []int{max(need, 2*l.n), need} {
		if want > kp.maxPages(l) || !g.room(kp.pageBytes(l)*uint64(want)+g.scratch) ||
			!g.keepsSlot(kp.pageBytes(l)*uint64(want-l.n)) {
			continue
		}
		if g.growKVLayer(kp, l, want) == nil {
			return len(l.free) >= n
		}
	}
	return false
}

// keepAhead grows layer l back to the free pages PrewarmKV grew it ahead
// by, where the prefix cache's unheld pages took them (keepNamed). Callers
// hold g.mu, between steps.
func (g *devTier) keepAhead(kp *kvPool, l *kvLayerPool) {
	if g.kvAhead > 0 && kp != nil && kp.p > 0 {
		g.keepNamed(kp, l, (g.kvAhead+kp.p-1)/kp.p)
	}
}

// heldNamed says some named page of the layer is held by a sequence.
func (l *kvLayerPool) heldNamed() bool {
	for _, h := range l.sh.holds {
		if h > 0 {
			return true
		}
	}
	return false
}

// publishKV names sequence s's page j of layer li by key, held by s. A key
// already naming another page is left as it is. Callers hold g.mu.
func (g *devTier) publishKV(kp *kvPool, l *kvLayerPool, s seqID, j int, key string) bool {
	if j >= len(l.owned[s]) {
		return false
	}
	id := l.owned[s][j]
	if id == 0 || key == "" {
		return false
	}
	if _, ok := l.sh.byKey[key]; ok {
		return true
	}
	if l.sh.holds == nil {
		l.sh.holds, l.sh.byKey = map[uint32]int{}, map[string]uint32{}
	}
	if _, ok := l.sh.holds[id]; !ok {
		l.sh.holds[id] = 1 // s holds it
	}
	l.sh.byKey[key] = id
	return true
}

// mapKV puts the page key names into sequence s's page j of layer l, held by
// s: the sequence's run is extended to j and an id it held there before is
// let go. False when no page answers to key. Callers hold g.mu.
func (g *devTier) mapKV(kp *kvPool, l *kvLayerPool, s seqID, j int, key string) (bool, error) {
	id, ok := l.sh.byKey[key]
	if !ok {
		return false, nil
	}
	if err := g.ensureRange(kp, s, j+1); err != nil {
		return false, err
	}
	for len(l.owned[s]) <= j {
		l.owned[s] = append(l.owned[s], 0)
	}
	if old := l.owned[s][j]; old != id {
		l.releaseID(old)
		l.sh.holds[id]++
		l.owned[s][j] = id
	}
	g.tabPend = append(g.tabPend, tabWrite{l.tab, kp.ranges[s].off + j, l.owned[s][j : j+1]})
	g.KVPagesMapped++
	return true, nil
}

// cowPages gives sequence s a page of its own for every named page of l it
// is about to write, positions [start, end): the page copied on the card,
// the sequence's table pointing at the copy, its hold on the named one
// dropped. Callers hold g.mu, outside any submission.
func (g *devTier) cowPages(sid uint64, kp *kvPool, l *kvLayerPool, s seqID, start, end int) error {
	if len(l.sh.holds) == 0 || start < 0 || g.KVCowFault {
		return nil
	}
	P := kp.p
	ids := l.owned[s]
	for j := start / P; j < len(ids) && j*P < end; j++ {
		id := ids[j]
		if id == 0 || !l.isNamed(id) {
			continue
		}
		// A page wholly past start holds nothing the sequence keeps: a fresh
		// one, not a copy.
		fresh := j*P >= start
		if len(l.free) < 1 {
			l.fence()
		}
		if len(l.free) < 1 && l.reclaimNamed(1) == 0 {
			if err := g.growFree(kp, l, 1); err != nil && !g.evictForRoom(sid, kp, l, 1) {
				return err
			}
		}
		if len(l.free) < 1 {
			return fmt.Errorf("%w: no page to copy a shared page into", ErrKVCapacity)
		}
		nid := l.free[len(l.free)-1]
		l.free = l.free[:len(l.free)-1]
		kw, vw := l.geom.kWords(P), l.geom.vWords(P)
		if !fresh {
			if err := g.dev.Copy(l.k, int(nid)*kw*4, l.k, int(id)*kw*4, kw*4); err != nil {
				return err
			}
			if vw > 0 {
				if err := g.dev.Copy(l.v, int(nid)*vw*4, l.v, int(id)*vw*4, vw*4); err != nil {
					return err
				}
			}
			g.KVPagesCopied++
		}
		l.releaseID(id)
		ids[j] = nid
		g.tabPend = append(g.tabPend, tabWrite{l.tab, kp.ranges[s].off + j, ids[j : j+1]})
	}
	return nil
}

// PublishKV is publishKV for the zero session (nn.KVShareDevice).
func (g *devTier) PublishKV(li, page int, key string) bool { return g.publishKVs(0, li, page, key) }

// MapKV is mapKV for the zero session (nn.KVShareDevice).
func (g *devTier) MapKV(li, page int, key string, end int) bool {
	return g.mapKVs(0, li, page, key, end)
}

// SharePage is the zero session's page of block li (nn.KVShareDevice).
func (g *devTier) SharePage(li int) int { return g.kvPageOf(0, li) }

// kvPageOf is the positions a page of block li's pool holds for session sid,
// 0 where its pages are not named.
func (g *devTier) kvPageOf(sid uint64, li int) int {
	sv := g.as(sid)
	if sv == nil {
		return 0
	}
	defer sv.done()
	g = sv
	g.mu.Lock()
	defer g.mu.Unlock()
	if kp, l := g.sharedLayer(li); l != nil {
		return kp.p
	}
	return 0
}

// PublishKVs is publishKV for the zero session over many names.
func (g *devTier) PublishKVs(lis, pages []int32, keys []string) bool {
	return g.publishKVsBatch(0, lis, pages, keys)
}

// publishKVsBatch names session sid's pages, name i block lis[i]'s page
// pages[i] by keys[i], under one lock: a prompt's names in one call.
func (g *devTier) publishKVsBatch(sid uint64, lis, pages []int32, keys []string) bool {
	sv := g.as(sid)
	if sv == nil {
		return false
	}
	defer sv.done()
	g = sv
	g.mu.Lock()
	defer g.mu.Unlock()
	ok := true
	for i := range lis {
		kp, l := g.sharedLayer(int(lis[i]))
		if l == nil || !g.publishKV(kp, l, seqID{sid, 0}, int(pages[i]), keys[i]) {
			ok = false
		}
	}
	return ok
}

// sharedLayer is block li's pool at the type this view's session holds it
// as, when its pages can be named: paged. A name is of one type's pages,
// so a prefix is held once per device per type. Callers hold g.mu.
func (g *devTier) sharedLayer(li int) (*kvPool, *kvLayerPool) {
	kp := g.kvp
	if kp == nil {
		return nil, nil
	}
	l := kp.layers[g.viewKey(li)]
	if l == nil {
		return nil, nil
	}
	return kp, l
}

// publishKVs names session sid's page of block li by key.
func (g *devTier) publishKVs(sid uint64, li, page int, key string) bool {
	sv := g.as(sid)
	if sv == nil {
		return false
	}
	defer sv.done()
	g = sv
	g.mu.Lock()
	defer g.mu.Unlock()
	kp, l := g.sharedLayer(li)
	if l == nil {
		return false
	}
	return g.publishKV(kp, l, seqID{sid, 0}, page, key)
}

// mapKVs puts the page key names into session sid's page of block li, the
// sequence then written to end at least.
func (g *devTier) mapKVs(sid uint64, li, page int, key string, end int) bool {
	sv := g.as(sid)
	if sv == nil {
		return false
	}
	defer sv.done()
	g = sv
	g.mu.Lock()
	defer g.mu.Unlock()
	kp, l := g.sharedLayer(li)
	if l == nil {
		return false
	}
	s := seqID{sid, 0}
	ok, err := g.mapKV(kp, l, s, page, key)
	if err != nil {
		g.fail(fmt.Sprintf("MapKV: block %d: %v", li, err))
		return false
	}
	if ok {
		if kp.written == nil {
			kp.written = map[seqID]int{}
		}
		kp.written[s] = max(kp.written[s], end)
		g.keepAhead(kp, l)
	}
	return ok
}

// namesPages reports that a layer's pool names a page for the prefix cache.
func (g *devTier) namesPages() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed || g.kvp == nil {
		return false
	}
	for _, l := range g.kvp.layers {
		if l != nil && len(l.sh.holds) > 0 {
			return true
		}
	}
	return false
}
