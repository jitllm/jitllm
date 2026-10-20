package maintier

import (
	"fmt"
	"unsafe"
)

// Eviction: a pool that cannot grow sends pages home rather than refusing
// (docs/design/device-kv-paging.md, "Errors and eviction").
//
// A sequence's pages leave oldest first, the same page in every layer that
// holds the sequence, so its evicted prefix is one count (kvPool.evicted) and
// one set of descriptors still serves every layer. An evicted page's id in the
// sequence's run is 0 -- the dummy -- and its bytes are the layer's home copy.
// Attention over a history with an evicted prefix streams (pagedstream.go):
// a pass uploads as many evicted pages as the layer has free ids into them,
// runs over them, and the passes' partials fold into one. A pass costs a Sync,
// so its width is every page nobody holds at that moment -- memory that would
// otherwise sit idle -- rather than a fixed count, and Config.KVStreamPages
// forces it (SetKVStreamPages at run time). Eviction keeps one id free beyond
// the call's own growth: the fewest a pass can make progress with. Only a pool
// too small for that -- the dummy and the page being written -- refuses, and
// relocation answers the refusal as it answers any.
//
// Residency is planned by the host, which evicted the pages and knows which
// they are, rather than faulted from inside a kernel: every attention layer
// reads its whole history, so there is no sparse read for a fault to save. The
// in-kernel fault the design describes is what a windowed or sparse read wants,
// and is not built.

// wrote records the positions a completed submission's rows wrote.
func (kp *kvPool) wrote(rows []pagedRow) {
	for _, r := range rows {
		if r.pad {
			continue
		}
		if kp.written == nil {
			kp.written = map[seqID]int{}
		}
		kp.written[r.s] = max(kp.written[r.s], r.pos+1)
	}
}

// evictable is how many more of s's pages can leave in layer l: those wholly
// below what s has written. A full page is never written again, and the page a
// call writes into is not full, so no write lands on a page at home.
func (kp *kvPool) evictable(l *kvLayerPool, s seqID) int {
	if l.noEvict {
		return 0
	}
	return min(len(l.owned[s]), kp.written[s]/kp.p) - kp.evicted[s]
}

// evictVictim is the sequence whose oldest resident page goes next: another
// session's first -- its history is cold for this call -- then the current
// session's, and only one with a page to spare in every layer that holds it.
// A session stepping on the device keeps its pages (stepping). Only a session
// of one sequence gives pages up: a batch's rows run together
// and one row cannot stream while the others read their pages in place.
// ok is false when nothing can leave.
func (g *devTier) evictVictim(sid uint64, kp *kvPool) (seqID, bool) {
	perSess := map[uint64]int{}
	for s := range kp.ranges {
		perSess[s.sid]++
	}
	var best seqID
	found, bestCur := false, true
	for s := range kp.ranges {
		if perSess[s.sid] != 1 {
			continue
		}
		spare, held := true, false
		for _, l := range kp.layers {
			if len(l.owned[s]) == 0 {
				continue
			}
			held = true
			if kp.evictable(l, s) < 1 {
				spare = false
				break
			}
		}
		if !held || !spare {
			continue
		}
		cur := s.sid == sid
		// Deterministic across map order: other sessions first, then the
		// lowest session id.
		if !found || bestCur && !cur || cur == bestCur && s.sid < best.sid {
			best, found, bestCur = s, true, cur
		}
	}
	return best, found
}

// stepping reports that session other is in a call on this device -- its
// pages taken and its table staged for a submission that has not completed --
// and is not self, whose own call is the one asking. Its pages are not
// evicted under it: the submission would read the dummy page where its
// history was. Callers hold g.mu.
func (g *devTier) stepping(other, self uint64) bool {
	if other == self {
		return false
	}
	ds := g.sess[other]
	return ds != nil && ds.cur != nil
}

// evictableIn is how many pages eviction could free in layer l: every page
// of every sequence evictVictim may choose, counted where it is spare in
// every layer that holds it.
func (g *devTier) evictableIn(sid uint64, kp *kvPool, l *kvLayerPool) int {
	perSess := map[uint64]int{}
	for s := range kp.ranges {
		perSess[s.sid]++
	}
	n := 0
	for s := range kp.ranges {
		if perSess[s.sid] != 1 || len(l.owned[s]) == 0 {
			continue
		}
		spare := kp.evictable(l, s)
		for _, x := range kp.layers {
			if len(x.owned[s]) > 0 {
				spare = min(spare, kp.evictable(x, s))
			}
		}
		// Only a page with an id frees one here: a windowed layer's pages
		// released behind its window go home as nothing.
		e := kp.evicted[s]
		for j := e; j < e+spare; j++ {
			if l.owned[s][j] != 0 {
				n++
			}
		}
	}
	return n
}

// evictPage sends s's oldest resident page home in every layer that holds s,
// freeing one id in each. Callers hold g.mu, between submissions: it waits for
// every submission in flight first (s may be another session's, stepping),
// so nothing reads the page and its id is free at once.
func (g *devTier) evictPage(kp *kvPool, s seqID) error {
	g.quiesce()
	if err := g.flushTabs(); err != nil {
		return err
	}
	j := kp.evicted[s]
	off := kp.ranges[s].off
	for li, l := range kp.layers {
		if len(l.owned[s]) <= j {
			continue
		}
		id := l.owned[s][j]
		if l.home == nil {
			l.home = map[seqID][][]float32{}
		}
		if id == 0 {
			// Released behind a windowed layer's window: there is nothing to
			// send, and nothing to read back.
			l.home[s] = append(l.home[s], nil)
			continue
		}
		kw, vw := l.geom.kWords(kp.p), l.geom.vWords(kp.p)
		page, err := g.readPages(l, l.k, []uint32{id}, kw)
		if err != nil {
			return fmt.Errorf("evicting block %d's page %d: %w", li, j, err)
		}
		if vw > 0 {
			vp, err := g.readPages(l, l.v, []uint32{id}, vw)
			if err != nil {
				return fmt.Errorf("evicting block %d's page %d: %w", li, j, err)
			}
			page = append(page, vp...)
		}
		l.home[s] = append(l.home[s], page)
		l.owned[s][j] = 0
		if l.isNamed(id) {
			l.releaseID(id) // others may hold it: this sequence's view went home
		} else {
			l.free = append(l.free, id)
		}
		if err := l.tab.WriteAt((off+j)*4, u32view([]uint32{0})); err != nil {
			return err
		}
	}
	if kp.evicted == nil {
		kp.evicted = map[seqID]int{}
	}
	kp.evicted[s]++
	g.KVEvictions++
	return nil
}

// evictForRoom sends pages home until layer l has want free ids and one more
// to stream through, and reports whether it did. All or nothing: a shortfall
// sends nothing home, since pages at home with no room left to stream them
// back would refuse every later call rather than this one.
// Callers hold g.mu, between submissions.
func (g *devTier) evictForRoom(sid uint64, kp *kvPool, l *kvLayerPool, want int) bool {
	short := want + 1 - len(l.free)
	if short <= 0 {
		return true
	}
	if g.evictableIn(sid, kp, l) < short {
		return false
	}
	for len(l.free) < want+1 {
		s, ok := g.evictVictim(sid, kp)
		if !ok {
			return false
		}
		if err := g.evictPage(kp, s); err != nil {
			g.LastErr = err.Error()
			return false
		}
	}
	return true
}

// streamRoom makes sure every layer the session holds keeps a free id once a
// call's pages are taken, when anything is at home: streaming needs one.
func (g *devTier) streamRoom(sid uint64) error {
	kp := g.kvPages()
	if len(kp.evicted) == 0 {
		return nil
	}
	return g.heldLayers(sid, func(li int, l *kvLayerPool) error {
		if len(l.free) == 0 && !g.evictForRoom(sid, kp, l, 0) {
			return fmt.Errorf("%w: block %d has no free page to stream evicted history through", ErrKVCapacity, li)
		}
		return nil
	})
}

// streamWidth is how many evicted pages one pass over layer l uploads, of
// left still to stream: every free id, or Config.KVStreamPages when forced.
func (g *devTier) streamWidth(l *kvLayerPool, left int) int {
	w := len(l.free)
	if f := g.KVStreamPages; f > 0 && f < w {
		w = f
	}
	return min(w, left)
}

// rewind tells the pool that sequence s's history now ends at pos: the
// sequence restarted lower (a State reset and taking a new prompt, a row
// retired and reused, a rollback), and its positions from pos on are about to
// be written again. Pages at home past pos are not its history any more, and
// a page a call writes into must be on the card, so the evicted prefix comes
// down to the pages wholly below pos: the page pos lies inside, when it is at
// home, comes back to the card with its home copy (its rows below pos are
// still the sequence's), and every page past it is released, as trimSeq
// releases them, for the call to take fresh. The written mark comes down to
// pos, so no page about to be written counts as full and goes home.
//
// Left as it was -- the count and the mark were only ever raised -- the rows
// written into a page at home landed on the dummy, the stream uploaded the old
// copy over them, and a layer joining later (pagedMigrate) took a shorter
// prefix than the count every other layer streams: the index past a layer's
// home copies in streamDecode.
//
// All or nothing: the one id per layer the page pos lies inside needs is
// taken first and given back if any layer cannot have it, so a refused call
// finds the pool as it was and rewinds again when retried. Callers hold g.mu,
// between submissions.
func (g *devTier) rewind(sid uint64, kp *kvPool, s seqID, pos int) error {
	e, keep := kp.evicted[s], pos/kp.p
	if e <= keep {
		if kp.written[s] > pos {
			kp.written[s] = pos
		}
		return nil
	}
	if err := g.flushTabs(); err != nil {
		return err
	}
	// Nothing of s is evictable while ids are found below.
	kp.written[s] = pos
	n := keep
	if pos%kp.p != 0 {
		n = keep + 1
		var took []*kvLayerPool
		for li, l := range kp.layers {
			// A windowed layer's page released behind its window has no copy;
			// the call's windowPages gives the page it writes a fresh id.
			if len(l.owned[s]) <= keep || l.home[s][keep] == nil {
				continue
			}
			if err := g.allocAt(sid, kp, l, s, keep); err != nil {
				for _, t := range took {
					t.free = append(t.free, t.owned[s][keep])
					t.owned[s][keep] = 0
				}
				// Only the ids just taken are queued (flushed above).
				clear(g.tabPend)
				g.tabPend = g.tabPend[:0]
				return fmt.Errorf("block %d: bringing page %d's home copy back for a rewind to %d: %w", li, keep, pos, err)
			}
			took = append(took, l)
		}
		for _, l := range took {
			part := l.home[s][keep]
			kw, vw := l.geom.kWords(kp.p), l.geom.vWords(kp.p)
			id := int(l.owned[s][keep])
			if err := l.k.WriteAt(id*kw*4, f32b(part[:kw])); err != nil {
				return err
			}
			if vw > 0 {
				if err := l.v.WriteAt(id*vw*4, f32b(part[kw:])); err != nil {
					return err
				}
			}
		}
	}
	for _, l := range kp.layers {
		if ids := l.owned[s]; len(ids) > n {
			l.fenceIDs(ids[n:])
			l.owned[s] = ids[:n]
		}
		if len(l.home[s]) > keep {
			l.home[s] = l.home[s][:keep]
		}
		if l.rel[s] > n {
			l.rel[s] = n
		}
	}
	kp.evicted[s] = keep
	g.roomGen.Add(1) // released pages are room (kvCompact), as trimKV's are
	return g.flushTabs()
}

// forgetFrom drops s's history from page keep on: evicted pages go from home
// and the counts shrink to match. Callers hold g.mu.
func (kp *kvPool) forgetFrom(s seqID, keep int) {
	for _, l := range kp.layers {
		if h := l.home[s]; len(h) > keep {
			if keep == 0 {
				delete(l.home, s)
			} else {
				l.home[s] = h[:keep]
			}
		}
	}
	if kp.evicted[s] > keep {
		kp.evicted[s] = keep
	}
	if kp.written[s] > keep*kp.p {
		kp.written[s] = keep * kp.p
	}
	if keep == 0 {
		delete(kp.evicted, s)
		delete(kp.written, s)
	}
}

// residentIDs appends ids' resident entries -- every id but 0, which marks a
// page at home -- to dst.
func residentIDs(dst, ids []uint32) []uint32 {
	for _, id := range ids {
		if id != 0 {
			dst = append(dst, id)
		}
	}
	return dst
}

// f32of views b as float32 words.
func f32of(b []byte) []float32 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
}

// SetKVStreamPages sets, at run time, how many evicted pages one pass over
// evicted history uploads (Config.KVStreamPages; 0 is every free page). It is
// a goal rather than an interruption: a submission already running keeps the
// width it was prepared with, and each device adopts the new one at its next
// call's preparation -- the first point at which no pass is in flight.
func (g *GPU) SetKVStreamPages(n int) {
	for _, d := range g.devs {
		d.mu.Lock()
		d.KVStreamPages = max(n, 0)
		d.mu.Unlock()
	}
}
