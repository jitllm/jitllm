package maintier

import "github.com/jitllm/jitllm/jit/gpu/backend"

// A hybrid's restore points on the card: the prefix cache keeps a linear
// block's recurrent summary at the positions a prompt may resume from
// (model's recpoint.go), named like the pages beside it. A block placed here
// keeps them here, where its state lives: SnapRec copies the session's state
// into a snapshot on the card, LoadRec copies one back into the session's
// seat. Nothing crosses the bus. The snapshots of a block are a bounded set,
// the oldest reused for the next (recSnapCap), freed with the block.

// recSnapKey names one block's snapshot.
type recSnapKey struct {
	li  int
	key string
}

// recSnap is one snapshot: the convolution window and the delta state of one
// sequence, recC and recS floats.
type recSnap struct {
	c, s backend.Buf
	key  recSnapKey
}

// recSnapCap is how many snapshots a block keeps: as many as fit 32 MiB, at
// least 4 and at most 64.
func (g *devTier) recSnapCap() int {
	b := (g.recC + g.recS) * 4
	if b <= 0 {
		return 0
	}
	return min(max(32<<20/b, 4), 64)
}

// recSeatOf is session sid's seat in block li's pools, one row: the pair and
// the seat. False when the block keeps no recurrent state for sid here.
func (g *devTier) recSeatOf(sid uint64, li int) (*layer, *recPair, recSeat, bool) {
	l := g.layers[li]
	rp := l.recOf(sid)
	st, ok := g.recSeats[sid]
	if l == nil || rp == nil || !ok || st.rows != 1 || l.pool == nil || g.recC == 0 {
		return nil, nil, recSeat{}, false
	}
	return l, rp, st, true
}

// snapRec copies session sid's state on block li into the snapshot named key.
// Callers hold g.mu, outside any submission.
func (g *devTier) snapRec(sid uint64, li int, key string) bool {
	l, rp, st, ok := g.recSeatOf(sid, li)
	if !ok {
		return false
	}
	k := recSnapKey{li, key}
	sn := g.recSnaps[k]
	if sn == nil {
		if sn = g.recSnapSlot(li); sn == nil {
			return false
		}
		sn.key = k
		g.recSnaps[k] = sn
		g.recSnapOrder = append(g.recSnapOrder, k)
	}
	sh := rp.cur
	if g.recShared {
		sh = 0
	}
	// Queued, and run with the rest of the point's blocks in one call
	// (flushSnaps).
	g.snapList = append(g.snapList, backend.CopyRange{Dst: sn.c, Src: l.pool.c[rp.cur], SrcOff: st.base * g.recC * 4, N: g.recC * 4})
	if g.recS > 0 {
		g.snapList = append(g.snapList, backend.CopyRange{Dst: sn.s, Src: l.pool.s[sh], SrcOff: st.base * g.recS * 4, N: g.recS * 4})
	}
	g.snapCount++
	return true
}

// flushSnaps runs the copies snapRec queued, as one list (backend.CopyAll),
// and counts the points they kept: the blocks queued before a refusal are
// kept still. It reports ok, false too when the list fails.
func (g *devTier) flushSnaps(ok bool) bool {
	if len(g.snapList) > 0 {
		if err := backend.CopyAll(g.dev, g.snapList); err != nil {
			g.LastErr = "SnapRec: " + err.Error()
			ok = false
		} else {
			g.RecPointsKept += g.snapCount
		}
	}
	clear(g.snapList)
	g.snapList, g.snapCount = g.snapList[:0], 0
	return ok
}

// recSnapSlot is a snapshot buffer for block li: a new one while the block
// keeps fewer than recSnapCap, else its oldest, forgotten. Callers hold g.mu.
func (g *devTier) recSnapSlot(li int) *recSnap {
	if g.recSnaps == nil {
		g.recSnaps = map[recSnapKey]*recSnap{}
	}
	n := 0
	for _, k := range g.recSnapOrder {
		if k.li == li {
			n++
		}
	}
	if n >= g.recSnapCap() {
		for i, k := range g.recSnapOrder {
			if k.li != li {
				continue
			}
			sn := g.recSnaps[k]
			delete(g.recSnaps, k)
			g.recSnapOrder = append(g.recSnapOrder[:i], g.recSnapOrder[i+1:]...)
			return sn
		}
	}
	c, err := g.dev.Alloc(max(g.recC, 1) * 4)
	if err != nil {
		return nil
	}
	sn := &recSnap{c: c}
	if g.recS > 0 {
		if sn.s, err = g.dev.Alloc(g.recS * 4); err != nil {
			c.Free()
			return nil
		}
	}
	g.charge(g.recSnapBytes())
	return sn
}

// recSnapBytes is one snapshot's charge.
func (g *devTier) recSnapBytes() uint64 { return uint64(max(g.recC, 1)+g.recS) * 4 }

// hasRec says block li keeps the snapshot named key. Callers hold g.mu.
func (g *devTier) hasRec(li int, key string) bool {
	_, ok := g.recSnaps[recSnapKey{li, key}]
	return ok
}

// loadRec copies the snapshot named key into session sid's seat on block li,
// both halves. Callers hold g.mu, outside any submission.
func (g *devTier) loadRec(sid uint64, li int, key string) bool {
	sn := g.recSnaps[recSnapKey{li, key}]
	l, _, st, ok := g.recSeatOf(sid, li)
	if sn == nil || !ok {
		return false
	}
	for i := 0; i < 2; i++ {
		if err := g.dev.Copy(l.pool.c[i], st.base*g.recC*4, sn.c, 0, g.recC*4); err != nil {
			g.LastErr = "LoadRec: " + err.Error()
			return false
		}
		if l.pool.s[i] == nil || g.recS == 0 {
			continue
		}
		if err := g.dev.Copy(l.pool.s[i], st.base*g.recS*4, sn.s, 0, g.recS*4); err != nil {
			g.LastErr = "LoadRec: " + err.Error()
			return false
		}
	}
	g.RecPointsLoaded++
	return true
}

// freeRecSnaps frees block li's snapshots. Callers hold g.mu.
func (g *devTier) freeRecSnaps(li int) {
	kept := g.recSnapOrder[:0]
	for _, k := range g.recSnapOrder {
		if k.li != li {
			kept = append(kept, k)
			continue
		}
		sn := g.recSnaps[k]
		delete(g.recSnaps, k)
		freeKV2(sn.c, sn.s)
		g.refund(g.recSnapBytes())
	}
	g.recSnapOrder = kept
}

// SnapRecs is snapRec for every block of lis, under one lock.
func (g *devTier) SnapRecs(lis []int32, key string) bool { return g.snapRecs(0, lis, key) }

func (g *devTier) snapRecs(sid uint64, lis []int32, key string) bool {
	sv := g.as(sid)
	if sv == nil {
		return false
	}
	defer sv.done()
	g = sv
	g.mu.Lock()
	defer g.mu.Unlock()
	for _, li := range lis {
		if !g.snapRec(sid, int(li), key) {
			return g.flushSnaps(false)
		}
	}
	return g.flushSnaps(true)
}

// SnapRec, HasRec and LoadRec are the zero session's (nn.RecShareDevice).
func (g *devTier) SnapRec(li int, key string) bool { return g.recShareCall(0, li, key, 0) }
func (g *devTier) HasRec(li int, key string) bool  { return g.recShareCall(0, li, key, 1) }
func (g *devTier) LoadRec(li int, key string) bool { return g.recShareCall(0, li, key, 2) }

// recShareCall runs op (0 snap, 1 has, 2 load) for session sid.
func (g *devTier) recShareCall(sid uint64, li int, key string, op int) bool {
	sv := g.as(sid)
	if sv == nil {
		return false
	}
	defer sv.done()
	g = sv
	g.mu.Lock()
	defer g.mu.Unlock()
	switch op {
	case 0:
		return g.flushSnaps(g.snapRec(sid, li, key))
	case 1:
		return g.hasRec(li, key)
	}
	return g.loadRec(sid, li, key)
}
