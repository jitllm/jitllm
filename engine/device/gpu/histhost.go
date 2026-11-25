package gpu

import (
	"fmt"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// A history read across the CPU and a card. A block placed here reading a
// block a session runs on the CPU device reads a copy of each history of it
// it reads, the card holding it for the session (makeHostCopy): a pool layer
// of the source's at the geometry the reader reads it at (hostHists), made
// empty with the reader and written from the CPU's rows -- or, for a history
// past the source's rows, its entries (writeEntries) -- by the engine before
// each run reading it, from the first position it lacks (hostCopyFrom) to
// the run's end (writeHostRows), whole pages in the card's layout, so the
// copy pages at the card's size whatever the CPU's is. A block on the CPU
// device reading a block placed here reads the CPU's own copy, which the
// engine reads from here (readHostRows) as the block's rows are written.
//
// The copy is a history of the card like any other: grown, sent home and
// released behind a window as the card pages. It is held once on the card a
// session, and goes when no block here reads it (settleCopies), or becomes
// the block's own pages when the block comes here.

// hostHist is one history of a block the CPU device runs that a copy of
// it here holds: its index, its geometry, and the window (chunked) and tail
// it is read at.
type hostHist struct {
	h       int
	geom    kvGeom
	win     int
	chunked bool
	tail    int
}

// hostHists appends to dst the histories of block src a block reading it
// here reads from a copy of them as the CPU's, each once: where it binds
// src's histories in place of its own (kvSrc), its rows' at rgeom, window
// win (chunked) and tail, and each past them as src keeps it (kept); and
// each other history of src it reads past those (more, as bound states
// each, a rows' history in rgeom's typed pool, as the CPU device's form
// reads it).
func hostHists(dst []hostHist, src, kvSrc int, rgeom kvGeom, win int, chunked bool, tail int, kept []graph.KeptHistory, more []graph.HistoryRead, bound []graph.KeptHistory) []hostHist {
	add := func(x hostHist) {
		for _, y := range dst {
			if y.h == x.h {
				return
			}
		}
		dst = append(dst, x)
	}
	if src == kvSrc {
		add(hostHist{geom: rgeom, win: win, chunked: chunked, tail: tail})
		for _, k := range kept {
			add(hostHist{h: k.Hist, geom: keptGeom(k, rgeom.pool)})
		}
	}
	for j, r := range more {
		switch {
		case r.Block != src || j >= len(bound):
		case r.Hist == 0:
			add(hostHist{geom: kvGeom{kvRow: bound[j].Width, pool: rgeom.pool}})
		default:
			add(hostHist{h: r.Hist, geom: keptGeom(bound[j], rgeom.pool)})
		}
	}
	return dst
}

// keptGeom is the page geometry of history k, one past its block's rows
// (pagedLayer), beside rows held in pool rows: binary16 where those are and
// its values are its keys, as the CPU device holds it, floats otherwise.
func keptGeom(k graph.KeptHistory, rows composer.Pool) kvGeom {
	g := kvGeom{kvRow: k.Width, mla: k.ValuesInKeys, period: k.Period}
	if k.ValuesInKeys && rows == (composer.Pool{Half: true}) {
		g.pool = rows
	}
	return g
}

// hostHistsOf is hostHists for placed block l reading block src for
// session sid: its rows at the type the session's copy of them holds, where
// it holds one, the type l was placed with otherwise. Callers hold g.mu.
func (g *devTier) hostHistsOf(dst []hostHist, sid uint64, l *layer, src int) []hostHist {
	rg := l.rgeom
	if ds := g.sess[sid]; ds != nil {
		if t, ok := ds.hist[src]; ok {
			rg.pool = codePool(t)
		}
	}
	return hostHists(dst, src, l.kvSrc, rg, l.win, l.chunked, l.rtail, l.rkept, l.kvMore, l.rbound)
}

// lacksHists reports that this device holds no pool layer of one of
// session sid's histories hs of block src. Callers hold g.mu.
func (g *devTier) lacksHists(sid uint64, src int, hs []hostHist) bool {
	k0 := typedKey(src, g.hostType(sid, src, hs))
	for _, x := range hs {
		if g.kvp == nil || g.kvp.layers[histKey(k0, x.h)] == nil {
			return true
		}
	}
	return false
}

// hostType is the type session sid's copy hs of block src's histories
// holds its rows as: their geometry's where hs reads them, the type the
// session holds them as here otherwise. Callers hold g.mu.
func (g *devTier) hostType(sid uint64, src int, hs []hostHist) int {
	for _, x := range hs {
		if x.h == 0 {
			return poolCode(x.geom.pool)
		}
	}
	return g.histType(sid, src)
}

// makeHostCopy gives d a copy of session sid's histories hs of block src,
// which the session runs on the CPU device: empty, for the engine to write.
// A copy d holds already is widened by the histories of hs it lacks, each
// written from the sequence's start (hostCopyFrom). p is the reading
// block's plan, nil once it is placed.
func (g *GPU) makeHostCopy(sid uint64, src int, d *devTier, hs []hostHist, p *nn.LayerPlan) error {
	dv := d.bareView(sid)
	dv.mu.Lock()
	defer dv.mu.Unlock()
	widen := dv.histCopies[src][sid] && dv.hostOf[src][sid]
	if p != nil {
		dv.initKVCap(p)
	}
	kp := dv.kvPages()
	t := dv.hostType(sid, src, hs)
	dv.setHistType(sid, src, t)
	for _, x := range hs {
		key, geom := histKey(typedKey(src, t), x.h), x.geom
		if geom.kvRow <= 0 {
			return fmt.Errorf("block %d's history %d is read on %s at no width", src, x.h, label(d.dev, d.ord))
		}
		if tl := kp.layers[key]; tl != nil {
			if tl.geom != geom {
				return fmt.Errorf("block %d's history %d is laid out %+v on %s, and a block reading it there reads %+v", src, x.h, tl.geom, label(d.dev, d.ord), geom)
			}
		} else {
			if err := dv.addKVLayer(kp, key, geom, 2); err != nil {
				return err
			}
			tl := kp.layers[key]
			tl.win, tl.chunked, tl.tail = x.win, x.chunked, x.tail
			if x.win > 0 {
				kp.chunked = x.chunked
			}
		}
		if err := dv.formGeomConv(geom, kp.p); err != nil {
			return err
		}
	}
	if !widen {
		if dv.histCopies == nil {
			dv.histCopies = map[int]map[uint64]bool{}
		}
		if dv.histCopies[src] == nil {
			dv.histCopies[src] = map[uint64]bool{}
		}
		if dv.hostOf == nil {
			dv.hostOf = map[int]map[uint64]bool{}
		}
		if dv.hostOf[src] == nil {
			dv.hostOf[src] = map[uint64]bool{}
		}
		dv.histCopies[src][sid] = true
		dv.hostOf[src][sid] = true
		dv.copyN.Add(1)
	}
	if err := dv.copySeat(kp, src); err != nil {
		dv.forgetCopy(sid, src)
		return err
	}
	if !widen {
		dv.HistoryCopies++
	}
	return nil
}

// forgetHost lets the record of session sid's copy of block src's history
// as the CPU's go. Callers hold g.mu.
func (g *devTier) forgetHost(sid uint64, src int) {
	if !g.hostOf[src][sid] {
		return
	}
	delete(g.hostOf[src], sid)
	if len(g.hostOf[src]) == 0 {
		delete(g.hostOf, src)
	}
	for key, m := range g.hostRows {
		if keyBlock(key) != src {
			continue
		}
		for s := range m {
			if s.sid == sid {
				delete(m, s)
			}
		}
		if len(m) == 0 {
			delete(g.hostRows, key)
		}
	}
}

// hostCopies appends to buf every device holding session sid's copy of
// block src's history as the CPU's.
func (g *GPU) hostCopies(sid uint64, src int, buf []*devTier) []*devTier {
	g.mu.Lock()
	ds := append(buf, g.devs...)
	g.mu.Unlock()
	out := ds[:len(buf)]
	for _, d := range ds[len(buf):] {
		if d.copyN.Load() == 0 {
			continue
		}
		d.mu.Lock()
		has := d.hostOf[src][sid]
		d.mu.Unlock()
		if has {
			out = append(out, d)
		}
	}
	return out
}

// hostCopyFrom is the first position of session sid's sequence at base of
// block src's history h that a copy of it here as the CPU's lacks, at most
// start, at the start of the page it falls in on every device holding one;
// false where none does.
func (g *GPU) hostCopyFrom(sid uint64, src, h, base, start int) (int, bool) {
	var buf [ownersStack]*devTier
	ds := g.hostCopies(sid, src, buf[:0])
	if len(ds) == 0 {
		return 0, false
	}
	from := max(start, 0)
	s := seqID{sid, base}
	for _, d := range ds {
		d.mu.Lock()
		key := histKey(d.sessKey(sid, src), h)
		if d.kvp == nil || d.kvp.layers[key] == nil {
			d.mu.Unlock()
			return 0, false
		}
		f := min(from, d.hostRows[key][s])
		if d.kvp != nil && d.kvp.p > 0 {
			f = f / d.kvp.p * d.kvp.p
		}
		d.mu.Unlock()
		from = min(from, f)
	}
	return from, true
}

// copyHostRows writes positions [from, to) of session sid's sequence at base
// of block src's history h, as nn.HistoryCopier lays them out, into every
// copy of it held as the CPU's. start is where the call about to run writes
// from: the sequence's record on the card comes down to it first.
func (g *GPU) copyHostRows(sid uint64, src, h, base, from, start, to int, k, v []float32) bool {
	var buf [ownersStack]*devTier
	for _, d := range g.hostCopies(sid, src, buf[:0]) {
		dv := d.bareView(sid)
		dv.mu.Lock()
		err := dv.writeHostRows(seqID{sid, base}, src, h, from, start, to, k, v)
		if err != nil {
			dv.fail(fmt.Sprintf("block %d's history copy on %s: %v", src, label(d.dev, d.ord), err))
		}
		dv.mu.Unlock()
		if err != nil {
			return false
		}
	}
	return true
}

// writeHostRows is copyHostRows on one device (writeRows, writeEntries),
// the copy then holding the sequence to to. Callers hold g.mu.
func (g *devTier) writeHostRows(s seqID, src, h, from, start, to int, k, v []float32) error {
	if g.pagedFaulted("copystale") {
		return nil
	}
	key := histKey(g.sessKey(s.sid, src), h)
	var pages int
	var err error
	if h == 0 {
		pages, err = g.writeRows(s, src, from, start, to, k, v)
	} else {
		pages, err = g.writeEntries(s, key, from, start, to, k)
	}
	if err != nil {
		return err
	}
	if g.hostRows == nil {
		g.hostRows = map[int]map[seqID]int{}
	}
	if g.hostRows[key] == nil {
		g.hostRows[key] = map[seqID]int{}
	}
	g.hostRows[key][s] = to
	g.HostCopyPages += pages
	return nil
}

// writeRows writes positions [from, to) of sequence s of block src's
// history, k and v [to-from][kvRow] floats, into this device's pages of it:
// the pages [from, to) falls in, built whole in the device's layout from the
// rows -- a page's rows outside them zeros, from at a page's start or past
// what any call reads -- and written where the sequence holds them: a page
// at home in host memory there, one released behind a window not at all.
// start, when not -1, is where the call about to run writes from: the
// sequence's record comes down to it first. It is the pages written.
// Callers hold g.mu.
func (g *devTier) writeRows(s seqID, src, from, start, to int, k, v []float32) (int, error) {
	kp := g.kvp
	if kp == nil || kp.layers[g.sessKey(s.sid, src)] == nil {
		return 0, fmt.Errorf("no copy of block %d's history", src)
	}
	if to <= from {
		return 0, nil
	}
	tl := kp.layers[g.sessKey(s.sid, src)]
	P, kvRow := kp.p, tl.geom.kvRow
	if len(k) < (to-from)*kvRow || len(v) < (to-from)*kvRow {
		return 0, fmt.Errorf("%d positions of %d floats do not fit the rows' %d/%d", to-from, kvRow, len(k), len(v))
	}
	if start >= 0 {
		if err := g.rewind(s.sid, kp, s, start); err != nil {
			return 0, err
		}
	}
	j0, j1 := from/P, (to+P-1)/P
	if err := g.cowPages(s.sid, kp, tl, s, from, j1*P); err != nil {
		return 0, err
	}
	if have := len(tl.owned[s]); have < j1 {
		if err := g.allocPages(s.sid, kp, tl, s, j1-have); err != nil {
			return 0, err
		}
	}
	if err := g.flushTabs(); err != nil {
		return 0, err
	}
	kw, vw := tl.geom.kWords(P), tl.geom.vWords(P)
	var hv *pageConv
	conv, keys := tl.geom.convOf(P)
	if keys {
		var err error
		if hv, err = g.keptConv(conv); err != nil {
			return 0, err
		}
	}
	// A typed page's keys are by position, narrowed as its values are: kf
	// is a page's keys as floats.
	byPos, kf := tl.geom.mla || keys, kw
	if keys {
		kf = P * kvRow
	}
	page := g.hostScratch(kf + P*kvRow)
	kPage, vPage := page[:kf], page[kf:]
	pages := 0
	for j := j0; j < j1; j++ {
		id := tl.owned[s][j]
		home := id == 0 && j < len(tl.home[s]) && tl.home[s][j] != nil
		if id == 0 && !home {
			continue // released behind the window: nothing reads it
		}
		clear(page)
		for t := max(j*P, from); t < min((j+1)*P, to); t++ {
			r := (t - from) * kvRow
			if byPos {
				copy(kPage[(t%P)*kvRow:], k[r:r+kvRow])
			} else {
				for e := range kvRow {
					kPage[e*P+t%P] = k[r+e]
				}
			}
			if !tl.geom.mla {
				copy(vPage[(t%P)*kvRow:], v[r:r+kvRow])
			}
		}
		if home {
			h := tl.home[s][j]
			if keys {
				words, err := hv.narrowPage(kPage, nil, 0)
				if err != nil {
					return 0, err
				}
				copy(h[:kw], words)
			} else {
				copy(h[:kw], kPage)
			}
			words := vPage[:vw]
			if hv != nil && vw > 0 {
				var err error
				if words, err = hv.narrowPage(vPage, nil, 0); err != nil {
					return 0, err
				}
			}
			copy(h[kw:], words)
			pages++
			continue
		}
		if keys {
			if _, err := hv.narrowPage(kPage, tl.k, int(id)*kw*4); err != nil {
				return 0, err
			}
		} else if err := tl.k.WriteAt(int(id)*kw*4, f32b(kPage)); err != nil {
			return 0, err
		}
		switch {
		case vw == 0:
		case hv != nil:
			if _, err := hv.narrowPage(vPage, tl.v, int(id)*vw*4); err != nil {
				return 0, err
			}
		default:
			if err := tl.v.WriteAt(int(id)*vw*4, f32b(vPage[:vw])); err != nil {
				return 0, err
			}
		}
		pages++
	}
	return pages, nil
}

// copyRepaged is copySeqPages between two devices whose pools page apart:
// positions [from, to) of sequence s of each of block src's histories read
// from sv's pages -- its rows' as rows, one past them as its entries -- and
// written into dv's, from the start of dv's page from falls in. start is
// copySeqPages' from. Callers hold both locks.
func copyRepaged(sv, dv *devTier, s seqID, src, from, start, to int) error {
	from = from / dv.kvp.p * dv.kvp.p
	if to <= from {
		return nil
	}
	if start >= 0 {
		if err := dv.rewind(s.sid, dv.kvp, s, start); err != nil {
			return err
		}
	}
	sk, dk := sv.sessKey(s.sid, src), dv.sessKey(s.sid, src)
	for h := range sv.kvp.blockHists(src) {
		key, dkey := histKey(sk, h), histKey(dk, h)
		fl := sv.kvp.layers[key]
		if fl == nil {
			continue
		}
		if dv.kvp.layers[dkey] == nil {
			return fmt.Errorf("block %d's history %d has no copy on %s", src, h, label(dv.dev, dv.ord))
		}
		to := min(to, len(fl.owned[s])*sv.kvp.p)
		if to <= from {
			continue
		}
		n := (to - from) * fl.geom.kvRow
		if h > 0 {
			n = entryFloats(fl.geom, from, to)
		}
		k, v := dv.copyRows(n)
		var pages int
		var err error
		if h == 0 {
			if err = sv.readRows(s, src, from, to, k, v); err == nil {
				pages, err = dv.writeRows(s, src, from, -1, to, k, v)
			}
		} else if err = sv.readEntries(s, key, from, to, k); err == nil {
			pages, err = dv.writeEntries(s, dkey, from, -1, to, k)
		}
		dv.HistoryCopyPages += pages
		if err != nil {
			return err
		}
	}
	return nil
}

// copyRows is the rows of n floats each for k and v a copy between two
// cards crosses through, grown to twice what they held, never shrunk.
func (g *devTier) copyRows(n int) ([]float32, []float32) {
	if len(g.rowsK) < n {
		n2 := max(n, 2*len(g.rowsK))
		g.rowsK, g.rowsV = make([]float32, n2), make([]float32, n2)
	}
	return g.rowsK[:n], g.rowsV[:n]
}

// entryFloats is the floats the entries of positions [from, to) of a
// history past a block's rows of geometry geom cross as: each entry its
// keys, then its values where they are not its keys
// (nn.HistoryCopier.CopyEntries). from is at a page's start.
func entryFloats(geom kvGeom, from, to int) int {
	per := max(1, geom.period)
	n := (to/per - from/per) * geom.kvRow
	if !geom.mla {
		n *= 2
	}
	return max(n, 0)
}

// readEntries reads the entries of positions [from, to) of sequence s of
// the history past a block's rows that pool key holds here into e, as
// entryFloats lays them out: entry j, of position (j+1)*period-1, at
// (j-from/period) entries in. from is at a page's start. Callers hold g.mu.
func (g *devTier) readEntries(s seqID, key, from, to int, e []float32) error {
	kp := g.kvp
	if kp == nil || kp.layers[key] == nil {
		return fmt.Errorf("block %d keeps no history %d here", keyBlock(key), keyHist(key))
	}
	pl := kp.layers[key]
	P, per, w := kp.p, max(1, pl.geom.period), pl.geom.kvRow
	kw, vw, kf := pl.geom.kWords(P), pl.geom.vWords(P), pl.geom.slots(P)*w
	ew := w
	if vw > 0 {
		ew = 2 * w
	}
	j0, j1 := from/per, to/per
	if j1 <= j0 {
		return nil
	}
	hv, err := g.entryConv(pl, P)
	if err != nil {
		return err
	}
	if len(e) < (j1-j0)*ew {
		return fmt.Errorf("%d entries of %d floats do not fit %d", j1-j0, ew, len(e))
	}
	ids := pl.owned[s]
	last := ((j1-1)+1)*per - 1
	if len(ids) <= last/P {
		return fmt.Errorf("the sequence holds %d entry page(s) and %d positions need %d", len(ids), to, last/P+1)
	}
	page := g.hostScratch(kf + vw)
	kPage, vPage := page[:kf], page[kf:]
	cur := -1
	for j := j0; j < j1; j++ {
		q := (j+1)*per - 1
		if pg := q / P; pg != cur {
			cur = pg
			clear(page)
			switch id := ids[pg]; {
			case id == 0 && pg < len(pl.home[s]) && len(pl.home[s][pg]) >= kw+vw && hv != nil:
				if err := hv.widenPage(nil, 0, pl.home[s][pg][:kw], kPage); err != nil {
					return err
				}
			case id == 0 && pg < len(pl.home[s]) && len(pl.home[s][pg]) >= kw+vw:
				copy(page, pl.home[s][pg][:kw+vw])
			case id == 0:
			case hv != nil:
				if err := hv.widenPage(pl.k, int(id)*kw*4, nil, kPage); err != nil {
					return err
				}
				g.HostReadPages++
			default:
				if err := g.dev.Copy(pl.stage, 0, pl.k, int(id)*kw*4, kw*4); err != nil {
					return err
				}
				if err := pl.stage.Read(f32b(kPage)); err != nil {
					return err
				}
				if vw > 0 {
					if err := g.dev.Copy(pl.stage, 0, pl.v, int(id)*vw*4, vw*4); err != nil {
						return err
					}
					if err := pl.stage.Read(f32b(vPage)); err != nil {
						return err
					}
				}
				g.HostReadPages++
			}
		}
		at, o := q%P/per*w, (j-j0)*ew
		copy(e[o:o+w], kPage[at:at+w])
		if vw > 0 {
			copy(e[o+w:o+ew], vPage[at:at+w])
		}
	}
	return nil
}

// writeEntries is writeRows for the history past a block's rows that pool
// key holds here: the entries of positions [from, to) from e (readEntries'
// layout), the pages they fall in built whole, entries outside them zeros.
// It is the pages written. Callers hold g.mu.
func (g *devTier) writeEntries(s seqID, key, from, start, to int, e []float32) (int, error) {
	kp := g.kvp
	if kp == nil || kp.layers[key] == nil {
		return 0, fmt.Errorf("no copy of block %d's history %d", keyBlock(key), keyHist(key))
	}
	if to <= from {
		return 0, nil
	}
	tl := kp.layers[key]
	P, per, w := kp.p, max(1, tl.geom.period), tl.geom.kvRow
	kw, vw, kf := tl.geom.kWords(P), tl.geom.vWords(P), tl.geom.slots(P)*w
	hv, err := g.entryConv(tl, P)
	if err != nil {
		return 0, err
	}
	ew := w
	if vw > 0 {
		ew = 2 * w
	}
	j0, j1 := from/per, to/per
	if len(e) < max(j1-j0, 0)*ew {
		return 0, fmt.Errorf("%d entries of %d floats do not fit %d", j1-j0, ew, len(e))
	}
	if start >= 0 {
		if err := g.rewind(s.sid, kp, s, start); err != nil {
			return 0, err
		}
	}
	p0, p1 := from/P, (to+P-1)/P
	if err := g.cowPages(s.sid, kp, tl, s, from, p1*P); err != nil {
		return 0, err
	}
	if have := len(tl.owned[s]); have < p1 {
		if err := g.allocPages(s.sid, kp, tl, s, p1-have); err != nil {
			return 0, err
		}
	}
	if err := g.flushTabs(); err != nil {
		return 0, err
	}
	page := g.hostScratch(kf + vw)
	kPage, vPage := page[:kf], page[kf:]
	pages := 0
	for pg := p0; pg < p1; pg++ {
		id := tl.owned[s][pg]
		home := id == 0 && pg < len(tl.home[s]) && tl.home[s][pg] != nil
		if id == 0 && !home {
			continue
		}
		clear(page)
		for j := max(j0, (pg*P+1+per-1)/per-1); j < j1; j++ {
			q := (j+1)*per - 1
			if q/P != pg {
				break
			}
			at, o := q%P/per*w, (j-j0)*ew
			copy(kPage[at:at+w], e[o:o+w])
			if vw > 0 {
				copy(vPage[at:at+w], e[o+w:o+ew])
			}
		}
		switch {
		case home && hv != nil:
			words, err := hv.narrowPage(kPage, nil, 0)
			if err != nil {
				return 0, err
			}
			copy(tl.home[s][pg], words)
			pages++
			continue
		case home:
			copy(tl.home[s][pg], page)
			pages++
			continue
		case hv != nil:
			if _, err := hv.narrowPage(kPage, tl.k, int(id)*kw*4); err != nil {
				return 0, err
			}
		default:
			if err := tl.k.WriteAt(int(id)*kw*4, f32b(kPage)); err != nil {
				return 0, err
			}
		}
		if vw > 0 {
			if err := tl.v.WriteAt(int(id)*vw*4, f32b(vPage)); err != nil {
				return 0, err
			}
		}
		pages++
	}
	return pages, nil
}

// readHostRows reads positions [from, to) of session sid's sequence at base
// of block src's history h, which the session runs here, into k and v as
// nn.HistoryCopier lays them out: a page at a time through the layer's read
// reserve, a page at home from host memory, one released behind a window as
// zeros.
func (g *GPU) readHostRows(sid uint64, src, h, base, from, to int, k, v []float32) bool {
	g.mu.Lock()
	o := g.at(sid, src)
	g.mu.Unlock()
	if o == nil {
		return false
	}
	ov := o.bareView(sid)
	ov.mu.Lock()
	var err error
	if h == 0 {
		err = ov.readRows(seqID{sid, base}, src, from, to, k, v)
	} else {
		err = ov.readEntries(seqID{sid, base}, histKey(ov.sessKey(sid, src), h), from, to, k)
	}
	if err != nil {
		ov.fail(fmt.Sprintf("block %d's history read for the CPU device on %s: %v", src, label(o.dev, o.ord), err))
	}
	ov.mu.Unlock()
	return err == nil
}

// readRows is readHostRows on the device holding the block. Callers hold
// g.mu.
func (g *devTier) readRows(s seqID, src, from, to int, k, v []float32) error {
	kp := g.kvp
	pl := kp.layerOf(g.sessKey(s.sid, src))
	if pl == nil {
		return fmt.Errorf("block %d keeps no history here", src)
	}
	if to <= from {
		return nil
	}
	P, kvRow := kp.p, pl.geom.kvRow
	if len(k) < (to-from)*kvRow || len(v) < (to-from)*kvRow {
		return fmt.Errorf("%d positions of %d floats do not fit the rows' %d/%d", to-from, kvRow, len(k), len(v))
	}
	ids := pl.owned[s]
	if len(ids) < (to+P-1)/P {
		return fmt.Errorf("the sequence holds %d page(s) and %d positions need %d", len(ids), to, (to+P-1)/P)
	}
	kw, vw := pl.geom.kWords(P), pl.geom.vWords(P)
	var hv *pageConv
	conv, keys := pl.geom.convOf(P)
	if keys {
		var err error
		if hv, err = g.keptConv(conv); err != nil {
			return err
		}
	}
	byPos, kf := pl.geom.mla || keys, kw
	if keys {
		kf = P * kvRow
	}
	page := g.hostScratch(kf + P*kvRow)
	kPage, vPage := page[:kf], page[kf:]
	pages := 0
	for j := from / P; j*P < to; j++ {
		id := ids[j]
		clear(page)
		switch {
		case id == 0 && j < len(pl.home[s]) && pl.home[s][j] != nil:
			h := pl.home[s][j]
			if keys {
				if err := hv.widenPage(nil, 0, h[:kw], kPage); err != nil {
					return err
				}
			} else {
				copy(kPage, h[:kw])
			}
			switch {
			case vw == 0:
			case hv != nil:
				if err := hv.widenPage(nil, 0, h[kw:], vPage); err != nil {
					return err
				}
			default:
				copy(vPage[:vw], h[kw:])
			}
		case id == 0:
		default:
			if keys {
				if err := hv.widenPage(pl.k, int(id)*kw*4, nil, kPage); err != nil {
					return err
				}
			} else if err := g.dev.Copy(pl.stage, 0, pl.k, int(id)*kw*4, kw*4); err != nil {
				return err
			} else if err := pl.stage.Read(f32b(kPage)); err != nil {
				return err
			}
			switch {
			case vw == 0:
			case hv != nil:
				if err := hv.widenPage(pl.v, int(id)*vw*4, nil, vPage); err != nil {
					return err
				}
			default:
				if err := g.dev.Copy(pl.stage, 0, pl.v, int(id)*vw*4, vw*4); err != nil {
					return err
				}
				if err := pl.stage.Read(f32b(vPage[:vw])); err != nil {
					return err
				}
			}
			pages++
		}
		for t := max(j*P, from); t < min((j+1)*P, to); t++ {
			r := (t - from) * kvRow
			if pl.geom.mla {
				row := kPage[(t%P)*kvRow : (t%P+1)*kvRow]
				copy(k[r:r+kvRow], row)
				copy(v[r:r+kvRow], row)
				continue
			}
			if byPos {
				copy(k[r:r+kvRow], kPage[(t%P)*kvRow:(t%P+1)*kvRow])
			} else {
				for e := range kvRow {
					k[r+e] = kPage[e*P+t%P]
				}
			}
			copy(v[r:r+kvRow], vPage[(t%P)*kvRow:(t%P+1)*kvRow])
		}
	}
	g.HostReadPages += pages
	return nil
}

// shareHostCopy names, on each device holding session sid's copy of block
// li's history as the CPU's, the copy's page the prefix cache names key on
// the CPU device, so the copy another session holds there maps it rather
// than holding the page twice: published (end -1) where the copy holds the
// page whole; mapped (end where the history ends then), the copy taking the
// page its device names by key. False where a device holds no copy, or
// none answers to key.
func (g *GPU) shareHostCopy(sid uint64, li, page int, key string, end int) bool {
	var buf [ownersStack]*devTier
	ds := g.hostCopies(sid, keyBlock(li), buf[:0])
	if len(ds) == 0 || key == "" {
		return false
	}
	s := seqID{sid, 0}
	for _, d := range ds {
		dv := d.bareView(sid)
		dv.mu.Lock()
		ok, err := dv.shareHostPage(s, li, page, key, end)
		if err != nil {
			dv.fail(fmt.Sprintf("block %d's history copy on %s: %v", keyBlock(li), label(d.dev, d.ord), err))
		}
		dv.mu.Unlock()
		if !ok {
			return false
		}
	}
	return true
}

// shareHostPage is shareHostCopy on one device. Callers hold g.mu.
func (g *devTier) shareHostPage(s seqID, li, page int, key string, end int) (bool, error) {
	kp := g.kvp
	li = g.sessKey(s.sid, li)
	if kp == nil || kp.layers[li] == nil {
		return false, nil
	}
	l, P := kp.layers[li], kp.p
	held := g.hostRows[li][s]
	if end < 0 {
		return held >= (page+1)*P && g.publishKV(kp, l, s, page, key), nil
	}
	ok, err := g.mapKV(kp, l, s, page, key)
	if err != nil || !ok {
		return false, err
	}
	if held >= page*P {
		if g.hostRows == nil {
			g.hostRows = map[int]map[seqID]int{}
		}
		if g.hostRows[li] == nil {
			g.hostRows[li] = map[seqID]int{}
		}
		g.hostRows[li][s] = max(held, (page+1)*P)
	}
	if kp.written == nil {
		kp.written = map[seqID]int{}
	}
	kp.written[s] = max(kp.written[s], end)
	return true, g.flushTabs()
}

// hostCopyPage is the positions a page of session sid's copy of block li's
// history holds on a device holding it as the CPU's, 0 where none does.
func (g *GPU) hostCopyPage(sid uint64, li int) int {
	var buf [ownersStack]*devTier
	for _, d := range g.hostCopies(sid, keyBlock(li), buf[:0]) {
		d.mu.Lock()
		p := 0
		if d.kvp != nil {
			p = d.kvp.p
		}
		d.mu.Unlock()
		return p
	}
	return 0
}

// hostScratch is the device's page of host floats, at least n long: a
// copy's write and a read for the CPU's go through it, so a token's
// allocates nothing. Callers hold g.mu.
func (g *devTier) hostScratch(n int) []float32 {
	if cap(g.hostPage) < n {
		g.hostPage = make([]float32, n)
	}
	return g.hostPage[:n]
}

// CopyFrom is hostCopyFrom for the zero session (nn.HistoryCopier).
func (g *GPU) CopyFrom(li, h, base, start int) (int, bool) {
	return g.hostCopyFrom(0, li, h, base, start)
}

// CopyRows is copyHostRows for the zero session (nn.HistoryCopier).
func (g *GPU) CopyRows(li, h, base, from, start, to int, k, v []float32) bool {
	return g.copyHostRows(0, li, h, base, from, start, to, k, v)
}

// ReadRows is readHostRows for the zero session (nn.HistoryCopier).
func (g *GPU) ReadRows(li, h, base, from, to int, k, v []float32) bool {
	return g.readHostRows(0, li, h, base, from, to, k, v)
}

func (s *gpuSession) CopyFrom(li, h, base, start int) (int, bool) {
	defer s.leave(s.step())
	return s.g.hostCopyFrom(s.sid, li, h, base, start)
}

func (s *gpuSession) CopyRows(li, h, base, from, start, to int, k, v []float32) bool {
	defer s.leave(s.step())
	return s.g.copyHostRows(s.sid, li, h, base, from, start, to, k, v)
}

func (s *gpuSession) ReadRows(li, h, base, from, to int, k, v []float32) bool {
	defer s.leave(s.step())
	return s.g.readHostRows(s.sid, li, h, base, from, to, k, v)
}
