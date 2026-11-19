package gpu

import (
	"fmt"

	"github.com/jitllm/jitllm/nn"
)

// A history read across the CPU and a card. A block placed here reading a
// block a session runs on the CPU device reads a copy of that block's
// history the card holds for the session (makeHostCopy): the source's pool
// layer at the reader's geometry, made empty with the reader and written
// from the CPU's rows by the engine before each run reading it -- from the
// first position it lacks (hostCopyFrom) to the run's end (writeHostRows),
// whole pages in the card's layout, so the copy pages at the card's size
// whatever the CPU's is. A block on the CPU device reading a block placed
// here reads the CPU's own copy, which the engine reads from here
// (readHostRows) as the block's rows are written.
//
// The copy is a history of the card like any other: grown, sent home and
// released behind a window as the card pages. It is held once on the card a
// session, and goes when no block here reads it (settleCopies), or becomes
// the block's own pages when the block comes here.

// makeHostCopy gives d a copy of session sid's history of block src, which
// the session runs on the CPU device, at geometry geom, the window win
// (chunked) and tail of the block reading it: empty, for the engine to
// write. p is the reading block's plan, nil once it is placed.
func (g *GPU) makeHostCopy(sid uint64, src int, d *devTier, geom kvGeom, win int, chunked bool, tail int, p *nn.LayerPlan) error {
	dv := d.bareView(sid)
	dv.mu.Lock()
	defer dv.mu.Unlock()
	if p != nil {
		dv.initKVCap(p)
	}
	kp := dv.kvPages()
	key := histKey(src, 0)
	if tl := kp.layers[key]; tl != nil {
		if tl.geom != geom {
			return fmt.Errorf("block %d's history is laid out %+v on %s, and a block reading it there reads %+v", src, tl.geom, label(d.dev, d.ord), geom)
		}
	} else {
		if err := dv.addKVLayer(kp, key, geom, 2); err != nil {
			return err
		}
		tl := kp.layers[key]
		tl.win, tl.chunked, tl.tail = win, chunked, tail
		if win > 0 {
			kp.chunked = chunked
		}
	}
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
	if err := dv.copySeat(kp, src); err != nil {
		dv.forgetCopy(sid, src)
		return err
	}
	dv.HistoryCopies++
	return nil
}

// readGeom is the geometry, window and tail placed block l reads its
// source's history at, for a copy of it made here.
func (l *layer) readGeom() (kvGeom, int, bool, int) {
	return l.rgeom, l.win, l.chunked, l.rtail
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
	for s := range g.hostRows[src] {
		if s.sid == sid {
			delete(g.hostRows[src], s)
		}
	}
	if len(g.hostRows[src]) == 0 {
		delete(g.hostRows, src)
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
// block src's history that a copy of it here as the CPU's lacks, at most
// start, at the start of the page it falls in on every device holding one;
// false where none does.
func (g *GPU) hostCopyFrom(sid uint64, src, base, start int) (int, bool) {
	var buf [ownersStack]*devTier
	ds := g.hostCopies(sid, src, buf[:0])
	if len(ds) == 0 {
		return 0, false
	}
	from := max(start, 0)
	s := seqID{sid, base}
	for _, d := range ds {
		d.mu.Lock()
		f := min(from, d.hostRows[src][s])
		if d.kvp != nil && d.kvp.p > 0 {
			f = f / d.kvp.p * d.kvp.p
		}
		d.mu.Unlock()
		from = min(from, f)
	}
	return from, true
}

// copyHostRows writes positions [from, to) of session sid's sequence at base
// of block src's history, k and v [to-from][kvRow] floats, into every copy
// of it held as the CPU's. start is where the call about to run writes from:
// the sequence's record on the card comes down to it first.
func (g *GPU) copyHostRows(sid uint64, src, base, from, start, to int, k, v []float32) bool {
	var buf [ownersStack]*devTier
	for _, d := range g.hostCopies(sid, src, buf[:0]) {
		dv := d.bareView(sid)
		dv.mu.Lock()
		err := dv.writeHostRows(seqID{sid, base}, src, from, start, to, k, v)
		if err != nil {
			dv.LastErr = fmt.Sprintf("block %d's history copy on %s: %v", src, label(d.dev, d.ord), err)
		}
		dv.mu.Unlock()
		if err != nil {
			return false
		}
	}
	return true
}

// writeHostRows is copyHostRows on one device (writeRows), the copy then
// holding the sequence to to. Callers hold g.mu.
func (g *devTier) writeHostRows(s seqID, src, from, start, to int, k, v []float32) error {
	if g.pagedFaulted("copystale") {
		return nil
	}
	pages, err := g.writeRows(s, src, from, start, to, k, v)
	if err != nil {
		return err
	}
	if g.hostRows == nil {
		g.hostRows = map[int]map[seqID]int{}
	}
	if g.hostRows[src] == nil {
		g.hostRows[src] = map[seqID]int{}
	}
	g.hostRows[src][s] = to
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
	if kp == nil || kp.layers[src] == nil {
		return 0, fmt.Errorf("no copy of block %d's history", src)
	}
	if to <= from {
		return 0, nil
	}
	tl := kp.layers[src]
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
	var hv *pageHalves
	if tl.geom.f16 && vw > 0 {
		var err error
		if hv, err = g.newPageHalves(kvRow, P); err != nil {
			return 0, err
		}
		defer hv.close()
	}
	page := g.hostScratch(kw + P*kvRow)
	kPage, vPage := page[:kw], page[kw:]
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
			if tl.geom.mla {
				copy(kPage[(t%P)*kvRow:], k[r:r+kvRow])
				continue
			}
			for e := range kvRow {
				kPage[e*P+t%P] = k[r+e]
			}
			copy(vPage[(t%P)*kvRow:], v[r:r+kvRow])
		}
		if home {
			words := vPage[:vw]
			if hv != nil {
				var err error
				if words, err = hv.narrowPage(vPage, nil, 0); err != nil {
					return 0, err
				}
			}
			h := tl.home[s][j]
			copy(h[:kw], kPage)
			copy(h[kw:], words)
			pages++
			continue
		}
		if err := tl.k.WriteAt(int(id)*kw*4, f32b(kPage)); err != nil {
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
// positions [from, to) of sequence s of block src's history read as rows
// from sv's pages and written into dv's, from the start of dv's page from
// falls in. start is copySeqPages' from. Callers hold both locks.
func copyRepaged(sv, dv *devTier, s seqID, src, from, start, to int) error {
	from = from / dv.kvp.p * dv.kvp.p
	if to <= from {
		return nil
	}
	n := (to - from) * sv.kvp.layers[src].geom.kvRow
	if len(dv.rowsK) < n {
		n2 := max(n, 2*len(dv.rowsK))
		dv.rowsK, dv.rowsV = make([]float32, n2), make([]float32, n2)
	}
	k, v := dv.rowsK[:n], dv.rowsV[:n]
	if err := sv.readRows(s, src, from, to, k, v); err != nil {
		return err
	}
	pages, err := dv.writeRows(s, src, from, start, to, k, v)
	dv.HistoryCopyPages += pages
	return err
}

// readHostRows reads positions [from, to) of session sid's sequence at base
// of block src's history, which the session runs here, into k and v
// [to-from][kvRow] floats: a page at a time through the layer's read
// reserve, a page at home from host memory, one released behind a window as
// zeros.
func (g *GPU) readHostRows(sid uint64, src, base, from, to int, k, v []float32) bool {
	g.mu.Lock()
	o := g.at(sid, src)
	g.mu.Unlock()
	if o == nil {
		return false
	}
	ov := o.bareView(sid)
	ov.mu.Lock()
	err := ov.readRows(seqID{sid, base}, src, from, to, k, v)
	if err != nil {
		ov.LastErr = fmt.Sprintf("block %d's history read for the CPU device on %s: %v", src, label(o.dev, o.ord), err)
	}
	ov.mu.Unlock()
	return err == nil
}

// readRows is readHostRows on the device holding the block. Callers hold
// g.mu.
func (g *devTier) readRows(s seqID, src, from, to int, k, v []float32) error {
	kp := g.kvp
	if kp == nil || kp.layers[src] == nil {
		return fmt.Errorf("block %d keeps no history here", src)
	}
	if to <= from {
		return nil
	}
	pl := kp.layers[src]
	P, kvRow := kp.p, pl.geom.kvRow
	if len(k) < (to-from)*kvRow || len(v) < (to-from)*kvRow {
		return fmt.Errorf("%d positions of %d floats do not fit the rows' %d/%d", to-from, kvRow, len(k), len(v))
	}
	ids := pl.owned[s]
	if len(ids) < (to+P-1)/P {
		return fmt.Errorf("the sequence holds %d page(s) and %d positions need %d", len(ids), to, (to+P-1)/P)
	}
	kw, vw := pl.geom.kWords(P), pl.geom.vWords(P)
	var hv *pageHalves
	if pl.geom.f16 && vw > 0 {
		var err error
		if hv, err = g.newPageHalves(kvRow, P); err != nil {
			return err
		}
		defer hv.close()
	}
	page := g.hostScratch(kw + P*kvRow)
	kPage, vPage := page[:kw], page[kw:]
	pages := 0
	for j := from / P; j*P < to; j++ {
		id := ids[j]
		clear(page)
		switch {
		case id == 0 && j < len(pl.home[s]) && pl.home[s][j] != nil:
			h := pl.home[s][j]
			copy(kPage, h[:kw])
			if hv != nil {
				if err := hv.widenPage(nil, 0, h[kw:], vPage); err != nil {
					return err
				}
			} else {
				copy(vPage[:vw], h[kw:])
			}
		case id == 0:
		default:
			if err := g.dev.Copy(pl.stage, 0, pl.k, int(id)*kw*4, kw*4); err != nil {
				return err
			}
			if err := pl.stage.Read(f32b(kPage)); err != nil {
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
			for e := range kvRow {
				k[r+e] = kPage[e*P+t%P]
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
	ds := g.hostCopies(sid, li, buf[:0])
	if len(ds) == 0 || key == "" {
		return false
	}
	s := seqID{sid, 0}
	for _, d := range ds {
		dv := d.bareView(sid)
		dv.mu.Lock()
		ok, err := dv.shareHostPage(s, li, page, key, end)
		if err != nil {
			dv.LastErr = fmt.Sprintf("block %d's history copy on %s: %v", li, label(d.dev, d.ord), err)
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
	for _, d := range g.hostCopies(sid, li, buf[:0]) {
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
func (g *GPU) CopyFrom(li, base, start int) (int, bool) { return g.hostCopyFrom(0, li, base, start) }

// CopyRows is copyHostRows for the zero session (nn.HistoryCopier).
func (g *GPU) CopyRows(li, base, from, start, to int, k, v []float32) bool {
	return g.copyHostRows(0, li, base, from, start, to, k, v)
}

// ReadRows is readHostRows for the zero session (nn.HistoryCopier).
func (g *GPU) ReadRows(li, base, from, to int, k, v []float32) bool {
	return g.readHostRows(0, li, base, from, to, k, v)
}

func (s *gpuSession) CopyFrom(li, base, start int) (int, bool) {
	defer s.leave(s.step())
	return s.g.hostCopyFrom(s.sid, li, base, start)
}

func (s *gpuSession) CopyRows(li, base, from, start, to int, k, v []float32) bool {
	defer s.leave(s.step())
	return s.g.copyHostRows(s.sid, li, base, from, start, to, k, v)
}

func (s *gpuSession) ReadRows(li, base, from, to int, k, v []float32) bool {
	defer s.leave(s.step())
	return s.g.readHostRows(s.sid, li, base, from, to, k, v)
}
