package gpu

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// A history read across cards. A block whose graph reads another block's
// history (graph.Graph.ReadsHistoryOf, ReadsMore) may run on another device
// than that block: the reader's device then holds a copy of the source's
// history for the session -- the source's pool layers, at the same keys, so
// the reader binds them as it binds a source beside it -- and the copy is
// appended as the source appends: before a run of a block reading it, the
// pages the run's rows wrote on the source's device are copied across
// (appendCopies). A history migrated in or a prefix mapped in reaches every
// copy beside the source (refreshCopies).
//
// A device holds a block's history for a session once: its own pages where
// the block runs there for the session, otherwise the copy. The copies
// follow the routes (settleCopies): one a reader needs is made from the
// source's whole history, one no reader needs goes, and one whose block now
// runs on its device is that block's own pages from then on. A device the
// source leaves keeps the pages a reader there reads as the copy.
//
// The copy pages as any history on its device: grown, sent home and read
// back in passes, released behind a window. Its pages are copied whole and
// raw, so the two pools must page alike: the same positions a page and the
// same layout; a pair that does not is refused when the copy would be made.

// readsOf is the blocks whose histories a block li of graph g reads beside or
// in place of its own, appended to dst.
func readsOf(dst []int, li int, g *graph.Graph) []int {
	if g == nil {
		return dst
	}
	if n := g.ReadsHistoryOf(); n >= 0 && n != li && !slices.Contains(dst, n) {
		dst = append(dst, n)
	}
	for _, r := range g.ReadsMore() {
		if r.Block != li && !slices.Contains(dst, r.Block) {
			dst = append(dst, r.Block)
		}
	}
	return dst
}

// reads reports whether placed block l reads block src's history.
func (l *layer) reads(src int) bool {
	if l == nil {
		return false
	}
	if l.kvSrc == src {
		return true
	}
	for _, r := range l.kvMore {
		if r.Block == src {
			return true
		}
	}
	return false
}

// ownsHistoryOf reports that session sid's history of block src here is the
// block's own: placed here, with the session's pages. Callers hold g.mu.
func (g *devTier) ownsHistoryOf(sid uint64, src int) bool {
	l := g.layers[src]
	if l == nil {
		return false
	}
	kvp := l.kv[sid]
	return kvp != nil && kvp.paged
}

// holdsHistoryOf reports that a block placed here may read block src's
// history for session sid: the block placed here, or a copy of its history.
// Callers hold g.mu.
func (g *devTier) holdsHistoryOf(sid uint64, src int) bool {
	if l := g.layers[src]; l != nil && l.ok {
		return true
	}
	return g.histCopies[src][sid]
}

// copySeat raises block src's pool layers' floors to the sessions holding
// them -- its own and the copies' -- and grows them there. Callers hold g.mu.
func (g *devTier) copySeat(kp *kvPool, src int) error {
	seats := len(g.histCopies[src])
	if l := g.layers[src]; l != nil {
		for sid, kvp := range l.kv {
			if kvp.paged && !g.histCopies[src][sid] {
				seats++
			}
		}
	}
	for h := range kp.blockHists(src) {
		l := kp.layers[histKey(src, h)]
		if l == nil {
			continue
		}
		l.floor = 1 + seats*max(l.ctx, 1)
		if l.n < l.floor {
			if err := g.growKVLayer(kp, l, l.floor); err != nil {
				return err
			}
		}
	}
	return nil
}

// lockPair takes two devices' locks in one order, whichever is the source;
// unlockPair gives them back.
func lockPair(a, b *devTier) {
	if b.ord < a.ord {
		a, b = b, a
	}
	a.mu.Lock()
	b.mu.Lock()
}

func unlockPair(a, b *devTier) {
	b.mu.Unlock()
	a.mu.Unlock()
}

// makeHistCopy gives d a copy of session sid's history of block src, which
// runs on o: src's pool layers at o's geometry, window and tail, and every
// page of the session's sequences there copied across. p is the plan of the
// block that reads it, nil once that block is placed: a device the copy is
// the first history of starts its pool at it, as a block of its own would,
// and pages it as o does.
func (g *GPU) makeHistCopy(sid uint64, src int, o, d *devTier, p *nn.LayerPlan) error {
	sv, dv := o.bareView(sid), d.bareView(sid)
	lockPair(sv, dv)
	defer unlockPair(sv, dv)
	fkp := sv.kvp
	if fkp == nil || fkp.layers[src] == nil {
		return fmt.Errorf("block %d keeps no history on %s", src, label(o.dev, o.ord))
	}
	if p != nil {
		dv.initKVCap(p)
	}
	if dv.kvp == nil {
		dv.kvp = newKVPool(fkp.p, dv.maxBuffer())
	}
	dkp := dv.kvp
	if dkp.p != fkp.p {
		return fmt.Errorf("block %d's history pages %d positions on %s and %d on %s: a copy is the pages as they are",
			src, fkp.p, label(o.dev, o.ord), dkp.p, label(d.dev, d.ord))
	}
	for h := range fkp.blockHists(src) {
		key := histKey(src, h)
		fl := fkp.layers[key]
		if fl == nil {
			continue
		}
		if tl := dkp.layers[key]; tl != nil {
			if tl.geom != fl.geom {
				return fmt.Errorf("block %d's history %d is laid out %+v on %s and %+v on %s", src, h, fl.geom, label(o.dev, o.ord), tl.geom, label(d.dev, d.ord))
			}
			continue
		}
		if err := dv.addKVLayer(dkp, key, fl.geom, 2); err != nil {
			return err
		}
		tl := dkp.layers[key]
		tl.win, tl.chunked, tl.tail, tl.ctx = fl.win, fl.chunked, fl.tail, fl.ctx
		if fl.win > 0 {
			dkp.chunked = fl.chunked
		}
	}
	if dv.histCopies == nil {
		dv.histCopies = map[int]map[uint64]bool{}
	}
	if dv.histCopies[src] == nil {
		dv.histCopies[src] = map[uint64]bool{}
	}
	dv.histCopies[src][sid] = true
	dv.copyN.Add(1)
	if err := dv.copySeat(dkp, src); err != nil {
		dv.forgetCopy(sid, src)
		return err
	}
	for s := range fkp.layers[src].owned {
		if s.sid != sid {
			continue
		}
		if err := copySeqPages(sv, dv, s, src, 0, -1, -1); err != nil {
			dv.forgetCopy(sid, src)
			return err
		}
	}
	dv.HistoryCopies++
	return nil
}

// forgetCopy lets session sid's copy of block src's history here go: its
// pages, unless they are the block's own now (it runs here for the session),
// and the pool layers no one holds pages in once the block is not placed
// here. Callers hold g.mu.
func (g *devTier) forgetCopy(sid uint64, src int) {
	if !g.histCopies[src][sid] {
		return
	}
	delete(g.histCopies[src], sid)
	if len(g.histCopies[src]) == 0 {
		delete(g.histCopies, src)
	}
	g.forgetHost(sid, src)
	g.copyN.Add(-1)
	kp := g.kvp
	if kp == nil || g.ownsHistoryOf(sid, src) {
		return
	}
	n := kp.blockHists(src)
	for h := range n {
		g.pagedLeave(histKey(src, h), sid, nil)
	}
	if g.layers[src] != nil || len(g.histCopies[src]) > 0 {
		if err := g.copySeat(kp, src); err != nil {
			g.LastErr = err.Error()
		}
		return
	}
	for h := n - 1; h >= 0; h-- {
		g.dropKVLayer(kp, histKey(src, h))
	}
}

// dropCopiesOf lets every copy session sid holds here go. Callers hold g.mu.
func (g *devTier) dropCopiesOf(sid uint64) {
	for src, m := range g.histCopies {
		if m[sid] {
			g.forgetCopy(sid, src)
		}
	}
}

// keepCopies is what a release of block li leaves of its pool layer key
// while a session holds a copy of its history here: that session's pages,
// every other session's released. Callers hold g.mu.
func (g *devTier) keepCopies(key, li int) {
	l := g.kvp.layers[key]
	if l == nil {
		return
	}
	for s := range l.owned {
		if !g.histCopies[li][s.sid] {
			g.pagedLeave(key, s.sid, nil)
		}
	}
}

// copySeqPages copies sequence s's pages [j0, j1) of every history of block
// src from sv's pools into dv's, which take the pages they lack; j1 -1 is
// every page sv holds. from, when not -1, is where the sequence's rows start
// on this call: dv's record of the sequence comes down to it first, as the
// call's own would (rewind), so no page copied is released under it. A page
// sv holds at home goes from there; one released behind a window on either
// side is not copied, and nothing reads it. Callers hold both locks.
func copySeqPages(sv, dv *devTier, s seqID, src, j0, j1, from int) error {
	fkp, dkp := sv.kvp, dv.kvp
	if fkp == nil || dkp == nil {
		return fmt.Errorf("block %d's history has no pool on one of the devices", src)
	}
	if from >= 0 {
		if err := dv.rewind(s.sid, dkp, s, from); err != nil {
			return err
		}
	}
	P := fkp.p
	pages := 0
	for h := range fkp.blockHists(src) {
		key := histKey(src, h)
		fl, tl := fkp.layers[key], dkp.layers[key]
		if fl == nil {
			continue
		}
		if tl == nil {
			return fmt.Errorf("block %d's history %d has no copy on %s", src, h, label(dv.dev, dv.ord))
		}
		hi := len(fl.owned[s])
		if j1 >= 0 {
			hi = min(hi, j1)
		}
		if hi <= j0 {
			continue
		}
		// A page the prefix cache names on the copy's device is the other
		// holders' too: written, it is copied first (kvshare.go).
		if from >= 0 {
			if err := dv.cowPages(s.sid, dkp, tl, s, from, hi*P); err != nil {
				return err
			}
		}
		if have := len(tl.owned[s]); have < hi {
			if err := dv.allocPages(s.sid, dkp, tl, s, hi-have); err != nil {
				return err
			}
		}
		if err := dv.flushTabs(); err != nil {
			return err
		}
		kw, vw := fl.geom.kWords(P), fl.geom.vWords(P)
		sids, dids := fl.owned[s], tl.owned[s]
		for j := j0; j < hi; {
			if sids[j] == 0 || dids[j] == 0 {
				if err := copyHomePage(sv, dv, fl, tl, s, j, kw, vw); err != nil {
					return err
				}
				j++
				continue
			}
			k := j + 1
			for k < hi && sids[k] != 0 && dids[k] != 0 {
				k++
			}
			for _, half := range [2]struct {
				d, s  backend.Buf
				words int
			}{{tl.k, fl.k, kw}, {tl.v, fl.v, vw}} {
				if half.words == 0 {
					continue
				}
				if err := copyRun(dv, half.d, dids[j:k], sv, fl, half.s, sids[j:k], half.words); err != nil {
					return err
				}
			}
			pages += k - j
			j = k
		}
	}
	dv.HistoryCopyPages += pages
	return nil
}

// copyRun copies pages sids of sv's buffer sb (pool layer fl) into pages dids
// of dv's buffer db, words a page: card to card where the backend copies
// between its own devices (copyPages), and otherwise a page at a time through
// fl's read reserve and a host buffer dv keeps, so a token's append allocates
// nothing. A copy made whole goes through copyPages, which reads groups of
// pages at once. Callers hold both locks.
func copyRun(dv *devTier, db backend.Buf, dids []uint32, sv *devTier, fl *kvLayerPool, sb backend.Buf, sids []uint32, words int) error {
	if peerCopies(dv, sv) || len(sids) > copyRunPages || fl.stage == nil {
		_, err := copyPages(dv, db, dids, sv, fl, sb, sids, words)
		return err
	}
	if cap(dv.copyHost) < words*4 {
		dv.copyHost = make([]byte, words*4)
	}
	host := dv.copyHost[:words*4]
	for i, id := range sids {
		if err := sv.dev.Copy(fl.stage, 0, sb, int(id)*words*4, words*4); err != nil {
			return err
		}
		if err := fl.stage.Read(host); err != nil {
			return err
		}
		if err := db.WriteAt(int(dids[i])*words*4, host); err != nil {
			return err
		}
	}
	return nil
}

// copyRunPages is the most pages copyRun takes a page at a time: a token's
// append is a page or two a history, a copy made whole is every page.
const copyRunPages = 4

// peerCopies reports that dv's backend copies from sv card to card
// (backend.PeerCopier), as one backend's devices do.
func peerCopies(dv, sv *devTier) bool {
	_, ok := dv.dev.(backend.PeerCopier)
	return ok && dv.dev.API() == sv.dev.API()
}

// copyHomePage copies page j of sequence s where one side holds it at home
// or released: a source page at home is written to the copy's page from
// there, a copy's page at home takes the source's bytes there, and a page
// released behind a window on either side is left. Callers hold both locks.
func copyHomePage(sv, dv *devTier, fl, tl *kvLayerPool, s seqID, j, kw, vw int) error {
	sid, did := fl.owned[s][j], tl.owned[s][j]
	var page []float32
	switch {
	case sid == 0 && j < len(fl.home[s]) && fl.home[s][j] != nil:
		page = fl.home[s][j]
	case sid == 0:
		return nil
	default:
		k, err := sv.readPages(fl, fl.k, []uint32{sid}, kw)
		if err != nil {
			return err
		}
		v, err := sv.readPages(fl, fl.v, []uint32{sid}, vw)
		if err != nil {
			return err
		}
		page = append(k, v...)
	}
	switch {
	case did != 0:
		if err := tl.k.WriteAt(int(did)*kw*4, f32b(page[:kw])); err != nil {
			return err
		}
		if vw > 0 {
			return tl.v.WriteAt(int(did)*vw*4, f32b(page[kw:kw+vw]))
		}
	case j < len(tl.home[s]) && tl.home[s][j] != nil:
		tl.home[s][j] = append([]float32(nil), page[:kw+vw]...)
	}
	return nil
}

// readsInto appends the blocks whose histories placed block l, block li,
// reads to dst.
func (l *layer) readsInto(dst []int, li int) []int {
	if l.kvSrc != li && !slices.Contains(dst, l.kvSrc) {
		dst = append(dst, l.kvSrc)
	}
	for _, r := range l.kvMore {
		if r.Block != li && !slices.Contains(dst, r.Block) {
			dst = append(dst, r.Block)
		}
	}
	return dst
}

// copiesFor gives device d the copies block li of plan p reads there for
// session sid: of each block whose history it reads that runs on another
// device for the session -- a card, or the CPU device for the history it
// reads in place of its own (makeHostCopy). made is the copies it made, for
// the caller to let go (dropCopies) when d then declines the block; a copy
// that cannot be made is why d cannot take it.
func (g *GPU) copiesFor(sid uint64, d *devTier, li int, p *nn.LayerPlan) (made []int, err error) {
	for _, src := range readsOf(nil, li, p.Graph) {
		if src >= li {
			continue
		}
		g.mu.Lock()
		o := g.at(sid, src)
		g.mu.Unlock()
		if o == d || o == nil && src != p.Graph.ReadsHistoryOf() {
			continue
		}
		d.mu.Lock()
		have, host := d.histCopies[src][sid], d.hostOf[src][sid]
		if have && host != (o == nil) {
			d.forgetCopy(sid, src)
			have = false
		}
		d.mu.Unlock()
		if have {
			continue
		}
		if o == nil {
			win, chunked := p.Graph.AttendWindow()
			if err := g.makeHostCopy(sid, src, d, d.kvGeomOf(p), win, chunked, historyTail(p.Graph), p); err != nil {
				g.dropCopies(sid, d, made)
				return nil, err
			}
			made = append(made, src)
			continue
		}
		if err := g.makeHistCopy(sid, src, o, d, p); err != nil {
			g.dropCopies(sid, d, made)
			return nil, err
		}
		made = append(made, src)
	}
	return made, nil
}

// dropCopies lets session sid's copies of blocks srcs on d go.
func (g *GPU) dropCopies(sid uint64, d *devTier, srcs []int) {
	if len(srcs) == 0 {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	for _, src := range srcs {
		d.forgetCopy(sid, src)
	}
}

// settleCopies makes the copies the devices hold for session sid the ones its
// routes read: a device running a block for the session that reads a block
// running on another device holds a copy of that block's history, and no
// device holds any other. A copy no block reads goes, one whose block runs on
// its device now is that block's own pages, and one a device lacks is made
// from the source's whole history -- a device the source just left makes it
// over the pages it held. Every placement that changes the session's routes
// ends with it.
func (g *GPU) settleCopies(sid uint64) error {
	g.mu.Lock()
	at := map[int]*devTier{}
	for li := range g.own {
		at[li] = g.at(sid, li)
	}
	for li := range g.route[sid] {
		at[li] = g.at(sid, li)
	}
	devs := slices.Clone(g.devs)
	g.mu.Unlock()
	type hcopy struct {
		d, o *devTier
		src  int
		l    *layer
	}
	var add, drop []hcopy
	var srcs []int
	for _, d := range devs {
		d.mu.Lock()
		// A source the session runs on the CPU device is wanted as the
		// CPU's copy (o nil), of the history read in place of a block's own.
		want := map[int]*devTier{}
		by := map[int]*layer{}
		for li, l := range d.layers {
			if l == nil || !l.ok || at[li] != d {
				continue
			}
			srcs = l.readsInto(srcs[:0], li)
			for _, src := range srcs {
				if o := at[src]; o != d && (o != nil || src == l.kvSrc) {
					want[src], by[src] = o, l
				}
			}
		}
		for src, m := range d.histCopies {
			if !m[sid] {
				continue
			}
			if o, ok := want[src]; ok && (o == nil) == d.hostOf[src][sid] {
				delete(want, src)
				continue
			}
			drop = append(drop, hcopy{d: d, src: src})
		}
		for src, o := range want {
			add = append(add, hcopy{d: d, o: o, src: src, l: by[src]})
		}
		d.mu.Unlock()
	}
	for _, c := range drop {
		c.d.mu.Lock()
		c.d.forgetCopy(sid, c.src)
		c.d.mu.Unlock()
	}
	for _, c := range add {
		var err error
		if c.o == nil {
			geom, win, chunked, tail := c.l.readGeom()
			err = g.makeHostCopy(sid, c.src, c.d, geom, win, chunked, tail, nil)
		} else {
			err = g.makeHistCopy(sid, c.src, c.o, c.d, nil)
		}
		if err != nil {
			err = fmt.Errorf("tier: a copy of block %d's history on %s: %w", c.src, label(c.d.dev, c.d.ord), err)
			c.d.mu.Lock()
			c.d.LastErr = err.Error()
			c.d.mu.Unlock()
			return err
		}
	}
	return nil
}

// noteCopies is settleCopies after a placement that has happened: a copy
// that could not be made is its device's last error, and the block reading
// it is declined there when it runs.
func (g *GPU) noteCopies(sid uint64) {
	if g.copiesIdle(sid) {
		return
	}
	g.settleCopies(sid)
}

// copiesIdle reports that no device holds a copy and no placed block reads
// another block's history: a model without such reads never settles.
func (g *GPU) copiesIdle(sid uint64) bool {
	g.mu.Lock()
	devs := slices.Clone(g.devs)
	g.mu.Unlock()
	for _, d := range devs {
		if d.copyN.Load() > 0 {
			return false
		}
		d.mu.Lock()
		reads := false
		for li, l := range d.layers {
			if l != nil && (l.kvSrc != li || len(l.kvMore) > 0) {
				reads = true
				break
			}
		}
		d.mu.Unlock()
		if reads {
			return false
		}
	}
	return true
}

// shareCopies names, on each device holding session sid's copy of block
// li's history, the copy's page the prefix cache names key on the block's
// device, so a copy another session holds there maps it rather than holding
// the page twice: published there (end -1), the copy's page is named too;
// mapped there (end where the history ends), the copy maps the page its
// device names by key, or takes the source's and names it.
func (g *GPU) shareCopies(sid uint64, li, page int, key string, end int) bool {
	if key == "" {
		return true
	}
	g.mu.Lock()
	o := g.at(sid, li)
	devs := slices.Clone(g.devs)
	g.mu.Unlock()
	if o == nil {
		return true
	}
	for _, d := range devs {
		if d == o || d.copyN.Load() == 0 {
			continue
		}
		d.mu.Lock()
		has := d.histCopies[li][sid]
		d.mu.Unlock()
		if !has {
			continue
		}
		sv, dv := o.bareView(sid), d.bareView(sid)
		lockPair(sv, dv)
		err := shareCopy(sv, dv, seqID{sid, 0}, li, page, key, end)
		if err != nil {
			dv.LastErr = fmt.Sprintf("block %d's history copy on %s: %v", li, label(d.dev, d.ord), err)
		}
		unlockPair(sv, dv)
		if err != nil {
			return false
		}
	}
	return true
}

// shareCopy is shareCopies on one copy. Callers hold both locks.
func shareCopy(sv, dv *devTier, s seqID, li, page int, key string, end int) error {
	kp := dv.kvp
	l := kp.layers[li]
	if l == nil {
		return fmt.Errorf("no copy of block %d's history", li)
	}
	if end < 0 {
		dv.publishKV(kp, l, s, page, key)
		return nil
	}
	ok, err := dv.mapKV(kp, l, s, page, key)
	if err != nil {
		return err
	}
	if !ok {
		if err := copySeqPages(sv, dv, s, li, page, page+1, -1); err != nil {
			return err
		}
		dv.publishKV(kp, l, s, page, key)
	}
	if kp.written == nil {
		kp.written = map[seqID]int{}
	}
	kp.written[s] = max(kp.written[s], end)
	return dv.flushTabs()
}

// copiesRead appends to dst the blocks whose histories blocks [lo, hi) read
// here from session sid's copies. Callers hold g.mu.
func (g *devTier) copiesRead(sid uint64, lo, hi int, dst []int) []int {
	for src, m := range g.histCopies {
		if !m[sid] {
			continue
		}
		for li := lo; li < hi; li++ {
			if g.layers[li].reads(src) {
				dst = append(dst, src)
				break
			}
		}
	}
	return dst
}

// readsCopies reports that a run of rs reads a copy session sid holds of a
// history another device writes.
func (g *GPU) readsCopies(sid uint64, rs []run) bool {
	for _, r := range rs {
		if r.dev.copyN.Load() == 0 {
			continue
		}
		var buf [copiesStack]int
		r.dev.mu.Lock()
		n := len(r.dev.copiesRead(sid, r.lo, r.hi, buf[:0]))
		r.dev.mu.Unlock()
		if n > 0 {
			return true
		}
	}
	return false
}

// copiesStack is the copies a run reads that appendSeq holds on the stack:
// more allocate, fewer do not.
const copiesStack = 8

// appendCopies brings the copies device d holds of session sid's histories
// that blocks [lo, hi) read up to rows [pos, pos+n): the pages those rows
// are in, copied from the device each source runs on, which ran them first
// -- a block reads only blocks below it, and a run is called after the runs
// below it.
func (g *GPU) appendCopies(sid uint64, d *devTier, lo, hi, pos, n int) bool {
	if d.copyN.Load() == 0 || lo >= hi || n <= 0 || d.pagedFaulted("copystale") {
		return true
	}
	return g.appendSeq(seqID{sid, 0}, d, lo, hi, pos, pos+n)
}

// appendSeq is appendCopies for sequence s's rows [start, end).
func (g *GPU) appendSeq(s seqID, d *devTier, lo, hi, start, end int) bool {
	var buf [copiesStack]int
	d.mu.Lock()
	srcs := d.copiesRead(s.sid, lo, hi, buf[:0])
	d.mu.Unlock()
	for _, src := range srcs {
		g.mu.Lock()
		o := g.at(s.sid, src)
		g.mu.Unlock()
		if o == nil || o == d {
			continue
		}
		sv, dv := o.bareView(s.sid), d.bareView(s.sid)
		lockPair(sv, dv)
		P := sv.kvp.p
		err := copySeqPages(sv, dv, s, src, start/P, (end-1)/P+1, start)
		if err != nil {
			dv.LastErr = fmt.Sprintf("block %d's history copy on %s: %v", src, label(d.dev, d.ord), err)
		}
		unlockPair(sv, dv)
		if err != nil {
			return false
		}
	}
	return true
}

// appendRows is appendCopies for a step's rows: each row's sequence (its
// session, sid nil for self's, and the base of its slots) brought up to the
// rows the step writes of it.
func (g *GPU) appendRows(self uint64, d *devTier, lo, hi int, pos, slot []int, sid []uint64) bool {
	if d.copyN.Load() == 0 || lo >= hi || d.pagedFaulted("copystale") {
		return true
	}
	seqOf := func(r int) seqID {
		s := self
		if sid != nil {
			s = sid[r]
		}
		return seqID{s, slot[r] - pos[r]}
	}
	for r := 0; r < len(pos); {
		s, start, end := seqOf(r), pos[r], pos[r]+1
		k := r + 1
		for k < len(pos) && seqOf(k) == s {
			start, end = min(start, pos[k]), max(end, pos[k]+1)
			k++
		}
		if !g.appendSeq(s, d, lo, hi, start, end) {
			return false
		}
		r = k
	}
	return true
}

// refreshCopies brings every copy of session sid's history of block li up to
// pages [j0, j1) of sequence s on the device the block runs on: a history
// migrated in, or a page mapped in from the prefix cache. from, when not -1,
// is where the sequence's history ends there now.
func (g *GPU) refreshCopies(sid uint64, li int, s seqID, j0, j1, from int) bool {
	g.mu.Lock()
	o := g.at(sid, li)
	devs := slices.Clone(g.devs)
	g.mu.Unlock()
	if o == nil {
		return true
	}
	for _, d := range devs {
		if d == o || d.copyN.Load() == 0 {
			continue
		}
		d.mu.Lock()
		has := d.histCopies[li][sid]
		d.mu.Unlock()
		if !has {
			continue
		}
		sv, dv := o.bareView(sid), d.bareView(sid)
		lockPair(sv, dv)
		err := copySeqPages(sv, dv, s, li, j0, j1, from)
		if err != nil {
			dv.LastErr = fmt.Sprintf("block %d's history copy on %s: %v", li, label(d.dev, d.ord), err)
		}
		unlockPair(sv, dv)
		if err != nil {
			return false
		}
	}
	return true
}

// copiesHeld is the copies this device holds, one a block and session.
// Callers hold g.mu.
func (g *devTier) copiesHeld() int {
	n := 0
	for _, m := range g.histCopies {
		n += len(m)
	}
	return n
}
