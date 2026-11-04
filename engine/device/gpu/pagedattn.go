package gpu

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/nn"
)

// Paged attention history on the device (docs/design/device-kv-paging.md).
//
// A device holds one kvPool: per attention layer its own pages (K and V
// buffers, ids, free list) and a persistent table arena, a sequence's run at
// the same offset in every layer's arena. A sequence is a session and the
// base of its rows (seqID), so a LayersRows batch's sequences and a single
// Instance are the same thing here. A session holds pages only in the layers it
// runs here, so leaving a layer frees that layer's pages. Nothing a kernel is
// built from depends on how long a history is: the rows' descriptors are data
// written before each submission (the only per-token upload), the tables take
// a new id at a page boundary, and growth appends pages. That is what lets the
// capacity machinery (sesscap.go) go: every session shares one scratch.
//
// Every block's attention reads its history through the composer's form,
// one row per descriptor; a sliding window is only the descriptor's start,
// so a windowed layer reads a second descriptor set rather than a second
// form.

// maxKVPage is the largest page the pool picks by itself.
const maxKVPage = 256

// pagedScratch is one scratch's paged history: the per-call descriptor
// buffers the composer's forms read.
type pagedScratch struct {
	// pRow/pRowW are the rows' descriptors, the second with each row's
	// window start; every layer reads them, each with its own table arena.
	pRow, pRowW backend.Buf
	// rows, desc and descW are the per-call staging, reused so a token
	// allocates nothing.
	rows        []pagedRow
	desc, descW []uint32
	// held says the current call's session holds an attention block here,
	// so its descriptors are staged (pagedPrep); keys is the call's key
	// count, rounded to the recording grain.
	held bool
	keys int
	// st is the current call's pass over evicted history (pagedstream.go), nil
	// when every page it reads is resident.
	st *pagedStream
	// stream is what st points at when set, and streamD each descriptor set's
	// words for every pass, written with the call's staged writes
	// (pagedStream.writes): both reused call to call, so a decode token that
	// streams allocates nothing once the widest call has been made.
	stream  pagedStream
	streamD [2][]uint32
	// streamDesc are the passes' descriptor buffers, full [0] and windowed
	// [1], grown to the most passes a call has run.
	streamDesc [2][]backend.Buf
	// run is a composed form's running partial over the passes
	// (composer.History.Passes): run[0] the empty partial it starts from, run[1]
	// and run[2] the folds' two sides, each runN floats.
	run  [3]backend.Buf
	runN int
}

// kvPage is the pool's page size, decided when the pool is made and fixed for
// its life: Config.KVPage, or the smallest power of two from 64 that covers
// the context, up to maxKVPage -- a page longer than the context would hold
// history no sequence can reach.
func (g *devTier) kvPage() int {
	if g.kvp != nil {
		return g.kvp.p
	}
	if g.KVPage > 0 {
		return g.KVPage
	}
	p := 64
	for p < maxKVPage && p < g.kvCap {
		p *= 2
	}
	return p
}

// pagedPlan reports that a plan's attention history is paged on this device:
// every text attention, MLA's latent included, but not a tower's, whose keys
// live for one encode and are not history.
func (g *devTier) pagedPlan(p *nn.LayerPlan) bool {
	return g.paged && !p.NonCausal
}

// kvGeomOf is a plan's page geometry. MLA's is one row-major latent row a
// position, float32 (the paged kernels read no packed latent), and no V
// buffer: the value is the row's prefix.
func (g *devTier) kvGeomOf(p *nn.LayerPlan) kvGeom {
	latent := p.MLA()
	return kvGeom{kvRow: p.KVRow(), f16: g.KVF16 && !latent, mla: latent}
}

// kvPages is the device's KV pool, made on first use. Callers hold g.mu.
func (g *devTier) kvPages() *kvPool {
	if g.kvp == nil {
		g.kvp = newKVPool(g.kvPage(), g.maxBuffer())
	}
	return g.kvp
}

// initPagedScratch builds bs's paged set: the rows' descriptors, full and
// windowed, and the entry descriptors of a block that compresses its
// history (entdesc.go). Callers hold g.mu.
func (g *devTier) initPagedScratch(bs *blockScratch) error {
	defer g.scratchWin().close() // the scratch's own buffers (scratch.go)
	p := &bs.p
	pg := &pagedScratch{}
	bs.pkv = pg
	var err error
	if pg.pRow, err = g.dev.Alloc(bs.rows * composer.RowWords * 4); err != nil {
		return err
	}
	if p.SWAWindow > 0 {
		if pg.pRowW, err = g.dev.Alloc(bs.rows * composer.RowWords * 4); err != nil {
			return err
		}
	}
	return nil
}

// freePaged releases a scratch's paged set.
func freePaged(pg *pagedScratch) {
	if pg == nil {
		return
	}
	for _, bufs := range pg.streamDesc {
		for _, b := range bufs {
			b.Free()
		}
	}
	for _, b := range pg.run {
		if b != nil {
			b.Free()
		}
	}
	for _, b := range []backend.Buf{pg.pRow, pg.pRowW} {
		if b != nil {
			b.Free()
		}
	}
}

// seqEnd is one sequence a call writes and reads: every position below end
// must have a page. start is the first position the call writes, or -1 for a
// reservation that writes nothing yet: a windowed layer releases its pages
// behind start's window, and only a call that writes knows where that is.
type seqEnd struct {
	s          seqID
	end, start int
}

// seqEnds collapses a call's rows into dst, one entry per sequence: its last
// position plus one. A prompt chunk's rows are one sequence, so the pages are
// checked once and not once a row in every layer -- 28 x 512 map reads on a
// 512-row prompt, twice. A row of a sequence already seen extends that entry
// (a chunk's rows are in order; a batch's are distinct sequences).
func seqEnds(dst []seqEnd, rows []pagedRow) []seqEnd {
	dst = dst[:0]
	for _, r := range rows {
		if r.pad {
			continue
		}
		if n := len(dst); n > 0 && dst[n-1].s == r.s {
			dst[n-1].end = max(dst[n-1].end, r.pos+1)
			dst[n-1].start = min(dst[n-1].start, r.pos)
			continue
		}
		dst = append(dst, seqEnd{r.s, r.pos + 1, r.pos})
	}
	return dst
}

// heldLayers calls f for every pool layer session sid holds history
// on: a session takes pages only where it runs.
func (g *devTier) heldLayers(sid uint64, f func(li int, l *kvLayerPool) error) error {
	if g.kvp == nil {
		return nil
	}
	for li, l := range g.kvp.layers {
		if bl := g.layers[li]; bl != nil {
			if kvp := bl.kv[sid]; kvp != nil && kvp.paged {
				if err := f(li, l); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// pagedAppend gives every sequence the pages its positions need in every
// layer the session holds, before a submission: a layer that grows may page
// weights out (roomOrReclaim) and drops the captured graph, neither of which
// can happen once blocks are paged in for a submission. A layer that cannot
// grow refuses with ErrKVCapacity, and what was already taken stays taken.
// Callers hold g.mu.
func (g *devTier) pagedAppend(sid uint64, seqs []seqEnd) error {
	kp := g.kvPages()
	for _, se := range seqs {
		if se.start >= 0 {
			if err := g.rewind(sid, kp, se.s, se.start); err != nil {
				return err
			}
		}
	}
	err := g.heldLayers(sid, func(li int, l *kvLayerPool) error {
		for _, se := range seqs {
			need := (l.positions(se.end) + kp.p - 1) / kp.p
			// A named page this call writes into is copied first (kvshare.go).
			if err := g.cowPages(sid, kp, l, se.s, se.start, se.end); err != nil {
				return err
			}
			if have := len(l.owned[se.s]); need > have {
				if err := g.allocPages(sid, kp, l, se.s, need-have); err != nil {
					return err
				}
			}
			if l.win > 0 && se.start >= 0 && !g.KVKeepWindow {
				if err := g.windowPages(sid, kp, li, l, se); err != nil {
					return err
				}
			}
		}
		return nil
	})
	if err == nil {
		err = g.streamRoom(sid)
	}
	if ferr := g.flushTabs(); err == nil {
		err = ferr
	}
	return err
}

// windowPages keeps a windowed layer's pages to what se's rows read: a page
// the call writes into that was released (a row restarting at a lower
// position) gets a fresh id, a page it reads that was released is refused --
// its history is gone -- and every page behind the window of
// se.start-nn.KVWindowSlack is released, so the layer holds the window and
// not the context. Callers hold g.mu, between submissions.
func (g *devTier) windowPages(sid uint64, kp *kvPool, li int, l *kvLayerPool, se seqEnd) error {
	s, P := se.s, kp.p
	need := (se.end + P - 1) / P
	ev := kp.evicted[s]
	for j := se.start / P; j < need; j++ {
		if l.owned[s][j] == 0 && j >= ev {
			if err := g.allocAt(sid, kp, l, s, j); err != nil {
				return err
			}
		}
	}
	fault := g.pagedFaulted("winrelease")
	for j := l.keyStart(se.start) / P; j < se.start/P && !fault; j++ {
		if l.owned[s][j] == 0 && (j >= ev || l.home[s][j] == nil) {
			return fmt.Errorf("block %d: sequence %v reads position %d's window from page %d, released "+
				"behind it", li, s, se.start, j)
		}
	}
	first := min(l.keyStart(max(0, se.start-nn.KVWindowSlack))/P, len(l.owned[s]))
	if fault {
		// The violation: the page the window starts in goes too, the one being
		// written into when the window lies inside it.
		first = min(l.keyStart(se.start)/P+1, len(l.owned[s]))
	}
	if l.rel == nil {
		l.rel = map[seqID]int{}
	}
	for j := l.rel[s]; j < first; j++ {
		if id := l.owned[s][j]; id != 0 {
			l.releaseID(id)
			l.owned[s][j] = 0
			g.tabPend = append(g.tabPend, tabWrite{l.tab, kp.ranges[s].off + j, l.owned[s][j : j+1]})
			g.KVWindowReleased++
		}
		if j < len(l.home[s]) {
			l.home[s][j] = nil // in host memory, and behind the window all the same
		}
	}
	l.rel[s] = max(l.rel[s], first)
	return nil
}

// allocAt gives sequence s a fresh id for its page j in layer l, which holds
// none (it was released behind a window). Callers hold g.mu, outside any
// submission.
func (g *devTier) allocAt(sid uint64, kp *kvPool, l *kvLayerPool, s seqID, j int) error {
	if len(l.free) < 1 {
		l.reclaimNamed(1)
	}
	if len(l.free) < 1 {
		if err := g.growKVLayer(kp, l, l.n+1-len(l.free)); err != nil && !g.evictForRoom(sid, kp, l, 1) {
			return err
		}
	}
	id := l.free[len(l.free)-1]
	l.free = l.free[:len(l.free)-1]
	l.owned[s][j] = id
	g.tabPend = append(g.tabPend, tabWrite{l.tab, kp.ranges[s].off + j, l.owned[s][j : j+1]})
	if l.rel[s] > j {
		l.rel[s] = j
	}
	return nil
}

// pagedAppendRows is pagedAppend for a call's rows, from outside g.mu.
func (g *devTier) pagedAppendRows(sid uint64, rows []pagedRow) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.paged {
		return true
	}
	g.pgSeqs = seqEnds(g.pgSeqs, rows)
	if err := g.pagedAppend(sid, g.pgSeqs); err != nil {
		g.LastErr = err.Error()
		return false
	}
	return true
}

// pagedPrep readies bs for a call's rows: a check that every row's page is
// there (pagedAppend took them), and whether the session holds an attention
// block here to stage descriptors for. Callers hold g.mu.
func (g *devTier) pagedPrep(sid uint64, bs *blockScratch, rows []pagedRow) error {
	kp, pg := g.kvPages(), bs.pkv
	held := false
	g.pgSeqs = seqEnds(g.pgSeqs, rows)
	err := g.heldLayers(sid, func(li int, l *kvLayerPool) error {
		held = true
		for _, se := range g.pgSeqs {
			if (l.positions(se.end)-1)/kp.p >= len(l.owned[se.s]) {
				return fmt.Errorf("block %d: sequence %v holds %d page(s) and writes position %d",
					li, se.s, len(l.owned[se.s]), se.end-1)
			}
		}
		return nil
	})
	pg.held = held && err == nil
	if !pg.held {
		return err
	}
	// The key count rounded to the recording grain, so a captured graph
	// serves every token of a grain (graphKey.keys).
	pg.keys = msaKeys(&bs.p, idxKeys(&bs.p, (pagedKeys(rows)+scoreGrain-1)/scoreGrain*scoreGrain))
	return nil
}

// pagedRow is one query row of a call: its sequence and its position.
// A padding row has pad set and reads and writes only the dummy page.
type pagedRow struct {
	s   seqID
	pos int
	pad bool
}

// pagedStage fills pg's descriptor slabs for a call's rows, full and
// windowed: each row's sequence run, the same offset in every layer's arena.
// Callers hold g.mu and have run pagedPrep for the same rows.
func (g *devTier) pagedStage(pg *pagedScratch, p *nn.LayerPlan, rows []pagedRow) {
	kp := g.kvPages()
	pg.desc, pg.descW = pg.desc[:0], pg.descW[:0]
	for _, r := range rows {
		if r.pad {
			pg.desc = append(pg.desc, 0, 0, 0, 0)
			if p.SWAWindow > 0 {
				pg.descW = append(pg.descW, 0, 0, 0, 0)
			}
			continue
		}
		o := uint32(kp.ranges[r.s].off)
		end := uint32(r.pos + 1)
		// A row of a bidirectional run attends to the run's end (bidir.go);
		// its window still starts from its own position, as the reference's
		// does, which the start word here carries apart from the end.
		last := end
		if e := bidirCount(g.bidir, r.pos); e > 0 && !p.BidirOff {
			last = uint32(e)
		}
		pg.desc = append(pg.desc, o, 0, last, uint32(r.pos))
		if p.SWAWindow > 0 {
			// The window as a start: the last W keys, or a chunk of W aligned
			// keys.
			var lo uint32
			if w := uint32(p.SWAWindow); p.SWAChunked {
				lo = (end - 1) / w * w
			} else {
				lo = end - min(end, w)
			}
			pg.descW = append(pg.descW, o, lo, last, uint32(r.pos))
		}
	}
}

// windowDesc is the descriptors block l's attention reads: the windowed ones
// where its graph attends a window, the full ones elsewhere.
func (g *devTier) windowDesc(pg *pagedScratch, l *layer) backend.Buf {
	// A scratch with no windowed set is one whose context fits inside the
	// window (model.windowFor): its rows' windows are the full ones.
	if l.win > 0 && pg.pRowW != nil && !g.pagedFaulted("window") {
		return pg.pRowW
	}
	return pg.pRow
}

// pagedRows fills dst with a call's rows: one sequence's positions pos.. for a
// decode or a prefill chunk, or a LayersRows step's sequences, with R-nrow
// padding rows. dst is reused, so a token allocates nothing.
func (g *devTier) pagedRows(sid uint64, dst []pagedRow, R, nrow, pos int, rag *ragStep) []pagedRow {
	rows := slices.Grow(dst[:0], R)[:R]
	for r := range rows {
		switch {
		case rag != nil && r < len(rag.pos):
			base := rag.slot[r] - rag.pos[r]
			if g.pagedFaulted("alias") {
				base = rag.slot[0] - rag.pos[0]
			}
			s := sid
			if rag.sid != nil && !g.pagedFaulted("session") {
				s = rag.sid[r]
			}
			at := rag.pos[r]
			if r == 1 && g.pagedFaulted("pos") {
				at /= 2
			}
			rows[r] = pagedRow{s: seqID{s, base}, pos: at}
		case rag == nil && r < nrow:
			rows[r] = pagedRow{s: seqID{sid, 0}, pos: pos + r}
		default:
			rows[r] = pagedRow{pad: true}
		}
	}
	return rows
}

// pagedKeys is the most keys any of a call's rows attends.
func pagedKeys(rows []pagedRow) int {
	n := 1
	for _, r := range rows {
		if !r.pad {
			n = max(n, r.pos+1)
		}
	}
	return n
}

// pagedLayer gives block li its place in the pool, and session sid its
// seat there: the layer's buffers and table arena, and its working set -- the
// dummy and one page per attached session, since a session with no page
// cannot run a token. That is a first frame, not a reservation of context:
// past it the layer grows as sequences write, and a growth that does not fit
// pages weights out or refuses (ErrKVCapacity). The session's history on the
// block is the marker kvPair{paged: true}. Callers hold g.mu.
func (g *devTier) pagedLayer(sid uint64, l *layer, li int, p *nn.LayerPlan) error {
	kp := g.kvPages()
	seats := 1
	for o, kvp := range l.kv {
		if kvp.paged && o != sid {
			seats++
		}
	}
	if err := g.addKVLayer(kp, li, g.kvGeomOf(p), 1+seats); err != nil {
		return err
	}
	// The window paged attention applies to this block (pagedStage's descW).
	kp.layers[li].win, kp.layers[li].chunked = l.win, l.chunked
	// A selection over the whole history (the lightning indexer, MiniMax-M3's
	// and QSA's block selections) is not served by a streamed pass, so its
	// pages cannot go to host memory.
	kp.layers[li].noEvict = selectsHistory(p)
	// A history that cannot leave the card is held at the context asked for,
	// charged here with the block, so the budget declines the block at
	// placement rather than a call failing for want of a page mid-prompt. It
	// is the layer's floor (pagedSeat), so a compaction for another layer's
	// room does not give it back.
	if selectsHistory(p) {
		kp.layers[li].ctx = (g.capped(p).MaxSeq + kp.p - 1) / kp.p
	}
	if l.kv == nil {
		l.kv = map[uint64]*kvPair{}
	}
	fresh := l.kv[sid] == nil
	l.kv[sid] = &kvPair{paged: true}
	if _, ok := kp.layers[li]; ok {
		if err := g.pagedSeat(kp, li, l); err != nil {
			if fresh {
				delete(l.kv, sid)
			}
			return err
		}
	}
	return nil
}

// selectsHistory reports a block whose attention selects from its whole
// history: the lightning indexer, MiniMax-M3's and QSA's block selections.
func selectsHistory(p *nn.LayerPlan) bool { return p.IdxHeads != 0 || p.QSA() }

// contextPages is the pages a block that holds its context needs for seats
// sequences of the plan's longest context, and the dummy.
func contextPages(kp *kvPool, maxSeq, seats int) int {
	return 1 + seats*((maxSeq+kp.p-1)/kp.p)
}

// pagedSeat raises layer li's floor to its attached sessions' working set --
// or their contexts, where the layer holds them (contextPages) -- and grows it
// there. Callers hold g.mu.
func (g *devTier) pagedSeat(kp *kvPool, li int, bl *layer) error {
	l := kp.layers[li]
	seats := 0
	for _, kvp := range bl.kv {
		if kvp.paged {
			seats++
		}
	}
	l.floor = 1 + seats*max(l.ctx, 1)
	if l.n < l.floor {
		return g.growKVLayer(kp, l, l.floor)
	}
	return nil
}

// pagedLayerBytes is what placing another attention block costs: a layer at
// the working set a session brings, and its table arena.
func (g *devTier) pagedLayerBytes(p *nn.LayerPlan) uint64 {
	geom := g.kvGeomOf(p)
	words := 256
	if g.kvp != nil {
		words = g.kvp.tabWords
	}
	pages := uint64(1 + g.sessions())
	if selectsHistory(p) && g.kvp != nil {
		// Held at the context asked for (pagedLayer).
		pages = uint64(contextPages(g.kvp, g.capped(p).MaxSeq, int(g.sessions())))
	}
	n := uint64(geom.kWords(g.kvPage())+geom.vWords(g.kvPage()))*4*pages + uint64(words)*4 +
		kvStageBytes(geom, g.kvPage())
	return n
}

// dropKVLayer frees layer li's pages, table and buffers and refunds them. The
// sequences' runs stay while another layer holds pages of them. Callers hold
// g.mu.
func (g *devTier) dropKVLayer(kp *kvPool, li int) {
	l, ok := kp.layers[li]
	if !ok {
		return
	}
	g.freeKVLayer(l)
	delete(kp.layers, li)
	for s := range l.owned {
		kp.dropRange(s)
	}
}

// pagedLeave gives back session sid's pages in layer li to the layer's free
// list, so what the session held there is room again -- the next sequence's
// pages, or given back to the budget by kvCompact when something else needs
// it: the point of a session yielding one block while another session keeps
// it. Callers hold g.mu, between submissions: the released pages are fenced
// until every submission that could read them has completed (fence).
func (g *devTier) pagedLeave(li int, sid uint64, bl *layer) {
	if g.kvp == nil {
		return
	}
	l := g.kvp.layers[li]
	if l == nil {
		return
	}
	for s := range l.owned {
		if s.sid == sid {
			l.releasePages(s)
			g.kvp.dropRange(s)
			g.roomGen.Add(1) // free pages are room (kvCompact)
		}
	}
	l.fence()
	if bl != nil {
		// The seat goes with the session: pagedSeat recounts.
		if err := g.pagedSeat(g.kvp, li, bl); err != nil {
			g.LastErr = err.Error()
		}
	}
}

// kvCompact makes room for n bytes out of the pool's free pages: every layer
// but keep is compacted to what its sequences hold and its floor, one at a
// time until n fits, and it reports whether n fits. Released pages otherwise
// stay in the pool -- the pages one request gives back are the next one's, and
// a pool that shrank on every release grew again on every prompt, an
// allocation and a copy a layer a request. keep is a layer being grown, whose
// ids its caller is still counting. Callers hold g.mu, outside any submission.
func (g *devTier) kvCompact(n uint64, keep *kvLayerPool) bool {
	if g.room(n) {
		return true
	}
	if g.kvp == nil {
		return false
	}
	for _, l := range g.kvp.layers {
		if l == nil || l == keep {
			continue
		}
		l.fence()
		if err := g.shrinkKVLayer(g.kvp, l); err != nil {
			g.LastErr = err.Error()
		}
		if g.room(n) {
			return true
		}
	}
	return false
}

// trimSeq keeps the pages sequence s needs for pos positions in every layer
// and releases the rest, reporting whether any went. Callers hold g.mu.
func (kp *kvPool) trimSeq(s seqID, pos int) bool {
	keep := (pos + kp.p - 1) / kp.p
	trimmed := false
	for _, l := range kp.layers {
		ids := l.owned[s]
		if keep >= len(ids) {
			continue
		}
		l.fenceIDs(ids[keep:])
		if keep == 0 {
			delete(l.owned, s)
		} else {
			l.owned[s] = ids[:keep]
		}
		if l.rel[s] > keep {
			l.rel[s] = keep
		}
		trimmed = true
	}
	kp.forgetFrom(s, keep)
	if kp.written[s] > pos {
		kp.written[s] = pos
	}
	kp.dropRange(s)
	return trimmed
}

// pagedMigrate moves layer li's history for session sid's sequence
// between host and device: k and v are [pos][kvRow] float32 on the host. To
// the device each page is laid out on the host in the layer's layout and
// written at its offset; from the device the sequence's pages are gathered
// into one buffer by a generated kernel and read back. Callers hold g.mu.
func (g *devTier) pagedMigrate(sid uint64, li, base int, k, v []float32, pos int, toDevice bool) error {
	kp := g.kvPages()
	pl, ok := kp.layers[li]
	if !ok {
		return fmt.Errorf("block %d has no pages on this device", li)
	}
	s := seqID{sid, base}
	kvRow, P := pl.geom.kvRow, kp.p
	g.KVPositionsMigrated += pos
	if pos*kvRow > len(k) || pos*kvRow > len(v) {
		return fmt.Errorf("%d positions of %d floats do not fit the host's %d/%d", pos, kvRow, len(k), len(v))
	}
	npages := (pos + P - 1) / P
	// The sequence's oldest pages may be in host memory (kvevict.go): the same count
	// in every layer, so a layer joining keeps them there too.
	// Sent up, the host's position is the history: past it the device's
	// record is a sequence that has since restarted lower. Read back, pos is
	// only how much of the history is wanted (a prefix a store saves), and the
	// device keeps the rest.
	if toDevice {
		if err := g.rewind(sid, kp, s, pos); err != nil {
			return err
		}
	}
	e := min(kp.evicted[s], npages)
	if toDevice && e > 0 && len(pl.owned[s]) == 0 {
		if err := g.ensureRange(kp, s, e); err != nil {
			return err
		}
		pl.owned[s] = make([]uint32, e)
		if err := pl.tab.WriteAt(kp.ranges[s].off*4, u32view(pl.owned[s])); err != nil {
			return err
		}
	}
	if toDevice {
		if have := len(pl.owned[s]); npages > have {
			if err := g.allocPages(sid, kp, pl, s, npages-have); err != nil {
				return err
			}
		}
		if err := g.flushTabs(); err != nil {
			return err
		}
	}
	ids := pl.owned[s]
	if len(ids) < npages {
		return fmt.Errorf("the sequence holds %d page(s) and %d positions need %d", len(ids), pos, npages)
	}
	kw, vw := pl.geom.kWords(P), pl.geom.vWords(P)
	// A binary16 V pool's pages are narrowed and widened on the card.
	var hv *pageHalves
	if pl.geom.f16 && vw > 0 {
		var err error
		if hv, err = g.newPageHalves(kvRow, P); err != nil {
			return err
		}
		defer hv.close()
	}
	if toDevice {
		kPage := make([]float32, kw)
		vPage := make([]float32, P*kvRow)
		for j := 0; j < npages; j++ {
			clear(kPage)
			clear(vPage)
			for t := j * P; t < min(pos, (j+1)*P); t++ {
				if pl.geom.mla {
					// MLA's row is row-major: the latent and its key, as the
					// host holds them.
					copy(kPage[(t%P)*kvRow:], k[t*kvRow:(t+1)*kvRow])
					continue
				}
				for e := 0; e < kvRow; e++ {
					kPage[e*P+t%P] = k[t*kvRow+e]
				}
				copy(vPage[(t%P)*kvRow:], v[t*kvRow:(t+1)*kvRow])
			}
			if ids[j] == 0 {
				if pl.home == nil {
					pl.home = map[seqID][][]float32{}
				}
				vWords := vPage[:vw] // MLA has none: the value is the key row's prefix
				if hv != nil {
					var err error
					if vWords, err = hv.narrowPage(vPage, nil, 0); err != nil {
						return err
					}
				}
				page := append(append(make([]float32, 0, kw+vw), kPage...), vWords...)
				pl.home[s] = append(pl.home[s][:j], page)
				continue
			}
			if err := pl.k.WriteAt(int(ids[j])*kw*4, f32b(kPage)); err != nil {
				return err
			}
			if vw == 0 {
				continue
			}
			if hv != nil {
				if _, err := hv.narrowPage(vPage, pl.v, int(ids[j])*vw*4); err != nil {
					return err
				}
				continue
			}
			if err := pl.v.WriteAt(int(ids[j])*vw*4, f32b(vPage)); err != nil {
				return err
			}
		}
		return nil
	}
	kAll, err := g.readPages(pl, pl.k, ids[e:npages], kw)
	if err != nil {
		return err
	}
	var vAll []float32
	switch {
	case hv != nil:
		// Widened on the card a page at a time, those in host memory too.
		vAll = make([]float32, npages*P*kvRow)
		for j := 0; j < npages; j++ {
			out := vAll[j*P*kvRow : (j+1)*P*kvRow]
			if j < e {
				if page := pl.home[s][j]; page != nil {
					if err := hv.widenPage(nil, 0, page[kw:], out); err != nil {
						return err
					}
				}
				continue
			}
			if err := hv.widenPage(pl.v, int(ids[j])*vw*4, nil, out); err != nil {
				return err
			}
		}
	case vw > 0:
		if vAll, err = g.readPages(pl, pl.v, ids[e:npages], vw); err != nil {
			return err
		}
	}
	if e > 0 {
		// The pages in host memory ahead of the ones read from the card. A windowed
		// layer's page released behind its window has no copy: zeros, which
		// nothing reads (the host trusts this history only from its window).
		hk, hvs := make([]float32, 0, npages*kw), make([]float32, 0, npages*vw)
		for _, page := range pl.home[s][:e] {
			if page == nil {
				hk, hvs = append(hk, make([]float32, kw)...), append(hvs, make([]float32, vw)...)
				continue
			}
			hk, hvs = append(hk, page[:kw]...), append(hvs, page[kw:]...)
		}
		kAll = append(hk, kAll...)
		if hv == nil {
			vAll = append(hvs, vAll...)
		}
	}
	for t := 0; t < pos; t++ {
		j, o := t/P, t%P
		if pl.geom.mla {
			// Row-major, and the value is the row's prefix. The host's latent
			// pages alias k and v and are written from both, so both carry
			// the row: a v left zero overwrites the k just restored.
			row := kAll[j*kw+o*kvRow : j*kw+(o+1)*kvRow]
			copy(k[t*kvRow:(t+1)*kvRow], row)
			copy(v[t*kvRow:(t+1)*kvRow], row)
			continue
		}
		for e := 0; e < kvRow; e++ {
			k[t*kvRow+e] = kAll[j*kw+e*P+o]
		}
		// Widened, a binary16 page's floats are a float page's.
		copy(v[t*kvRow:(t+1)*kvRow], vAll[j*P*kvRow+o*kvRow:j*P*kvRow+(o+1)*kvRow])
	}
	return nil
}

// readPages gathers pages ids of one of layer l's buffers, words each, into
// one staging buffer on the device and reads it back. When the card refuses
// that buffer the pages are read back a page at a time through the layer's
// reserve (kvLayerPool.stage): a park or an eviction is how a full card makes
// room, so it must not need room of its own.
func (g *devTier) readPages(l *kvLayerPool, src backend.Buf, ids []uint32, words int) ([]float32, error) {
	if len(ids) == 0 || words == 0 {
		return nil, nil
	}
	out := make([]float32, len(ids)*words)
	dst, err := g.dev.Alloc(len(ids) * words * 4)
	if err != nil {
		if l.stage == nil {
			return nil, err
		}
		g.KVStageReads++
		return out, g.readStaged(l, src, ids, words, out)
	}
	defer dst.Free()
	if err := g.gatherInto(src, dst, ids, words); err != nil {
		return nil, err
	}
	return out, dst.Read(f32b(out))
}

// readStaged is readPages through l's reserve, a page a pass: nothing is
// allocated on the device.
func (g *devTier) readStaged(l *kvLayerPool, src backend.Buf, ids []uint32, words int, out []float32) error {
	for i, id := range ids {
		if err := g.dev.Copy(l.stage, 0, src, int(id)*words*4, words*4); err != nil {
			return err
		}
		if err := l.stage.Read(f32b(out[i*words : (i+1)*words])); err != nil {
			return err
		}
	}
	return nil
}

// PerSequenceKV reports that this device pages its history (nn.SeqKVDevice).
func (g *devTier) PerSequenceKV() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.paged
}

// MigrateKVSeq is migrateKVSeq for the zero session: a caller that never attached.
func (g *devTier) MigrateKVSeq(li, base int, k, v []float32, pos int, toDevice bool) bool {
	return g.migrateKVSeq(0, li, base, k, v, pos, toDevice)
}

// migrateKVSeq moves layer li's history for session sid's sequence at
// base between host and device (nn.SeqKVDevice).
func (g *devTier) migrateKVSeq(sid uint64, li, base int, k, v []float32, pos int, toDevice bool) bool {
	sv := g.as(sid)
	if sv == nil {
		return false
	}
	defer sv.done()
	g = sv
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.paged {
		g.LastErr = "MigrateKVSeq: this device keeps the contiguous cache"
		return false
	}
	// A KV-sharing block has no history of its own to move: it reads its
	// source's, which moves with the source.
	if l := g.layers[li]; pos <= 0 || l != nil && l.kvSrc != li {
		return true
	}
	if err := g.pagedMigrate(sid, li, base, k, v, pos, toDevice); err != nil {
		g.LastErr = fmt.Sprintf("MigrateKVSeq: block %d, sequence at %d: %v", li, base, err)
		return false
	}
	return true
}

// ReserveKVSeqs is reserveKVSeqs for the zero session: a caller that never attached.
func (g *devTier) ReserveKVSeqs(bases, ends []int) bool {
	return g.reserveKVSeqs(0, bases, ends)
}

// reserveKVSeqs takes the pages session sid's sequences at bases need
// for positions up to ends (nn.SeqKVDevice).
func (g *devTier) reserveKVSeqs(sid uint64, bases, ends []int) bool {
	// No scratch is read: the session's pages and table run are all it
	// touches, so it takes no lane.
	g = g.bareView(sid)
	g.mu.Lock()
	defer g.mu.Unlock()
	g.asking = true
	defer func() { g.asking = false }()
	if !g.paged {
		g.LastErr = "ReserveKVSeqs: this device keeps the contiguous cache"
		return false
	}
	g.pgSeqs = g.pgSeqs[:0]
	for i, b := range bases {
		g.pgSeqs = append(g.pgSeqs, seqEnd{seqID{sid, b}, ends[i], -1})
	}
	if err := g.pagedAppend(sid, g.pgSeqs); err != nil {
		g.LastErr = err.Error()
		return false
	}
	return true
}

// pagedHold gives session sid's sequence the pages pos positions of
// its history need, in every layer it holds, growing a layer that is short.
// It takes them rather than counting free ones as room: every caller reserves
// positions it is about to write, and a free page is exactly what kvCompact
// gives back when another layer grows -- counted and not taken, it was gone by
// the time the token asked for it. Callers hold g.mu.
func (g *devTier) pagedHold(sid uint64, pos int) error {
	g.pgSeqs = append(g.pgSeqs[:0], seqEnd{seqID{sid, 0}, pos, -1})
	return g.pagedAppend(sid, g.pgSeqs)
}

// kvPoolBytes is what the pool holds against the budget: every layer's
// buffers and tables, the dummy pages included. Callers hold g.mu.
func (g *devTier) kvPoolBytes() uint64 {
	if g.kvp == nil {
		return 0
	}
	return g.kvp.charged()
}

// pagedHeldBy is what session sid's pages cost on this device, over every
// layer it holds.
func (g *devTier) pagedHeldBy(sid uint64) uint64 {
	if g.kvp == nil {
		return 0
	}
	var n uint64
	for _, l := range g.kvp.layers {
		for s, ids := range l.owned {
			if s.sid == sid {
				// Resident pages: an id of 0 is in host memory or released.
				for _, id := range ids {
					if id != 0 {
						n += g.kvp.pageBytes(l)
					}
				}
			}
		}
	}
	return n
}
