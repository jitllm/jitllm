package tier

import (
	"errors"
	"fmt"
	"slices"
	"unsafe"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// ErrKVCapacity is a device that cannot give a sequence another page of
// attention history: the budget, the backend's buffer limit or the index
// width stops a layer's pool growing. Nothing changed; the caller may evict,
// move blocks, wait for room or give up, and the model is not down. See
// docs/design/device-kv-paging.md.
var ErrKVCapacity = errors.New("tier: no room for another page of attention history")

// kvGeom is what one attention layer's page holds: kvRow floats of K per
// position (transposed within the page), and V beside it -- f32, binary16
// packed two to a word, or none under MLA, where V is K's prefix.
type kvGeom struct {
	kvRow int
	f16   bool
	mla   bool
}

// kWords and vWords are one page's K and V in 32-bit words.
func (k kvGeom) kWords(p int) int { return p * k.kvRow }
func (k kvGeom) vWords(p int) int {
	switch {
	case k.mla:
		return 0
	case k.f16:
		return p * k.kvRow / 2
	}
	return p * k.kvRow
}

// seqID names a sequence: a session, and the base of its rows' slots (0 for a
// single State).
type seqID struct {
	sid  uint64
	base int
}

// kvLayerPool is one attention layer's pages on a device. Page ids are the
// LAYER's: a physical slot in this layer's K and V buffers, so a session that
// leaves this layer frees exactly this layer's pages, and every layer grows and
// shrinks on its own. Id 0 is the dummy, zero-filled and never owned (padded
// rows write it, empty rows read it).
type kvLayerPool struct {
	geom kvGeom
	k, v backend.Buf
	n    int // pages the buffers hold, the dummy included
	free []uint32
	// fenced are ids released while a submission may still read them, and
	// fencedAt the ticket each was released at (subClock.seq then); fence()
	// frees them once nothing that could read them is in flight. clk is the
	// device's clock, nil in a pool built outside one, where a fence frees
	// everything.
	fenced   []uint32
	fencedAt []uint64
	clk      *subClock
	owned    map[seqID][]uint32
	// tab is the layer's table arena, kvPool.tabWords u32: sequence s's ids
	// at ranges[s].off, in position order, entry 0 the dummy.
	tab backend.Buf
	// charged is what the layer holds against the budget: its buffers at n
	// pages, its table arena and its read reserve.
	charged uint64
	// stage and stageTab are the reserve a read of the layer's pages goes
	// through when the card refuses a buffer for it (readPages): one page of
	// the wider half, and the gather's one-id table. They are made with the
	// layer, while the card has room, and charged with it: a park or an
	// eviction runs on a full card, which is exactly when an allocation for
	// the read is refused (a 4 GiB card whose budget the server set past
	// what the driver had free).
	stage, stageTab backend.Buf
	// sh is the layer's pages the prefix cache names (kvshare.go).
	sh sharedIDs
	// floor is the fewest pages a shrink leaves: the dummy and a page for each
	// session attached to the layer (its working set).
	floor int
	// home holds sequence s's evicted pages at home, page j's K words then its
	// V words, in the device's layout (kvevict.go); owned[s][j] is 0 for each.
	home map[seqID][][]float32
	// win is the block's window in keys, 0 for one that reads its whole
	// history; chunked makes it aligned chunks. A windowed layer releases the
	// pages behind its window (devTier.windowPages): their ids are 0 in the
	// sequence's run and, below the evicted prefix, their home copy is nil.
	win     int
	chunked bool
	// rel is where windowPages' walk starts for sequence s: every page below
	// it was released.
	rel map[seqID]int
	// rate is set on a DeepSeek V4 block's entries (entKey): its positions
	// are entries, one per rate positions of the block's, and of is the block.
	// noEvict keeps the pool's pages on the card (ds4.go).
	rate, of int
	noEvict  bool
}

// keyStart is the first key position pos attends on this layer: pagedStage's
// window start, or 0 on a layer with no window.
func (l *kvLayerPool) keyStart(pos int) int {
	switch {
	case l.win <= 0 || pos <= 0:
		return 0
	case l.chunked:
		return pos / l.win * l.win
	}
	return max(0, pos+1-l.win)
}

// positions is how many of the pool's own positions a sequence that has
// written end positions holds: end, or on an entries pool the entries they
// closed -- at least one, so the first entry's page is always there to read.
func (l *kvLayerPool) positions(end int) int {
	if l.rate > 0 {
		return max(end/l.rate, 1)
	}
	return end
}

// tabRange is a sequence's run in every layer's table arena.
type tabRange struct{ off, cap int }

// kvPool is a device's paged attention history: one kvLayerPool per attention
// layer, and the table layout they share. A sequence's run starts at the SAME
// offset in every layer's arena, so one set of row descriptors serves every
// layer of a submission and only the table bound per layer differs; the ids in
// it are each layer's own.
type kvPool struct {
	p int // positions per page, fixed for the pool's life
	// maxBytes is the backend's largest buffer (Vulkan's storage-buffer range,
	// Metal's maxBufferLength), 0 for no limit: a buffer past it would bind
	// short and read the wrong memory.
	maxBytes uint64
	layers   map[int]*kvLayerPool
	ranges   map[seqID]tabRange
	// tabWords is every arena's size; tabTop the first word never handed out
	// (word 0 is the dummy entry); tabFree runs given back, reused first-fit.
	tabWords, tabTop int
	tabFree          []tabRange
	// evicted is how many of sequence s's leading pages are at home, the same
	// count in every layer that holds s (kvevict.go).
	evicted map[seqID]int
	// written is the positions of sequence s a completed submission wrote:
	// only pages wholly below it may leave, so a page a call is about to write
	// is never sent home under it.
	written map[seqID]int
}

// newKVPool starts a pool of p-position pages with no layers. maxBuf is the
// backend's largest buffer in bytes, 0 for no limit.
func newKVPool(p int, maxBuf uint64) *kvPool {
	return &kvPool{p: p, maxBytes: maxBuf, layers: map[int]*kvLayerPool{},
		ranges: map[seqID]tabRange{}, tabWords: 256, tabTop: 1}
}

// pageBytes is one page of one layer.
func (kp *kvPool) pageBytes(l *kvLayerPool) uint64 {
	return uint64(l.geom.kWords(kp.p)+l.geom.vWords(kp.p)) * 4
}

// maxPages is the most pages a layer's buffer may hold: kernels index a pool
// in 32-bit words, and a buffer past the backend's limit binds short.
func (kp *kvPool) maxPages(l *kvLayerPool) int {
	words := uint64(1<<32 - 1)
	if kp.maxBytes > 0 {
		words = min(words, kp.maxBytes/4)
	}
	return int(words / uint64(max(1, l.geom.kWords(kp.p), l.geom.vWords(kp.p))))
}

// charged is the whole pool's charge.
func (kp *kvPool) charged() uint64 {
	var n uint64
	for _, l := range kp.layers {
		n += l.charged
	}
	return n
}

// addKVLayer gives attention layer li its buffers at n pages -- the dummy and
// n-1 free, allocated at that size so placing a block costs no resize -- and a
// table arena. Callers hold g.mu.
func (g *devTier) addKVLayer(kp *kvPool, li int, geom kvGeom, n int) error {
	if _, ok := kp.layers[li]; ok {
		return nil
	}
	n = max(n, 1)
	l := &kvLayerPool{geom: geom, n: n, owned: map[seqID][]uint32{}, clk: &g.clk}
	for id := n - 1; id >= 1; id-- {
		l.free = append(l.free, uint32(id))
	}
	l.reserveFences()
	need := kp.pageBytes(l)*uint64(n) + uint64(kp.tabWords)*4 + kvStageBytes(geom, kp.p)
	if !g.kvCompact(need, nil) && !g.roomOrReclaim(need) {
		return fmt.Errorf("%w: a layer's %d page(s) and table need %d bytes", ErrKVCapacity, n, need)
	}
	var err error
	if l.k, l.v, err = g.allocLayerPool(geom, kp.p, n); err != nil {
		return err
	}
	if l.tab, err = g.dev.Alloc(kp.tabWords * 4); err == nil {
		err = l.tab.Write(make([]byte, kp.tabWords*4))
	}
	if err == nil {
		err = g.makeStage(l, kp.p)
	}
	if err != nil {
		freeLayerBufs(l)
		return err
	}
	g.charge(need)
	l.charged = need
	kp.layers[li] = l
	return nil
}

// kvStageBytes is a layer's read reserve: a page of its wider half and a
// one-id table.
func kvStageBytes(geom kvGeom, p int) uint64 {
	return uint64(max(geom.kWords(p), geom.vWords(p))+1) * 4
}

// makeStage gives l its read reserve and compiles the one-page gathers it
// runs, so a read through it compiles and allocates nothing. Callers hold
// g.mu.
func (g *devTier) makeStage(l *kvLayerPool, p int) error {
	kw, vw := l.geom.kWords(p), l.geom.vWords(p)
	var err error
	if l.stage, err = g.dev.Alloc(max(kw, vw) * 4); err != nil {
		return err
	}
	if l.stageTab, err = g.dev.Alloc(4); err != nil {
		return err
	}
	for _, w := range []int{kw, vw} {
		if w == 0 {
			continue
		}
		if _, err = g.gatherKernel(w, 1); err != nil {
			return err
		}
	}
	return nil
}

// maxBuffer is the device's largest buffer, 0 when it has no such limit; a
// pool on this device is created with it (newKVPool).
func (g *devTier) maxBuffer() uint64 {
	if bl, ok := g.dev.(backend.BufferLimited); ok {
		return bl.MaxBuffer()
	}
	return 0
}

// allocLayerPool allocates a layer's K and V buffers for n pages, zeroed: the
// dummy page is read as history by an empty row, and zero is finite.
func (g *devTier) allocLayerPool(geom kvGeom, p, n int) (k, v backend.Buf, err error) {
	kw, vw := geom.kWords(p)*n, geom.vWords(p)*n
	if k, err = g.dev.Alloc(kw * 4); err != nil {
		return nil, nil, err
	}
	if err = k.Write(make([]byte, kw*4)); err == nil && vw > 0 {
		if v, err = g.dev.Alloc(vw * 4); err == nil {
			err = v.Write(make([]byte, vw*4))
		}
	}
	if err != nil {
		k.Free()
		if v != nil {
			v.Free()
		}
		return nil, nil, err
	}
	return k, v, nil
}

// allocPages gives sequence s count more pages in layer l, all or none. The
// layer grows when its free list is short -- a new buffer pair, the old pages
// copied across on the device, every captured graph dropped since it names the
// old buffers -- and refuses with ErrKVCapacity when the budget or the
// backend's limit stops it. The new ids are appended to s's run in the layer's
// table arena. Callers hold g.mu, outside any submission.
func (g *devTier) allocPages(sid uint64, kp *kvPool, l *kvLayerPool, s seqID, count int) error {
	if count <= 0 {
		return nil
	}
	// The table run first: a run that moves may grow every layer's table
	// arena, and the room for that comes from compacting the layers
	// (kvCompact), this one included, which gives its free pages back. Taken
	// after the free list was made long enough, it emptied the list again.
	have := len(l.owned[s])
	if err := g.ensureRange(kp, s, have+count); err != nil {
		return err
	}
	// The prefix cache's unheld pages first: they are a cache, and room for
	// them is the first a sequence takes back (kvshare.go).
	if len(l.free) < count {
		l.reclaimNamed(count - len(l.free))
	}
	if len(l.free) < count {
		// Growth first -- a page-out of streamed weights or more pool -- and
		// when nothing can grow, cold history goes home (kvevict.go). A
		// driver's refusal inside the budget is no room either.
		err := g.growKVLayer(kp, l, l.n+count-len(l.free))
		switch {
		case err == nil:
			g.kvTight = false
		case !errors.Is(err, ErrKVCapacity):
			// The budget allowed it and the driver did not: the card's
			// own ceiling, which KVRoom then reads (kvTight).
			g.kvTight = true
		}
		if err != nil && !g.evictForRoom(sid, kp, l, count) {
			return err
		}
	}
	take := l.free[len(l.free)-count:]
	l.owned[s] = append(l.owned[s], take...)
	l.free = l.free[:len(l.free)-count]
	g.tabPend = append(g.tabPend, tabWrite{l.tab, kp.ranges[s].off + have, l.owned[s][have:]})
	return nil
}

// tabWrite is a page-table update allocPages queued: ids at word off of tab.
// A write to a table is a hop to the device's owner goroutine and a wait, so
// a call's pages -- one run per layer -- go in one session (flushTabs), which
// was a measurable share of a 512-row prompt's host time as one write a layer.
type tabWrite struct {
	tab backend.Buf
	off int
	ids []uint32
}

// flushTabs writes the queued table updates, in order, in one session.
// Everything that reads or rewrites a table or renumbers ids flushes first,
// so a queued write never lands after one that superseded it; ids alias the
// sequences' runs, which only such functions change. Callers hold g.mu,
// outside any submission.
func (g *devTier) flushTabs() error {
	if len(g.tabPend) == 0 {
		return nil
	}
	// The session's body is made once and kept: a prompt that maps or
	// copies pages writes tables every call, and a closure a call is an
	// allocation a call.
	if g.flushFn == nil {
		g.flushFn = func(s backend.Session) {
			for _, w := range g.tabPend {
				if g.flushErr = s.WriteAt(w.tab, w.off*4, u32view(w.ids)); g.flushErr != nil {
					return
				}
			}
		}
	}
	g.flushErr = nil
	g.dev.Session(g.flushFn)
	err := g.flushErr
	clear(g.tabPend)
	g.tabPend = g.tabPend[:0]
	return err
}

// growKVLayer makes layer l's buffers hold at least want pages: doubling when
// that fits, else exactly want, since a doubling is an amortisation and not a
// requirement. Nothing changes unless all of it succeeds.
func (g *devTier) growKVLayer(kp *kvPool, l *kvLayerPool, want int) error {
	limit := kp.maxPages(l)
	if want > limit {
		return fmt.Errorf("%w: %d page(s) wanted, the backend's buffer limit allows %d", ErrKVCapacity, want, limit)
	}
	// The pages are copied into the new buffers: a submission in flight
	// writing a row into the old ones would be lost.
	if err := g.quiesce(); err != nil {
		return err
	}
	pb := kp.pageBytes(l)
	fits := func(n uint64) bool { return g.stateFits(n, l) }
	// A doubling stops at what a scheduler promised (PromiseKV): the pages in
	// use and the promise's, with one spare. The budget past it is left to
	// the batched scratches the admitted rows' steps run in, which a pool
	// doubled into the whole card refused, padding every step to the widest.
	grow := l.n * 2
	if g.kvPromise > 0 {
		grow = min(grow, l.n-len(l.free)+(g.kvPromise+kp.p-1)/kp.p+1)
	}
	for _, n := range []int{min(max(grow, want, 2), limit), want} {
		if n < want {
			continue
		}
		var k, v backend.Buf
		var err error
		switch {
		case fits(pb * uint64(n)):
			// Copied on the device: the old buffers are live until the copy is
			// done, so the peak is both pairs.
			if k, v, err = g.allocLayerPool(l.geom, kp.p, n); err != nil {
				return err
			}
			kw, vw := l.geom.kWords(kp.p)*l.n, l.geom.vWords(kp.p)*l.n
			if !g.restride(l.k, k, 1, kw, kw, kw) || vw > 0 && !g.restride(l.v, v, 1, vw, vw, vw) {
				freeKV2(k, v)
				return fmt.Errorf("tier: carrying a KV layer across a resize failed: %s", g.LastErr)
			}
			g.dropGraph()
			freeKV2(l.k, l.v)
		case fits(pb * uint64(n-l.n)):
			// Only the grown layer fits, not both: through the host, the old
			// pair freed before the new one is made (compactViaHost).
			order := make([]uint32, l.n)
			for i := range order {
				order[i] = uint32(i)
			}
			if k, v, err = g.compactViaHost(kp, l, order, n); err != nil {
				return err
			}
		default:
			continue
		}
		l.k, l.v = k, v
		for id := n - 1; id >= l.n; id-- {
			l.free = append(l.free, uint32(id))
		}
		delta := pb * uint64(n-l.n)
		g.charge(delta)
		l.charged += delta
		l.n = n
		l.reserveFences()
		g.KVGrows++
		return nil
	}
	return fmt.Errorf("%w: a layer's %d more page(s) need %d bytes", ErrKVCapacity, want-l.n, pb*uint64(want-l.n))
}

// ensureRange gives sequence s a run of at least pages words in every layer's
// table arena. A run that has to move takes a new offset: the ids go across
// from the host copies (owned), and the old run is given back.
func (g *devTier) ensureRange(kp *kvPool, s seqID, pages int) error {
	r, ok := kp.ranges[s]
	if ok && r.cap >= pages {
		return nil
	}
	if err := g.flushTabs(); err != nil {
		return err
	}
	want := max(pages, 2*r.cap, 4)
	nr := tabRange{cap: want}
	if fit := slices.IndexFunc(kp.tabFree, func(f tabRange) bool { return f.cap >= want }); fit >= 0 {
		nr = kp.tabFree[fit]
		kp.tabFree = slices.Delete(kp.tabFree, fit, fit+1)
	} else {
		if kp.tabTop+want > kp.tabWords {
			if err := g.growTables(kp, max(2*kp.tabWords, kp.tabTop+want)); err != nil {
				return err
			}
		}
		nr.off = kp.tabTop
		kp.tabTop += want
	}
	for _, l := range kp.layers {
		if ids := l.owned[s]; len(ids) > 0 {
			if err := l.tab.WriteAt(nr.off*4, u32view(ids)); err != nil {
				return err
			}
		}
	}
	if ok {
		kp.tabFree = append(kp.tabFree, r)
	}
	kp.ranges[s] = nr
	return nil
}

// growTables replaces every layer's table arena with one of words u32,
// rewritten from the host copies. It drops the captured graph, which names the
// old arenas.
func (g *devTier) growTables(kp *kvPool, words int) error {
	type swap struct {
		l   *kvLayerPool
		tab backend.Buf
	}
	var done []swap
	undo := func(err error) error {
		for _, s := range done {
			s.tab.Free()
		}
		return err
	}
	extra := uint64(words-kp.tabWords) * 4
	if !g.stateFits(extra*uint64(len(kp.layers)), nil) {
		return fmt.Errorf("%w: the page tables need %d more bytes a layer", ErrKVCapacity, extra)
	}
	host := make([]uint32, words)
	for _, l := range kp.layers {
		clear(host)
		for s, ids := range l.owned {
			copy(host[kp.ranges[s].off:], ids)
		}
		tab, err := g.dev.Alloc(words * 4)
		if err == nil {
			if err = tab.Write(u32view(host)); err != nil {
				tab.Free()
			}
		}
		if err != nil {
			return undo(err)
		}
		done = append(done, swap{l, tab})
	}
	g.dropGraph()
	for _, s := range done {
		s.l.tab.Free()
		s.l.tab = s.tab
		g.charge(extra)
		s.l.charged += extra
	}
	kp.tabWords = words
	return nil
}

// releasePages gives sequence s's pages in layer l back. They are fenced, not
// free: a submission still in flight may read them.
func (l *kvLayerPool) releasePages(s seqID) {
	l.fenceIDs(l.owned[s])
	delete(l.owned, s)
	delete(l.home, s)
	delete(l.rel, s)
}

// reserveFences sizes the fence lists for every page of the layer, so
// releasing pages never grows them on a warm path.
func (l *kvLayerPool) reserveFences() {
	l.fenced = slices.Grow(l.fenced, l.n-len(l.fenced))
	l.fencedAt = slices.Grow(l.fencedAt, l.n-len(l.fencedAt))
}

// fenceIDs fences every id of ids that is a page on the card (0 is one at
// home), at the ticket the next submission takes.
func (l *kvLayerPool) fenceIDs(ids []uint32) {
	for _, id := range ids {
		if id != 0 {
			l.releaseID(id) // a named page is held, not fenced (kvshare.go)
		}
	}
}

// fenceID fences one page.
func (l *kvLayerPool) fenceID(id uint32) {
	var at uint64
	if l.clk != nil {
		at = l.clk.seq
	}
	l.fenced = append(l.fenced, id)
	l.fencedAt = append(l.fencedAt, at)
}

// fence frees the fenced pages nothing in flight can read: every one when no
// submission is in flight, else those released before the oldest submission in
// flight took its ticket. A page released at ticket t waits for every
// submission holding t or less, the one about to take t included -- what
// "after the next completed submission" was when submissions ran one at a
// time.
func (l *kvLayerPool) fence() {
	if l.clk == nil || l.clk.live == 0 {
		l.free = append(l.free, l.fenced...)
		l.fenced, l.fencedAt = l.fenced[:0], l.fencedAt[:0]
		return
	}
	kept := 0
	for i, id := range l.fenced {
		if l.fencedAt[i] < l.clk.oldest {
			l.free = append(l.free, id)
			continue
		}
		l.fenced[kept], l.fencedAt[kept] = id, l.fencedAt[i]
		kept++
	}
	l.fenced, l.fencedAt = l.fenced[:kept], l.fencedAt[:kept]
}

// dropRange gives s's table run back once no layer holds a page of it.
func (kp *kvPool) dropRange(s seqID) {
	for _, l := range kp.layers {
		if len(l.owned[s]) > 0 {
			return
		}
	}
	if r, ok := kp.ranges[s]; ok {
		kp.tabFree = append(kp.tabFree, r)
		delete(kp.ranges, s)
	}
	kp.forgetFrom(s, 0)
}

// shrinkKVLayer gives back layer l's pages beyond the ones in use and its
// floor, whatever ids they hold: the live pages are compacted into a fresh,
// smaller buffer pair by one gather (dummy first, then each live page), the
// sequences' ids renumbered and their table runs rewritten. Ids are stable only
// while a submission can read them, and none can here: callers hold g.mu,
// outside any submission, with the layer's released pages fenced. The
// transient peak is both buffer pairs; a card that cannot hold it keeps the
// larger layer and says so.
func (g *devTier) shrinkKVLayer(kp *kvPool, l *kvLayerPool) error {
	if err := g.flushTabs(); err != nil {
		return err
	}
	live := 1
	for _, ids := range l.owned {
		for _, id := range ids {
			if id != 0 {
				live++
			}
		}
	}
	// Named pages: the unheld ones go first, and a layer whose named pages a
	// sequence holds is not compacted (an id held twice would be gathered
	// twice).
	l.reclaimNamed(l.n)
	if l.heldNamed() {
		return nil
	}
	// A layer with history at home keeps one id free to stream it through.
	// streamRoom ensured it; a compaction forced by another layer's growth
	// took it back, and the next call found no page to stream through.
	want := live
	if kp.anyEvicted() {
		want++
	}
	n := max(want, l.floor, 1)
	if n >= l.n || len(l.fenced) > 0 {
		return nil
	}
	// The live pages are gathered into new buffers and renumbered under
	// every sequence: nothing in flight may write or read the old ones.
	if err := g.quiesce(); err != nil {
		return err
	}
	pb := kp.pageBytes(l)
	// The gather order: new id j is old id order[j], the dummy first.
	seqs := make([]seqID, 0, len(l.owned))
	for s := range l.owned {
		seqs = append(seqs, s)
	}
	slices.SortFunc(seqs, func(a, b seqID) int {
		if a.sid != b.sid {
			if a.sid < b.sid {
				return -1
			}
			return 1
		}
		return a.base - b.base
	})
	order := make([]uint32, 1, live)
	for _, s := range seqs {
		order = residentIDs(order, l.owned[s])
	}
	var k, v backend.Buf
	if g.room(pb * uint64(n)) {
		var err error
		if k, v, err = g.allocLayerPool(l.geom, kp.p, n); err != nil {
			return err
		}
		err = g.gatherInto(l.k, k, order, l.geom.kWords(kp.p))
		if vw := l.geom.vWords(kp.p); err == nil && vw > 0 {
			err = g.gatherInto(l.v, v, order, vw)
		}
		if err != nil {
			freeKV2(k, v)
			return err
		}
		g.dropGraph()
		freeKV2(l.k, l.v)
	} else {
		// No room for both pairs, which is exactly when compaction is asked
		// for (kvCompact): the live pages go through the host instead, the old
		// pair is freed before the new one is made, and the peak is the host
		// copy rather than device memory.
		var err error
		if k, v, err = g.compactViaHost(kp, l, order, n); err != nil {
			return err
		}
	}
	l.k, l.v = k, v
	next := uint32(1)
	for _, s := range seqs {
		ids := l.owned[s]
		for i := range ids {
			if ids[i] != 0 { // 0 is a page at home (kvevict.go)
				ids[i] = next
				next++
			}
		}
		if err := l.tab.WriteAt(kp.ranges[s].off*4, u32view(ids)); err != nil {
			return err
		}
	}
	l.free = l.free[:0]
	for id := n - 1; id >= live; id-- {
		l.free = append(l.free, uint32(id))
	}
	back := pb * uint64(l.n-n)
	g.refund(back)
	l.charged -= back
	l.n = n
	return nil
}

// compactViaHost is shrinkKVLayer's gather when the card cannot hold the old
// and the new buffer pair at once: the layer is read to the host, freed, and
// made again at n pages with page j holding old page order[j]. A failure after
// the free leaves the layer with no buffers, which is reported and not
// survivable -- its history is gone -- so it is refused before the free
// wherever it can be.
func (g *devTier) compactViaHost(kp *kvPool, l *kvLayerPool, order []uint32, n int) (k, v backend.Buf, err error) {
	kw, vw := l.geom.kWords(kp.p), l.geom.vWords(kp.p)
	// The layer is read home whole: nothing in flight may still write it.
	if err := g.quiesce(); err != nil {
		return nil, nil, err
	}
	hk, err := readPages(l.k, kw, l.n)
	if err != nil {
		return nil, nil, err
	}
	hv, err := readPages(l.v, vw, l.n)
	if err != nil {
		return nil, nil, err
	}
	g.dropGraph()
	freeKV2(l.k, l.v)
	l.k, l.v = nil, nil
	if k, v, err = g.allocLayerPool(l.geom, kp.p, n); err != nil {
		// The card refused the new size where the budget allowed it. The old
		// size was just given back, so the history goes back in at it: a
		// layer left with no buffers made the next grow read a nil one.
		if rk, rv, rerr := g.restoreLayer(l, kp.p, hk, hv); rerr == nil {
			l.k, l.v = rk, rv
			return nil, nil, fmt.Errorf("tier: a KV layer could not grow to %d pages (it keeps its %d): %w", n, l.n, err)
		}
		return nil, nil, fmt.Errorf("tier: a KV layer freed for compaction could not be made again: %w", err)
	}
	if err = writePagesIn(k, hk, kw, order); err == nil {
		err = writePagesIn(v, hv, vw, order)
	}
	if err != nil {
		// As above: the history goes back in at the old size rather than
		// leaving the layer with no buffers for the next read.
		freeKV2(k, v)
		if rk, rv, rerr := g.restoreLayer(l, kp.p, hk, hv); rerr == nil {
			l.k, l.v = rk, rv
		}
		return nil, nil, err
	}
	return k, v, nil
}

// restoreLayer makes l's buffers again at its own l.n pages and writes back the
// host copy compactViaHost took, page for page.
func (g *devTier) restoreLayer(l *kvLayerPool, page int, hk, hv []byte) (k, v backend.Buf, err error) {
	if k, v, err = g.allocLayerPool(l.geom, page, l.n); err != nil {
		return nil, nil, err
	}
	same := make([]uint32, l.n)
	for i := range same {
		same[i] = uint32(i)
	}
	if err = writePagesIn(k, hk, l.geom.kWords(page), same); err == nil {
		err = writePagesIn(v, hv, l.geom.vWords(page), same)
	}
	if err != nil {
		freeKV2(k, v)
		return nil, nil, err
	}
	return k, v, nil
}

// readPages reads n pages of words each from b to the host; nothing when the
// half has no words (MLA's V).
func readPages(b backend.Buf, words, n int) ([]byte, error) {
	if words == 0 {
		return nil, nil
	}
	h := make([]byte, words*4*n)
	return h, b.Read(h)
}

// writePagesIn writes host page order[j] to page j of b.
func writePagesIn(b backend.Buf, host []byte, words int, order []uint32) error {
	pw := words * 4
	for j, id := range order {
		if pw == 0 {
			break
		}
		if err := b.WriteAt(j*pw, host[int(id)*pw:int(id+1)*pw]); err != nil {
			return err
		}
	}
	return nil
}

// gatherInto copies pages order[j] of src, words each, to page j of dst. The
// kernel is compiled per shape and kept, as restride's are.
func (g *devTier) gatherInto(src, dst backend.Buf, order []uint32, words int) error {
	n := len(order)
	k, err := g.gatherKernel(words, n)
	if err != nil {
		return err
	}
	tab, err := g.dev.Alloc(n * 4)
	if err != nil {
		return err
	}
	defer tab.Free()
	if err := tab.Write(u32view(order)); err != nil {
		return err
	}
	return k.Launch((n*words+127)/128, 128, src, dst, tab)
}

// gatherKernel is the gather of n pages of words each, compiled once and kept
// with the restrides.
func (g *devTier) gatherKernel(words, n int) (backend.Kernel, error) {
	key := [4]int{-1, words, n, 0}
	if k := g.restrides[key]; k != nil {
		return k, nil
	}
	ker, err := oldkernels.PagedGatherPages(words, n)
	if err != nil {
		return nil, err
	}
	k, err := g.dev.Compile(ker)
	if err != nil {
		return nil, err
	}
	if g.restrides == nil {
		g.restrides = map[[4]int]backend.Kernel{}
	}
	g.restrides[key] = k
	return k, nil
}

// freeKVLayer frees one layer's buffers and refunds its charge, and forgets
// its queued table writes.
func (g *devTier) freeKVLayer(l *kvLayerPool) {
	g.tabPend = slices.DeleteFunc(g.tabPend, func(w tabWrite) bool { return w.tab == l.tab })
	freeLayerBufs(l)
	g.refund(l.charged)
	l.charged = 0
}

func freeLayerBufs(l *kvLayerPool) {
	freeKV2(l.k, l.v)
	freeKV2(l.tab, l.stage)
	if l.stageTab != nil {
		l.stageTab.Free()
	}
	l.k, l.v, l.tab, l.stage, l.stageTab = nil, nil, nil, nil, nil
	l.sh = sharedIDs{}
}

// freeKV2 frees a K buffer and, when there is one, its V buffer.
func freeKV2(k, v backend.Buf) {
	if k != nil {
		k.Free()
	}
	if v != nil {
		v.Free()
	}
}

// freeKVPool gives every buffer back and refunds the charge. Callers hold g.mu.
func (g *devTier) freeKVPool(kp *kvPool) {
	if kp == nil {
		return
	}
	for li, l := range kp.layers {
		g.freeKVLayer(l)
		delete(kp.layers, li)
	}
}

// u32view is v's bytes, without a copy: the per-token staging writes through
// it so a token allocates nothing.
func u32view(v []uint32) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), 4*len(v))
}

// anyEvicted reports that some sequence has pages at home.
func (kp *kvPool) anyEvicted() bool {
	for _, e := range kp.evicted {
		if e > 0 {
			return true
		}
	}
	return false
}
