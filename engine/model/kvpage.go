package model

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"slices"
	"strconv"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/cpu"
)

// The KV cache is paged, and its pages are its own.
//
// It is not the weight pager: a weight page is fixed at conversion, never
// changes and always has a copy on disk, while KV pages grow one position per
// token per layer, the tail page is written every token, and until something
// stores a page there is no other copy. Consequences:
//
//   - Pages behind the tail are immutable, so they are safe to store, share or
//     drop.
//   - There is no fault on the write path: a new page is needed exactly when
//     pos%P == 0, so the write path allocates.
//   - Reads span every page: attention at position n reads 0..n (see kvSpan).
//
// The page size is per layer: a linear-attention layer gets no pages, a
// sliding-window layer one page of its window, and an ordinary attention
// layer a run of positions aligned to the score kernel's tile.

// ErrNoPage is what a store returns when it simply does not hold the page.
//
// A miss means "compute it"; any other error means the store is broken, and
// treating the two alike would silently drop a sequence's history.
var ErrNoPage = errors.New("model: no such kv page")

// KVStore is where a KV page lives when no tier is holding it.
//
// The sink is the caller's io.Writer, so a store only moves bytes and never
// learns about device buffers or layouts; io.Copy's ReaderFrom/WriterTo lets a
// destination that can alias memory skip the copy.
//
// cacheId is opaque to the engine and identifies one history. The caller can
// make it content-addressed ("<model>/<sha256 of the prompt prefix>"), which
// is how a later process finds an earlier one's pages; a counter or a
// Scheduler row would address another prompt's pages. Two sessions sharing a
// key must share a token prefix, model, KV element width and page geometry.
// The engine cannot check that. Implementations must be safe for concurrent
// use.
type KVStore interface {
	// Get writes page `index` of `layer` into page. It returns ErrNoPage when
	// the store does not hold it.
	Get(cacheId string, layer int, index int, page io.Writer) error
	// Set takes the page's bytes. Only sealed pages are ever offered.
	Set(cacheId string, layer int, index int, page io.Reader) error
	// Drop forgets everything held for one cache.
	Drop(cacheId string) error
}

// FillStore is a KVStore that can hold several pages which are byte prefixes
// of one buffer over a single copy of it. SetFills takes, for one layer and
// page index, src and for each i the page keys[i] = src[:lens[i]]; lens are
// in bytes and none exceeds len(src). It has the effect of one Set per key.
type FillStore interface {
	KVStore
	SetFills(layer, index int, src []byte, keys []string, lens []int) error
}

// NoStore holds nothing: every Get misses and every Set is discarded.
//
// It is the default: the cache is resident-only and a run that fits pays
// nothing for the store.
type NoStore struct{}

// Get misses every page: it returns ErrNoPage.
func (NoStore) Get(string, int, int, io.Writer) error { return ErrNoPage }

// Set discards the page.
func (NoStore) Set(string, int, int, io.Reader) error { return nil }

// Drop has nothing to forget.
func (NoStore) Drop(string) error { return nil }

// kvKeyTile is how far past a row's causal width the score kernel may write, and
// therefore the alignment a page boundary has to respect. jit/gpu/tier carries
// the same constant for the score-row padding it forces (sstride = MaxSeq + it).
const kvKeyTile = 32

// kvPageTarget is the positions a page holds when nothing constrains it.
//
// It is a target, not a tuning result: EmitAttnAcc re-reads the V window once
// per column block per page, bounding it from below, and a page is the granule
// of growth and spill, bounding it from above. 256 is a multiple of every key
// tile and a power of two. A model's kvPage option overrides it (tests only).
const kvPageTarget = 256

// kvPagePositions is how many positions one page of layer li holds, or 0 when
// the layer keeps no attention history at all. pin, when positive, replaces
// kvPageTarget (the model's kvPage option).
func kvPagePositions(c *Config, li, pin int) int {
	if !c.LayerKind(li).Attends() {
		// A recurrent layer keeps a fixed-size summary, not a history.
		return 0
	}
	// Only a window shorter than the context sizes a page (Phi-3.5 declares
	// a window larger than its context, which would be gigabytes a layer).
	if c.SWA(li) && c.SWAWindow > 0 && (c.NCtx == 0 || c.SWAWindow < c.NCtx) {
		return align(c.SWAWindow, kvKeyTile)
	}
	p := align(kvPageTarget, kvKeyTile)
	if pin > 0 {
		p = pin
	}
	if c.NCtx > 0 && p > c.NCtx {
		p = align(c.NCtx, kvKeyTile)
	}
	return p
}

func align(n, to int) int {
	if to <= 0 || n <= 0 {
		return n
	}
	return ((n + to - 1) / to) * to
}

// kvPages is one layer's paged history for one State.
//
// A page holds every sequence slot, laid out as the flat cache was with maxSeq
// replaced by P, so kvLayout addresses within a page unchanged.
type kvPages struct {
	p    int // positions per page; 0 means this layer keeps none
	pp   int // float32 elements per page (nseq * p * kvDim, scaled by elem width)
	k, v [][]float32
	// latent makes the value the same memory as the key, which is what MLA
	// is: a cached row is the latent followed by the shared rotary key, and
	// the value is the latent prefix. It is an alias, not a copy. vSpan
	// returns kSpan's base, and the accumulate kernel is emitted for
	// KVLoraRank while scores use the whole row (nn.AddAttnKV).
	latent bool
	// lastSealed is how many pages have been offered to the store. A page is
	// offered once, when it stops being the one the token writes into.
	lastSealed int
	// written is the high-water position this layer has actually had written
	// into it. Resident is not written: the page straddling the end of a
	// restored or migrated run is resident and partly zeros, and sealing it
	// would store those zeros as history.
	written int
	// out[n] marks a page trim evicted with history still in it. A write into
	// it faults it back first (State.kvWritable): in a batch the page holds
	// every row's slots, and a readmitted row writes into pages the others
	// sealed long ago, so allocating a zeroed one would wipe their history.
	// A page dropped on purpose (a restore, a truncation) is not out.
	out []bool
	// pool is where a growing page comes from: the pages a closed State on
	// the same model gave back, or a fresh one. nil is always fresh.
	pool *kvPagePool
	// win is a sliding or chunked layer's window, 0 for a layer that reads its
	// whole history; chunked says the window is aligned chunks (llama4).
	//
	// A windowed layer's pages behind every row's window are never read
	// again, so release hands them to spare and the next page grows out of
	// it: the layer holds the window and a page of slack, not the context.
	win     int
	chunked bool
	// gone is where release's walk starts: every page below it was released
	// or never written. A page grown below it (a row restarting in a batch)
	// lowers it again.
	gone int
	// genuine is the first position whose bytes are history. Positions below
	// it may be zeros a device sent back for pages it had released, so a page
	// starting below it is never stored: under content addressing it would
	// be another session's history.
	genuine int
	// spare is released pages, reused before the pool or a fresh one.
	spare [][]float32
	// shared[n] says page n aliases sh's single copy (kvshare.go): it is
	// never written in place, never pooled, and its reference is dropped
	// wherever the page is let go. cowFault is the sharing gate's violation:
	// a write into a shared page lands in place.
	shared   []bool
	sh       *SharedStore
	cowFault bool
}

// keyStart is the first key position pos attends on this layer: the window's
// start, as Config.AttnWindow places it, or 0 on a layer with no window.
func (s *kvPages) keyStart(pos int) int {
	switch {
	case s.win <= 0 || pos <= 0:
		return 0
	case s.chunked:
		return pos / s.win * s.win
	}
	return max(0, pos+1-s.win)
}

// liveFrom is the first page a windowed layer must keep while its slowest row
// is at low: the page holding the window start of nn.KVWindowSlack positions
// earlier, so a rewind that short (a speculation round's rejected rows) still
// finds its window resident. 0 on a layer with no window.
func (s *kvPages) liveFrom(low int) int {
	if s.win <= 0 || s.p == 0 {
		return 0
	}
	return s.keyStart(max(0, low-nn.KVWindowSlack)) / s.p
}

// release hands every page below liveFrom(low) to spare. Callers pass the
// lowest position any row with history still writes (State.advance), so no
// row reads a released page again.
//
// fault is a gate's violation: the page the window starts in goes too, with
// no slack, so the window reads a released page.
func (s *kvPages) release(low int, fault bool) {
	first := s.liveFrom(low)
	if fault {
		first = s.keyStart(low)/s.p + 1
	}
	for n := s.gone; n < first && n < len(s.k); n++ {
		s.recycle(n)
	}
	s.gone = max(s.gone, first)
}

// recycle moves page n's buffers to spare and leaves it non-resident. Spare
// holds at most what a window and its slack turn over (spareMax); a page past
// that goes to the collector.
func (s *kvPages) recycle(n int) {
	if s.isShared(n) {
		s.drop(n) // others read it: the reference goes, the memory stays
		return
	}
	for _, b := range [2][]float32{s.k[n], s.v[n]} {
		if b != nil && len(b) == s.pp && len(s.spare) < s.spareMax() {
			s.spare = append(s.spare, b)
		}
		if s.latent {
			break // the value is the key's memory
		}
	}
	s.drop(n)
}

// spareMax is the most released buffers a windowed layer keeps: the K and V
// of the pages a window and its slack span, twice over.
func (s *kvPages) spareMax() int {
	if s.win <= 0 || s.p == 0 {
		return 0
	}
	return 2 * 2 * ((s.win+nn.KVWindowSlack)/s.p + 2)
}

// page0 is a zeroed page: a released one first, then the pool's.
func (s *kvPages) page0() []float32 {
	if n := len(s.spare); n > 0 {
		b := s.spare[n-1]
		s.spare[n-1] = nil
		s.spare = s.spare[:n-1]
		clear(b)
		return b
	}
	return s.pool.get(s.pp)
}

// reserve sizes the page index for maxSeq positions, so a decode crossing a
// page boundary does not grow it.
func (s *kvPages) reserve(maxSeq int) {
	if s.p == 0 || maxSeq <= 0 {
		return
	}
	n := (maxSeq + s.p - 1) / s.p
	s.k = slices.Grow(s.k, n)
	s.v = slices.Grow(s.v, n)
	s.out = slices.Grow(s.out, n)
	s.spare = slices.Grow(s.spare, s.spareMax())
}

// drop lets page n go for good: nothing is to be faulted back into it.
func (s *kvPages) drop(n int) {
	s.unshare(n)
	s.k[n], s.v[n] = nil, nil
	if n < len(s.out) {
		s.out[n] = false
	}
}

// resident says page n is here now. It is read-only because attention heads
// fan out across the pool: a worker may ask "is it here", never "put it here".
func (s *kvPages) resident(n int) bool {
	return n < len(s.k) && s.k[n] != nil && n < len(s.v) && s.v[n] != nil
}

// page returns the page index and the position within it.
func (s *kvPages) page(pos int) (int, int) { return pos / s.p, pos % s.p }

// grow makes sure page n exists, allocating it if this is the first time the
// sequence has reached it. It is called on the write path, where the need is
// known rather than discovered.
func (s *kvPages) grow(n int) {
	s.extend(n)
	// A page below the released ones: a batch row restarting from zero, whose
	// pages the others had left. Release walks from here again.
	if n < s.gone {
		s.gone = n
	}
	// It refills a hole, not only the tail: eviction and a restored prefix
	// leave nil slots inside the slice.
	if s.k[n] == nil {
		s.k[n] = s.page0()
	}
	if s.latent {
		// The value IS the key's prefix; see the field.
		s.v[n] = s.k[n]
		return
	}
	if s.v[n] == nil {
		s.v[n] = s.page0()
	}
}

// extend makes index n addressable without making it resident. The fault path
// wants this and not grow: it fills the page from a store, and a buffer
// allocated here could be read as history if a failure path kept it.
func (s *kvPages) extend(n int) {
	for len(s.k) <= n {
		s.k = append(s.k, nil)
		s.v = append(s.v, nil)
	}
	for len(s.out) <= n {
		s.out = append(s.out, false)
	}
	if s.sh != nil {
		for len(s.shared) <= n {
			s.shared = append(s.shared, false)
		}
	}
}

// bytes is what this layer's pages occupy right now -- what has actually been
// committed, not what the context could reach.
func (s *kvPages) bytes() uint64 {
	if s.p == 0 {
		return 0
	}
	// Counted by what is resident, not by the slice's length: eviction nils a
	// slot and leaves the length unchanged.
	// A shared page is the store's, held once however many sessions read it
	// (KVSharedBytes counts it).
	var n uint64
	for i := range s.k {
		if s.isShared(i) {
			continue
		}
		if s.k[i] != nil {
			n++
		}
		if i < len(s.v) && s.v[i] != nil {
			n++
		}
	}
	return n * uint64(s.pp) * 4
}

// kvSpan is one page's share of a read window: which page, where in it the run
// starts, and how many positions of it to read.
type kvSpan struct {
	page, first, n int
}

// spans splits the position window [w0, w0+want) into per-page runs, in
// increasing position order.
//
// The order is load-bearing: the accumulate kernel seeds from Out for every
// span after the first (nn.AttnAccInto), so increasing position order
// reproduces the FMA sequence of one contiguous call and paged equals unpaged
// bit for bit.
//
// The readers below walk it one span at a time (span), which costs nothing at
// any context length; a slice of them would grow on this per-head, per-token
// path once a window passed the array it started in -- four pages, 1024
// positions, which a picture's prompt alone can pass.
func (s *kvPages) spans(dst []kvSpan, w0, want int) []kvSpan {
	if s.p == 0 || want <= 0 {
		return nil
	}
	out := dst[:0]
	for pos, end := w0, w0+want; pos < end; {
		sp := s.span(pos, end)
		out = append(out, sp)
		pos += sp.n
	}
	return out
}

// span is the run of [pos, end) inside pos's page: one step of spans' walk.
func (s *kvPages) span(pos, end int) kvSpan {
	pg, off := s.page(pos)
	return kvSpan{page: pg, first: off, n: min(s.p-off, end-pos)}
}

// kvCache is a State's whole history: one kvPages per layer, plus the identity
// the store knows it by.
type kvCache struct {
	id     string
	layers []kvPages
	// ent is DeepSeek V4's compressed history, one per block (ds4kv.go):
	// addressed by entry, not position, and empty on every block without a
	// compressor and on every other model. entRates is each block's rate.
	ent      []kvPages
	entRates []int
	store    KVStore
	// budget caps the resident bytes. Zero never evicts.
	budget uint64
	// ns is the namespace for content-addressed pages: set it and a page is named
	// by the tokens that produced it rather than by this session. seq is those
	// tokens and bh caches one boundary hash per page size. All three stay empty
	// for a session that never names a namespace, which is the default.
	ns string
	// geom is the digest of everything a page's byte count cannot tell apart.
	geom []byte
	// named is the image rows seq refers to: a negative id -1-i in seq is
	// named[i] (nameRow), the row of a picture the page keys hash by its
	// full name rather than by a number.
	named                                                                  []namedRow
	seq                                                                    []int32
	bh                                                                     map[int][]string
	stored, fetched, evicted, failed, mismatched, unmigratable, syncFailed int64
	// tailStored counts the partial pages written at a prompt boundary, which
	// is a different event from a sealed page and is the one a short prompt
	// produces. See sealTail.
	tailStored int64
	// recurrent says this model carries state outside the pages, so a restore
	// point needs a summary as well as pages. See sealRecurrentTail.
	recurrent bool
	// lo is the first block this cache holds pages for: zero for a trunk, the
	// first prediction block for a draft State (State.lo). layers is indexed by
	// block, so the slice runs from zero and the blocks below lo stay empty.
	lo int
	// keepWindow and relFault are the window gate's two arms beside the
	// shipping one (TestWindowedLayersHoldTheirWindow): the control keeps
	// every page, the violation releases the page a window starts in.
	keepWindow, relFault bool
	// walkWindow is the prefix gate's violation (TestPrefixCacheRestoresAWindowedLayer):
	// a windowed layer's coverage is walked from zero, as a full layer's is.
	walkWindow bool
	// share is the store this session's sealed pages are held once in
	// (State.ShareKV), nil when it shares nothing; shareRefused counts the
	// device moves a shared page refused.
	share        *SharedStore
	shareRefused int64
}

func newKVCache(c *Config, id string, nseq int, l kvLayout, store KVStore, pin int) *kvCache {
	return newKVCacheRange(c, id, nseq, l, store, pin, 0, c.NLayer)
}

// newKVCacheRange is newKVCache for a State that runs blocks [lo, hi) of its
// model (State.lo): the pages are indexed by block, and the blocks below lo
// have none.
func newKVCacheRange(c *Config, id string, nseq int, l kvLayout, store KVStore, pin, lo, hi int) *kvCache {
	if store == nil {
		store = NoStore{}
	}
	kc := &kvCache{id: id, store: store, layers: make([]kvPages, hi), lo: lo,
		bh: map[int][]string{}, geom: geomDigest(c, nseq, l)}
	// A model with a layer that keeps no pages keeps its history somewhere
	// else, and that somewhere has its own restore points.
	for li := kc.lo; li < len(kc.layers); li++ {
		if c.LayerKind(li).Recurrent() {
			kc.recurrent = true
			break
		}
	}
	for li := kc.lo; li < len(kc.layers); li++ {
		p := kvPagePositions(c, li, pin)
		kc.layers[li].p = p
		// DeepSeek V4's value is its key: one row, as MLA's.
		kc.layers[li].latent = c.MLA() || c.DSV4()
		if p > 0 {
			kc.layers[li].pp = l.atLayer(c, li).slots(nseq * p * c.KVRowAt(li))
			// The window Config.AttnWindow applies, whatever sized the page.
			if c.SWA(li) && c.SWAWindow > 0 {
				kc.layers[li].win, kc.layers[li].chunked = c.SWAWindow, c.SWAChunked
			}
		}
	}
	kc.initEnt(c, nseq, l, pin)
	return kc
}

// release frees every windowed layer's pages behind the window of low
// (kvPages.release).
func (kc *kvCache) release(low int) {
	if kc.keepWindow {
		return
	}
	for li := kc.lo; li < len(kc.layers); li++ {
		if s := &kc.layers[li]; s.win > 0 {
			s.release(low, kc.relFault)
		}
	}
}

// reserve sizes every layer's page index for maxSeq positions, and every
// compressed history's for the entries those positions fold.
func (kc *kvCache) reserve(maxSeq int) {
	for li := kc.lo; li < len(kc.layers); li++ {
		kc.layers[li].reserve(maxSeq)
	}
	for li := range kc.ent {
		if r := kc.entRate(li); r > 0 {
			kc.ent[li].reserve(maxSeq/r + 1)
		}
	}
}

// Bytes is what this cache occupies now. It grows with the conversation; it is
// not a reservation.
func (kc *kvCache) Bytes() uint64 {
	var n uint64
	for i := kc.lo; i < len(kc.layers); i++ {
		n += kc.layers[i].bytes()
	}
	return n
}

// layout is kvLayout addressed within one page: the same value with maxSeq
// replaced by the page's own position count. At P == maxSeq there is one page
// and its bytes are the flat cache, element for element.
func (s *kvPages) layout(l kvLayout) kvLayout { l.maxSeq = s.p; return l }

// kSpan and vSpan are the run of one page that a kvSpan names, for one
// (slot, kv head), ready to hand to a kernel as a base pointer.
func (s *kvPages) kSpan(l kvLayout, sp kvSpan, slot, kvHead int) []float32 {
	return s.k[sp.page][s.layout(l).At(slot, sp.first, kvHead):]
}

func (s *kvPages) vSpan(l kvLayout, sp kvSpan, slot, kvHead int) []float32 {
	return s.v[sp.page][s.layout(l).At(slot, sp.first, kvHead):]
}

// at is one (slot, position, kv head) vector, for the scalar fallbacks.
func (s *kvPages) at(l kvLayout, slot, pos, kvHead int) (k, v []float32) {
	pg, off := s.page(pos)
	i := s.layout(l).At(slot, off, kvHead)
	return s.k[pg][i:], s.v[pg][i:]
}

// write stores one position's whole kvDim-wide vector into the page that holds
// it, growing the page list if this position starts a new one.
func (s *kvPages) write(l kvLayout, slot, pos int, k, v []float32) {
	if s.p == 0 {
		return // a linear-attention layer keeps no history
	}
	pg, off := s.page(pos)
	s.grow(pg)
	s.cow(pg) // a shared page is never written in place
	pl := s.layout(l)
	pl.Write(s.k[pg], slot, off, k)
	pl.Write(s.v[pg], slot, off, v)
	if pos+1 > s.written {
		s.written = pos + 1
	}
}

// gather copies positions [0, n) of one slot into a flat contiguous buffer, and
// scatter puts them back.
//
// They exist for relocation: nn.LayerDevice.MigrateKV takes a layer's history
// as one contiguous run (backend.Buf has no offset on Write), so the seam
// converts between tokens rather than pushing pages onto the tier.
func (s *kvPages) gather(l kvLayout, slot, n int, k, v []float32) bool {
	if s.p == 0 || n <= 0 {
		return true
	}
	// A shared page does not move to a device for one of its holders: the
	// block stays home (kvshare.go).
	if s.anyShared() {
		return false
	}
	pl := s.layout(l)
	kvDim := l.kvDim()
	// A windowed layer's pages below what its window still reaches were
	// released: nothing reads those positions again, so they travel as zeros.
	live := s.liveFrom(n)
	for pos := 0; pos < n; pos++ {
		pg, off := s.page(pos)
		if pg < max(live, s.gone) && !s.resident(pg) {
			o, w := l.slots(pos*kvDim), l.slots(kvDim)
			clear(k[o : o+w])
			clear(v[o : o+w])
			continue
		}
		// Resident, not merely addressable: eviction leaves nil slots. Nothing
		// has faulted for a migration, so refuse and let the caller keep the
		// block where it is.
		if !s.resident(pg) {
			return false
		}
		i := pl.At(slot, off, 0)
		o := l.slots(pos * kvDim)
		w := l.slots(kvDim)
		copy(k[o:o+w], s.k[pg][i:i+w])
		copy(v[o:o+w], s.v[pg][i:i+w])
	}
	return true
}

func (s *kvPages) scatter(l kvLayout, slot, n int, k, v []float32) bool {
	if s.p == 0 || n <= 0 {
		return true
	}
	// Grow every page, not only the last: grow(last) allocates only `last`,
	// and a placed block's host pages are all nil on demotion. A windowed
	// layer grows only what its window still reaches: below that, the device
	// may hold nothing (it releases those pages too) and the host would read
	// nothing.
	last, _ := s.page(n - 1)
	live := s.liveFrom(n)
	for pg := live; pg <= last; pg++ {
		s.grow(pg)
	}
	if s.win > 0 {
		// What came back below the window of n-1-KVWindowSlack is the device's
		// to vouch for and it does not: it releases pages behind that.
		s.genuine = max(s.genuine, s.keyStart(max(0, n-1-nn.KVWindowSlack)))
	}
	pl := s.layout(l)
	kvDim := l.kvDim()
	for pos := live * s.p; pos < n; pos++ {
		pg, off := s.page(pos)
		s.cow(pg)
		i := pl.At(slot, off, 0)
		o := l.slots(pos * kvDim)
		w := l.slots(kvDim)
		copy(s.k[pg][i:i+w], k[o:o+w])
		copy(s.v[pg][i:i+w], v[o:o+w])
	}
	if n > s.written {
		s.written = n
	}
	return true
}

// nextCacheID hands out a fresh identity for every State's history. It is
// unique and deliberately not reusable: a session that wants its pages found
// again names itself with SetCacheKey. It is not a Scheduler row, which admit
// reuses as soon as its sequence retires.
var nextCacheID = func() func() string {
	// A per-process nonce: a bare counter restarts in every process, and a
	// second process would address the first one's pages in a shared store.
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		// A process that cannot read randomness must not fall back to a
		// guessable name; the timestamp is only a tiebreak beside the counter.
		binary.LittleEndian.PutUint64(b[:], uint64(time.Now().UnixNano()))
	}
	nonce := hex.EncodeToString(b[:])
	var n atomic.Int64
	return func() string {
		return "session-" + nonce + "-" + strconv.FormatInt(n.Add(1), 10)
	}
}()

// migrateKV moves block li's history between the host's pages and the tier.
//
// It gathers the pages into one contiguous run on the way out and scatters
// them back on the way in (see gather).
func (s *State) migrateKV(li, pos int, toDevice bool) bool {
	// A KV-sharing block has no history of its own: it reads its source's,
	// which moves with the source (fixKVGroups keeps them together).
	if s.ld == nil || s.c.KVShared(li) {
		return true
	}
	pg := &s.kv.layers[li]
	if pg.p == 0 || pos <= 0 {
		// A linear-attention block has no history, and nothing to move at
		// position zero is success -- the same answer the tier gives.
		return s.ld.MigrateKV(li, nil, nil, 0, toDevice)
	}
	if s.nseq > 1 {
		return s.migrateKVRows(li, toDevice)
	}
	// A shared page does not move to a device for one of its holders: the
	// block stays home (kvshare.go).
	if toDevice && pg.anyShared() {
		s.kv.shareRefused++
		return false
	}
	kvl := s.kvlAt(li)
	k, v, fk, fv := kvMigBufs(kvl, pos)
	if toDevice && !pg.gather(kvl, 0, pos, k, v) {
		return false
	}
	if !s.migrateKVAs(kvl, pos, k, v, fk, fv, toDevice, func() bool {
		return s.ld.MigrateKV(li, fk, fv, pos, toDevice)
	}) {
		return false
	}
	// A DeepSeek V4 block's entries travel with its rows (ds4dev.go).
	if !s.migrateEnt(li, []int{pos}, func(int) int { return 0 }, toDevice) {
		return false
	}
	if toDevice {
		return true
	}
	if !pg.scatter(kvl, 0, pos, k, v) {
		return false
	}
	// What scatter left out lies behind the window: released, as far as
	// seal and the read walk are concerned.
	pg.gone = max(pg.gone, pg.liveFrom(pos))
	return true
}

// kvMigBufs is the buffers a migration of n positions of layout kvl moves
// through: k and v in the host cache's format, which gather fills and scatter
// reads, and fk and fv in the float32 a device takes (MigrateKV's contract).
// They are the same buffers unless the cache is q8.
func kvMigBufs(kvl kvLayout, n int) (k, v, fk, fv []float32) {
	k = make([]float32, kvl.slots(n*kvl.kvDim()))
	v = make([]float32, len(k))
	if kvl.fmt != cpu.KVQ8 {
		return k, v, k, v
	}
	fk = make([]float32, n*kvl.kvDim())
	fv = make([]float32, len(fk))
	return k, v, fk, fv
}

// migrateKVAs runs move between the q8 rows and the float32 a device holds: a
// q8 history leaves widened to d*q (nn.WidenKVRows) and comes home quantized
// again (nn.QuantizeKVRows). Widening then quantizing gives the same int8
// back and each scale to within an ulp (the block's amax widens to 127*d), so
// a history that travels and returns keeps its rows; what the device wrote
// while it held the block is quantized as the host would have quantized it.
// The device holds float32 pages: a q8 cache resident on a device is not
// implemented (kvtype.go).
func (s *State) migrateKVAs(kvl kvLayout, n int, k, v, fk, fv []float32, toDevice bool, move func() bool) bool {
	rows := n * kvl.nKV
	if kvl.fmt == cpu.KVQ8 && toDevice {
		s.jit.WidenKVRows(fk, k, kvl.headDim, rows)
		s.jit.WidenKVRows(fv, v, kvl.headDim, rows)
	}
	if !move() {
		return false
	}
	if kvl.fmt == cpu.KVQ8 && !toDevice {
		s.jit.QuantizeKVRows(k, fk, kvl.headDim, rows)
		s.jit.QuantizeKVRows(v, fv, kvl.headDim, rows)
	}
	return true
}

// migrateKVRows is migrateKV for a batch: every row's history, not row 0's.
//
// The device keeps one cache per block with row r at positions
// r*maxSeq+pos (nn.RowsDevice), so the rows travel as one flat run in that
// layout and MigrateKV's contract is unchanged: the positions it moves are
// device positions, up to the end of the last row with history. Each row
// contributes its own live prefix, bpos[r], which is why the high-water
// s.pos is not used; the gaps between rows are positions nothing reads.
func (s *State) migrateKVRows(li int, toDevice bool) bool {
	pg := &s.kv.layers[li]
	if sd, ok := s.ld.(nn.SeqKVDevice); ok && sd.PerSequenceKV() {
		return s.migrateKVSeqs(sd, li, toDevice)
	}
	ext := 0
	for r, p := range s.bpos {
		if p > 0 {
			ext = r*s.maxSeq + p
		}
	}
	if ext == 0 {
		return s.ld.MigrateKV(li, nil, nil, 0, toDevice)
	}
	// The next rows step reserves the whole batch anyway (LayersRows); asking
	// here is what lets the device hold rows past its current capacity before
	// any step has grown it.
	if toDevice && !s.ld.ReserveKV(s.nseq*s.maxSeq) {
		return false
	}
	kvl := s.kvlAt(li)
	kvDim := kvl.kvDim()
	k, v, fk, fv := kvMigBufs(kvl, ext)
	for r, p := range s.bpos {
		if !toDevice || p == 0 {
			continue
		}
		o := kvl.slots(r * s.maxSeq * kvDim)
		if !pg.gather(kvl, r, p, k[o:], v[o:]) {
			return false
		}
	}
	if !s.migrateKVAs(kvl, ext, k, v, fk, fv, toDevice, func() bool {
		return s.ld.MigrateKV(li, fk, fv, ext, toDevice)
	}) {
		return false
	}
	if toDevice {
		return true
	}
	for r, p := range s.bpos {
		if p == 0 {
			continue
		}
		o := kvl.slots(r * s.maxSeq * kvDim)
		if !pg.scatter(kvl, r, p, k[o:], v[o:]) {
			return false
		}
	}
	return true
}

// migrateKVSeqs is migrateKVRows on a device that keeps each row's history
// as its own sequence (nn.SeqKVDevice): every row with history moves on its
// own, by the base of its slots, and nothing is reserved -- the device takes
// the pages a row's positions need as it loads them.
func (s *State) migrateKVSeqs(sd nn.SeqKVDevice, li int, toDevice bool) bool {
	pg := &s.kv.layers[li]
	kvl := s.kvlAt(li)
	for r, p := range s.bpos {
		if p == 0 {
			continue
		}
		k, v, fk, fv := kvMigBufs(kvl, p)
		if toDevice && !pg.gather(kvl, r, p, k, v) {
			return false
		}
		if !s.migrateKVAs(kvl, p, k, v, fk, fv, toDevice, func() bool {
			return sd.MigrateKVSeq(li, r*s.maxSeq, fk, fv, p, toDevice)
		}) {
			return false
		}
		if !toDevice && !pg.scatter(kvl, r, p, k, v) {
			return false
		}
	}
	// A DeepSeek V4 block's entries travel with its rows (ds4dev.go).
	return s.migrateEnt(li, s.bpos, func(r int) int { return r * s.maxSeq }, toDevice)
}

// migrateRec moves one placed linear block's running summary between the host
// arrays and the device: migrateKV's twin, and every release path must call
// both or the host resumes from a summary that never saw a token.
//
// A batch moves every row at once: rconv/rstate hold the rows back to back,
// row r at r*convStateLen and r*deltaStateLen, which is the device's own
// per-row layout (nn.LayerDevice.MigrateRec).
func (s *State) migrateRec(li int, toDevice bool) bool {
	if s.ld == nil || s.rconv == nil || li >= len(s.rconv) || s.rconv[li] == nil {
		return true // an attention block, or a session with no recurrence
	}
	return s.migrateRecLayer(li, toDevice)
}

// migrateRecLayer is LayerDevice.MigrateRec on block li's host arrays, with
// the one part of a recurrent state the host keeps for itself:
// Qwen3.8-Flash-Next's n-gram predecessors, which the host hashes whichever
// side runs the block, so the device's copy of them is never the live one
// and a migration home keeps the host's.
func (s *State) migrateRecLayer(li int, toDevice bool) bool {
	if toDevice || !s.c.NGramAt(li) {
		return s.ld.MigrateRec(li, s.rconv[li], s.rstate[li], toDevice)
	}
	nseq := len(s.rstate[li]) / s.dg.stateStride()
	np := s.c.NGram - 1
	keep := make([]float32, nseq*np)
	for slot := 0; slot < nseq; slot++ {
		prev, _ := s.ngramState(li, slot)
		copy(keep[slot*np:], prev)
	}
	if !s.ld.MigrateRec(li, s.rconv[li], s.rstate[li], false) {
		return false
	}
	for slot := 0; slot < nseq; slot++ {
		prev, _ := s.ngramState(li, slot)
		copy(prev, keep[slot*np:(slot+1)*np])
	}
	return true
}

// The paged read walk. Every attention read in model/ goes through these four,
// so the page arithmetic exists once. The scores are written contiguously, so
// softmax still normalises over the whole attended range, and the spans are
// consumed in order (see spans).

// kvScores fills af[0:want] with dot(q, k[t]) for t in [w0, w0+want).
func (s *State) kvScores(fast *nn.AttnSet, li, slot, kvh int, q, af []float32, w0, want int) {
	pg := &s.kv.layers[li]
	for pos, end := w0, w0+want; pos < end && pg.p > 0; {
		sp := pg.span(pos, end)
		if !pg.resident(sp.page) {
			return
		}
		fast.AttnScores(af[pos-w0:], pg.kSpan(s.kvlAt(li), sp, slot, kvh), q, sp.n)
		pos += sp.n
	}
}

// kvScores2 is kvScores for two query heads sharing a kv head. It reports false
// when no paired kernel served the whole walk, and the caller makes two single
// passes -- all or nothing, so a window is never half paired.
func (s *State) kvScores2(fast *nn.AttnSet, li, slot, kvh int, q0, q1, af0, af1 []float32, w0, want int) bool {
	pg := &s.kv.layers[li]
	if pg.p == 0 || want <= 0 {
		return false
	}
	for pos, end := w0, w0+want; pos < end; {
		x := pg.span(pos, end)
		if !pg.resident(x.page) {
			return false
		}
		at := pos - w0
		if !fast.AttnScores2(af0[at:], af1[at:], pg.kSpan(s.kvlAt(li), x, slot, kvh), q0, q1, x.n) {
			return false
		}
		pos += x.n
	}
	return true
}

// kvAcc accumulates V over the window into out.
func (s *State) kvAcc(fast *nn.AttnSet, li, slot, kvh int, out, af []float32, w0, want int) {
	pg := &s.kv.layers[li]
	for pos, end := w0, w0+want; pos < end && pg.p > 0; {
		sp := pg.span(pos, end)
		if !pg.resident(sp.page) {
			return
		}
		v := pg.vSpan(s.kvlAt(li), sp, slot, kvh)
		if pos == w0 {
			fast.AttnAcc(out, v, af, sp.n)
		} else {
			fast.AttnAccInto(out, v, af[pos-w0:], sp.n)
		}
		pos += sp.n
	}
}

// kvAcc2 is kvAcc for two heads at once, all or nothing like kvScores2.
func (s *State) kvAcc2(fast *nn.AttnSet, li, slot, kvh int, o0, o1, af0, af1 []float32, w0, want int) bool {
	pg := &s.kv.layers[li]
	if pg.p == 0 || want <= 0 {
		return false
	}
	// Page 0 overwrites and the rest add in, as in the single-head walk.
	for pos, end := w0, w0+want; pos < end; {
		x := pg.span(pos, end)
		if !pg.resident(x.page) {
			return false
		}
		at := pos - w0
		v := pg.vSpan(s.kvlAt(li), x, slot, kvh)
		ok := false
		if pos == w0 {
			ok = fast.AttnAcc2(o0, o1, v, af0[at:], af1[at:], x.n)
		} else {
			ok = fast.AttnAcc2Into(o0, o1, v, af0[at:], af1[at:], x.n)
		}
		if !ok {
			// All or nothing: a window half-paired would have the first spans
			// already written into o0/o1, so the caller's two single passes
			// must start from a clean slate. kvAcc's page-0 call does that.
			return false
		}
		pos += x.n
	}
	return true
}

// ============================ residency ============================
//
// A sealed page is written through to the store, which makes eviction a free()
// rather than a transfer; spilling at eviction time would put a write on the
// path already short of memory. With NoStore a resident-only session pays
// nothing beyond one no-op call per sealed page.

// sealed reports whether page n of this layer can no longer change. Only the
// page holding the newest position is mutable; everything behind it is final.
func (s *kvPages) sealed(n, pos int) bool {
	cur, _ := s.page(pos)
	return n < cur
}

// pageBytes is one page's size in bytes.
func (s *kvPages) pageBytes() uint64 { return uint64(s.pp) * 4 }

// kvBytesOf views a page as bytes for the store, without copying.
func kvBytesOf(p []float32) []byte {
	if len(p) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&p[0])), len(p)*4)
}

// SetKVBudget caps what this session's history may hold resident, in bytes.
// Zero, the default, never evicts. A nonzero budget needs a store: trim only
// evicts pages the store already holds, so without one the cap would silently
// do nothing (and evicting anyway would lose history).
func (s *State) SetKVBudget(b uint64) error {
	if b > 0 {
		if _, ok := s.kv.store.(NoStore); ok {
			return fmt.Errorf("model: a KV budget needs a store to evict into -- " +
				"call SetKVStore first, or the cap would drop history rather than spill it")
		}
	}
	s.kv.budget = b
	return nil
}

// SetKVStore hands the session a backing store for its pages.
func (s *State) SetKVStore(st KVStore) {
	if st == nil {
		st = NoStore{}
	}
	s.kv.store = st
}

// SetCacheKey names a namespace for this session's history so a later one can
// find it. The default key is process-local (see nextCacheID). With a
// namespace, each page is stored under H(the tokens that produced it) inside
// it, so PrefillCached can find the longest cached prefix of any prompt.
//
// The namespace must carry whatever must match for a page to be valid and the
// hash does not cover: the model, its quantization and the KV element width
// ("tinyllama/q3_K_M/f32"). The engine cannot check it.
//
// It is also the tenancy dial: the hash is scoped inside the namespace, so
// what goes in front of it decides who shares with whom:
//
//	"llama3/q4/f32"              every session shares every prefix
//	"llama3/q4/f32/tenant-42"    shared within one tenant, isolated between them
//	"llama3/q4/f32/sess-<id>"    shared with nobody -- a private cache
//
// A cache hit is observable, so a shared namespace is a side channel on the
// prefix; per-user namespaces trade sharing for isolation. A tiered design
// (shared public prefix, private tail) is not built.
//
// Call it before the first token.
func (s *State) SetCacheKey(ns string) error {
	// A batched page holds every row's slots, so no single row's tokens can
	// name it.
	if ns != "" && s.nseq > 1 {
		return fmt.Errorf("model: a %d-row batch cannot use a cache namespace: one page "+
			"holds every row's slots, so no single sequence's tokens can name it", s.nseq)
	}
	// It wants a fresh sequence: mid-flight, the token log would restart at the
	// current position and name every later page by a shifted window.
	if ns != "" && s.pos > 0 {
		return fmt.Errorf("model: SetCacheKey wants a fresh sequence, this one is at "+
			"position %d -- the tokens before it were never recorded and the pages "+
			"after it would be named by a shifted window", s.pos)
	}
	s.kv.ns = ns
	return nil
}

// KVRestored is how many prompt positions the last PrefillCached took from the
// store instead of computing. Zero is a miss, and a caller that does not look at
// it cannot tell a working cache from a silently useless one.
func (s *State) KVRestored() int { return s.kvSkipped }

// CacheNamespace is what SetCacheKey named, or "" when content addressing is
// off. A content-addressed session has no single key: its pages are one key
// each (count them with FileStore.Total).
func (s *State) CacheNamespace() string { return s.kv.ns }

// RestoreKV pulls back every page the store holds under this session's key and
// returns the number of positions that are now valid to attend over.
//
// The answer is the minimum across layers: every layer attends over the same
// positions, and page sizes differ per layer, so the coarsest page decides.
// Layers with more coverage are truncated rather than kept, or their high
// lastSealed would stop seal() offering the rewritten pages and the store
// would keep serving stale ones. A linear layer's summary is restored
// separately (State.faultRecurrentAt).
func (kc *kvCache) restore() (int, int, error) {
	if _, ok := kc.store.(NoStore); ok {
		return 0, 0, nil
	}
	// No namespace means no restore: the key is the only check that the pages
	// belong to these tokens.
	if kc.ns == "" {
		return 0, 0, nil
	}
	best := -1
	for li := kc.lo; li < len(kc.layers); li++ {
		pg := &kc.layers[li]
		if pg.p == 0 {
			// A linear layer has no pages; its summary is restored by
			// State.faultRecurrentAt. Coverage is over the attention layers.
			continue
		}
		// Drop residency first so the probe asks the store: fault() returns nil
		// for a resident page, and a reused State's leftovers would count as
		// hits.
		for n := range pg.k {
			pg.drop(n)
		}
		pg.k, pg.v, pg.out, pg.lastSealed = pg.k[:0], pg.v[:0], pg.out[:0], 0
		pg.gone, pg.genuine = 0, 0
		if !kc.covers(li) {
			// A windowed layer is not walked from zero: it stores only what
			// its window reached, and restoreWin faults that once the
			// coverage is known.
			continue
		}
		n := 0
		for {
			err := kc.fault(li, n)
			if errors.Is(err, ErrNoPage) {
				break
			}
			if err != nil {
				return 0, 0, err
			}
			n++
		}
		if cov := n * pg.p; best < 0 || cov < best {
			best = cov
		}
	}
	// DeepSeek V4's coverage is its entries' (ds4kv.go): returned as a tail,
	// so restoreKV clamps it in positions.
	if kc.entRestores() {
		t, err := kc.entRestore()
		return t, t, err
	}
	if best < 0 {
		return 0, 0, nil
	}
	// The partial tail (sealTail's page, keyed by its fill), which is the whole
	// prompt on a short chat turn. The fill is probed downward from this
	// prompt's remainder, exact when the prompt repeats, and is all-or-nothing
	// across layers.
	tail := 0
	if best >= 0 && kc.ns != "" {
		probe := -1
		for li := kc.lo; li < len(kc.layers); li++ {
			if kc.covers(li) {
				probe = li
				break
			}
		}
		if probe >= 0 {
			pg := &kc.layers[probe]
			n := best / pg.p
			max := len(kc.seq) - n*pg.p
			if max > pg.p-1 {
				max = pg.p - 1
			}
			// One lookup per chunk boundary, as vLLM and LMCache do. A hybrid
			// probes only the exact end: its recurrent summary exists at one
			// fill per prompt (see sealRecurrentTail), so any shorter hit would
			// be given up anyway.
			probes := kvChunkProbes(max)
			if kc.recurrent {
				probes = []int{max}
			}
			for _, f := range probes {
				if kc.faultFill(probe, n, f) == nil {
					tail = f
					break
				}
			}
		}
		if tail > 0 {
			for li := kc.lo; li < len(kc.layers); li++ {
				pg := &kc.layers[li]
				if !kc.covers(li) || li == probe {
					continue
				}
				if err := kc.faultFill(li, best/pg.p, tail); err != nil {
					tail = 0
					break
				}
			}
		}
		if tail == 0 {
			// Drop whatever the probe brought in: a page present and unusable
			// would be mistaken for history.
			for li := kc.lo; li < len(kc.layers); li++ {
				pg := &kc.layers[li]
				if !kc.covers(li) {
					continue
				}
				if n := best / pg.p; n < len(pg.k) {
					pg.drop(n)
				}
			}
		}
	}
	if best+tail <= 0 {
		return 0, 0, nil
	}
	for li := kc.lo; li < len(kc.layers); li++ {
		pg := &kc.layers[li]
		if !kc.covers(li) {
			continue // a linear layer has no pages to truncate, a windowed one none yet
		}
		keep := best / pg.p
		for n := keep + 1; n < len(pg.k); n++ {
			pg.drop(n)
		}
		if tail == 0 && keep < len(pg.k) {
			pg.drop(keep)
		}
		pg.lastSealed = keep
		if w := keep*pg.p + tail; w > pg.written {
			pg.written = w // a restored page came out of the store whole
		}
	}
	return best + tail, tail, nil
}

// covers reports whether layer li's stored pages decide how much of a prefix
// restores: every attention layer but a windowed one, which keeps (and
// stores) only what its window reaches. A range of blocks that is all
// windowed has nothing else to decide by, and walks them as before.
func (kc *kvCache) covers(li int) bool {
	s := &kc.layers[li]
	if s.p == 0 {
		return false
	}
	if s.win == 0 || kc.walkWindow {
		return true
	}
	// DeepSeek V4: every block's rows slide, and its entries decide
	// (entRestore).
	if kc.entRestores() {
		return false
	}
	for i := kc.lo; i < len(kc.layers); i++ {
		if x := &kc.layers[i]; x.p > 0 && x.win == 0 {
			return false
		}
	}
	return true
}

// restoreWin faults every windowed layer's pages the restored prefix of n
// positions reads from (the page its window starts in on: nothing rewinds
// behind a restored prompt), out of a store whose prefix covers
// cover >= n positions: a page wholly inside cover was stored whole, the one
// straddling it by its fill. It reports false, having kept nothing it
// faulted, when any of them is missing.
func (kc *kvCache) restoreWin(n, cover int) (bool, error) {
	for li := kc.lo; li < len(kc.layers); li++ {
		s := &kc.layers[li]
		if s.p == 0 || kc.covers(li) || n <= 0 {
			continue
		}
		first, last := s.keyStart(n)/s.p, (n-1)/s.p
		for j := first; j <= last; j++ {
			var err error
			if (j+1)*s.p <= cover {
				err = kc.fault(li, j)
			} else {
				err = kc.faultFill(li, j, cover-j*s.p)
			}
			if errors.Is(err, ErrNoPage) {
				kc.dropWin()
				return false, nil
			}
			if err != nil {
				return false, err
			}
		}
		// Below first is not history: nothing was faulted there, and no page
		// starting there may be stored as if it were.
		s.gone, s.genuine = first, first*s.p
	}
	return true, nil
}

// dropWin forgets every page a windowed layer holds, which a restore that
// cannot serve its window gives back.
func (kc *kvCache) dropWin() {
	for li := kc.lo; li < len(kc.layers); li++ {
		if s := &kc.layers[li]; s.p > 0 && !kc.covers(li) {
			for j := range s.k {
				s.drop(j)
			}
		}
	}
}

// dropTail forgets the partial page at coverage n, which is how a caller gives
// a tail back when it cannot use it.
func (kc *kvCache) dropTail(n int) {
	for li := kc.lo; li < len(kc.layers); li++ {
		pg := &kc.layers[li]
		if pg.p == 0 {
			continue
		}
		idx := n / pg.p
		if idx < len(pg.k) {
			pg.drop(idx)
		}
		if w := idx * pg.p; pg.written > w {
			pg.written = w
		}
	}
}

// PrefillCached runs a prompt, skipping whatever prefix of it the store already
// holds under this session's key.
//
// It checks that every layer can serve the positions it skips; it trusts that
// the skipped tokens produced those pages, which the key (see SetCacheKey)
// asserts. RoPE is applied before the cache write, so a stored key carries its
// absolute position and only a prefix is reusable.
func (s *State) PrefillCached(tokens []int32) ([]float32, error) {
	defer s.m.enterPager()()
	if s.batched {
		return nil, errBatch{}
	}
	if len(tokens) == 0 {
		return nil, errEmptyPrompt{}
	}
	if s.seqPos(0) != 0 {
		return nil, fmt.Errorf("model: PrefillCached wants a fresh sequence, this one is at position %d", s.seqPos(0))
	}
	return s.prefillCachedKey(tokens, len(tokens), func(n int) ([]float32, error) { return s.prefillInto(0, tokens[n:]) })
}

// PrefillCachedMixed is PrefillCached for a prompt of token and embedding
// spans -- a chat turn with a picture in its history.
//
// A page is named by the rows that produced it, so every row needs an id: a
// token's own, and for an image row the picture's name (Span.Key, ImageKey),
// the row's index in it and, on a model whose rotary is multi-axis, its
// (t, h, w) -- one image's rows under another grid or at another rotary
// position are different rows (kvCache.nameRow). The pages after the image
// are then named as text pages are, so a second turn about the same picture
// restores everything up to its own new tokens.
//
// An embedding span with no key stops the naming there: its rows and every
// row after them are prefilled, never restored. A bidirectional run (Gemma
// 3's image rows) is restored whole or not at all: its rows' history depends
// on rows after them, which the run's name covers, and a restore that ended
// inside it would leave the tail a chunk that cuts the run.
func (s *State) PrefillCachedMixed(spans ...Span) ([]float32, error) {
	defer s.m.enterPager()()
	if s.batched {
		return nil, errBatch{}
	}
	if s.seqPos(0) != 0 {
		return nil, fmt.Errorf("model: PrefillCachedMixed wants a fresh sequence, this one is at position %d", s.seqPos(0))
	}
	spans = named(spans)
	rows := 0
	for _, sp := range spans {
		rows += sp.n(s.c.NEmbd)
	}
	if err := s.bidirRuns(0, spans); err != nil {
		return nil, err
	}
	defer func() {
		if len(s.bidir) > 0 {
			if kd, ok := s.ld.(nn.KeyRunDevice); ok {
				kd.SetKeyRuns(nil, nil)
			}
		}
		s.bidir = s.bidir[:0]
	}()
	rp, next, err := s.spanRope(0, 0, spans)
	if err != nil {
		return nil, err
	}
	key := s.spanKey(0, spans, rp)
	s.markRope(0, rows, next-rows)
	return s.prefillCachedKey(key, rows, func(n int) ([]float32, error) {
		// Only now are the rows needed, and only from n on: a picture the
		// restore covered whole runs no tower block.
		enc, err := s.pictureRows(spans, n)
		if err != nil {
			return nil, err
		}
		src, err := s.spanSrc(enc)
		if err != nil {
			return nil, err
		}
		undoDeep, err := s.deepRuns(enc, n)
		if err != nil {
			return nil, err
		}
		defer undoDeep()
		var tail [][4]int
		if rp != nil {
			tail = rp[n:]
		}
		defer s.tokensOf(enc)()
		s.spanTok = s.spanTok[min(n, len(s.spanTok)):]
		return s.prefillSrc(0, src[n:], nil, tail)
	})
}

// spanKey is the id per row the pages of a mixed prompt starting at position
// p0 are named by: a token's own id, and for an image row a reference to the
// row's name in the cache's table (kvCache.nameRow). It stops at the first
// embedding span with no key, so it may be shorter than the prompt.
func (s *State) spanKey(p0 int, spans []Span, rp [][4]int) []int32 {
	nembd := s.c.NEmbd
	var key []int32
	for _, sp := range spans {
		if !sp.rows() {
			key = append(key, sp.Tokens...)
			continue
		}
		if sp.Key == nil || sp.Key.zero() {
			return key
		}
		for i := 0; i < sp.n(nembd); i++ {
			var at [4]int
			if rp != nil {
				at = rp[len(key)]
			}
			key = append(key, s.kv.nameRow(*sp.Key, i, at, p0+len(key)))
		}
	}
	return key
}

// prefillCachedKey is PrefillCached over a prompt of rows rows whose key is
// the id per row the pages are named by: the tokens for a text prompt, and for
// a mixed one (PrefillCachedMixed) a token's id or an image row's name. A key
// shorter than the prompt restores no further than it reaches. run prefills
// the rows from n on, after the restore.
func (s *State) prefillCachedKey(key []int32, rows int, run func(n int) ([]float32, error)) ([]float32, error) {
	n, full, err := s.restoreKey(key, rows, true)
	if err != nil {
		return nil, err
	}
	if full != nil && n == rows {
		// Nothing to prefill: the position is already past the prompt and the
		// logits are the ones that run produced. The tail is already in the
		// store under this key, so there is nothing to seal either.
		return full, nil
	}
	lg, err := run(n)
	if err != nil {
		return nil, err
	}
	if err := s.SealPrompt(lg); err != nil {
		return nil, err
	}
	return lg, nil
}

// RestorePrefix is the restore half of PrefillCached: it restores the
// longest prefix of tokens the store holds under this session's key, leaving
// at least the last token to run, and returns how many positions it took.
// The State is then at that position, and the caller runs tokens[n:] by any
// path that records its ids (Prefill, Forward, StepRuns) and calls SealPrompt
// with the last logits, so the next request finds this prompt too. A server's
// step loop restores this way and then runs the rest as a row.
func (s *State) RestorePrefix(tokens []int32) (int, error) {
	defer s.m.enterPager()()
	if s.batched {
		return 0, errBatch{}
	}
	if len(tokens) == 0 {
		return 0, errEmptyPrompt{}
	}
	if s.seqPos(0) != 0 {
		return 0, fmt.Errorf("model: RestorePrefix wants a fresh sequence, this one is at position %d", s.seqPos(0))
	}
	n, _, err := s.restoreKey(tokens, len(tokens), false)
	return n, err
}

// SealPrompt offers the store what a prompt just run left: the pages a
// placed block holds on its device, the partial last page, a hybrid's
// recurrent summary at the position, and the logits lg the prompt ended on.
// PrefillCached calls it itself; a caller that restored with RestorePrefix
// calls it once the rest of the prompt has run. Without a store or a cache
// key it does nothing.
func (s *State) SealPrompt(lg []float32) error {
	if err := s.syncKVFromDevice(); err != nil {
		return err
	}
	// Seal the prompt's own partial page too: seal only offers pages the
	// position has passed, and a short prompt would store nothing.
	s.kv.tailStored += int64(s.kv.sealTail(s.pos))
	s.sealRecurrentTail(s.pos)
	s.sealLogits(s.pos, lg)
	return nil
}

// restoreKey restores what the store holds of a prompt of rows rows named by
// key and advances the State past it, returning the positions restored and,
// for a whole-prompt hit when logits says the caller can take one, the
// logits the prompt ended on. Without them a whole-prompt hit gives its last
// position back, so something is left to run.
func (s *State) restoreKey(key []int32, rows int, logits bool) (int, []float32, error) {
	tokens := key
	// The tokens come first: restore() names the pages by H(tokens[0:boundary]).
	s.kv.seq, s.kv.bh = append(s.kv.seq[:0], tokens...), map[int][]string{}
	// Before the restore reserves the prefix on the card, so a shorter prompt
	// than the last sequence's gives its room back first.
	s.relocateFresh(rows)
	// The whole prompt may restore, because sealLogits stores the final
	// logits: a full hit with its logits leaves nothing to run. (A hybrid's
	// summary cannot be shortened by one position to make room for a token.)
	n, err := s.restoreKV(len(tokens))
	if err != nil {
		return 0, nil, err
	}
	full := []float32(nil)
	if n == rows {
		lg := make([]float32, s.c.NVocab)
		if logits && s.faultLogits(n, lg) {
			full = lg
		} else {
			// No logits for this boundary: give the last position back so
			// something is left to run. A hybrid cannot shorten a restored
			// tail, so it gives the tail up entirely -- restoreKV's rule.
			if n, err = s.restoreKV(rows - 1); err != nil {
				return 0, nil, err
			}
		}
	}
	// A restore that ends inside a bidirectional run gives the run back
	// whole: the rows left to prefill would be a chunk that cuts it.
	for _, r := range s.bidir {
		if r.Lo < n && n < r.Hi {
			if n, err = s.restoreKV(r.Lo); err != nil {
				return 0, nil, err
			}
			full = nil
		}
	}
	// Past the restored prefix the sequence is whatever the caller goes on to
	// generate, so the tail of the prompt stays and the rest is appended as it
	// arrives. Trimming to n would lose the tail this call is about to prefill.
	s.kv.seq = append(s.kv.seq[:0], tokens...)
	// A restored prefix is pushed onto the card for placed blocks, whose
	// history lives there (host pages are written only for host blocks). This
	// runs before advance, which seals and trims, so gather finds the pages
	// resident. Reserve the card's KV first, since it otherwise grows only as
	// tokens run; a migration that still cannot happen is a miss (a full
	// prefill), not a failed run.
	if n > 0 && s.ld != nil && s.devCount() > 0 && !s.ld.ReserveKV(n) {
		s.kv.unmigratable++
		n = 0
	}
	for _, li := range s.placedFrom(s.lo) {
		if n == 0 {
			break
		}
		if !s.migrateKV(li, n, true) {
			s.kv.unmigratable++
			n = 0
			break
		}
	}
	// The recurrent summary goes up too: restore() faults it into the host's
	// rconv/rstate, and a placed linear block reads its own device copy.
	if n > 0 && !s.pushRecToDevice() {
		s.kv.unmigratable++
		n = 0
	}
	if n > 0 {
		s.advance(0, n)
	}
	s.kvSkipped = n
	return n, full, nil
}

// syncKVFromDevice brings a placed block's history down into the host pages so
// that seal has something to store.
//
// runPrep writes host pages only for host blocks, so without this a fully
// offloaded model would store nothing. It runs once, after the prompt, not per
// token, to keep device round trips off decode; positions generated after it
// are not stored for placed blocks.
func (s *State) syncKVFromDevice() error {
	if s.devCount() == 0 || s.kv.ns == "" || s.pos == 0 {
		return nil
	}
	if _, ok := s.kv.store.(NoStore); ok {
		return nil
	}
	// The recurrent summary comes home too: sealRecurrentTail writes through
	// s.rconv/s.rstate, which are stale for a placed linear block.
	if err := s.syncRecFromDevice(); err != nil {
		return err
	}
	for _, li := range s.placedFrom(s.lo) {
		if !s.migrateKV(li, s.pos, false) {
			// A capability gap, not a wrong answer: the f16 V cache cannot be
			// copied as float32 and MigrateKV says so. The run continues with
			// nothing cached for those blocks rather than failing.
			s.kv.syncFailed++
			return nil
		}
	}
	s.kv.seal(s.pos)
	return nil
}

// syncRecFromDevice brings every placed linear block's summary into the host's
// rconv/rstate, where every host consumer reads it: the recurrent twin of
// syncKVFromDevice.
func (s *State) syncRecFromDevice() error {
	if s.rconv == nil || s.devCount() == 0 || s.ld == nil {
		return nil
	}
	for _, li := range s.placedFrom(s.lo) {
		if s.rconv[li] == nil {
			continue // not a linear block
		}
		if !s.migrateRecLayer(li, false) {
			// A capability gap, not a wrong answer: nothing is cached for this
			// run rather than a summary that did not come from it.
			s.kv.syncFailed++
			return nil
		}
	}
	return nil
}

// pushRecToDevice sends every placed linear block's restored summary back to
// the device that runs it. It is syncRecFromDevice with the arrow reversed.
func (s *State) pushRecToDevice() bool {
	if s.rconv == nil || s.devCount() == 0 || s.ld == nil {
		return true
	}
	for _, li := range s.placedFrom(s.lo) {
		if s.rconv[li] == nil {
			continue
		}
		if !s.migrateRecLayer(li, true) {
			return false
		}
	}
	return true
}

// restoreKV restores at most max positions, which is how the caller keeps a
// token to actually run: a prompt entirely in cache still has to produce
// logits, and Prefill of nothing is an error.
func (s *State) restoreKV(max int) (int, error) {
	n, tail, err := s.kv.restore()
	if err != nil {
		return 0, err
	}
	cover := n
	// Clamp in positions for a partial page and in whole pages otherwise. A
	// hybrid gives the tail up rather than clamping it: its recurrent summary
	// covers exactly the positions it was taken after, and resuming earlier
	// would apply a token twice.
	if n > max {
		if tail > 0 && s.rconv != nil && s.recurrentBoundary() > 0 {
			s.kv.dropTail(n)
			n, tail = n-tail, 0
		}
		if n > max {
			if tail > 0 {
				n = max
			} else {
				for li := s.kv.lo; li < len(s.kv.layers); li++ {
					if p := s.kv.layers[li].p; p > 0 {
						n = (max / p) * p
					}
				}
			}
		}
	}
	if n <= 0 {
		return 0, nil
	}
	// Both halves of a hybrid's history, or neither: without the summary for
	// this boundary, the prefix is given up and the prompt prefilled in full.
	if p := s.recurrentBoundary(); s.rconv != nil && p > 0 {
		// A tail's summary is named by its fill, exactly as its pages are; a
		// whole boundary's is named by the boundary it closes.
		idx, key := n/p-1, ""
		if al := s.attnLayer(); al >= 0 {
			if tail > 0 {
				idx = n / p
				key = s.kv.pageKeyFill(al, idx, n-idx*p)
			} else {
				key = s.kv.pageKey(al, idx)
			}
		}
		if idx < 0 || key == "" || !s.faultRecurrentAt(idx, key) {
			s.kv.unmigratable++
			return 0, nil
		}
	}

	// The windowed layers' pages, once the length is known: only what n's
	// window reads was ever stored, so that is all a restore can bring back.
	if ok, err := s.kv.restoreWin(n, cover); err != nil || !ok {
		return 0, err
	}

	// Truncate again at the clamped length, for the reason restore() truncates:
	// a page past the position we will resume from would never be re-sealed.
	for li := s.kv.lo; li < len(s.kv.layers); li++ {
		pg := &s.kv.layers[li]
		if pg.p == 0 {
			continue // a linear layer has no pages to truncate
		}
		// Keep the page holding position n-1 (for a partial n, the tail).
		keep := n / pg.p
		if n%pg.p != 0 {
			keep++
		}
		for j := keep; j < len(pg.k); j++ {
			pg.drop(j)
		}
		// Only whole pages have been offered to the store; the tail is written
		// through by sealTail and stays mutable.
		pg.lastSealed = n / pg.p
		pg.written = n
	}
	s.kv.entTruncate(n)
	return n, nil
}

// KVBytes is what this session's history holds resident right now. It grows with
// the conversation and falls when pages are evicted; it is not a reservation.
func (s *State) KVBytes() uint64 { return s.kv.Bytes() }

// KVPageStats is pages written to the store, fetched back, and evicted.
func (s *State) KVPageStats() (stored, fetched, evicted int64) {
	return s.kv.stored, s.kv.fetched, s.kv.evicted
}

// KVDeviceSyncFailures is how many placed blocks would not hand their history
// back, so nothing was cached for them. It is not a store failure: the store
// was never reached.
func (s *State) KVDeviceSyncFailures() int64 { return s.kv.syncFailed }

// KVPrefixUnmigratable is how many times a restored prefix had to be given up
// because it could not be moved onto a device (or its recurrent summary was
// missing): a reuse of 0 with a full store is otherwise unexplainable.
func (s *State) KVPrefixUnmigratable() int64 { return s.kv.unmigratable }

// KVStoreMismatches is how many pages came back the wrong shape and were dropped.
// It is separate from KVStoreFailures because the causes differ: a mismatch is a
// stale or foreign cache directory and costs a prefill, a failure is a store that
// cannot write and costs the budget.
func (s *State) KVStoreMismatches() int64 { return s.kv.mismatched }

// KVStoreFailures is how many times a write-through was refused by the store. A
// store that is failing every Set is otherwise indistinguishable from no store
// at all: the budget stops working and nothing else changes.
func (s *State) KVStoreFailures() int64 { return s.kv.failed }

// seal offers every page of every layer that has just become immutable to the
// store, once each.
func (kc *kvCache) seal(pos int) {
	if _, ok := kc.store.(NoStore); ok {
		return // nothing to write through to; do not walk the layers
	}
	for li := kc.lo; li < len(kc.layers); li++ {
		s := &kc.layers[li]
		if s.p == 0 {
			continue
		}
		cur, _ := s.page(pos)
		// lastSealed means "stored", not "walked past": stop at the first page
		// that cannot be stored, so a later pull-down from the card can still
		// seal it, and trim (which evicts below the mark) never drops a page
		// the store lacks.
		n := s.lastSealed
		if n > cur {
			n = cur // a rollback makes pages mutable again; see below
		}
		for ; n < cur; n++ {
			// A windowed layer's page that is not history: released behind
			// the window, or below what a device vouched for. Nothing is
			// stored for it, and a restore reads only the window (restoreWin).
			if s.win > 0 && (n < s.gone && !s.resident(n) || s.resident(n) && n*s.p < s.genuine) {
				continue
			}
			if n >= len(s.k) || s.k[n] == nil {
				break
			}
			// It must be full: a page is named by the whole boundary's tokens.
			if s.written < (n+1)*s.p {
				break
			}
			key := kc.pageKey(li, n)
			if key == "" {
				break // the tokens behind this page are not known; it cannot be named
			}
			if kc.share != nil {
				// Held once: the page becomes the store's copy, or the
				// store's copy replaces it (kvshare.go).
				kc.publishSealed(li, n, key)
				kc.stored += 2
				continue
			}
			if err := kc.store.Set(key, li*2, n, bytes.NewReader(kvBytesOf(s.k[n]))); err != nil {
				kc.failed++
				break
			}
			if err := kc.store.Set(key, li*2+1, n, bytes.NewReader(kvBytesOf(s.v[n]))); err != nil {
				kc.failed++
				break
			}
			kc.stored += 2
		}
		// The mark falls as well as rises: a rollback makes a page mutable
		// again, and trim must not treat it as evictable.
		s.lastSealed = n
	}
	kc.entSeal(pos)
}

// trim evicts sealed pages, oldest first, until the resident set is inside the
// budget. A page is only a candidate once the store has it.
func (kc *kvCache) trim(pos int) {
	if kc.budget == 0 || kc.Bytes() <= kc.budget {
		return
	}
	for li := kc.lo; li < len(kc.layers); li++ {
		s := &kc.layers[li]
		if s.p == 0 {
			continue
		}
		for n := 0; n < s.lastSealed && kc.Bytes() > kc.budget; n++ {
			if n >= len(s.k) || s.k[n] == nil {
				continue
			}
			s.unshare(n)
			s.k[n], s.v[n] = nil, nil
			s.out[n] = true
			kc.evicted += 2
		}
	}
}

// fault brings page n of layer li back from the store. It is called from the
// read path, which is the only place a missing page can be discovered -- the
// write path allocates, because growth is known rather than found.
func (kc *kvCache) fault(li, n int) error {
	s := &kc.layers[li]
	if n < len(s.k) && s.k[n] != nil {
		return nil
	}
	s.extend(n)
	// The slot is addressable, not resident (extend, not grow), so every
	// failure path below leaves it nil: a zeroed page present would read as
	// history.
	s.k[n], s.v[n] = nil, nil
	key := kc.pageKey(li, n)
	if key == "" {
		return ErrNoPage // nothing names this page, so nothing can hold it
	}
	if kc.share != nil {
		return kc.faultShared(li, n, key)
	}
	// The length is checked here, where the geometry is: a short page would
	// leave zeros read back as keys the model never wrote.
	k := make([]float32, s.pp)
	v := make([]float32, s.pp)
	kw := newSliceWriter(kvBytesOf(k))
	if err := kc.store.Get(key, li*2, n, kw); err != nil {
		return kc.miss(key, err)
	}
	if kw.n != len(kw.b) {
		return kc.miss(key, shortPage(li, n, kw.n, len(kw.b)))
	}
	vw := newSliceWriter(kvBytesOf(v))
	if err := kc.store.Get(key, li*2+1, n, vw); err != nil {
		return kc.miss(key, err)
	}
	if vw.n != len(vw.b) {
		return kc.miss(key, shortPage(li, n, vw.n, len(vw.b)))
	}
	s.k[n], s.v[n] = k, v
	s.out[n] = false
	kc.fetched += 2
	return nil
}

// faultFill is fault for a page stored under its fill rather than as a whole
// one. A short read is still a miss.
func (kc *kvCache) faultFill(li, n, fill int) error {
	if kc.share != nil {
		return ErrNoPage // a sharing session stores no partial page (sealTail)
	}
	s := &kc.layers[li]
	key := kc.pageKeyFill(li, n, fill)
	if key == "" {
		return ErrNoPage
	}
	s.extend(n)
	s.k[n], s.v[n] = nil, nil
	// The buffer is a whole page and the read is its filled prefix; restore's
	// coverage stops before the zero tail.
	row := s.pp / s.p
	want := fill * row
	if fill >= s.p || want <= 0 || want > s.pp {
		return ErrNoPage
	}
	k := make([]float32, s.pp)
	v := make([]float32, s.pp)
	kw := newSliceWriter(kvBytesOf(k[:want]))
	if err := kc.store.Get(key, li*2, n, kw); err != nil {
		return kc.miss(key, err)
	}
	if kw.n != len(kw.b) {
		return kc.miss(key, shortPage(li, n, kw.n, len(kw.b)))
	}
	vw := newSliceWriter(kvBytesOf(v[:want]))
	if err := kc.store.Get(key, li*2+1, n, vw); err != nil {
		return kc.miss(key, err)
	}
	if vw.n != len(vw.b) {
		return kc.miss(key, shortPage(li, n, vw.n, len(vw.b)))
	}
	s.k[n], s.v[n] = k, v
	s.out[n] = false
	kc.fetched += 2
	return nil
}

// kvChunk is the cache's addressing granularity, in positions, and it is
// deliberately not the page size: the page size is set by the attention
// kernel, while what a prefix cache can name is a different question. 16 is
// vLLM's block size, not a tuned value; a shared prefix shorter than a chunk
// misses, as it does in vLLM, SGLang and LMCache.
const kvChunk = 16

// kvChunkFills is the fills a partial page is sealed at: every chunk boundary
// it has passed, plus the exact end.
//
// The exact end covers the identical prompt, the only restore point a hybrid
// has (see sealRecurrentTail).
func kvChunkFills(fill int) []int {
	var out []int
	for f := kvChunk; f < fill; f += kvChunk {
		out = append(out, f)
	}
	if fill > 0 {
		out = append(out, fill)
	}
	return out
}

// kvChunkProbes is the same set read backwards: the fills a restore asks for,
// largest first, so an identical prompt hits on the first question and a shared
// prefix on the next few: one Get per chunk, stopping at the first hit.
func kvChunkProbes(max int) []int {
	if max <= 0 {
		return nil
	}
	out := []int{max}
	for f := max / kvChunk * kvChunk; f >= kvChunk; f -= kvChunk {
		if f != max {
			out = append(out, f)
		}
	}
	return out
}

// sealTail offers the page the position is currently inside, named by how many
// positions it holds, which makes a prompt shorter than a page cacheable.
//
// Only the filled prefix is stored. That is a byte prefix only at nseq == 1,
// which SetCacheKey guarantees by refusing a namespace for a batch.
//
// It does not advance lastSealed: the page is still mutable, and marking it
// sealed would make trim treat it as evictable and stop seal offering the full
// page later.
func (kc *kvCache) sealTail(pos int) int {
	if _, ok := kc.store.(NoStore); ok {
		return 0
	}
	if kc.ns == "" || pos <= 0 {
		return 0
	}
	// A partial page is still being written, so it cannot be held once: a
	// sharing session's divergent page is its own, and the next session
	// computes its copy (kvshare.go).
	if kc.share != nil {
		return 0
	}
	stored := 0
	for li := kc.lo; li < len(kc.layers); li++ {
		s := &kc.layers[li]
		if s.p == 0 {
			continue
		}
		cur, _ := s.page(pos)
		have := s.written - cur*s.p
		if have <= 0 || have >= s.p {
			continue // nothing partial here, or seal() already owns it
		}
		if cur >= len(s.k) || s.k[cur] == nil || s.v[cur] == nil {
			continue
		}
		if cur*s.p < s.genuine {
			continue // part of it is zeros a device sent for a released page
		}
		row := s.pp / s.p // floats per position, which is nseq*kvDim at nseq 1
		var keys []string
		var lens []int
		for _, fill := range kvChunkFills(have) {
			key := kc.pageKeyFill(li, cur, fill)
			if key == "" {
				continue
			}
			n := fill * row
			if n > len(s.k[cur]) || n > len(s.v[cur]) {
				continue
			}
			keys, lens = append(keys, key), append(lens, n)
		}
		if len(keys) == 0 {
			continue
		}
		// The fills are byte prefixes of one another. A store that can hold
		// them over one copy is offered them so (FillStore); any other takes
		// each as a page of its own.
		if fs, ok := kc.store.(FillStore); ok {
			top := lens[len(lens)-1]
			bk, bv := kvBytesOf(s.k[cur][:top]), kvBytesOf(s.v[cur][:top])
			for i := range lens {
				lens[i] *= 4 // float32 elements to bytes
			}
			if err := fs.SetFills(li*2, cur, bk, keys, lens); err != nil {
				kc.failed++
				continue
			}
			if err := fs.SetFills(li*2+1, cur, bv, keys, lens); err != nil {
				kc.failed++
				continue
			}
			stored += 2 * len(keys)
			continue
		}
		for i, key := range keys {
			n := lens[i]
			if err := kc.store.Set(key, li*2, cur, bytes.NewReader(kvBytesOf(s.k[cur][:n]))); err != nil {
				kc.failed++
				continue
			}
			if err := kc.store.Set(key, li*2+1, cur, bytes.NewReader(kvBytesOf(s.v[cur][:n]))); err != nil {
				kc.failed++
				continue
			}
			stored += 2
		}
	}
	return stored + kc.entSealTail(pos)
}

// miss turns "these bytes are not a page of this shape" into a miss, and drops
// the offending key on the way past, so a page left by a differently
// configured process costs one prefill rather than failing that prefix
// forever. A genuine I/O failure still stops the request.
func (kc *kvCache) miss(key string, err error) error {
	if !errors.Is(err, errPageMismatch) {
		return err
	}
	kc.mismatched++
	kc.store.Drop(key) // best effort: a store that cannot forget still misses
	return ErrNoPage
}

// sliceWriter is the io.Writer a resident page presents to a store: it fills a
// buffer the engine already owns and refuses anything that does not fit.
//
// A page whose size does not match the geometry is from a different
// configuration, and accepting it would attend over a reinterpreted history.
type sliceWriter struct {
	b []byte
	n int
}

func newSliceWriter(b []byte) *sliceWriter { return &sliceWriter{b: b} }

func (w *sliceWriter) Write(p []byte) (int, error) {
	if w.n+len(p) > len(w.b) {
		return 0, fmt.Errorf("%w: kv page is %d bytes, the store offered at least %d",
			errPageMismatch, len(w.b), w.n+len(p))
	}
	copy(w.b[w.n:], p)
	w.n += len(p)
	return len(p), nil
}

// kvFault brings a page back if it was evicted. On a resident-only session -- no
// store, no budget -- it is one nil compare.
func (s *State) kvFault(li, page int) bool {
	pg := &s.kv.layers[li]
	if page < len(pg.k) && pg.k[page] != nil {
		return true
	}
	if err := s.kv.fault(li, page); err != nil {
		// A page the store cannot return is a hole in the sequence's past, so
		// the request fails. The error is recorded because this may run in a
		// pool closure with no error path; kvCheck surfaces it at the entry
		// point. Returning false stops the walk over the nil page.
		s.kvErr.CompareAndSwap(nil, &err)
		return false
	}
	return true
}

// kvWritable brings back the page pos falls in before it is written, when trim
// evicted it with history still in it. On a session with no budget it is one
// bounds check.
func (s *State) kvWritable(li, pos int) {
	pg := &s.kv.layers[li]
	if pg.p == 0 {
		return
	}
	if n, _ := pg.page(pos); n < len(pg.out) && pg.out[n] {
		s.kvFault(li, n)
	}
}

// The prefix solver: page n of layer li is named by H(tokens[0 : (n+1)*p])
// inside the namespace, so the store answers "which prefix is cached". A hit
// proves every token up to that boundary matches, the longest prefix is found
// by walking boundaries until a miss, and sessions sharing a prefix share its
// pages. The hash is cumulative (LMCache's chunk_hash) because RoPE is applied
// before the cache write: a span is reusable only as a prefix.

// geomDigest is everything a page's byte count cannot distinguish. Different
// head factorizations (tinyllama, Qwen2-1.5B and gemma-2b all have kvDim 256)
// and head-major versus row-major layouts produce same-sized pages, so the
// geometry is hashed into every key and a mismatch is a clean miss. Model
// identity (two fine-tunes of one architecture) remains the namespace's job.
func geomDigest(c *Config, nseq int, l kvLayout) []byte {
	h := sha256.New()
	fmt.Fprintf(h, "kvgeom1|nseq=%d|elem=%s|hm=%t|nkv=%d|hd=%d|L=%d|embd=%d|head=%d|rot=%d|ffn=%d|vocab=%d|rope=%g|swa=%d/%d|arch=%s|exp=%d/%d",
		nseq, l.elemTag(), l.headMajor, c.NKVHead, c.HeadDim, c.NLayer,
		c.NEmbd, c.NHead, c.NRot, c.NFFN, c.NVocab, c.RopeBase,
		c.SWAWindow, c.SWAPeriod, c.Arch, c.NExpert, c.NExpertUsed)
	// The sliding layers' own geometry, only where it is their own: every
	// other model's digest, and so every page it stored, is unchanged.
	if c.GeomSplit() {
		fmt.Fprintf(h, "|swageom=%d/%d/%d", c.NKVHeadSWA, c.HeadDimSWA, c.NRotSWA)
	}
	return h.Sum(nil)
}

// pageKeyFill names page n of layer li when it holds `fill` positions rather
// than a whole page.
//
// The fill is in the digest, not only in the prefix, so a partial page cannot
// alias a full one. fill == p returns the ordinary key, so a full page is
// stored under one name only.
func (kc *kvCache) pageKeyFill(li, n, fill int) string {
	return kc.pageKeyFillP(kc.layers[li].p, n, fill)
}

// pageKeyFillP is pageKeyFill for a page of p positions, whoever's it is: a
// DeepSeek V4 block's entry page covers p = its entries times its rate
// (ds4kv.go).
func (kc *kvCache) pageKeyFillP(p, n, fill int) string {
	if fill >= p {
		return kc.pageKeyP(p, n)
	}
	if kc.ns == "" || fill <= 0 {
		return ""
	}
	end := n*p + fill
	if end > len(kc.seq) {
		return ""
	}
	x := sha256.New()
	x.Write(kc.geom)
	fmt.Fprintf(x, "|p=%d|fill=%d|", p, fill)
	kc.writeSeq(x, end)
	return kc.ns + "/" + hex.EncodeToString(x.Sum(nil)[:16])
}

func (kc *kvCache) pageKey(li, n int) string { return kc.pageKeyP(kc.layers[li].p, n) }

// pageKeyP is pageKey for a page of p positions; see pageKeyFillP.
func (kc *kvCache) pageKeyP(p, n int) string {
	if kc.ns == "" {
		return kc.id // no namespace: one key for the session, no sharing
	}
	end := (n + 1) * p
	if end > len(kc.seq) {
		return "" // the tokens for this page are not all known yet
	}
	h := kc.bh[p]
	for len(h) <= n {
		// One sha256 per boundary, computed once and kept.
		e := (len(h) + 1) * p
		if e > len(kc.seq) {
			break
		}
		x := sha256.New()
		x.Write(kc.geom) // the shape first, so a geometry change can never alias
		fmt.Fprintf(x, "|p=%d|", p)
		kc.writeSeq(x, e)
		h = append(h, hex.EncodeToString(x.Sum(nil)[:16]))
	}
	kc.bh[p] = h
	if n >= len(h) {
		return ""
	}
	return kc.ns + "/" + h[n]
}

// int32bytes views the ids as bytes for hashing. The byte order is this machine's
// and that is correct: a page written on one box and read on another would differ in
// far more than the hash.
func int32bytes(v []int32) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
}

// note records the ids this sequence has consumed, which is what pageKey hashes.
// It is a no-op without a namespace, so nothing is tracked for a session that
// does not want content addressing.
func (kc *kvCache) note(pos int, ids ...int32) {
	if kc.ns == "" {
		return
	}
	// seq[i] is the id at position i: a token's, or a named picture row's
	// (nameRow; PrefillMixed notes every row of a mixed prompt). Only rows
	// with no name advance the position with no id -- ForwardEmbd's, and an
	// embedding span with no Key -- leaving a gap; appending across it would
	// shift every later id. Stop instead: pageKey refuses boundaries past
	// len(seq), so nothing after the gap is cached.
	if pos > len(kc.seq) {
		return
	}
	if pos < len(kc.seq) {
		kc.seq = kc.seq[:pos] // a rollback rewrites the tail
		for p := range kc.bh {
			if n := pos / p; n < len(kc.bh[p]) {
				kc.bh[p] = kc.bh[p][:n]
			}
		}
	}
	kc.seq = append(kc.seq, ids...)
}

// reset drops every page and forgets what was sealed, so a State that is being
// reused starts with nothing resident and nothing claimed.
//
// The token stream goes with the pages, because it names them: paths that
// record no ids (ForwardEmbd, PrefillMixed) would otherwise store their pages
// under the previous prompt's hash.
func (kc *kvCache) reset() {
	for li := kc.lo; li < len(kc.layers); li++ {
		s := &kc.layers[li]
		for n := range s.k {
			s.keepOnReset(n)
		}
		s.k, s.v, s.out, s.lastSealed, s.written = s.k[:0], s.v[:0], s.out[:0], 0, 0
	}
	kc.entTruncate(0)
	kc.seq, kc.named = kc.seq[:0], kc.named[:0]
	if kc.bh == nil {
		kc.bh = map[int][]string{}
	}
	clear(kc.bh)
}

// keepOnReset lets page n go as reset does, its memory kept in spare for the
// next sequence to grow into: a State reused prompt after prompt (a Decider's)
// would otherwise hand the collector its whole history each time and make it
// again. A layer with no window keeps what it held; a windowed layer what its
// window turns over (spareMax), so it still holds its window and not the
// context. A shared page is the store's: its reference goes.
func (s *kvPages) keepOnReset(n int) {
	if s.isShared(n) {
		s.drop(n)
		return
	}
	for _, b := range [2][]float32{s.k[n], s.v[n]} {
		if b != nil && len(b) == s.pp && (s.win <= 0 || len(s.spare) < s.spareMax()) {
			s.spare = append(s.spare, b)
		}
		if s.latent {
			break // the value is the key's memory
		}
	}
	s.drop(n)
}

// kvEnsureWindow faults every page of layer li covering [w0, w0+want) into
// residence, on one goroutine, before the attention heads fan out. Faulting
// from inside the pool is a data race (extend can reallocate the slice another
// worker holds), so residency is decided here and workers only read.
func (s *State) kvEnsureWindow(li, w0, want int) {
	pg := &s.kv.layers[li]
	if pg.p == 0 {
		return
	}
	// A windowed layer's released pages are behind every live row's window
	// (kvPages.release): a caller asking from zero for a superset of the rows'
	// windows asks for nothing there.
	if lo := pg.gone * pg.p; w0 < lo {
		want -= lo - w0
		w0 = lo
	}
	for pos, end := w0, w0+want; pos < end; {
		sp := pg.span(pos, end)
		s.kvFault(li, sp.page)
		pos += sp.n
	}
}

// kvCheck reports the first unrecoverable page fault of this step, and clears
// it. Every entry point calls it before returning logits, so it is also where a
// step's placement moves (growth, relocation, a demotion, the KV a device grew)
// reach the host's budgets (followHost).
func (s *State) kvCheck() error {
	s.followHost()
	if p := s.kvErr.Swap(nil); p != nil {
		return *p
	}
	return nil
}

// The recurrent half of a hybrid's history.
//
// A hybrid's past is in two places and only the attention half is paged. The
// recurrent summary is a function of exactly the tokens the page key names,
// so it is stored under the same name; it is constant in the context length.
//
// It rides the same KVStore: the layer index is offset past the k and v range
// (2*NLayer + li), so no interface changes and a store sees one more page kind.
func (s *State) recurrentLayer(li int) int { return 2*s.c.NLayer + li }

// logitsLayer is the store index the prompt's final logits go under. The
// attention planes take 0..2*NLayer-1 and the recurrent summaries the NLayer
// after them, so this sits past both and cannot collide with either.
func (s *State) logitsLayer() int { return 3 * s.c.NLayer }

// sealLogits stores the logits the prompt ended on, under the same fill-named
// key as its partial page.
//
// It makes a fully cached prompt possible: with the logits stored there is
// nothing left to run, which a hybrid needs because its summary cannot be
// shortened by one position. The cost is one vocabulary row per prompt.
func (s *State) sealLogits(pos int, lg []float32) {
	if s.kv.ns == "" || pos <= 0 || len(lg) == 0 {
		return
	}
	if _, ok := s.kv.store.(NoStore); ok {
		return
	}
	al := s.attnLayer()
	if al < 0 {
		return
	}
	p := s.kv.layers[al].p
	cur := pos / p
	key := s.kv.pageKeyFill(al, cur, pos-cur*p)
	if key == "" {
		return
	}
	if err := s.kv.store.Set(key, s.logitsLayer(), cur,
		bytes.NewReader(kvBytesOf(lg))); err != nil {
		s.kv.failed++
	}
}

// faultLogits reads back what sealLogits wrote, or reports that it cannot.
func (s *State) faultLogits(pos int, out []float32) bool {
	al := s.attnLayer()
	if al < 0 || s.kv.ns == "" || pos <= 0 || len(out) == 0 {
		return false
	}
	if _, ok := s.kv.store.(NoStore); ok {
		return false
	}
	p := s.kv.layers[al].p
	cur := pos / p
	key := s.kv.pageKeyFill(al, cur, pos-cur*p)
	if key == "" {
		return false
	}
	w := newSliceWriter(kvBytesOf(out))
	if err := s.kv.store.Get(key, s.logitsLayer(), cur, w); err != nil {
		return false
	}
	// Exact length, as in fault(): a short read would leave zeros.
	return w.n == len(w.b)
}

// recurrentBoundary is the page size the snapshots are aligned to: the
// attention layers' page, since the linear ones have none and the two halves
// must be restorable at the same position.
func (s *State) recurrentBoundary() int {
	for li := s.kv.lo; li < len(s.kv.layers); li++ {
		if p := s.kv.layers[li].p; p > 0 {
			return p
		}
	}
	return 0
}

// sealRecurrent stores every linear layer's summary when the position lands
// exactly on a page boundary.
//
// Exactly, because a summary cannot be sliced: the state for boundary n exists
// only when the position is n*p, which is why prefill snaps its chunk width to
// the boundary when a hybrid is being cached.
func (s *State) sealRecurrent(pos int) {
	if s.rconv == nil || s.kv.ns == "" || pos == 0 {
		return
	}
	if _, ok := s.kv.store.(NoStore); ok {
		return
	}
	p := s.recurrentBoundary()
	if p == 0 || pos%p != 0 {
		return
	}
	n := pos/p - 1 // the boundary this state closes
	for li := range s.rconv {
		if s.rconv[li] == nil {
			continue
		}
		key := s.kv.pageKey(s.attnLayer(), n)
		if key == "" {
			return
		}
		if err := s.kv.store.Set(key, s.recurrentLayer(li), n,
			bytes.NewReader(kvBytesOf(append(append([]float32{}, s.rconv[li]...), s.rstate[li]...)))); err != nil {
			s.kv.failed++
			return
		}
		s.kv.stored++
	}
}

// sealRecurrentTail snapshots the recurrent summary at a prompt boundary rather
// than at a page boundary, under the same fill-named key sealTail uses: a
// restored partial page needs a matching summary or no tail at all.
func (s *State) sealRecurrentTail(pos int) {
	if s.rconv == nil || s.kv.ns == "" || pos == 0 {
		return
	}
	if _, ok := s.kv.store.(NoStore); ok {
		return
	}
	p := s.recurrentBoundary()
	if p == 0 || pos%p == 0 {
		return // a whole boundary is sealRecurrent's, and it is the better copy
	}
	al := s.attnLayer()
	if al < 0 {
		return
	}
	cur := pos / p
	// One point, at the prompt's end: a recurrence cannot be rewound, so the
	// engine holds the state for this position only. Capturing points during
	// prefill is unbuilt; at ~79 MB a point on Qwen3-Next-80B it is costly.
	key := s.kv.pageKeyFill(al, cur, pos-cur*p)
	if key == "" {
		return
	}
	for li := range s.rconv {
		if s.rconv[li] == nil {
			continue
		}
		if err := s.kv.store.Set(key, s.recurrentLayer(li), cur,
			bytes.NewReader(kvBytesOf(append(append([]float32{}, s.rconv[li]...), s.rstate[li]...)))); err != nil {
			s.kv.failed++
			return
		}
		s.kv.stored++
	}
}

// attnLayer is any layer with pages, so that pageKey has a page size to hash.
func (s *State) attnLayer() int {
	for li := s.kv.lo; li < len(s.kv.layers); li++ {
		if s.kv.layers[li].p > 0 {
			return li
		}
	}
	return 0
}

// faultRecurrentAt brings every linear layer's summary back for position n,
// under key -- a boundary's page key, or a partial page's, which is named by
// its fill rather than by a boundary (see sealRecurrentTail). All of them or
// none: a hybrid with half its summaries restored is worse than one with none.
func (s *State) faultRecurrentAt(n int, key string) bool {
	if s.rconv == nil {
		return true
	}
	if key == "" || n < 0 {
		return false
	}
	type snap struct {
		li  int
		buf []float32
	}
	var got []snap
	for li := range s.rconv {
		if s.rconv[li] == nil {
			continue
		}
		buf := make([]float32, len(s.rconv[li])+len(s.rstate[li]))
		w := newSliceWriter(kvBytesOf(buf))
		if err := s.kv.store.Get(key, s.recurrentLayer(li), n, w); err != nil {
			return false
		}
		if w.n != len(w.b) {
			s.kv.mismatched++
			return false
		}
		got = append(got, snap{li, buf})
	}
	// Applied only once every layer has answered, so a failure half way leaves
	// the session's own summaries untouched.
	for _, g := range got {
		copy(s.rconv[g.li], g.buf[:len(s.rconv[g.li])])
		copy(s.rstate[g.li], g.buf[len(s.rconv[g.li]):])
	}
	return true
}

// namedRow is one image row as the page keys see it: the picture's name, the
// row's index in the run that name covers, and its rotary position where the
// model's rotary is multi-axis (zero where a row's position is its index,
// which the stream already says).
type namedRow struct {
	key ImageKey
	row int32
	at  [4]int32
}

// namePosFault is a gate's violation: an image row's rotary position left
// out of its name. False in every real run.
var namePosFault bool

// nameRow records image row row of the picture named k, at rotary position
// at, as the row at cache position pos, and returns the id seq holds for it:
// negative, so no vocabulary id is one. A row named again at the same position
// reuses its entry; the table never grows past the rows the sequence holds.
func (kc *kvCache) nameRow(k ImageKey, row int, at [4]int, pos int) int32 {
	r := namedRow{key: k, row: int32(row)}
	if !namePosFault {
		r.at = [4]int32{int32(at[0]), int32(at[1]), int32(at[2]), int32(at[3])}
	}
	if pos < len(kc.seq) && kc.seq[pos] < 0 {
		if i := int(-1 - kc.seq[pos]); i < len(kc.named) && kc.named[i] == r {
			return kc.seq[pos]
		}
	}
	kc.named = append(kc.named, r)
	return -int32(len(kc.named))
}

// writeSeq hashes the ids of positions [0, e) into h: a run of token ids as
// they are, and an image row as a tag no id run can produce followed by its
// whole name -- the picture's digest, never a truncation of it -- its row and
// its position.
func (kc *kvCache) writeSeq(h io.Writer, e int) {
	seq := kc.seq[:e]
	for len(seq) > 0 {
		i := 0
		for i < len(seq) && seq[i] >= 0 {
			i++
		}
		if i > 0 {
			h.Write(int32bytes(seq[:i]))
			seq = seq[i:]
			continue
		}
		r := kc.named[-1-seq[0]]
		var b [4 + 32 + 4 + 16]byte
		// "IMG\xff" read as a little-endian int32 is negative, which no
		// token id is, so no run of ids hashes to the tag.
		copy(b[:4], "IMG\xff")
		copy(b[4:36], r.key[:])
		binary.LittleEndian.PutUint32(b[36:], uint32(r.row))
		for j, v := range r.at {
			binary.LittleEndian.PutUint32(b[40+4*j:], uint32(v))
		}
		h.Write(b[:])
		seq = seq[1:]
	}
}
