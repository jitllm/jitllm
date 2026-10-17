package jlm

import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

// ReadThreads is how many concurrent ReadAt calls one page-in issues.
//
// A page fault is a queue of one. An NVMe is not saturated by one reader
// however large the request; it needs several requests outstanding, which one
// mmap fault cannot produce. Measured on an NVMe with 586 MiB pages: one
// ReadAt reaches under half the rate of sixteen concurrent ones, and an mmap
// fault falls between them. That ratio is why this package reads rather than
// maps.
const ReadThreads = 16

// File is an open container.
//
// It is read, not mapped: besides the rate above, mapping left residency to
// the kernel and to madvise's alignment rules (MADV_DONTNEED refuses an
// unaligned start and fails the whole call). Explicit reads make residency a
// decision this package makes and can be asked about.
type File struct {
	Path string
	H    *Header
	FP   Fingerprint
	// ver is the version the container was opened at: Version, or an older
	// one a migration reads (migrate.go).
	ver uint32

	// mu is the one lock over residency: two States decoding on a container
	// that evicts would otherwise race in claim(). It guards pages, filled,
	// free, resident, recent, pins, pageTotal and the counters, and not the file
	// reads: EnsureRanges claims and pins under it, drops it, reads, and marks at
	// the end. On a model that fits it is one uncontended pair per block per
	// token.
	mu sync.Mutex
	// noHuge and nodeMask are this File's Placement; see SetPlacement.
	noHuge   bool
	nodeMask []uint64
	// nodes is a one-node mask for each of Placement.Nodes when they spread
	// the memory: expert page x is bound whole to nodes[x%len(nodes)]
	// (claim), the rest interleaved.
	nodes [][]uint64
	// pins[b] is how many leases hold block b resident. A pinned page is never
	// a victim; see Hold.
	pins []int
	// chunk is this File's fill granularity: the header's, or SetChunk's. It
	// is per-File because the residency bitmap is sized from it.
	chunk uint64
	// reads counts readAt calls, the quantity the pager is priced in. Atomic
	// because EnsureRanges' run goroutines call it unlocked.
	reads int64
	// inIO, peakIO and readBytes are ReadConcurrency's and ReadBytes'
	// counters: requests outstanding now, the most ever, and bytes asked for.
	inIO, peakIO, readBytes atomic.Int64
	// inPages and peakPages are PreloadDepth's: pages Preload has in flight.
	inPages, peakPages atomic.Int64
	// preloadRead stands in for EnsurePage in Preload, for a gate that holds
	// a page mid-read; nil is EnsurePage.
	preloadRead func(i int) error
	// preN counts running Preloads, for SetBudget to wait out
	// (settlePreload), under preMu; preCond is signalled as each ends.
	preMu   sync.Mutex
	preCond sync.Cond
	preN    int
	// waitNs is wall time a caller spent blocked in EnsureRanges waiting for
	// its runs to land: serial time on the decode path, which is what a
	// prefetch would have to hide. It is the only exposed-I/O number on the
	// host path; a rate over a wall that includes compute is only a floor.
	waitNs int64
	// filled[b][c] says chunk c of block b's page holds real bytes. A resident
	// page is not necessarily a FILLED page: see EnsureRanges.
	filled [][]bool
	// busy[b][c] says chunk c of block b is being read by a caller that
	// planned it (planRunsBusy); landed wakes the callers that joined it.
	busy   [][]bool
	landed sync.Cond
	// specOn, specHeld and specCap are the speculative occupancy (readsvc.go),
	// guarded by mu; spec is the queue and its workers.
	specOn   []bool
	specHeld int
	specCap  int
	spec     specSvc
	// readHook stands in for the file read, for a gate that holds or counts
	// reads; nil reads the file.
	readHook func(dst []byte, off int64) error
	chunkIn  int64
	// spanBuf is planRuns' scratch for the chunk spans it sorts. Guarded by
	// f.mu, which planRuns holds for its whole body, so one array serves every
	// caller and a mixture's 24-range plan allocates nothing per token.
	spanBuf []run

	f *os.File
	// direct is a second handle opened O_DIRECT, used for page reads only (see
	// openDirect). nil when the filesystem refuses O_DIRECT, in which case
	// everything falls back to the buffered handle.
	direct *os.File
	meta   []byte // header through fingerprint; decodeKV slices Raw out of it
	// dense holds every non-block weight and is always resident: the embedding,
	// the output projection and the norms, small beside the pages.
	dense []byte
	pages [][]byte // one per block, nil when that page is not resident
	// offHeap and mapped are where frames and the dense region live (see
	// SetOffHeap): mapped is every live anonymous mapping, by its first byte,
	// each unmapped when the File lets go of it (release) and the rest at
	// Close.
	offHeap bool
	mapped  map[*byte][]byte
	// recycle is told the address range of every buffer whose bytes stop being
	// the ones a caller may have copied (OnRecycle), and recycled holds those
	// buffers until f.mu is dropped: the callback is not run under the lock.
	recycle  func(p unsafe.Pointer, n uintptr)
	recycled [][]byte
	// recycledSpare is the array flushRecycled handed back, kept for the next
	// retire so a page-in does not grow a fresh one. Guarded by mu.
	recycledSpare [][]byte
	// freeFilled holds the validity maps of evicted pages for claim to reuse:
	// a paged decode claims a frame per block per token. Guarded by mu.
	freeFilled [][]bool
	// io is the page reader's worker pool (pageio.go), started on the first
	// page-in and stopped at Close.
	io pageIO

	// Frames are the page budget, and they are reused buffers. "The model fits"
	// means the budget holds every page, so nothing is ever evicted; below that
	// a page-in takes a free frame or evicts one. A frame is recycled rather than
	// reallocated so the resident set is this budget's, not the allocator's, and
	// the GC is not handed a page of garbage per fault.
	//
	// The budget is bytes, not a frame count, because the text, vision and
	// expert page arrays have different page sizes (see Role.Vision).
	budget uint64 // 0 = unlimited
	// pageTotal is every page's size added up, computed once: page sizes are
	// fixed at conversion, so CanEvict does not re-walk them per token.
	pageTotal uint64
	// canEvict is "the budget is smaller than the pages", published so a caller
	// can ask once per token without taking f.mu. Written under it.
	canEvict atomic.Bool
	resident uint64 // page bytes currently held
	// free lists buffers by size so a page-in reuses rather than reallocates:
	// a fresh frame costs its page faults, a reused one none. freeBytes is
	// their total, and resident+freeBytes never exceeds the budget (take,
	// SetBudget); TrimFree hands every one back once a burst of page-ins ends.
	free      map[uint64][][]byte
	freeBytes uint64
	// extent[i] is how much of page i its tensors use, rounded up to Align:
	// what a frame for it holds and what the budget is charged. A page on
	// disk is the largest block's size, so a smaller block's tail is padding
	// nothing reads. Nil means every page is its array's full size.
	extent []uint64
	// fresh counts frames take allocated rather than reused: a page-in that
	// found no free frame of its size and paid for a new one's faults.
	fresh  int64
	recent int // the block used most recently: the MRU victim. -1 for none
	// used[i] is when page i was last claimed, on a counter that ticks per
	// claim: the LRU order over expert pages. Guarded by mu.
	used    []uint64
	tick    uint64
	faults  int64
	evicted int64

	tab  []Entry
	byID map[tensorID]int
	cfg  *Config
	vis  *Vision
	voc  *Vocab
}

// Open reads and validates a container's metadata. No page is read.
//
// Every offset in the file is checked before it slices anything (RULE 9).
func Open(path string) (*File, error) { return openAt(path, Version) }

// openAt is Open for a container of version want: the current one, or an
// older one a migration reads (migrate.go).
func openAt(path string, want uint32) (*File, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	st, err := fh.Stat()
	if err != nil {
		fh.Close()
		return nil, err
	}
	// The page handle. nil is supported (tmpfs and some network filesystems
	// refuse O_DIRECT) and costs only the second copy.
	direct := openDirect(path)
	size := uint64(st.Size())
	if size < HeaderBytes {
		fh.Close()
		closeDirect(direct)
		return nil, fmt.Errorf("jlm: %s is %d bytes, shorter than a header", path, size)
	}
	f := &File{Path: path, f: fh, direct: direct, offHeap: offHeap.Load()}

	hb := make([]byte, HeaderBytes)
	if _, err := fh.ReadAt(hb, 0); err != nil {
		fh.Close()
		closeDirect(direct)
		return nil, fmt.Errorf("jlm: reading the header of %s: %w", path, err)
	}
	h, err := decodeHeader(hb, size, want)
	if err != nil {
		fh.Close()
		closeDirect(direct)
		return nil, err
	}
	f.H = h
	f.ver = want

	// One read for everything between the header and the dense region: the KV
	// block, the tensor table and the fingerprint are contiguous by construction
	// (Convert lays them out that way) and together are a few hundred KiB.
	metaEnd := h.DenseOff
	if metaEnd < HeaderBytes || metaEnd > size {
		fh.Close()
		closeDirect(direct)
		return nil, fmt.Errorf("jlm: metadata ends at %d in a %d-byte file", metaEnd, size)
	}
	f.meta = make([]byte, metaEnd)
	copy(f.meta, hb)
	if metaEnd > HeaderBytes {
		if _, err := fh.ReadAt(f.meta[HeaderBytes:], HeaderBytes); err != nil {
			fh.Close()
			closeDirect(direct)
			return nil, fmt.Errorf("jlm: reading the metadata of %s: %w", path, err)
		}
	}
	if f.cfg, err = decodeConfigAt(f.meta[h.CfgOff:h.CfgOff+h.CfgLen], want); err != nil {
		fh.Close()
		closeDirect(direct)
		return nil, err
	}
	if f.vis, err = decodeVision(f.meta[h.VisOff : h.VisOff+h.VisLen]); err != nil {
		fh.Close()
		closeDirect(direct)
		return nil, err
	}
	if f.voc, err = decodeVocab(f.meta[h.VocOff : h.VocOff+h.VocLen]); err != nil {
		fh.Close()
		closeDirect(direct)
		return nil, err
	}
	if f.tab, err = decodeTable(f.meta[h.TabOff:h.TabOff+h.TabLen], h.NTensors, size); err != nil {
		fh.Close()
		closeDirect(direct)
		return nil, err
	}
	if h.FPLen > 0 {
		if f.FP, err = decodeFP(f.meta[h.FPOff : h.FPOff+h.FPLen]); err != nil {
			fh.Close()
			closeDirect(direct)
			return nil, err
		}
	}
	// The index is keyed by identity and a duplicate is a refusal: an
	// unreachable weight is a model that loads and is wrong.
	f.byID = make(map[tensorID]int, len(f.tab))
	for i := range f.tab {
		e := &f.tab[i]
		id := tensorID{e.Role, e.Block, e.Index}
		if j, dup := f.byID[id]; dup {
			fh.Close()
			closeDirect(direct)
			return nil, fmt.Errorf("jlm: %v block %d index %d appears twice (%q and %q)",
				e.Role, e.Block, e.Index, f.tab[j].Name, e.Name)
		}
		f.byID[id] = i
	}
	if f.cfg != nil {
		if err := f.cfg.CheckQKNorm(int(h.NBlocks), func(r Role, b int32) bool {
			return f.Has(r, b, -1)
		}); err != nil {
			fh.Close()
			closeDirect(direct)
			return nil, fmt.Errorf("%w (%s)", err, path)
		}
	}
	// Every span is bounds-checked once, here, so a page-in on the hot path
	// is a read and a slice, not a validation.
	for i := range f.tab {
		e := &f.tab[i]
		qs, d, sc, err := spans(e.Type, e.Dims, e.NDim)
		if err != nil {
			fh.Close()
			closeDirect(direct)
			return nil, fmt.Errorf("jlm: %s: %w", e.Name, err)
		}
		// A split bank is checked sheet by sheet: its planes are NExpert spans
		// ExpPageSize apart, each inside the expert page it starts in.
		if expertPaged(e) {
			if err := h.checkSheets(e); err != nil {
				fh.Close()
				closeDirect(direct)
				return nil, fmt.Errorf("jlm: %s: %w", e.Name, err)
			}
			continue
		}
		for _, s := range [3]struct{ off, n uint64 }{{e.QSOff, qs}, {e.DOff, d}, {e.SCOff, sc}} {
			if s.n == 0 {
				continue
			}
			if s.off+s.n > size || s.off+s.n < s.off {
				fh.Close()
				closeDirect(direct)
				return nil, fmt.Errorf("jlm: %s: span [%d,%d) past the %d-byte file",
					e.Name, s.off, s.off+s.n, size)
			}
		}
	}

	// The dense region is always resident: the host reads it directly every
	// token.
	if f.offHeap {
		f.dense = f.buf(int(h.DenseLen))
	} else {
		f.dense = make([]byte, h.DenseLen)
	}
	// The default Placement, before the read faults the pages in; a model's own
	// SetPlacement re-places it afterwards.
	placeMemory(f.dense, false, nil)
	if h.DenseLen > 0 {
		if err := f.readAt(f.dense, int64(h.DenseOff)); err != nil {
			f.dense = nil
			f.unmapAll()
			fh.Close()
			closeDirect(direct)
			return nil, fmt.Errorf("jlm: reading the dense region of %s: %w", path, err)
		}
	}
	// One block space over all the arrays: text blocks are 0..NBlocks-1, vision
	// blocks follow, then expert pages, so the pager, residency map, chunk
	// bitmap and eviction policy are written once. Only pageAt knows.
	f.pages = make([][]byte, int(h.NBlocks)+int(h.NVisBlocks)+int(h.NExpPages))
	f.used = make([]uint64, len(f.pages))
	if err := f.measureExtents(); err != nil {
		fh.Close()
		closeDirect(direct)
		return nil, fmt.Errorf("jlm: %s: %w", path, err)
	}
	f.free = map[uint64][][]byte{}
	f.recent = -1
	// The container's own granularity (decodeHeader already validated it);
	// File.SetChunk overrides it before the first page is read.
	f.chunk = h.Chunk
	return f, nil
}

// pageAt is where block li's page starts in the file and how many bytes it
// occupies. It is the only place that knows about the separate page arrays.
func (f *File) pageAt(li int) (off, size uint64) {
	off, size = f.pageSlot(li)
	if f.extent != nil && li < len(f.extent) && f.extent[li] > 0 {
		size = f.extent[li]
	}
	return off, size
}

// pageSlot is page li's offset in the file and its slot there: the array's
// fixed page size, padding included.
func (f *File) pageSlot(li int) (off, size uint64) {
	nb, nv := int(f.H.NBlocks), int(f.H.NVisBlocks)
	switch {
	case li >= nb+nv:
		return f.H.ExpDataOff + uint64(li-nb-nv)*f.H.ExpPageSize, f.H.ExpPageSize
	case li >= nb:
		return f.H.VisDataOff + uint64(li-nb)*f.H.VisPageSize, f.H.VisPageSize
	}
	return f.H.DataOff + uint64(li)*f.H.PageSize, f.H.PageSize
}

// measureExtents fills f.extent from the tensor table: for each block and
// vision page, the furthest byte any of its tensors reaches, rounded up to
// Align so a page-in stays a direct read. Expert pages hold one expert's
// sheets each and are all one size, so they keep it.
func (f *File) measureExtents() error {
	ext := make([]uint64, len(f.pages))
	for i := range f.Entries() {
		e := &f.Entries()[i]
		if e.Block < 0 || e.Role.Expanded() || expertPaged(e) {
			continue
		}
		li := f.BlockOf(e)
		if li < 0 || li >= int(f.H.NBlocks)+int(f.H.NVisBlocks) {
			continue
		}
		base, slot := f.pageSlot(li)
		qs, d, sc, err := spans(e.Type, e.Dims, e.NDim)
		if err != nil {
			return err
		}
		for _, s := range [3]struct{ off, n uint64 }{{e.QSOff, qs}, {e.DOff, d}, {e.SCOff, sc}} {
			if s.n == 0 || s.off < base {
				continue
			}
			end := s.off + s.n - base
			if end > slot {
				return fmt.Errorf("%s ends %d bytes into a %d-byte page", e.Name, end, slot)
			}
			ext[li] = max(ext[li], end)
		}
	}
	for i := range ext {
		ext[i] = (ext[i] + Align - 1) &^ (Align - 1)
	}
	f.extent = ext
	return nil
}

// expertPage reports whether page index li is in the expert array.
func (f *File) expertPage(li int) bool { return li >= int(f.H.NBlocks)+int(f.H.NVisBlocks) }

// ExpertPage is the page index holding expert x's sheet of bank entry e, or
// -1 when e is not split into expert pages or x is out of range. Pass it to
// EnsurePage, Hold and Sheet.
func (f *File) ExpertPage(e *Entry, x int) int {
	if !expertPaged(e) || f.H.ExpPageSize == 0 || e.QSOff < f.H.ExpDataOff {
		return -1
	}
	sheets, _ := sheetsOf(e.Dims, e.NDim)
	if x < 0 || uint64(x) >= sheets {
		return -1
	}
	p := int((e.QSOff-f.H.ExpDataOff)/f.H.ExpPageSize) + x
	if p >= int(f.H.NExpPages) {
		return -1
	}
	return int(f.H.NBlocks) + int(f.H.NVisBlocks) + p
}

// ExpertPaged reports whether e's sheets live in expert pages, which is when
// Span returns nothing for it and Sheet is how its bytes are reached.
func (f *File) ExpertPaged(e *Entry) bool { return expertPaged(e) && f.H.NExpPages > 0 }

// Sheet is expert x's sheet of bank entry e, three planes cut from its expert
// page, or nils when that page is not resident. Like Span, the slices belong
// to a frame the pager reuses: hold the page (Hold) for as long as they are
// read.
func (f *File) Sheet(e *Entry, x int) (qs, d, sc []byte) {
	li := f.ExpertPage(e, x)
	if li < 0 {
		return nil, nil, nil
	}
	buf := f.Page(li)
	if buf == nil {
		return nil, nil, nil
	}
	nq, nd, nsc, _, err := sheetPlanes(e)
	if err != nil {
		return nil, nil, nil
	}
	base, _ := f.pageAt(li)
	x0 := uint64(x) * f.H.ExpPageSize
	cut := func(off, n uint64) []byte {
		off += x0
		if n == 0 || off < base || off+n > base+uint64(len(buf)) {
			return nil
		}
		o := off - base
		return buf[o : o+n : o+n]
	}
	return cut(e.QSOff, nq), cut(e.DOff, nd), cut(e.SCOff, nsc)
}

// PageBounds is block i's page offset and size, for a caller that needs to
// reason about where a tensor sits inside its page.
func (f *File) PageBounds(li int) (off, size uint64) { return f.pageAt(li) }

// StreamGroups is how many pieces a streamed block's routed selection should be
// uploaded in, as the converter computed it. Zero means the container predates
// the field and the caller should pick its own.
func (f *File) StreamGroups() int {
	if f == nil || f.H == nil {
		return 0
	}
	return int(f.H.StreamGroups)
}

// BlockOf is the logical block index an entry's bytes live in. A vision role
// indexes the vision array, which sits after the text one.
func (f *File) BlockOf(e *Entry) int {
	if e.Role.Vision() {
		return int(f.H.NBlocks) + int(e.Block)
	}
	return int(e.Block)
}

// InDense reports whether e's bytes are in the dense region, which is read once
// and stays at one address until Close. Every other span is in a page frame,
// and the pager reuses a frame for another block's bytes once its page is out.
func (f *File) InDense(e *Entry) bool {
	return e.QSOff >= f.H.DenseOff && e.QSOff < f.H.DenseOff+f.H.DenseLen
}

// NPages is how many block pages this container has, both arrays together, and
// PageBytes how big one of them is.
func (f *File) NPages() int { return len(f.pages) }
func (f *File) PageBytes(li int) uint64 {
	if li < 0 || li >= len(f.pages) {
		return 0
	}
	_, size := f.pageAt(li)
	return size
}

// VisionBlocks is how many of them are the vision tower's.
func (f *File) VisionBlocks() int { return int(f.H.NVisBlocks) }

// take reuses a buffer of exactly this size, or allocates one. Aligned, because
// an unaligned frame silently demotes its page-in to buffered I/O.
func (f *File) take(size uint64) []byte {
	if l := f.free[size]; len(l) > 0 {
		b := l[len(l)-1]
		f.free[size] = l[:len(l)-1]
		f.freeBytes -= size
		return b
	}
	// Blocks of a model come in a few sizes (a Q4_K_M mixes Q6_K into some),
	// so a frame a little larger serves too: reusing it costs its slack, a
	// fresh one costs page faults and leaves this one to be unmapped.
	// a quarter is the slack allowed; a size-classed pool if the
	// sizes of one model ever spread wider.
	best := uint64(0)
	for n, l := range f.free {
		if len(l) > 0 && n > size && n <= size+size/4 && (best == 0 || n < best) &&
			(f.budget == 0 || f.resident+n <= f.budget) {
			best = n
		}
	}
	if best > 0 {
		l := f.free[best]
		b := l[len(l)-1]
		f.free[best] = l[:len(l)-1]
		f.freeBytes -= best
		return b[:size]
	}
	// No frame of this size: free frames of other sizes give way first, so a
	// mixture evicting expert pages to fit a block page does not hold both.
	if f.budget > 0 && f.resident+size <= f.budget {
		f.trimFree(f.budget - f.resident - size)
	}
	f.fresh++
	b := f.buf(int(size))
	placeMemory(b, f.noHuge, f.nodeMask)
	return b
}

// FreshFrames is how many frames page-ins allocated rather than reused.
func (f *File) FreshFrames() int64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.fresh
}

// evictOne drops the most recently used resident block and returns its bytes to
// the free list. See EnsurePage for why MRU rather than LRU.
func (f *File) evictOne() bool {
	// An expert page goes first, least recently used: routing is not a cyclic
	// scan (the next token reuses many of the same experts), and LRU replayed
	// on real routing hits most of the time. Block pages are the working set
	// every token scans and are the last thing to give up.
	victim, oldest := -1, ^uint64(0)
	for i := int(f.H.NBlocks) + int(f.H.NVisBlocks); i < len(f.pages); i++ {
		if f.pages[i] != nil && !f.pinned(i) && f.used[i] < oldest {
			victim, oldest = i, f.used[i]
		}
	}
	if victim < 0 {
		victim = f.recent
	}
	if victim < 0 || victim >= len(f.pages) || f.pages[victim] == nil || f.pinned(victim) {
		victim = -1
		for i := range f.pages {
			if f.pages[i] != nil && !f.pinned(i) {
				victim = i
				break
			}
		}
	}
	if victim < 0 {
		return false
	}
	f.unhold(victim)
	f.evicted++
	if f.recent == victim {
		f.recent = -1
	}
	return true
}

// makeRoom evicts until size more bytes fit, or until nothing is left to evict.
// One page always fits: a budget below a single page is a configuration error
// the caller cannot recover from mid-token, and thrashing one page is strictly
// better than failing the read.
func (f *File) makeRoom(li int, size uint64) {
	if f.budget == 0 {
		return
	}
	for f.resident+size > f.budget {
		save := f.pages[li]
		f.pages[li] = nil // never evict the block being faulted in
		ok := f.evictOne()
		f.pages[li] = save
		if !ok {
			return
		}
	}
}

// SetBudget caps how many BYTES of block pages may be resident at once.
//
// It is the only knob: a budget that holds every page is the fully-resident
// configuration, and there is no paging-off mode. It is bytes because the
// page arrays have different sizes.
//
// Residency survives a re-budget (it is called twice on a normal load), so
// widening is free and narrowing evicts exactly the excess; nilling every
// page here once meant loading the model twice.
func (f *File) SetBudget(bytes uint64) {
	// After everything, unlocked: a preload's in-flight pages are pinned and
	// only leave once their reads land.
	defer f.settlePreload()
	// Deferred first, so it runs after the unlock: see flushRecycled.
	defer f.flushRecycled()
	f.mu.Lock()
	defer f.mu.Unlock()
	f.budget = bytes
	f.setCanEvict()
	if bytes == 0 {
		return
	}
	for f.resident > f.budget {
		if !f.evictOne() {
			break
		}
	}
	if f.resident < f.budget {
		f.trimFree(f.budget - f.resident)
	} else {
		f.trimFree(0)
	}
}

// Budget is the byte cap, 0 when unlimited, and ResidentBytes what is held
// against it.
func (f *File) Budget() uint64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.budget
}

func (f *File) ResidentBytes() uint64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.resident
}

// ResidentPages is how many block pages are in memory.
func (f *File) ResidentPages() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	n := 0
	for i := range f.pages {
		if f.pages[i] != nil {
			n++
		}
	}
	return n
}

// Faults is what the budget has cost: pages read in, pages evicted.
func (f *File) Faults() (in, out int64) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.faults, f.evicted
}

// readAt fills dst from off with ReadThreads concurrent requests. See the
// constant: the depth IS the rate on an NVMe.
func (f *File) readAt(dst []byte, off int64) error {
	// The direct handle only takes aligned work: page reads are aligned by
	// construction (DataOff, PageSize and alignedBuf frames), metadata is not,
	// and an unaligned O_DIRECT read returns EINVAL. Calls are counted because a
	// pread is priced per call, not per byte: a chunk dose that cut the bytes
	// 16x left the decode rate unchanged.
	f.countRead()
	if f.readHook != nil {
		return f.readHook(dst, off)
	}
	h := f.f
	if f.direct != nil && directOK(off, len(dst)) && aligned(dst) {
		h = f.direct
	}
	g := ReadThreads
	// Below a few MiB the goroutines cost more than the queue depth buys, and
	// the metadata reads above go through here too.
	if len(dst) < 4<<20 || g < 2 {
		f.ioStart(len(dst))
		defer f.ioEnd()
		_, err := h.ReadAt(dst, off)
		return err
	}
	if n := runtime.GOMAXPROCS(0) * 4; g > n {
		g = n
	}
	chunk := (len(dst) + g - 1) / g
	// Every worker's slice must also be aligned for the direct handle, so the
	// chunk is rounded up to a block.
	if h == f.direct {
		chunk = (chunk + Align - 1) &^ (Align - 1)
	}
	errs := make([]error, g)
	var wg sync.WaitGroup
	for w := 0; w < g; w++ {
		lo := w * chunk
		hi := min(lo+chunk, len(dst))
		if lo >= hi {
			continue
		}
		wg.Add(1)
		go func(w, lo, hi int) {
			defer wg.Done()
			f.ioStart(hi - lo)
			defer f.ioEnd()
			_, errs[w] = h.ReadAt(dst[lo:hi], off+int64(lo))
		}(w, lo, hi)
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}

// EnsurePage makes block i resident, reading it if it is not.
//
// The victim is the most recently used page, deliberately. Blocks are visited
// 0..N-1 every token, so on a cyclic scan through more blocks than frames LRU
// or a clock evicts the page needed soonest and faults every block every
// token. Belady's optimum
// evicts the page whose next use is furthest away, which on a cyclic scan is
// the one just used, giving N-F faults per token instead of N. This follows
// from model/'s ascending visit order; another order would degrade it toward
// the clock, not make it wrong.
func (f *File) EnsurePage(i int) error {
	if i < 0 || i >= len(f.pages) {
		return fmt.Errorf("jlm: block %d of %d", i, len(f.pages))
	}
	// The whole page is a plan of one range through EnsureRanges, which also
	// fills a frame that an earlier EnsureRanges left partially filled (an early return on
	// "already allocated" would not).
	//
	// The kernel's page cache is kept: posix_fadvise(DONTNEED) was tried and
	// measured several times slower on a large mixture, because the cache
	// serves the page-ins a token takes.
	off, size := f.pageAt(i)
	return f.EnsureRanges(i, []Range{{Off: off, N: size}})
}

// ReleasePages gives back every page AND the frame buffers behind them, so a
// container that is finished with its weights stops costing what it holds.
//
// DropPage alone frees nothing when there is a frame pool: nilling
// f.pages[i] only forgets which block is in a buffer the pool still owns. The
// budget is remembered, so a later page-in refills the same pool. The caller
// must re-bind anything pointing into a page afterwards; every span it
// handed out is now nil.
func (f *File) ReleasePages() {
	defer f.flushRecycled()
	f.mu.Lock()
	defer f.mu.Unlock()
	for i := range f.pages {
		if f.pages[i] == nil {
			continue
		}
		f.release(f.pages[i])
		f.pages[i] = nil
		f.dropFilled(i)
	}
	f.resident, f.recent = 0, -1
	f.specOn, f.specHeld, f.busy = nil, 0, nil
	f.trimFree(0)
}

// DropPage releases block i's memory. The next Span on it reads nil until
// EnsurePage runs again. A held page stays: its frame is being read, and
// handing it to the next claim would put another page's bytes under the
// reader.
func (f *File) DropPage(i int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if i < 0 || i >= len(f.pages) || f.pages[i] == nil || f.pinned(i) {
		return
	}
	f.unhold(i)
	if f.recent == i {
		f.recent = -1
	}
}

// dropFilled forgets page i's validity map, keeping it for takeFilled.
// Callers hold f.mu.
func (f *File) dropFilled(i int) {
	if f.filled == nil || f.filled[i] == nil {
		return
	}
	f.freeFilled = append(f.freeFilled, f.filled[i])
	f.filled[i] = nil
}

// takeFilled is a cleared validity map of n chunks, reused from an evicted
// page where one is long enough. Callers hold f.mu.
func (f *File) takeFilled(n int) []bool {
	for k := len(f.freeFilled) - 1; k >= 0; k-- {
		if m := f.freeFilled[k]; cap(m) >= n {
			last := len(f.freeFilled) - 1
			f.freeFilled[k] = f.freeFilled[last]
			f.freeFilled[last] = nil
			f.freeFilled = f.freeFilled[:last]
			m = m[:n]
			clear(m)
			return m
		}
	}
	return make([]bool, n)
}

// unhold moves page i's frame to the free list and refunds what it was
// charged: the frame's capacity, which take may have given a page smaller
// than itself. Callers hold f.mu.
func (f *File) unhold(i int) {
	b := f.pages[i]
	n := uint64(cap(b))
	f.unspec(i)
	f.putFree(n, b[:n])
	f.pages[i] = nil
	f.dropFilled(i)
	if f.resident >= n {
		f.resident -= n
	} else {
		f.resident = 0
	}
}

// pinned, hold and drop are the pin count. Callers hold f.mu.
//
// A pin is also how a read runs unlocked: EnsureRanges pins the block, drops
// the lock, reads into the frame and unpins, so the frame cannot be taken.
func (f *File) pinned(i int) bool { return i < len(f.pins) && f.pins[i] > 0 }

func (f *File) hold(i int) {
	if f.pins == nil {
		f.pins = make([]int, len(f.pages))
	}
	if i >= 0 && i < len(f.pins) {
		f.pins[i]++
	}
}

func (f *File) drop(i int) {
	if i >= 0 && i < len(f.pins) && f.pins[i] > 0 {
		f.pins[i]--
	}
}

// Lease holds a block's page resident until it is released.
//
// Every span this package hands out is a slice of a reused frame, so a
// caller that keeps one across anything that can evict reads whichever block
// was faulted in next: fluent and wrong. Making bytes readable and using them
// are two steps, and a lease closes the window between them (a streamed
// expert bank ensures k sheets and uploads them one at a time).
//
// A zero Lease is valid and releases nothing, so a caller need not branch.
type Lease struct {
	f *File
	b int
}

// Hold makes rs readable in block b and pins the page until Release. Passing no
// ranges pins without reading, which is what a caller wants when the bytes are
// already there and only the frame is at risk.
func (f *File) Hold(b int, rs []Range) (Lease, error) {
	if f == nil || b < 0 || b >= len(f.pages) {
		return Lease{}, fmt.Errorf("jlm: block %d of %d", b, len(f.pages))
	}
	// The pin is taken before the read, and the order is the gate: the read
	// drops f.mu, so pinning afterwards could pin a recycled buffer claim()
	// handed back after an eviction. evictOne and makeRoom skip a pinned page,
	// so the frame the read fills is the frame the lease returns.
	f.mu.Lock()
	err := f.claim(b)
	if err == nil {
		f.hold(b)
	}
	f.mu.Unlock()
	f.flushRecycled()
	if err != nil {
		return Lease{}, err
	}
	if len(rs) > 0 {
		if err := f.EnsureRanges(b, rs); err != nil {
			f.mu.Lock()
			f.drop(b)
			f.mu.Unlock()
			return Lease{}, err
		}
	}
	return Lease{f: f, b: b}, nil
}

// HoldFilled pins block b and reports true when its whole page is already
// resident, and does nothing and reports false otherwise. It is Hold of the
// full page without the read, decided under one lock so the page cannot be
// evicted between the check and the pin -- which is what lets a caller take
// the warm pages inline and send only the cold ones out as reads in flight.
func (f *File) HoldFilled(b int) (Lease, bool) {
	if f == nil || b < 0 || b >= len(f.pages) {
		return Lease{}, false
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.pages[b] == nil || f.filled == nil || f.filled[b] == nil {
		return Lease{}, false
	}
	for _, ok := range f.filled[b] {
		if !ok {
			return Lease{}, false
		}
	}
	if f.claim(b) != nil {
		return Lease{}, false
	}
	f.hold(b)
	return Lease{f: f, b: b}, true
}

// Release drops the pin. It is idempotent so a deferred Release is safe beside
// an early one.
func (l *Lease) Release() {
	if l.f == nil {
		return
	}
	f, b := l.f, l.b
	l.f = nil
	f.mu.Lock()
	defer f.mu.Unlock()
	f.drop(b)
}

// CanEvict reports whether this container's budget is smaller than its pages,
// which is the only state in which a page-in takes someone else's frame.
//
// It is the condition a caller serialises on: when the model fits, sessions
// read spans concurrently forever and no frame changes hands; below that, one
// session's page-in can take a frame another is reading, and no lock here
// can help because the spans were handed out earlier. It is an atomic
// because it changes only in SetBudget and is asked once per token.
func (f *File) CanEvict() bool { return f != nil && f.canEvict.Load() }

// setCanEvict recomputes the flag. Callers hold f.mu.
func (f *File) setCanEvict() {
	// Summed once: page sizes are fixed at conversion.
	if f.pageTotal == 0 {
		for i := range f.pages {
			_, size := f.pageAt(i)
			f.pageTotal += size
		}
	}
	f.canEvict.Store(f.budget != 0 && f.budget < f.pageTotal)
}

func (f *File) countRead() { atomic.AddInt64(&f.reads, 1) }

// Reads is how many readAt calls the pager has issued, the quantity a
// page-in is priced in; see readAt.
func (f *File) Reads() int64 { return atomic.LoadInt64(&f.reads) }

// ReadWait is how long callers have been blocked in EnsureRanges waiting for
// reads. It is exposed serial time, not device busy time: two sessions
// waiting on the same I/O count it twice, as each really did stop.
func (f *File) ReadWait() time.Duration {
	if f == nil {
		return 0
	}
	return time.Duration(atomic.LoadInt64(&f.waitNs))
}

// Resident reports whether block i's page is in memory.
func (f *File) Resident(i int) bool {
	if f == nil {
		return false
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	return i >= 0 && i < len(f.pages) && f.pages[i] != nil
}

// Filled reports whether block i's whole page is in memory: Resident, and no
// range of it left unread by a partial EnsureRanges.
func (f *File) Filled(i int) bool {
	if f == nil {
		return false
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if i < 0 || i >= len(f.pages) || f.pages[i] == nil || f.filled == nil || f.filled[i] == nil {
		return false
	}
	for _, ok := range f.filled[i] {
		if !ok {
			return false
		}
	}
	return true
}

// LoadAll makes every block resident, reading the pages concurrently.
//
// It is a configuration, not a mode (paging is always on and a model that
// fits never evicts), and the baseline the paging curve is measured against.
func (f *File) LoadAll() error {
	// Pages are read one at a time and each is already split ReadThreads ways,
	// so the queue is full without a second level of fan-out competing for it.
	// The count is read once: pages is allocated at Open and only its
	// elements change.
	n := f.NPages()
	for i := 0; i < n; i++ {
		if err := f.EnsurePage(i); err != nil {
			return err
		}
	}
	return nil
}

func (f *File) Close() error {
	// Close is not concurrent with use. The slice header of f.pages is read
	// without the lock on the hot path (NPages, PageBytes, pageAt), so the
	// elements are cleared and the header left immutable for the File's life.
	f.stopSpec()
	defer f.flushRecycled()
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.f == nil {
		return nil
	}
	h, d := f.f, f.direct
	f.stopIO()
	// Every buffer goes, and the next allocation in this process may land on
	// any of these addresses: the dense region, the pages and the free frames.
	f.retire(f.dense)
	for i := range f.pages {
		if f.pages[i] != nil {
			f.retire(f.pages[i])
		}
		f.pages[i] = nil
	}
	for _, l := range f.free {
		for _, b := range l {
			f.retire(b)
		}
	}
	// free is emptied rather than nilled: evictOne writes to it, and a write
	// to a nil map panics.
	f.filled, f.resident, f.recent = nil, 0, -1
	f.specOn, f.specHeld, f.busy = nil, 0, nil
	if f.landed.L != nil {
		f.landed.Broadcast()
	}
	f.free, f.freeBytes = map[uint64][][]byte{}, 0
	f.f, f.direct, f.dense = nil, nil, nil
	f.unmapAll()
	closeDirect(d)
	return h.Close()
}

// Config, Vision and Vocab are the container's own sections: everything a
// caller needs to know what this model is, in no other format's vocabulary.
func (f *File) Config() *Config { return f.cfg }
func (f *File) Vision() *Vision { return f.vis }
func (f *File) Vocab() *Vocab   { return f.voc }

// Find resolves a tensor by what it is: (role, block, index). index is -1 for
// every role but the per-expert ones.
func (f *File) Find(role Role, block, index int32) (*Entry, bool) {
	if i, ok := f.byID[tensorID{role, block, index}]; ok {
		return &f.tab[i], true
	}
	return nil, false
}

// tensorID is the addressing triple.
type tensorID struct {
	role  Role
	block int32
	index int32
}

// Has reports whether a role is present at all, which is how the reading side
// asks questions like "is the head tied" without a nil dance.
func (f *File) Has(role Role, block, index int32) bool {
	_, ok := f.Find(role, block, index)
	return ok
}

// Entries is every tensor, in file order.
func (f *File) Entries() []Entry { return f.tab }

// SpanLenOf is how many bytes a tensor's three regions occupy, without a File
// or the page: every length is arithmetic on the entry. Measuring with
// len(Span(e)) instead silently reports zero for an absent page.
func SpanLenOf(e *Entry) (qs, d, sc uint64) {
	nq, nd, nsc, err := spans(e.Type, e.Dims, e.NDim)
	if err != nil {
		return 0, 0, 0
	}
	return nq, nd, nsc
}

// SpanLen is SpanLenOf as a File method.
func (f *File) SpanLen(e *Entry) (qs, d, sc uint64) {
	nq, nd, nsc, err := spans(e.Type, e.Dims, e.NDim)
	if err != nil {
		return 0, 0, 0
	}
	return nq, nd, nsc
}

// Span returns a tensor's three device-layout regions.
//
// It returns nil for a block whose page is out: residency is explicit, so a
// caller that has not paged the block in gets nothing rather than a stall.
func (f *File) Span(e *Entry) (qs, d, sc []byte) {
	// A split bank is not one span: its sheets are in expert pages, one page
	// per expert. Sheet is how they are reached.
	if f.ExpertPaged(e) {
		return nil, nil, nil
	}
	nq, nd, nsc, err := spans(e.Type, e.Dims, e.NDim)
	if err != nil {
		return nil, nil, nil
	}
	var base uint64
	var buf []byte
	// The region is decided by the offset, not the block index: a block's
	// norms are stored in the dense region (alwaysResident), and looking for
	// them in a page that may be out would return nil.
	if f.InDense(e) {
		base, buf = f.H.DenseOff, f.dense
	} else {
		li := f.BlockOf(e)
		// One locked read of the frame pointer, not Resident followed by an
		// unlocked index, which could slice a page another caller nilled.
		if buf = f.Page(li); buf == nil {
			return nil, nil, nil
		}
		base, _ = f.pageAt(li)
	}
	cut := func(off, n uint64) []byte {
		if n == 0 || off < base || off+n > base+uint64(len(buf)) {
			return nil
		}
		o := off - base
		return buf[o : o+n : o+n]
	}
	return cut(e.QSOff, nq), cut(e.DOff, nd), cut(e.SCOff, nsc)
}

// Page is block i's whole page, or nil when it is not resident.
//
// The slice is a frame the pager reuses; holding it across anything that
// can evict reads whichever block was faulted in next. Hold(i, nil) is how a
// caller says it will still be reading.
func (f *File) Page(i int) []byte {
	f.mu.Lock()
	defer f.mu.Unlock()
	if i < 0 || i >= len(f.pages) {
		return nil
	}
	return f.pages[i]
}

// U32Bytes views a packed word slice as bytes without copying it.
func U32Bytes(v []uint32) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
}

// Direct reports whether page reads bypass the kernel's page cache.
//
// It changes what a budget means: with buffered I/O every byte is resident
// twice (our frame and the kernel's cache, charged to the same cgroup), which
// sched.MemBudget's eight-tenths fraction leaves room for. Direct reads have
// no shadow, so that headroom can be spent on weights.
func (f *File) Direct() bool { return f != nil && f.direct != nil }

// checkSheets validates a split bank: every plane of expert 0's sheet lies
// inside one expert page, and the bank's last expert is still inside the
// expert array -- which bounds every sheet between, since they are the same
// offsets ExpPageSize apart.
func (h *Header) checkSheets(e *Entry) error {
	nq, nd, nsc, sheets, err := sheetPlanes(e)
	if err != nil {
		return err
	}
	if h.NExpPages == 0 || h.ExpPageSize == 0 {
		return fmt.Errorf("an expert bank in a container with no expert pages")
	}
	if e.QSOff < h.ExpDataOff {
		return fmt.Errorf("sheet at %d is before the expert pages at %d", e.QSOff, h.ExpDataOff)
	}
	first := (e.QSOff - h.ExpDataOff) / h.ExpPageSize
	start := h.ExpDataOff + first*h.ExpPageSize
	for _, s := range [3]struct{ off, n uint64 }{{e.QSOff, nq}, {e.DOff, nd}, {e.SCOff, nsc}} {
		if s.n == 0 {
			continue
		}
		if s.off < start || s.off+s.n > start+h.ExpPageSize || s.off+s.n < s.off {
			return fmt.Errorf("plane [%d,%d) leaves its expert page [%d,%d)",
				s.off, s.off+s.n, start, start+h.ExpPageSize)
		}
	}
	if sheets == 0 || first+sheets > uint64(h.NExpPages) {
		return fmt.Errorf("%d sheets from expert page %d run past the %d pages", sheets, first, h.NExpPages)
	}
	return nil
}
