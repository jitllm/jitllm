package jlm

import (
	"fmt"
	"sync/atomic"
	"time"
)

// Chunk is the fallback fill granularity. A page is the unit of placement; a
// chunk is the unit of reading: a resident page need not be filled, since a
// mixture's token touches ten of 512 experts. A real container's granularity
// comes from its header (see chunkFor); this serves a File built without one
// (the pager's tests). File.SetChunk overrides either, per File.
//
// The chunk is not the request size: planRuns coalesces adjacent missing
// chunks into one read, so a finer chunk changes only the residency bitmap
// and the rounding on a range that does not fill one.
const Chunk uint64 = 1 << 20

// SetChunk sets this File's fill granularity, overriding the container's own
// so a dose can be taken against the file it is dosing. It must be a power of
// two of at least 4096, and it must come before the first page is read: the
// residency bitmap is sized from it.
func (f *File) SetChunk(n uint64) error {
	if n < 4096 || n&(n-1) != 0 {
		return fmt.Errorf("jlm: a fill granularity of %d bytes; want a power of two of at least 4096", n)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.faults > 0 {
		return fmt.Errorf("jlm: %s has read %d page(s); the fill granularity is fixed by then", f.Path, f.faults)
	}
	f.chunk = n
	return nil
}

// chunkSize is this File's fill granularity, fixed at Open so the residency
// bitmap and every offset derived from it agree for the File's whole life.
func (f *File) chunkSize() uint64 {
	if f.chunk == 0 {
		return Chunk
	}
	return f.chunk
}

// chunksFor is how many chunks cover a page.
func (f *File) chunksFor(li int) int {
	if f.H == nil {
		return 0
	}
	_, size := f.pageAt(li)
	if size == 0 {
		return 0
	}
	return int((size + f.chunkSize() - 1) / f.chunkSize())
}

// Range is a byte range of the file a caller needs readable.
type Range struct{ Off, N uint64 }

// EnsureRanges makes many ranges of one block readable in one plan, reading
// only the chunks that are not already there.
//
// The bytes a caller did not ask for are not read and not valid: a matvec
// handed a span nobody ensured reads whatever the last block left, as
// numbers rather than a fault. Every consumer of a packed span must ensure it
// first (model.pageIn for a block's weights, model.moe for selected experts).
//
// One plan per block, not one read per tensor, for queue depth rather than
// bytes: readAt spreads a request across workers only above a few MiB, so a
// single expert or projection is one request at queue depth 1. Collecting
// every missing run first issues them together. This is neurostream's
// finding: planning once per layer rather than per projection removed
// nearly all of its stall.
func (f *File) EnsureRanges(b int, rs []Range) error {
	return f.ensureRanges(b, rs, false)
}

// ensureRanges is EnsureRanges for a demanded read (spec false) and for a
// speculative one (spec true, see readsvc.go).
func (f *File) ensureRanges(b int, rs []Range, spec bool) error {
	if len(rs) == 0 {
		return nil
	}
	// The lock covers the decision, not the I/O: holding it across a page read
	// would serialise sessions' file I/O. So the frame is claimed and pinned
	// under the lock, the reads run unlocked, and the validity map is marked at
	// the end; a pinned page is never a victim. A chunk another caller is
	// reading is marked busy: this caller reads the rest, then waits for those
	// chunks alone and never reads one twice.
	//
	// The two locked halves are their own functions so every Lock has a
	// deferred Unlock: a mutex left held hangs the next token with no error.
	pl := f.getPlan()
	defer f.putPlan(pl)
	for {
		page, base, runs, waits, err := f.planRunsBusy(b, rs, spec, pl.runs[:0])
		pl.runs = runs[:0]
		// Before the read: a copy of the frame's old bytes must be gone before
		// its new ones can be handed out.
		f.flushRecycled()
		if err != nil {
			return err
		}
		if len(runs) == 0 {
			if !waits {
				return nil
			}
			f.awaitLanded()
			continue
		}
		if err := f.readPlanned(b, pl, page, base, runs); err != nil {
			return err
		}
		if !waits {
			return nil
		}
	}
}

// readPlanned reads the runs planRunsBusy returned into page and marks them,
// releasing the pin the plan took.
func (f *File) readPlanned(b int, pl *ioPlan, page []byte, base uint64, runs []run) error {
	// The pin is released by a defer, not by markFilled, so a panic in the
	// reads cannot leave the block unevictable (which reads as a budget too
	// small, not as an error).
	defer f.unpin(b)
	_, psize := f.pageAt(b)

	// The clock covers the I/O only, not planRuns' lock wait. See File.waitNs.
	t0 := time.Now()
	defer func() { atomic.AddInt64(&f.waitNs, int64(time.Since(t0))) }()

	errs := f.readRuns(pl, page, base, psize, runs)
	return f.markFilled(b, runs, errs)
}

// awaitLanded blocks until some in-flight read lands (or fails), which is
// when a caller that joined it re-plans.
func (f *File) awaitLanded() {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.landed.L == nil {
		f.landed.L = &f.mu
	}
	f.landed.Wait()
}

// run is a contiguous stretch of missing chunks, which is one read request.
type run struct{ lo, hi int }

// chunkSpan is the half-open chunk range a byte range covers, clamped to the
// page. One function because planRuns walks the ranges twice -- once to bound
// the window and once to mark it -- and the two must agree.
func (f *File) chunkSpan(r Range, base uint64, nc int) (lo, hi int) {
	lo = int((r.Off - base) / f.chunkSize())
	hi = int((r.Off - base + r.N + f.chunkSize() - 1) / f.chunkSize())
	if hi > nc {
		hi = nc
	}
	return lo, hi
}

// planRuns claims and pins block b's frame and returns the chunk runs the
// caller must read. A non-empty result carries a pin that markFilled releases.
//
// runs is the caller's scratch, appended to and returned.
func (f *File) planRuns(b int, rs []Range, runs []run) (page []byte, base uint64, _ []run, err error) {
	page, base, runs, _, err = f.planRunsBusy(b, rs, false, runs)
	// A plan taken for inspection reads nothing, so it must not leave its
	// chunks marked as being read.
	f.mu.Lock()
	defer f.mu.Unlock()
	f.clearBusy(b, runs)
	return page, base, runs, err
}

// planRunsBusy is planRuns for a caller that will read the runs: it marks
// them busy so a second caller joins instead of reading them again, and
// reports in waits that some wanted chunk is being read by someone else.
// runs is the caller's scratch, appended to and returned.
func (f *File) planRunsBusy(b int, rs []Range, spec bool, runs []run) (page []byte, base uint64, _ []run, waits bool, err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if b < 0 || b >= len(f.pages) {
		return nil, 0, runs[:0], false, fmt.Errorf("jlm: block %d of %d", b, len(f.pages))
	}
	if err := f.claimFor(b, spec); err != nil {
		return nil, 0, runs[:0], false, err
	}
	base, psize := f.pageAt(b)
	have := f.filled[b]
	nc := len(have)
	busy := f.busyOf(b, nc)

	// The plan is built from the requested spans, never by sweeping the page: a
	// routed expert selection is scattered across the banks, so its bracketing
	// span is nearly the whole page, and a per-chunk sweep of it was all of
	// ensureExperts' warm cost (model.Model.ExpertIO splits plan from I/O).
	// Sorting and merging the spans is O(R log R + chunks asked for) and yields
	// the same runs: maximal stretches of chunks wanted and not yet filled.
	// Spans that touch (hi == lo) merge, as one contiguous stretch.
	//
	// The scratch is the File's and f.mu is held throughout, so it is
	// allocated once per File.
	sp := f.spanBuf[:0]
	for _, r := range rs {
		if r.N == 0 {
			continue
		}
		if r.Off < base || r.Off+r.N > base+psize {
			return nil, 0, runs[:0], false, fmt.Errorf("jlm: [%d,%d) is outside block %d's page",
				r.Off, r.Off+r.N, b)
		}
		if lo, hi := f.chunkSpan(r, base, nc); lo < hi {
			sp = append(sp, run{lo, hi})
		}
	}
	f.spanBuf = sp[:0] // keep the grown array whatever this call found
	if len(sp) == 0 {
		return nil, 0, runs[:0], false, nil
	}
	// Insertion sort, not sort.Slice: a plan is a couple of dozen nearly
	// sorted spans (bank by bank, ascending expert), and sort.Slice's
	// reflect-based swaps cost more than the sweep this replaced. model.moe
	// sorts its selection the same way.
	for i := 1; i < len(sp); i++ {
		v := sp[i]
		j := i - 1
		for j >= 0 && sp[j].lo > v.lo {
			sp[j+1] = sp[j]
			j--
		}
		sp[j+1] = v
	}
	cur := sp[0]
	for k := 1; k <= len(sp); k++ {
		if k < len(sp) && sp[k].lo <= cur.hi {
			if sp[k].hi > cur.hi {
				cur.hi = sp[k].hi
			}
			continue
		}
		for i := cur.lo; i < cur.hi; {
			if have[i] {
				i++
				continue
			}
			if busy[i] {
				waits = true
				i++
				continue
			}
			j := i
			for j < cur.hi && !have[j] && !busy[j] {
				j++
			}
			runs = append(runs, run{i, j})
			i = j
		}
		if k < len(sp) {
			cur = sp[k]
		}
	}
	if len(runs) == 0 {
		return nil, 0, runs[:0], waits, nil
	}
	for _, r := range runs {
		for k := r.lo; k < r.hi; k++ {
			busy[k] = true
		}
	}
	f.hold(b)
	return f.pages[b], base, runs, waits, nil
}

// unpin releases the pin planRuns took.
func (f *File) unpin(b int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.drop(b)
}

// markFilled records the runs that landed, clears their busy marks whatever
// happened, and wakes the callers that joined them.
func (f *File) markFilled(b int, runs []run, errs []error) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.clearBusy(b, runs)
	if f.landed.L == nil {
		f.landed.L = &f.mu
	}
	defer f.landed.Broadcast()
	var first error
	for ri, err := range errs {
		if err != nil {
			if first == nil {
				first = fmt.Errorf("jlm: block %d run %d: %w", b, ri, err)
			}
			continue
		}
		// Through f.filled rather than a captured slice header: DropPage and
		// ReleasePages do not consult the pin, and marking a freed map would
		// write memory nothing owns.
		if f.filled[b] == nil {
			continue
		}
		for k := runs[ri].lo; k < runs[ri].hi && k < len(f.filled[b]); k++ {
			f.filled[b][k] = true
		}
		f.chunkIn += int64(runs[ri].hi - runs[ri].lo)
	}
	return first
}

// busyOf is block b's in-flight map, sized to its chunks. Callers hold f.mu.
func (f *File) busyOf(b, nc int) []bool {
	if f.busy == nil {
		f.busy = make([][]bool, len(f.pages))
	}
	if len(f.busy[b]) != nc {
		f.busy[b] = make([]bool, nc)
	}
	return f.busy[b]
}

// clearBusy unmarks runs. Callers hold f.mu.
func (f *File) clearBusy(b int, runs []run) {
	if f.busy == nil || f.busy[b] == nil {
		return
	}
	for _, r := range runs {
		for k := r.lo; k < r.hi && k < len(f.busy[b]); k++ {
			f.busy[b][k] = false
		}
	}
}

// claim gives block b a frame without reading anything into it, evicting
// another block if the pool is full. It is EnsurePage's residency half with
// the read left out. Callers hold f.mu.
func (f *File) claim(b int) error { return f.claimFor(b, false) }

// claimFor is claim for a demanded page (spec false) or a speculative one.
// A speculative claim of a page that is not resident is charged to the
// speculative occupancy (specHeld) and refused with errSpecFull past
// specLimit; a demanded claim takes the charge off, the page now being the
// working set's.
func (f *File) claimFor(b int, spec bool) error {
	if spec && f.pages[b] == nil && f.specHeld >= f.specLimit() {
		return errSpecFull
	}
	f.tick++
	if f.used != nil {
		f.used[b] = f.tick
	}
	if f.pages[b] != nil {
		if !spec {
			f.unspec(b)
		}
		if !f.expertPage(b) {
			f.recent = b
		}
		return nil
	}
	if spec {
		f.markSpec(b)
	}
	f.faults++
	if f.filled == nil {
		f.filled = make([][]bool, len(f.pages))
	}
	_, size := f.pageAt(b)
	f.makeRoom(b, size)
	f.pages[b] = f.take(size)
	// An expert page whole on one node, the experts alternating over the
	// nodes, so a host computing an expert where its page lives reads only
	// local memory (nn's node-affine mixture regions).
	if len(f.nodes) > 1 && f.expertPage(b) {
		x := b - int(f.H.NBlocks) - int(f.H.NVisBlocks)
		bindNode(f.pages[b], f.nodes[x%len(f.nodes)])
	}
	// Whatever the frame held, it is about to hold block b; see OnRecycle.
	f.retire(f.pages[b])
	f.resident += uint64(cap(f.pages[b]))
	if !f.expertPage(b) {
		f.recent = b
	}
	f.filled[b] = f.takeFilled(f.chunksFor(b))
	return nil
}

// ChunkBytes is this container's fill granularity: its header's, or what
// SetChunk set. Byte counts derived from ChunksIn must use this.
func (f *File) ChunkBytes() uint64 { return f.chunkSize() }

// ChunksIn is how many Chunk-sized reads have been served, which is what a
// partial page-in actually costs -- Faults counts blocks, and a block whose
// token touched ten of 512 experts is not the same event as one read whole.
func (f *File) ChunksIn() int64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.chunkIn
}
