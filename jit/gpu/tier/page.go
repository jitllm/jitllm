package tier

import (
	"fmt"
	"time"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// Block paging: a device holds N block slots and swaps, instead of declining a
// block and losing it for the rest of the run, so a card smaller than the model
// still runs every block. See docs/design/block-paging.md.
//
// A block's weights have a backing store (the packed arena, or the container),
// so they can leave the card and come back byte-identical. Its KV cache is this
// session's own history and nothing else holds a copy; norms, biases and the
// router are kilobytes. So:
//
//	pageable    the quantised matrices
//	permanent   the KV cache, the norms, the biases, the router
//
// The permanent part bounds how many blocks a device can own; the pageable part
// is what the slots hold. Both come out of the same budget, so admitting a
// block shrinks the slot count (see slots). A device that cannot page (fewer
// than minSlots) declines per block exactly as before.
//
// The structure is always built and the budget is the control: every block
// gets a page record whether or not the model fits, so SetBudget can demote a
// model that currently fits (make room for another, hand the card back). On a
// model that fits eviction never fires: pageIn is a lookup, submit coalesces
// the whole range into one submission, and PageOuts stays at zero.

// minSlots is the fewest block slots a device can page with. One, because a
// swap needs one slot (evict, read in, run): a one-slot pager cannot overlap
// upload and compute, but refusing it sends the block to the host instead,
// and a card can be one page short of two slots.
const minSlots = 1

// page is one block's pageable weights: what a page-in uploads and a page-out
// frees, and nothing else. ws is the shape, not a copy of the bytes; ensure
// refreshes where the host holds them.
type page struct {
	ws    []nn.Weight // the seven matrices, in layer.tensors() order
	bytes uint64      // what the set costs on the device while it is resident
	// round is the driver's page rounding of those buffers beyond bytes, which
	// a page-in also has to find room for (scratch.go).
	round uint64
	in    bool // resident right now
	// uses is how many submissions -- running, or paged in and about to run
	// (submit) -- read the block: it is no victim while any does, whichever
	// session's prologue is looking for room. Under mu.
	uses int
	// ensure re-points every slice in ws at where the host holds those bytes
	// NOW, and is nil for a caller that does not page its own weights.
	// See nn.LayerWeights.Ensure.
	ensure func(*nn.LayerWeights) error
}

// tensors is the block's pageable weights in one fixed order, so that
// PrepLayer, ReleaseLayers and the pager cannot disagree about what a block is:
// a tensor missing from this list leaks on release and is never priced.
func (l *layer) tensors() []**resident {
	// The order matches weightList and prepLayer's ws/ms/bias order: the dense
	// seven, then the shared expert, the linear block, MLA and KDA's gates.
	return []**resident{&l.wq, &l.wk, &l.wv, &l.wo, &l.gate, &l.up, &l.down,
		&l.shGate, &l.shUp, &l.shDown, &l.ssmGate, &l.ssmBA, &l.ssmOut,
		&l.wqa, &l.wqb, &l.wkva, &l.wkb, &l.wvb,
		&l.ssmFA, &l.ssmFB, &l.ssmGA, &l.ssmGB,
		&l.pleGate, &l.pleProj,
		&l.idxQB, &l.idxK, &l.idxProj,
		&l.laurelL, &l.laurelR, &l.ssmIn,
		&l.idxQ,
		&l.woA, &l.compKV, &l.compGate, &l.idxCompKV, &l.idxCompGate,
		&l.mlaGate, &l.routedDown, &l.routedUp,
		&l.hcAD, &l.hcAU, &l.hcAI, &l.hcFD, &l.hcFU, &l.hcFI, &l.ngKey, &l.ngVal}
}

// pageSizeWhy is pageSize with the reason, naming the tensor and its format.
func (g *devTier) pageSizeWhy(ws []nn.Weight) (uint64, string, bool) {
	var total uint64
	for i, x := range ws {
		if len(x.Data) == 0 {
			continue
		}
		q, ok := quantOf(x.T)
		if !ok {
			why := "which has no device kernel"
			if kq, packed := kernels.QuantOf(x.T); packed {
				why = "and " + kernels.DeviceWhyNot(kq)
			}
			return 0, fmt.Sprintf("%s is %s, %s", weightRole(i), x.T, why), false
		}
		// At the buffers' footprint, as a page-in asks for them (pageIn): a
		// slot priced at the bytes alone counts slots the card does not have.
		f, err := g.planesFootprint(q, x.Rows, x.K)
		if err != nil {
			return 0, fmt.Sprintf("%s (%s %dx%d): %v", weightRole(i), x.T, x.Rows, x.K, err), false
		}
		total += f
	}
	return total, "", true
}

// weightRole names the slot a weight occupies in the fixed order PrepLayer
// builds, so a decline reads as a tensor rather than as an index.
func weightRole(i int) string {
	names := [...]string{"attn_q", "attn_k", "attn_v", "attn_output",
		"ffn_gate", "ffn_up", "ffn_down",
		"sh_exp_gate", "sh_exp_up", "sh_exp_down",
		"ssm_gate", "ssm_ba", "ssm_out",
		"attn_q_a", "attn_q_b", "attn_kv_a", "attn_k_b", "attn_v_b",
		"ssm_f_a", "ssm_f_b", "ssm_g_a", "ssm_g_b",
		"inp_gate", "proj",
		"indexer.attn_q_b", "indexer.attn_k", "indexer.proj",
		"laurel_l", "laurel_r", "ssm_in",
		"indexer.q_proj",
		"attn_output_a", "attn_compressor_kv", "attn_compressor_gate",
		"indexer_compressor_kv", "indexer_compressor_gate",
		"attn_gate", "ffn_routed_down", "ffn_routed_up",
		"hc_attn_down", "hc_attn_up", "hc_attn_inject", "hc_ffn_down", "hc_ffn_up", "hc_ffn_inject",
		"ngram_key", "ngram_value"}
	if i < len(names) {
		return names[i]
	}
	return fmt.Sprintf("weight %d", i)
}

// slots is how many blocks' weights this device can hold at once: the budget
// left after the permanent parts, divided by the widest block (not the mean,
// so the resident set stays under the budget). Both sides are at the buffers'
// footprint, the driver's rounding included, because that is what a page-in
// has to find room for and what a page-out gives back. Every owned block's KV and
// norms are permanent, so the slot count shrinks as blocks are admitted, and a
// model whose KV alone fills the card stops being admitted rather than
// thrashing.
//
// The pool is checked too, because two devices on one heap (an integrated GPU
// and the host) spend the same bytes; see Pool. Callers hold g.mu.
func (g *devTier) slots() int {
	if g.widest == 0 {
		return 0
	}
	g.settleRounding() // see room

	// Imported (wrapped) blocks hold no device bytes, but that is handled in
	// victim(), not here: the arena having a budget does not mean it has room,
	// so this counts what the budget buys if the bytes have to be held.
	// The permanent parts are everything charged that the slots do not hold.
	perm := uint64(0)
	if g.used > g.pageBytes {
		perm = g.used - g.pageBytes
	}
	perm += g.reserved()
	avail := uint64(0)
	if g.limit > perm {
		avail = g.limit - perm
	}
	if pl := g.pool.Limit(); pl > 0 {
		// What the pool would allow, plus what this device's own slots already
		// hold -- those bytes come back to the pool when a page goes out.
		free := g.pageBytes
		if u := g.pool.Used(); pl > u {
			free += pl - u
		}
		if free < avail {
			avail = free
		}
	}
	return int(avail / g.widest)
}

// slotsIfPerm is slots() with n more bytes held permanently: what the pager
// would have left if something that is not a page took that space. The output
// projection and a block slot compete for the same bytes; see PrepHead.
func (g *devTier) slotsIfPerm(n uint64) int {
	if g.widest == 0 {
		return 0
	}
	g.settleRounding() // see room

	perm := n
	if g.used > g.pageBytes {
		perm += g.used - g.pageBytes
	}
	avail := uint64(0)
	if g.limit > perm {
		avail = g.limit - perm
	}
	if pl := g.pool.Limit(); pl > 0 {
		free := g.pageBytes
		if u := g.pool.Used(); pl > u+n {
			free += pl - u - n
		}
		if free < avail {
			avail = free
		}
	}
	return int(avail / g.widest)
}

// victim is the resident block whose next use is furthest away, or -1. On the
// cyclic layer scan that is Belady's optimum and it is computable
// (sched.NextUse, shared with the host ring; see engine/sched/cyclic.go): LRU is
// pessimal there, while this pins a prefix of slots-1 blocks and thrashes only
// the last slot.
//
// A block outside the scan is never used again in it, so it goes first. Ties
// break on the lowest index, so the choice does not depend on map iteration
// order. Callers hold g.mu.
//
// streamed limits the candidates to streamed blocks (devTier.stream): making
// room for a streamed block, or for history, never evicts a resident one. Only
// the budget's owner (trim) and a demoted block coming back take any unpinned
// block.
func (g *devTier) victim(cur, lo, hi int, streamed bool) int {
	best, bestD := -1, -1
	for li, l := range g.layers {
		if l == nil || l.pg == nil || !l.pg.in || li == cur || g.pinned[li] || streamed && !g.stream[li] ||
			l.pg.uses > 0 {
			continue
		}
		// A block whose page-out frees nothing is not a candidate: wrapped
		// weights (resident.imported) reclaim zero bytes, and under trim()
		// evicting them would demote every block for nothing. A partially
		// wrapped block prices only its copied half, so it stays a candidate.
		if l.pg.bytes == 0 {
			continue
		}
		// The map is walked rather than the range (so this is not
		// sched.CyclicVictim): a device's blocks need not be contiguous, and
		// one outside the scan reports sched.Never and goes first.
		d := sched.NextUse(li, cur, lo, hi)
		if d > bestD || (d == bestD && li < best) {
			best, bestD = li, d
		}
	}
	return best
}

// dropTensors frees a block's pageable weights and gives the budget back. The
// entries leave g.res by identity rather than by a recomputed key; page-out and
// release share this walk so they cannot drift apart.
//
// The arena is deliberately not touched: dropping the pack would make the next
// page-in a repack. GPU.DropArena is the explicit way to give those host bytes
// back. Callers hold g.mu.
func (g *devTier) dropTensors(l *layer) uint64 {
	var freed uint64
	for _, rp := range l.tensors() {
		r := *rp
		if r == nil || !r.ok {
			continue
		}
		if r.shared {
			// One buffer behind several blocks: freeing it here would take it
			// from under every other block. Close frees it once; see
			// devTier.sharedCompact.
			*rp = nil
			continue
		}
		for k, v := range g.res {
			if v == r {
				delete(g.res, k)
				break
			}
		}
		if n := r.bytes(); n > 0 {
			g.refund(n)
			freed += n
		}
		g.freeResident(r)
	}
	return freed
}

// dropImported pages block li out if any of its weights are wrappers over host
// memory, which is what makes GPU.DropArena safe on a live tier. A copied block
// is left alone: the device holds its own copy, and releasing the arena entry
// only makes its next page-in repack. Callers must not hold g.mu.
func (g *devTier) dropImported(li int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	l := g.layers[li]
	if l == nil || l.pg == nil || !l.pg.in {
		return
	}
	for _, rp := range l.tensors() {
		if r := *rp; r != nil && r.ok && r.imported {
			g.pageOut(li)
			return
		}
	}
}

// pageOut frees block li's weights, leaving the block owned by this device and
// its history untouched. Callers hold g.mu.
func (g *devTier) pageOut(li int) uint64 {
	l := g.layers[li]
	if l == nil || l.pg == nil || !l.pg.in {
		return 0
	}
	// A recording names device addresses, which stop meaning anything once
	// these buffers are freed. See dropGraph.
	g.dropGraph()
	freed := g.dropTensors(l)
	l.pg.in = false
	g.pagesIn--
	g.pageBytes -= min(g.pageBytes, freed+l.pg.round)
	g.PageOuts++
	return freed
}

// reclaim pages blocks out until n bytes fit, and reports whether they do.
// cur is the block being loaded, which is never a candidate; [lo, hi) is the
// scan the victim policy is reading. The victims are streamed blocks, unless
// cur is a block that is not (one trim demoted, coming back), which may take
// any unpinned one as it always could. Callers hold g.mu.
func (g *devTier) reclaim(n uint64, cur, lo, hi int) bool {
	streamed := cur < 0 || g.stream[cur]
	for !g.room(n) {
		v := g.victim(cur, lo, hi, streamed)
		if v < 0 {
			// Every candidate is read by another session's submission: it
			// gives them back when it completes (submit), so wait for that
			// rather than refuse. The caller holds none itself here.
			if !g.victimHeld(cur, streamed) {
				return false
			}
			g.idle.Wait()
			continue
		}
		g.pageOut(v)
	}
	return true
}

// victimHeld reports a block victim would take but for a submission reading
// it (page.uses). Callers hold g.mu.
func (g *devTier) victimHeld(cur int, streamed bool) bool {
	for li, l := range g.layers {
		if l != nil && l.pg != nil && l.pg.in && li != cur && !g.pinned[li] && !(streamed && !g.stream[li]) &&
			l.pg.bytes > 0 && l.pg.uses > 0 {
			return true
		}
	}
	return false
}

// holdPages marks blocks [lo, hi) as read by a submission about to run, so
// no other session's prologue pages them out under it (victim), and reports
// whether every one is on the card to be held. Callers hold g.mu.
func (g *devTier) holdPages(lo, hi int) bool {
	for li := lo; li < hi; li++ {
		if l := g.layers[li]; l == nil || !l.ok || l.pg != nil && !l.pg.in {
			g.dropPages(lo, li)
			return false
		}
	}
	for li := lo; li < hi; li++ {
		if l := g.layers[li]; l.pg != nil {
			l.pg.uses++
		}
	}
	return true
}

// dropPages ends what holdPages began, and wakes a page-in waiting for room
// (reclaim). Callers hold g.mu.
func (g *devTier) dropPages(lo, hi int) {
	for li := lo; li < hi; li++ {
		if l := g.layers[li]; l != nil && l.pg != nil && l.pg.uses > 0 {
			l.pg.uses--
		}
	}
	g.idle.Broadcast()
}

// roomOrReclaim is room for n bytes of state that cannot be re-read -- KV
// history, recurrent summaries -- paging streamed blocks out for it when one
// slot is left to swap through: a page can come back, the state cannot.
// Callers hold g.mu.
func (g *devTier) roomOrReclaim(n uint64) bool {
	if g.room(n) || g.dropCaches(n) {
		return true
	}
	if len(g.stream) == 0 || g.slotsIfPerm(n+g.reserved()) < minSlots || !g.reclaim(n, -1, 0, 0) {
		return false
	}
	g.Slots = g.slots()
	return true
}

// keepsSlot reports that n more bytes held for good leave a device that
// streams the minSlots its streamed blocks swap through. It is asked whether or
// not n fits right now: room() counts the pages in as space, so state charged
// into it while a streamed block was paged in left the next page-in nothing --
// TestPrefillStreamsHostBlocks on a discrete card, where a prompt's sixteen
// streamed histories took the one slot and nothing ever paged in. Callers hold
// g.mu.
func (g *devTier) keepsSlot(n uint64) bool {
	return len(g.stream) == 0 || g.widest == 0 || g.slotsIfPerm(n+g.reserved()) >= minSlots
}

// stateFits makes room for n bytes of state that cannot be re-read -- KV
// pages, recurrent summaries -- by the KV pools' free pages and then the
// streamed blocks (kvCompact, roomOrReclaim), and refuses state that would
// take a streaming device's last slot (keepsSlot). It is the run-time test,
// for history a sequence grows into; a block's own first pages are held to
// the slot by its admission (prepLayer), where the block's weights are charged
// before they are counted as a page and this would read them as permanent.
// keep is a KV layer pool compaction must leave alone. Callers hold g.mu.
func (g *devTier) stateFits(n uint64, keep *kvLayerPool) bool {
	return (g.kvCompact(n, keep) || g.roomOrReclaim(n)) && g.keepsSlot(n)
}

// over reports that this device is holding more than it is allowed to, which
// only SetBudget (or prepLayer's unchecked permanent charges) can make true.
// Callers hold g.mu.
func (g *devTier) over() bool {
	if g.used > g.limit {
		return true
	}
	pl := g.pool.Limit()
	return pl > 0 && g.pool.Used() > pl
}

// trim pages blocks out until the card is inside its budget again, and refreshes
// the reported slot count.
//
// The condition is the budget, not the slot count: slots() divides by the
// widest block, so a model whose blocks differ in size can fit with fewer
// slots than blocks, and evicting on that number would demote a model that
// fits. On every model that fits this returns immediately. Callers hold g.mu.
func (g *devTier) trim() {
	// On a device that streams nothing, the unpack staging goes before any block:
	// it will never be reused, and prepLayer charges the KV cache and norms
	// with no room check, so the over-commit would otherwise be paid in blocks
	// (TestAModelThatFitsAdmitsTheSameBlocksEitherWay).
	if g.over() && len(g.stream) == 0 {
		g.freeRaw()
	}
	// Expert caches before blocks: a cache is copies of host pages, and its
	// block keeps running on the plain bank without it.
	if g.over() {
		g.dropCaches(0)
	}
	for g.over() && g.pagesIn > 0 {
		v := g.victim(-1, 0, 0, false)
		if v < 0 {
			break
		}
		g.pageOut(v)
	}
	// Otherwise the unpack staging goes last: it is charged to the
	// budget (unpack.go) so a shrink must be able to reclaim it, but losing it
	// costs every later page-in a host repack, while losing a block costs only
	// that block's next page-in. ensureRaw re-sizes it on the next page-in.
	if g.over() {
		g.freeRaw()
	}
	g.Slots = g.slots()
}

// pageIn brings block li's weights back onto the card, evicting by the policy in
// victim. [lo, hi) is the scan in progress, which is what makes the eviction
// choice optimal rather than a guess.
//
// The upload goes through resident(), the one path that asks the arena first:
// a retained block is a DMA of bytes already in the device layout, and one the
// arena declined is repacked and counted in Stats.Packs.
func (g *devTier) pageIn(li, lo, hi int) bool {
	if !g.pageInHold(li, lo, hi) {
		return false
	}
	g.mu.Lock()
	g.dropPages(li, li+1)
	g.mu.Unlock()
	return true
}

// pageInHold is pageIn for a submission about to read the block: it comes
// back held (holdPages), and the caller lets it go. Every wait on the way -- for the submissions in
// flight, for one reading the victims -- lets g.mu go, so another session may
// page the block in meanwhile, and the residency is asked again after them.
func (g *devTier) pageInHold(li, lo, hi int) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	l := g.layers[li]
	if l == nil || !l.ok {
		return false
	}
	if l.pg == nil || l.pg.in {
		return g.holdPages(li, li+1)
	}
	g.quiesce()
	if l.pg.in {
		return g.holdPages(li, li+1)
	}
	// The staging is asked for before the reclaim, best-effort: it is normally
	// already held and is absent only after trim took it. Without room the
	// page-in proceeds without it and packs on the host. Passing the bytes this
	// block still owes keeps the staging from taking the page-in's room.
	g.ensureRaw(l.pg.ws, l.pg.bytes)
	if need := l.pg.bytes + l.pg.round; !g.kvCompact(need, nil) && !g.reclaim(need, li, lo, hi) {
		g.LastErr = fmt.Sprintf("block %d needs %d bytes and nothing can be paged out (%d of %d used, %d slot(s))",
			li, l.pg.bytes+l.pg.round, g.used, g.limit, g.slots())
		return false
	}
	// Whatever the recording named is about to be joined by buffers that did not
	// exist when it was captured.
	g.dropGraph()
	// The room asked for above may have been waited for, and the block paged in
	// by another session meanwhile.
	if l.pg.in {
		return g.holdPages(li, li+1)
	}
	// The host bytes are asked for again before any is read: l.pg.ws holds
	// slices into a host page captured at admission, and the host pager may
	// since have reused that frame for another block, which would upload the
	// wrong weights. This is the one place that re-reads, in weightList order
	// so the refreshed set matches what tensors() frees.
	if l.pg.ensure != nil && !g.pagedFaulted(pagedStale) {
		var w nn.LayerWeights
		w.Ensure = l.pg.ensure
		if err := l.pg.ensure(&w); err != nil {
			g.LastErr = fmt.Sprintf("block %d page-in: %v", li, err)
			return false
		}
		got := weightList(&w)
		if len(got) != len(l.pg.ws) {
			g.LastErr = fmt.Sprintf("block %d page-in: the refreshed weight list is "+
				"%d long and the page's is %d", li, len(got), len(l.pg.ws))
			return false
		}
		// The shapes are checked: a refresh that returned a different block
		// would otherwise land as the right number of bytes in the right slots.
		for i, x := range got {
			if o := l.pg.ws[i]; x.Rows != o.Rows || x.K != o.K || x.T != o.T ||
				len(x.Data) != len(o.Data) {
				g.LastErr = fmt.Sprintf("block %d page-in: tensor %d refreshed as "+
					"%v %dx%d of %d bytes, admitted as %v %dx%d of %d",
					li, i, x.T, x.Rows, x.K, len(x.Data), o.T, o.Rows, o.K, len(o.Data))
				return false
			}
		}
		copy(l.pg.ws, got)
	}
	ts := l.tensors()
	// What a failed upload reports is this page-in's reason, not one an
	// earlier call left (a placement's first, refused pass sets LastErr).
	g.LastErr = ""
	for i, x := range l.pg.ws {
		if len(x.Data) == 0 {
			continue
		}
		q, ok := quantOf(x.T)
		if !ok {
			return false
		}
		r := g.resident(li, i, q, x.Data, x.Rows, x.K, x.Packed)
		if r == nil || !r.ok {
			if g.LastErr == "" {
				g.LastErr = fmt.Sprintf("block %d page-in: tensor %d not resident", li, i)
			}
			// A half-paged block is a block whose kernels would launch against
			// freed buffers, so it goes all the way back out.
			g.dropTensors(l)
			return false
		}
		*ts[i] = r
	}
	l.pg.in = true
	g.pagesIn++
	g.pageBytes += l.pg.bytes + l.pg.round
	g.PageIns++
	g.PageBytes += l.pg.bytes
	return g.holdPages(li, li+1)
}

// submit runs [lo, hi) as the fewest submissions the resident set allows,
// paging blocks in as it goes. It is the only way into layersOnce.
//
// On a model that fits this is one layersOnce call carrying the whole range and
// the head; the machinery costs hi-lo lookups. A paged range is cut where the
// next block is not resident, since a submission names its blocks' buffers:
// under the MRU policy that is one submission for the pinned prefix and one
// per streamed block, each one host/device crossing.
func (g *devTier) submit(sid uint64, bs *blockScratch, lo, hi, pos int, x, cs, csSWA []float32, head *nn.Head) bool {
	if lo >= hi {
		// An empty range, with or without a head: layersOnce owns that rule.
		return g.layersOnce(sid, bs, lo, hi, pos, x, cs, csSWA, head)
	}
	rows := 1
	if w := bs.p.ResidW(); w > 0 && len(x) > w {
		rows = len(x) / w
	}
	for li := lo; li < hi; {
		if !g.pageInHold(li, lo, hi) {
			return false
		}
		end := li + 1
		// PerLayerSubmit is the other arm of the submission comparison and it
		// asks for one block per Session, so it must not be coalesced away.
		// Otherwise a range is cut where the next block is not resident, and
		// where the run-time budget ends (subbudget.go). pageIn holds li; the
		// blocks the range takes on are held as they are taken, so no other
		// session's page-in sends one away before the submission reads it.
		if !g.PerLayerSubmit {
			g.mu.Lock()
			most := g.submitSpan(rows)
			for end < hi && end-li < most && g.holdPages(end, end+1) {
				end++
			}
			g.mu.Unlock()
		}
		h := (*nn.Head)(nil)
		if end >= hi {
			h = head
		}
		// Look-ahead belongs here and is not built: the next blocks' host reads
		// should be in flight during layersOnce, as an early EnsurePage into
		// the frame pool (jlm reads, it does not map).
		t0 := time.Now()
		ok := g.layersOnce(sid, bs, li, end, pos, x, cs, csSWA, h)
		g.mu.Lock()
		g.dropPages(li, end)
		if ok {
			g.noteSubmit(rows, end-li, time.Since(t0))
		}
		g.mu.Unlock()
		if !ok {
			return false
		}
		li = end
	}
	return true
}

// pinBytes is what this device's pinned blocks charge. Callers hold g.mu.
func (g *devTier) pinBytes() uint64 {
	var n uint64
	for li := range g.pinned {
		if l := g.layers[li]; l != nil && l.pg != nil {
			n += l.pg.bytes
		}
	}
	return n
}

// SetBudget retargets this device's weight budget while it is running, and
// reports how many blocks still have their weights resident. See GPU.SetBudget:
// shrinking pages weights out without changing which device owns a block, so no
// token moves; only a budget too small to page at all takes blocks away, and
// model/ sees that as Layers returning false. Callers must not hold g.mu.
func (g *devTier) SetBudget(bytes uint64) int {
	g.mu.Lock()
	defer g.mu.Unlock()
	if bytes > g.limit {
		g.roomGen.Add(1)
	}
	g.limit = bytes
	g.kvTight = false
	// Give back whatever the new budget no longer covers. A block paged out
	// here is demoted, not released: it keeps its owner and history and pages
	// back in on the next token if there is a slot.
	g.trim()
	// Below the slot floor the pages go all the way out, since there is
	// nothing to swap through: the next call returns false and model/'s
	// demotion path brings the KV history home.
	for g.slots() < minSlots && g.pagesIn > 0 {
		v := g.victim(-1, 0, 0, false)
		if v < 0 {
			break
		}
		g.pageOut(v)
	}
	// Below the slot floor nothing will page in again on this budget, so the
	// staging is pure held memory.
	if g.slots() < minSlots {
		g.freeRaw()
	}
	g.Slots = g.slots()
	return g.pagesIn
}

// ReleasesHostPage reports whether block li's weights are a real copy on the
// device, so the host may drop its own page of them rather than hold the block
// twice.
//
// It is false for a unified device, whose buffers alias host memory
// (backend.HostImport, resident.imported): dropping the host page there would
// be a use-after-free. It stays true under paging, because pageIn re-reads the
// block's bytes through nn.LayerWeights.Ensure, faulting the page back in.
//
// A streamed block is no: it reads its routed sheets from the host every
// token, and the expert pages they land in are that cache.
//
// An unknown answer is no: a caller that cannot ask must keep its page.
func (g *devTier) ReleasesHostPage(li int) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	l, ok := g.layers[li]
	if !ok {
		return false
	}
	if l.stream != nil {
		// A streamed block's base is on the card and its experts are read from
		// their own pages (a bank in expert pages hands sheets out through
		// Sheet), so the host's copy of its block page is dead weight -- 1.2
		// GiB a block on Kimi-K3. A bank kept in the block page (a float bank)
		// is read from there, and that page stays.
		return l.stream.w.Down.Packed != nil && l.stream.w.Down.Packed.Sheet != nil
	}
	any := false
	for _, pp := range l.tensors() {
		r := *pp
		if r == nil {
			continue
		}
		if !r.ok || r.imported {
			return false
		}
		any = true
	}
	return any
}
