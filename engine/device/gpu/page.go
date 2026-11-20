package gpu

import (
	"fmt"
	"time"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/nn"
	"github.com/jitllm/jitllm/sched"
)

// Block paging: a device holds N block slots and swaps, instead of declining a
// block and losing it for the rest of the run, so a card smaller than the model
// still runs every block. See docs/design/block-paging.md.
//
// A block's weights have a backing store (the packed arena, or the container),
// so they can leave the card and come back byte-identical. Its KV cache is this
// session's own history and nothing else holds a copy; norms and biases are
// kilobytes. So:
//
//	pageable    every matrix the block's graph binds, by role
//	permanent   the KV cache, the norms, the biases
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
	ws    []nn.Weight // the block's matrices, in layer.tensors() order
	bytes uint64      // what the set costs on the device while it is resident
	// round is the driver's page rounding of those buffers beyond bytes, which
	// a page-in also has to find room for (scratch.go).
	round uint64
	in    bool // resident right now
	// uses is how many submissions -- running, or paged in and about to run
	// (submit) -- read the block: it is no victim while any does, whichever
	// session's prologue is looking for room. Under mu.
	uses int
	// ensure re-points every slice in ws at where the card's frames (fr)
	// hold those bytes NOW, and is nil for a caller that does not page its
	// own weights. See nn.LayerWeights.Ensure.
	ensure func(*nn.LayerWeights, *jlm.Frames) error
	fr     *jlm.Frames
}

// tensors is the block's pageable weights in one fixed order (layer.mats),
// so that PrepLayer, ReleaseLayers and the pager cannot disagree about what a
// block is: a tensor missing from this list leaks on release and is never
// priced.
func (l *layer) tensors() []**resident {
	ts := make([]**resident, len(l.mats))
	for j := range l.mats {
		ts[j] = &l.mats[j].r
	}
	return ts
}

// pageSizeWhy is pageSize with the reason, naming the tensor and its format:
// ws is block l's matrices in l.mats' order.
func (g *devTier) pageSizeWhy(l *layer, ws []nn.Weight) (uint64, string, bool) {
	var total uint64
	for i, x := range ws {
		if len(x.Data) == 0 {
			continue
		}
		q, ok := quantOf(x.T)
		if !ok {
			return 0, fmt.Sprintf("%s is %s, which the device layout does not hold", l.mats[i].role, x.T), false
		}
		// At the buffers' footprint, as a page-in asks for them (pageIn): a
		// slot priced at the bytes alone counts slots the card does not have.
		f, err := g.planesFootprint(q, x.Rows, x.K)
		if err != nil {
			return 0, fmt.Sprintf("%s (%s %dx%d): %v", l.mats[i].role, x.T, x.Rows, x.K, err), false
		}
		total += f
	}
	return total, "", true
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
	// The slack is held back as room() holds it beside a page-in's bytes:
	// a slot it would refuse is no slot.
	perm := g.slack()
	if g.used > g.pageBytes {
		perm += g.used - g.pageBytes
	}
	perm += g.reserved()
	avail := uint64(0)
	if g.limit > perm {
		avail = g.limit - perm
	}
	if pl := g.pool.Limit(); pl > 0 {
		// What the pool would allow, plus what this device's own slots already
		// hold -- those bytes come back to the pool when a page goes out.
		free := g.pageBytes + g.claimed + g.swapHeld
		if u := g.pool.Used(); pl > u {
			free += pl - u
		}
		if free < avail {
			avail = free
		}
	}
	return int(avail / g.widest)
}

// holdsPages says a block placed on the card pages through its slots, paged
// in or out. Callers hold g.mu.
func (g *devTier) holdsPages() bool {
	for _, l := range g.layers {
		if l != nil && l.pg != nil {
			return true
		}
	}
	return false
}

// slotsIfPerm is slots() with n more bytes held permanently: what the pager
// would have left if something that is not a page took that space. The output
// projection and a block slot compete for the same bytes; see PrepHead.
func (g *devTier) slotsIfPerm(n uint64) int {
	if g.widest == 0 {
		return 0
	}
	g.settleRounding() // see room

	perm := n + g.slack()
	if g.used > g.pageBytes {
		perm += g.used - g.pageBytes
	}
	avail := uint64(0)
	if g.limit > perm {
		avail = g.limit - perm
	}
	if pl := g.pool.Limit(); pl > 0 {
		free := g.pageBytes + g.claimed + g.swapHeld
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
// (sched.NextUse, shared with the host ring; see sched/cyclic.go): LRU is
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
	if g.headVictim(cur, lo, hi, best, bestD) {
		return headSlot
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
			// One buffer behind several blocks: freed with the last of them,
			// or it outlives the model on the card (devTier.sharedCompact).
			*rp = nil
			freed += g.letGo(r)
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
	// A block given an expert cache after placement (cacheBlock) keeps the
	// bank it was placed on beside it, where no tensor slot names it.
	if st := l.stream; st != nil && st.cached() {
		for m, r := range st.plain {
			switch {
			case r == nil || !r.ok:
			case r.shared:
				freed += g.letGo(r)
			default:
				n := r.bytes()
				g.refund(n)
				freed += n
				g.freeResident(r)
			}
			st.plain[m] = nil
		}
		st.uncache()
	}
	return freed
}

// letGo is one block letting go of shared bank r: the last frees it and
// refunds its bytes, which it reports. Callers hold g.mu.
func (g *devTier) letGo(r *resident) uint64 {
	if r.users--; r.users > 0 {
		return 0
	}
	for k, v := range g.res {
		if v == r {
			delete(g.res, k)
			break
		}
	}
	n := r.planes()
	g.refund(n)
	g.freeResident(r)
	return n
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
	if li == headSlot {
		return g.pageOutHead()
	}
	l := g.layers[li]
	if l == nil || l.pg == nil || !l.pg.in {
		return 0
	}
	// A recording names device addresses, which stop meaning anything once
	// these buffers are freed. See dropGraph.
	g.dropGraph()
	g.keepBack = g.keepOf(li)
	freed := g.dropTensors(l)
	// A page narrower than the slot leaves it short: the rest is taken from
	// the pool while it is free.
	if g.keepBack > 0 && g.pool.claim(g.keepBack) {
		g.keepFreed(g.keepBack)
	}
	g.keepBack = 0
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

// streamForRoom makes room for n bytes of history where neither growth nor
// sending cold history home can (allocPages), by streaming blocks: the
// highest resident ones no one pinned or is reading, until n leaves them a
// slot to swap through (swapRoom), then the pager takes it (reclaim). One
// streamed block frees nothing -- it needs its slot back to run. Nothing stays
// marked when they cannot cover n. Callers hold g.mu, outside any submission.
func (g *devTier) streamForRoom(n uint64) bool {
	if g.stream == nil {
		g.stream = map[int]bool{}
	}
	var marked []int
	for li := len(g.layers) - 1; li >= 0 && (len(marked) < 2 || !g.swapRoom(n)); li-- {
		l := g.layers[li]
		if l == nil || l.pg == nil || !l.pg.in || l.pg.bytes == 0 || l.pg.uses > 0 || g.pinned[li] || g.stream[li] {
			continue
		}
		g.stream[li] = true
		marked = append(marked, li)
	}
	if len(marked) < 2 || !g.swapRoom(n) || !g.reclaim(n, -1, 0, 0) {
		for _, li := range marked {
			if l := g.layers[li]; l != nil && l.pg != nil && l.pg.in {
				delete(g.stream, li)
			}
		}
		return false
	}
	g.Slots = g.slots()
	return true
}

// swapRoom reports that n more bytes held for good leave the streamed blocks
// a slot to swap through: everything held but their paged-in pages, n, the
// slack every page-in leaves free (room) and the widest of them inside the
// budget. slots() divides the budget by the widest
// block of all and counts every paged-in block as swap space, which a block
// resident and unstreamed is not. Callers hold g.mu.
func (g *devTier) swapRoom(n uint64) bool {
	var in, widest uint64
	for li := range g.stream {
		l := g.layers[li]
		if l == nil || l.pg == nil {
			continue
		}
		b := l.pg.bytes + l.pg.round
		widest = max(widest, b)
		// A page a submission reads is no swap space until it lets go.
		if l.pg.in && l.pg.uses == 0 {
			in += b
		}
	}
	if widest == 0 {
		return true
	}
	held := g.used - min(g.used, in) + g.reserved() + g.slack()
	return held+n+widest <= g.limit
}

// keepsSlot reports that n more bytes held for good leave a device that
// streams the minSlots its streamed blocks swap through. It is asked whether or
// not n fits right now: room() counts the pages in as space, so state charged
// into it while a streamed block was paged in left the next page-in nothing --
// TestPrefillStreamsCPUBlocks on a discrete card, where a prompt's sixteen
// streamed histories took the one slot and nothing ever paged in. Callers hold
// g.mu.
func (g *devTier) keepsSlot(n uint64) bool {
	return len(g.stream) == 0 || g.widest == 0 || g.slotsIfPerm(n+g.reserved()) >= minSlots && g.swapRoom(n)
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

// stateHold is stateFits with the room held (claim) for the charges the
// caller is about to make: another device of the pool runs beside this one
// (GPU.pipeline) and spent the room found between the check and the charge.
// The room is made again when the other took it first. It returns the hold
// to restore (unclaim) once the charges are made, or why there is no room.
// Callers hold g.mu.
func (g *devTier) stateHold(n uint64, keep *kvLayerPool) (uint64, error) {
	return g.holdFor(n, keep, true)
}

// holdFor is stateHold, asking keepsSlot only when slot is set: a block's
// first pages are held to the slot by its admission (addKVLayer).
func (g *devTier) holdFor(n uint64, keep *kvLayerPool, slot bool) (uint64, error) {
	for g.kvCompact(n, keep) || g.roomOrReclaim(n) || g.fromPeers(n, false) {
		if slot && !g.keepsSlot(n) {
			break
		}
		if before, ok := g.claim(n); ok {
			return before, nil
		}
	}
	return g.claimed, g.shortOf(n)
}

// roomHold is room with the pool's bytes taken in the same step (claim).
// Callers hold g.mu.
func (g *devTier) roomHold(n uint64) (uint64, bool) {
	if !g.room(n) {
		return g.claimed, false
	}
	return g.claim(n)
}

// seatHold is a session's seat's room (roomSeat), or the room made as for a
// history growing, held (stateHold). Callers hold g.mu.
func (g *devTier) seatHold(need uint64, live int, hist uint64) (uint64, error) {
	if g.roomSeat(need, live, hist) {
		if before, ok := g.claim(need); ok {
			return before, nil
		}
	}
	return g.stateHold(need, nil)
}

// shortOf says why n bytes do not fit: the pool this device's memory comes
// from, or its own budget.
func (g *devTier) shortOf(n uint64) error {
	if g.pool != nil && g.pool.Limit() > 0 && g.pool.Used()+n > g.pool.Limit() {
		return fmt.Errorf("its pool %s has %d of %d bytes used by its devices", g.pool.Name(), g.pool.Used(), g.pool.Limit())
	}
	return fmt.Errorf("%d of %d bytes used", g.used, g.limit)
}

// fromPeers pages out the other devices' streamed blocks of this device's
// pool until n fits, and reports whether it does: a block no submission
// reads, on a peer with nothing in flight, and never a peer's last one in
// (keepOf). A busy peer is waited for, its lock and then its submissions in
// flight (drain), in lockPair's order: a peer after this device is waited
// for holding g.mu, since whoever holds it waits for nothing this device
// holds; for a peer before it g.mu is let go first (Stats.PeerLetGo), since
// its holder may be waiting for g.mu and its submissions end under its own
// lock, and taken again after the peer's. Where g.mu may not be let go
// (mayLetGo) a busy peer before it is passed over (Stats.PeerPassed). n fits
// this device's budget and its pool (room), or the pool alone where pool is
// set (a scratch window's hold, holdWin). Callers hold g.mu.
func (g *devTier) fromPeers(n uint64, pool bool) bool {
	for _, p := range g.peers {
		if g.fitsFor(n, pool) {
			return true
		}
		pd := p.dv
		after := pd.ord > g.ord
		let := !after && g.mayLetGo()
		switch {
		case after:
			p.mu.Lock()
		case p.mu.TryLock():
		case let:
			g.PeerLetGo++
			g.mu.Unlock()
			p.mu.Lock()
			g.mu.Lock()
		default:
			g.PeerPassed++
			continue
		}
		if len(pd.tickets) != 0 && !pd.prologueHeld {
			switch {
			case after:
				pd.drain()
			case let:
				g.PeerLetGo++
				g.mu.Unlock()
				pd.drain()
				g.mu.Lock()
			}
		}
		if len(pd.tickets) != 0 || pd.inSub || pd.prologueHeld {
			g.PeerPassed++
		} else {
			for !g.fitsFor(n, pool) {
				v := pd.victim(-1, 0, 0, true)
				if v < 0 || v == headSlot || pd.keepOf(v) > 0 {
					break
				}
				pd.pageOut(v)
				g.PeerPageOuts++
			}
			pd.Slots = pd.slots()
		}
		p.mu.Unlock()
	}
	return g.fitsFor(n, pool)
}

// mayLetGo reports that g.mu may be let go and taken again in the middle of
// the caller's operation, as reclaim's wait lets it go: not inside a scratch
// window, which measures what the device allocates until its close, nor in
// a submission or its prologue (devSess.inSub, prologueHeld), nor a hold
// taken with another device's (paired), whose holder may be this caller.
// Callers hold g.mu.
func (g *devTier) mayLetGo() bool {
	return g.winDepth == 0 && !g.inSub && !g.prologueHeld && g.paired == 0
}

// fitsFor is room, or the pool's alone where pool is set (fromPeers).
func (g *devTier) fitsFor(n uint64, pool bool) bool {
	if pool {
		return g.pool.room(n)
	}
	return g.room(n)
}

// keepOf is what the page-out of block li keeps in the pool rather than
// giving it back: an operation paging into the slot (openSlot) keeps what
// it asked for and the slot past it, and a device that streams keeps its
// slot (slotKeep). Only in a pool another device spends (one card under two
// backends): on its own a device keeps its slot in its budget (keepsSlot).
// Nothing is kept past a budget lowered under the device (trim). Callers
// hold g.mu.
func (g *devTier) keepOf(li int) uint64 {
	if g.pool == nil || g.pool.devs < 2 || g.over() {
		return 0
	}
	k := g.slotKeep(li)
	if g.slotNeed > 0 {
		// The page coming in fills the slot as far as it is wide.
		k -= min(k, g.slotNeed-min(g.slotNeed, g.reserved()+g.slack()))
		k += g.slotNeed
		return k - min(k, g.claimed)
	}
	return k - min(k, g.swapHeld)
}

// slotKeep is what a device that streams keeps of its pool (swapHeld) for
// its next page-in, with block li paged out: what a page-in of its widest
// streamed block asks (pageInHold's openSlot), less the streamed pages in,
// which give theirs back when they go. The bytes of the last page out were
// not it: a narrow page out left a wide one's page-in short, and neither
// held the rounding and slack the page-in asks. Callers hold g.mu.
func (g *devTier) slotKeep(li int) uint64 {
	if g.pool == nil || g.pool.devs < 2 || g.over() {
		return 0
	}
	var w, in uint64
	for s := range g.stream {
		l := g.layers[s]
		if l == nil || l.pg == nil {
			continue
		}
		b := l.pg.bytes + l.pg.round
		w = max(w, b)
		if s != li && l.pg.in {
			in += b
		}
	}
	if w == 0 {
		return 0
	}
	w += g.reserved() + g.slack()
	return w - min(w, in)
}

// openSlot begins an operation paging into the swap slot: the slot kept
// since the last page-out (swapHeld) becomes its hold, and its page-outs
// keep up to need there rather than giving the bytes to the pool, where
// another device would spend them before the uploads. It returns the hold
// closeSlot restores. Callers hold g.mu.
func (g *devTier) openSlot(need uint64) uint64 {
	open := g.claimed
	g.claimed += g.swapHeld
	g.swapHeld = 0
	g.slotNeed = need
	return open
}

// closeSlot ends what openSlot began, keeping the slot (slotKeep) of what
// the operation did not spend and giving the pool back the rest. Callers
// hold g.mu.
func (g *devTier) closeSlot(open uint64) {
	g.slotNeed = 0
	if k := g.slotKeep(-1); k > g.swapHeld && g.claimed > open {
		m := min(k-g.swapHeld, g.claimed-open)
		g.claimed -= m
		g.swapHeld += m
	}
	g.unclaim(open)
}

// dropSwap gives the pool the slot this device kept. Callers hold g.mu.
func (g *devTier) dropSwap() {
	g.pool.give(g.swapHeld)
	g.swapHeld = 0
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
	// Expert caches before blocks: a cache is copies of host pages, and its
	// block keeps running on the plain bank without it.
	if g.over() {
		g.dropSwap()
		g.dropCaches(0)
	}
	for g.over() && g.pagesIn > 0 {
		v := g.victim(-1, 0, 0, false)
		if v < 0 {
			break
		}
		g.pageOut(v)
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
	defer g.closeSlot(g.openSlot(l.pg.bytes + l.pg.round + g.reserved() + g.slack()))
	// The staging is asked for before the reclaim, best-effort: it is normally
	// already held and is absent only after trim took it. Without room the
	// page-in proceeds without it and packs on the host. Passing the bytes this
	// block still owes keeps the staging from taking the page-in's room.
	if need := l.pg.bytes + l.pg.round; !g.kvCompact(need, nil) && !g.reclaim(need, li, lo, hi) && !g.fromPeers(need, false) {
		// reclaim's wait for a held victim lets g.mu go: another session may
		// have paged this block in meanwhile, into the very room asked for.
		if l.pg.in {
			return g.holdPages(li, li+1)
		}
		g.LastErr = fmt.Sprintf("block %d needs %d bytes and nothing can be paged out (%d of %d used, %d slot(s))",
			li, l.pg.bytes+l.pg.round, g.used, g.limit, g.slots())
		g.noteCause(nn.FailRoom)
		return false
	}
	// The room is held from here to the last tensor's upload: another device
	// of the pool runs beside this one (GPU.pipeline) and would otherwise
	// spend it between the reclaim and the uploads.
	before, held := g.claim(l.pg.bytes + l.pg.round + g.reserved() + g.slack())
	if !held {
		if need := l.pg.bytes + l.pg.round; g.reclaim(need, li, lo, hi) || g.fromPeers(need, false) {
			before, held = g.claim(l.pg.bytes + l.pg.round + g.reserved() + g.slack())
		}
		if !held {
			if l.pg.in {
				return g.holdPages(li, li+1)
			}
			g.LastErr = fmt.Sprintf("block %d needs %d bytes and its pool %s has %d of %d used by its devices",
				li, l.pg.bytes+l.pg.round, g.pool.Name(), g.pool.Used(), g.pool.Limit())
			g.noteCause(nn.FailRoom)
			return false
		}
	}
	defer g.unclaim(before)
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
	// wrong weights. This is the one place that re-reads, in layer.mats order
	// so the refreshed set matches what tensors() frees.
	if l.pg.ensure != nil && !g.pagedFaulted("stale") {
		// Pinned across the upload below: the frame is the source, and a read
		// for another block into the same frames would otherwise evict it.
		if fr := l.pg.fr; fr != nil {
			pin, err := fr.Hold(li, nil)
			if err != nil {
				g.LastErr = fmt.Sprintf("block %d page-in: %v", li, err)
				return false
			}
			defer pin.Release()
		}
		var w nn.LayerWeights
		w.Ensure = l.pg.ensure
		if err := l.pg.ensure(&w, l.pg.fr); err != nil {
			g.LastErr = fmt.Sprintf("block %d page-in: %v", li, err)
			return false
		}
		got := l.pageList(&w)
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
		r := g.resident(li, i, nil, q, x.Data, x.Rows, x.K, x.Packed)
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
func (g *devTier) submit(sid uint64, bs *blockScratch, lo, hi, pos int, x []float32, cs nn.RopeTables, head *nn.Head) bool {
	if lo >= hi {
		// An empty range, with or without a head: layersOnce owns that rule.
		if head != nil && !g.headHold(hi, lo, hi, true) {
			return false
		}
		ok := g.layersOnce(sid, bs, lo, hi, pos, x, cs, head)
		if head != nil {
			g.headLetGo()
		}
		return ok
	}
	// apart is a streamed projection that found no room beside the run
	// carrying it: that run goes alone and the head after it, a call of its
	// own holding nothing else.
	apart := false
	rows := 1
	if w := bs.p.ResidW(); w > 0 && len(x) > w {
		rows = len(x) / w
	}
	// A marked block's second step writes its marked half: copied first.
	g.mu.Lock()
	g.keepMarkHalves(sid, lo, hi)
	g.mu.Unlock()
	for li := lo; li < hi; {
		if !g.pageInHold(li, lo, hi) {
			return false
		}
		end := li + 1
		// PerLayerSubmit is the other arm of the submission comparison and it
		// asks for one block per Session, so it must not be coalesced away.
		// Otherwise a range is cut where the next block is not resident, and
		// where the run-time budget ends (subbudget.go), and where the blocks'
		// key-run rule changes over a picture (cutsRuns). pageIn holds li; the
		// blocks the range takes on are held as they are taken, so no other
		// session's page-in sends one away before the submission reads it.
		if !g.PerLayerSubmit {
			g.mu.Lock()
			most := g.submitSpan(rows)
			for end < hi && end-li < most && !g.cutsRuns(li, end) && g.holdPages(end, end+1) {
				end++
			}
			g.mu.Unlock()
		}
		h := (*nn.Head)(nil)
		if end >= hi {
			h = head
		}
		if h != nil && !g.headHold(li, lo, hi, false) {
			h, apart = nil, true
		}
		// Look-ahead belongs here and is not built: the next blocks' host reads
		// should be in flight during layersOnce, as an early EnsurePage into
		// the frame pool (jlm reads, it does not map).
		t0 := time.Now()
		ok := g.layersOnce(sid, bs, li, end, pos, x, cs, h)
		if h != nil {
			g.headLetGo()
		}
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
	if apart {
		return g.submit(sid, bs, hi, hi, pos, x, cs, head)
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
		g.forgetRefusals()
	}
	lowered := bytes < g.limit
	g.limit = bytes
	g.kvTight = false
	// A lowered budget is paid from the KV pools' free pages first: every
	// layer gives back what its sequences do not hold (shrinkKVLayer), so a
	// history that grows past the budget evicts rather than filling pages the
	// budget no longer covers.
	if lowered && g.kvp != nil {
		for _, l := range g.kvp.layers {
			if l == nil {
				continue
			}
			l.fence()
			if err := g.shrinkKVLayer(g.kvp, l); err != nil {
				g.LastErr = err.Error()
			}
		}
	}
	// Give back whatever the new budget no longer covers. A block paged out
	// here is demoted, not released: it keeps its owner and history and pages
	// back in on the next token if there is a slot.
	g.trim()
	// Below the slot floor the pages go all the way out, since there is
	// nothing to swap through: the next call returns false and model/'s
	// demotion path moves the KV history to the CPU.
	for g.slots() < minSlots && g.pagesIn > 0 {
		v := g.victim(-1, 0, 0, false)
		if v < 0 {
			break
		}
		g.pageOut(v)
	}
	g.Slots = g.slots()
	// A raised budget is the tier tuned, outside any step: the per-row head
	// a squeeze held back for room forms in the room it makes.
	if !lowered {
		g.prepRowsReserved()
	}
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
		d := l.stream.w.Mat(jlm.RoleExpDownBank)
		return d.Packed != nil && d.Packed.Sheet != nil
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
