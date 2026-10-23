package tier

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// The device's own buffers -- the block scratches, the batched prompt's and
// the ragged step's among them, and the staging -- are charged to the budget
// the blocks use, and the widths the device will run are reserved when its
// first block is placed. A prompt then finds its scratch already there,
// rather than asking a card the blocks have filled for hundreds of
// megabytes and going a row at a time when it is refused.
//
// The charge is MEASURED: a window around everything that builds or frees a
// scratch reads the device's allocation count (backend.AllocCounter) before
// and after, and the difference -- less whatever charge and refund moved
// inside the window, which is a weight or a page already priced -- is the
// scratch's. Pricing each buffer by hand is the walk that fell behind every
// field added since (see freeScratch).

// allocated is the bytes the device's buffers hold, or 0 on a device that
// does not count them, where scratch is then not charged.
func (g *devTier) allocated() int64 {
	if g.cnt != nil {
		return int64(g.cnt.Allocated())
	}
	return 0
}

// footprint is what one n-byte buffer costs the card: n rounded up to the
// driver's granularity (backend.AllocCounter.FootprintOf), n where the device
// does not say.
func (g *devTier) footprint(n int) uint64 {
	if g.cnt != nil {
		return g.cnt.FootprintOf(n)
	}
	return uint64(n)
}

// planesFootprint is what a packed q tensor of rows x k costs the budget in
// its three buffers, rounding included.
func (g *devTier) planesFootprint(q kernels.Quant, rows, k int) (uint64, error) {
	qs, d, sc, err := kernels.PackedWords(q, rows, k)
	if err != nil {
		return 0, err
	}
	return g.planeFootprint(qs) + g.planeFootprint(d) + g.planeFootprint(sc), nil
}

// planeFootprint is one plane of words. An empty plane (Q4_0's and Q8_0's
// scales, say) is still a buffer: the upload hands the kernel a word for it
// (resident, allocBytes) and charges nothing, so what the budget spends on it
// is that word's rounding -- a page less four bytes on a card that rounds to
// 64 KiB, nothing on one that does not.
func (g *devTier) planeFootprint(words int) uint64 {
	if words == 0 {
		return g.footprint(4) - 4
	}
	return g.footprint(words * 4)
}

// settleRounding charges (or refunds) what the driver's page rounding costs
// beyond the bytes asked for (backend.AllocCounter's Footprint less
// Allocated): every buffer's charge, explicit or measured, is its size, and
// the card spends more. It runs after every charge, refund and scratch
// window, so it follows the allocations those account for. Callers hold g.mu.
func (g *devTier) settleRounding() {
	c := g.cnt
	if c == nil {
		return
	}
	d := int64(c.Footprint()) - int64(c.Allocated()) - int64(g.rounding)
	switch {
	case d > 0:
		g.rounding += uint64(d)
		g.used += uint64(d)
		g.pool.Take(uint64(d))
	case d < 0:
		n := min(uint64(-d), g.rounding, g.used)
		g.rounding -= n
		g.used -= n
		g.pool.Give(n)
		g.roomGen.Add(1)
	}
}

// scratchWin opens a window, closed by the mark it returns: what the device
// allocated or freed in between, less the explicit charges, is charged to (or
// refunded from) the scratch. Windows nest; only the outermost measures.
// Callers hold g.mu.
func (g *devTier) scratchWin() scratchMark {
	g.winDepth++
	if g.winDepth > 1 {
		return scratchMark{g: g, inner: true}
	}
	return scratchMark{g: g, a: g.allocated(), x: g.explicit}
}

// scratchMark is an open window: a value, so a deferred close allocates
// nothing on the paths a step runs every time (prepBatch, prepRagged).
type scratchMark struct {
	g     *devTier
	a, x  int64
	inner bool
}

// close ends the window, settling the outermost one.
func (m scratchMark) close() {
	g := m.g
	g.winDepth--
	if m.inner {
		return
	}
	g.settleScratch(g.allocated() - m.a - (g.explicit - m.x))
	g.settleRounding()
}

// settleScratch moves d bytes into (or out of) the scratch's share of both
// accounts. It does not touch explicit: it is not an explicit charge.
func (g *devTier) settleScratch(d int64) {
	switch {
	case d > 0:
		g.scratch += uint64(d)
		g.used += uint64(d)
		g.pool.Take(uint64(d))
	case d < 0:
		n := min(uint64(-d), g.scratch, g.used)
		g.scratch -= n
		g.used -= n
		g.pool.Give(n)
		g.roomGen.Add(1)
	}
}

// dropScratch frees bs and refunds what it held. Callers hold g.mu.
func (g *devTier) dropScratch(bs *blockScratch) {
	defer g.scratchWin().close()
	freeScratch(bs, g.devShared)
}

// dropBatch frees the batched scratch of one width.
func (g *devTier) dropBatch(w int) {
	if bs := g.bbs[w]; bs != nil {
		g.dropScratch(bs)
		delete(g.bbs, w)
	}
	if g.promptW == w {
		g.promptW = 0
	}
	if g.stepW == w {
		g.stepW = 0
	}
	g.forgetExact(w)
}

// maxExactWidths bounds the exact prompt widths a lane keeps beside its
// reservation. Each is a scratch the budget charged when it was built; the
// bound only stops a stream of prompts of every length from holding a
// scratch per grain width.
const maxExactWidths = 4

// forgetExact removes w from the exact widths kept.
func (g *devTier) forgetExact(w int) {
	for i, x := range g.exactWs {
		if x == w {
			g.exactWs = append(g.exactWs[:i], g.exactWs[i+1:]...)
			return
		}
	}
}

// keepExact builds the exact prompt width w if it is not built and marks it
// the most recently used, giving back the least recently used exact widths
// when the lane holds maxExactWidths or when the budget refuses w beside
// them. It returns the width the chunk runs in (w, or a kept wider width for
// a w given back before), 0 for none. Prompts whose lengths straddle a grain (a stream of 31- to 34-token
// requests runs at 32 and 64 rows) then build each width once, not once per
// crossing. Callers hold g.mu, outside any submission.
func (g *devTier) keepExact(w int) int {
	reserved := w == g.promptW || w == g.stepW
	if g.bbs[w] == nil {
		drop := func() {
			// After every submission that may read it.
			if err := g.quiesce(); err != nil {
				g.LastErr = err.Error()
				return
			}
			if g.exactDropped == nil {
				g.exactDropped = map[int]bool{}
			}
			g.exactDropped[g.exactWs[0]] = true
			g.dropBatch(g.exactWs[0])
		}
		// A width given back before runs in the narrowest kept wider one:
		// prompts of more lengths than the lane keeps otherwise cycled every
		// scratch, building and dropping one a prompt, and a warm prompt
		// allocated.
		if g.exactDropped[w] {
			c := 0
			for _, x := range g.exactWs {
				if x > w && (c == 0 || x < c) {
					c = x
				}
			}
			if c != 0 {
				g.forgetExact(c)
				g.exactWs = append(g.exactWs, c)
				return c
			}
		}
		for len(g.exactWs) >= maxExactWidths {
			drop()
		}
		for !g.prepBatch(w) {
			if len(g.exactWs) == 0 {
				return 0
			}
			drop()
		}
	}
	if !reserved {
		g.forgetExact(w)
		g.exactWs = append(g.exactWs, w)
	}
	return w
}

// dropAllScratch frees every scratch and the staging: what a device holding
// no block and no head keeps of its own. Callers hold g.mu and drop the graph.
func (g *devTier) dropAllScratch() {
	defer g.scratchWin().close()
	g.dropLanes()
	g.dropLaneScratch()
	for _, b := range []*backend.Buf{&g.aBuf, &g.axBuf, &g.outBuf, &g.xfBuf, &g.mvPart} {
		if *b != nil {
			(*b).Free()
			*b = nil
		}
	}
	g.aCap, g.axCap, g.outCap, g.xfCap, g.mvPartCap = 0, 0, 0, 0, 0
	g.shapeChanged()
}

// dropLaneScratch frees this view's lane: every set's scratches and the
// lane's staging. Callers hold g.mu inside a scratch window, outside any
// Session.
func (g *devTier) dropLaneScratch() {
	g.dropOtherGeom()
	for w := range g.bbs {
		g.dropBatch(w)
	}
	freeScratch(g.bs, g.devShared)
	g.bs = nil
	for _, b := range []*backend.Buf{&g.partBuf, &g.f16Buf} {
		if *b != nil {
			(*b).Free()
			*b = nil
		}
	}
	g.partCap, g.f16Cap = 0, 0
	for c := range g.valBufs {
		if g.valBufs[c] != nil {
			g.valBufs[c].Free()
			g.valBufs[c] = nil
		}
		g.valCaps[c] = 0
	}
}

// overBudget reports whether what is charged has passed the budget, the
// seats reserved for sessions included (room's test with nothing more).
func (g *devTier) overBudget() bool { return !g.room(0) }

// stepRows is the ragged step width reserved for, 0 for none.
func (g *devTier) stepRows() int {
	if g.StepRows <= 0 {
		return 0
	}
	return roundUp(min(g.StepRows, batchWidth), batchGrain)
}

func roundUp(n, to int) int { return (n + to - 1) / to * to }

// reserveScratch builds, at placement, the batched scratch a prompt chunk and
// a ragged step will run in, so the blocks placed after it are budgeted
// around it. The prompt width is halved until it fits: a narrower chunk is
// a prompt that takes more submissions, where no scratch at all is one that
// goes a row at a time. The step's width is the server's and is not halved
// -- a step of more rows than were reserved is refused whatever this does --
// so a step that does not fit is reported and reserved for nothing.
//
// need is what the block being placed will charge: the reservation leaves room
// for it, so the first block is never the one the scratch pushes off the card
// -- on a paging device, the one slot every block swaps through.
//
// Callers hold g.mu and have built g.bs.
func (g *devTier) reserveScratch(p *nn.LayerPlan, need uint64) {
	if g.NoScratchReserve || g.bs == nil || p.NonCausal {
		return
	}
	g.reserving = true
	defer func() { g.reserving = false }()
	want := batchWidth
	if g.PromptRows > 0 {
		want = roundUp(min(g.PromptRows, batchWidth), batchGrain)
	}
	if sw := g.stepRows(); sw > 0 && g.stepW != sw {
		if g.prepBatch(sw) && g.reserveRowsHead(g.bbs[sw], p) && g.reservePaged(g.bbs[sw]) && g.room(need) {
			g.stepW = sw
		} else {
			g.LastErr = fmt.Sprintf("a %d-row ragged step's scratch does not fit the budget beside the decode scratch (%d of %d bytes charged)",
				sw, g.used, g.limit)
			g.dropBatch(sw)
		}
	}
	if g.promptW >= want {
		return
	}
	for w := want; w >= batchGrain; w = w / 2 / batchGrain * batchGrain {
		if g.prepBatch(w) && g.reservePaged(g.bbs[w]) && g.room(need) {
			g.promptW = w
			return
		}
		if w != g.stepW {
			g.dropBatch(w)
		}
	}
}

// reservePaged builds bs's paged attention plan for the deepest history it
// can reach, so its partials and planes are at their widest before the blocks
// are placed: they grow with the key count (pagedPlanFor), and a chunk deep in
// a conversation would otherwise ask a full card for them. Shallower plans
// then compile into buffers already that size. Callers hold g.mu.
//
// It builds one plan, at the deepest keys: a plan whose split count fell as
// its keys grew would still grow its partials at the step; walk every key
// count if one ever does.
func (g *devTier) reservePaged(bs *blockScratch) bool {
	if bs == nil {
		return false
	}
	return true
}

// reserveRowsHead allocates a ragged step's per-row logits and tokens at
// placement, sized from the plan's vocabulary because the head arrives after
// the blocks (prepRowsHead reuses them once it does). A plan that does not say
// its vocabulary reserves nothing here and builds them at the first step.
func (g *devTier) reserveRowsHead(bs *blockScratch, p *nn.LayerPlan) bool {
	if bs == nil {
		return false
	}
	if p.Vocab <= 0 || bs.ragLogits != nil {
		return true
	}
	defer g.scratchWin().close()
	var err error
	if bs.ragLogits, err = g.dev.Alloc(bs.rows * p.Vocab * 4); err == nil {
		bs.ragVocab = p.Vocab
		bs.ragTok, err = g.dev.Alloc(bs.rows * 4)
	}
	// The capped twin a final softcap writes (prepRowsHead compiles the cap).
	if err == nil && p.FinalSoftcap != 0 {
		bs.ragCap, err = g.dev.Alloc(bs.rows * p.Vocab * 4)
	}
	if err != nil {
		freeRagOut(bs)
		g.LastErr = "rows: " + err.Error()
		return false
	}
	return true
}

// freeRagOut frees a ragged step's per-row logits and tokens, and the capped
// logits with the cap built for their width.
func freeRagOut(bs *blockScratch) {
	for _, b := range []*backend.Buf{&bs.ragLogits, &bs.ragTok, &bs.ragCap} {
		if *b != nil {
			(*b).Free()
			*b = nil
		}
	}
	if bs.ragCapK != nil {
		bs.ragCapK.Close()
		bs.ragCapK = nil
	}
	bs.ragVocab, bs.ragCapC = 0, 0
}

// batchFor is the batched scratch a chunk or step of width w runs in: its own
// width's when it is built; else the power of two at or above it, built when
// it fits the budget (prepBatch refuses one that does not); else the narrowest
// built scratch wider than w, which pads. It returns the width to use, 0 for
// none. Callers hold g.mu.
//
// Widths go in powers of two so a step loop whose row count moves every step
// builds a handful of scratches, not one per multiple of eight: on a V100 the
// widths of 64 rows' steps grew the scratch by 2.6 GiB and then, with the KV
// pool grown into the rest, every width after was refused and a 64-row decode
// step ran padded to the 512-row scratch, eight times its rows.
func (g *devTier) batchFor(w int) int {
	if w > 64 && (g.roomFaulted(roomNarrow) || g.roomFaulted(roomNarrowLie)) {
		g.LastErr = fmt.Sprintf("a %d-row batched scratch does not fit the budget (roomFault)", w)
		return 0
	}
	if g.bbs[w] != nil && g.prepBatch(w) {
		return w
	}
	b := w
	for b&(b-1) != 0 {
		b += b & -b
	}
	if b <= batchWidth && g.prepBatch(b) {
		return b
	}
	best := 0
	for bw := range g.bbs {
		if bw > w && (best == 0 || bw < best) {
			best = bw
		}
	}
	if best > 0 && g.prepBatch(best) {
		return best
	}
	for _, r := range []int{g.promptW, g.stepW} {
		if r > w && g.prepBatch(r) {
			return r
		}
	}
	return 0
}

// prewarmPromptWidths builds, at the end of a placement, the kernels of every
// grain width below the reserved prompt width (layersFor's exact prompt
// scratch), so a prompt of any length compiles nothing: at most
// promptW/batchGrain widths, each built and dropped again, its kernels kept in
// the shape-keyed caches. A device with no reserved width, or the exact width
// switched off, has nothing to warm.
func (g *GPU) prewarmPromptWidths() {
	for _, d := range g.devs {
		d.mu.Lock()
		d.prewarmPromptWidths()
		d.mu.Unlock()
	}
}

// prewarmPromptWidths is GPU.prewarmPromptWidths for one device. Callers hold
// g.mu.
func (g *devTier) prewarmPromptWidths() {
	if g.NoPromptPrewarm {
		return
	}
	g.prewarmPrompt()
	grain := g.promptGrain()
	if g.NoExactPrompt || g.promptW <= grain {
		return
	}
	for w := grain; w < g.promptW; w += grain {
		if g.bbs[w] != nil || w == g.stepW {
			continue
		}
		if g.prepBatch(w) {
			g.PromptWidthsWarmed++
			g.prewarmFused(g.bbs[w])
		}
		g.dropBatch(w)
	}
}

// noRoomKey is a batched width of one geometry set.
type noRoomKey struct {
	geo   geoKey
	width int
}

// budgetAt is a device's budget when a batched width did not fit it: the bytes
// charged, the limit and the history promised to admitted rows.
type budgetAt struct{ used, limit, promised uint64 }

// markNoRoom records that the batched width did not fit the budget as it now
// stands. Callers hold g.mu.
func (g *devTier) markNoRoom(width int) {
	if g.noRoom == nil {
		g.noRoom = map[noRoomKey]budgetAt{}
	}
	g.noRoom[noRoomKey{g.geoCur, width}] = budgetAt{g.used, g.limit, g.promisedBytes()}
}

// noRoomFor reports that the batched width was refused for the budget and
// nothing has given room back since: no fewer bytes charged, no higher limit,
// no less history promised. Building it again would compile every kernel of
// the width and allocate its buffers only to drop them, and a step that runs
// at the reserved width (batchFor) did that every step on a card the width
// did not fit. Callers hold g.mu.
//
// A wider width than one refused is refused with it: a scratch grows with its
// rows, and a bidirectional prompt asks for its own length every time (a
// width a prompt), which rebuilt and dropped a scratch per prompt.
func (g *devTier) noRoomFor(width int) bool {
	for k, at := range g.noRoom {
		if k.geo != g.geoCur || k.width > width {
			continue
		}
		if g.used < at.used || g.limit > at.limit || g.promisedBytes() < at.promised {
			delete(g.noRoom, k)
			continue
		}
		return true
	}
	return false
}
