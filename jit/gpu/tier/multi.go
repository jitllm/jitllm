package tier

import (
	"fmt"
	"os"
	"slices"
	"sort"
	"strings"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// This file is the router: which device gets a block, in what order they are
// asked, and what a change of device costs. It only places; it computes
// nothing. A block is still the unit and the device still declines per block;
// there are simply more devices to offer a block to before the host keeps it.

// GPU is the nn.LayerDevice a caller gets; devTier is the same interface over
// one card.
var (
	_ nn.LayerDevice   = (*GPU)(nil)
	_ nn.NamedDevice   = (*GPU)(nil)
	_ nn.RecRowsDevice = (*GPU)(nil)
	_ nn.NamedDevice   = (*gpuSession)(nil)
	_ nn.Session       = (*gpuSession)(nil)
	_ nn.RecRowsDevice = (*gpuSession)(nil)
)

// order ranks devices fastest first, by the same probe() choose() uses. The
// enumeration order is not good enough: a block placed on the slow device while
// the fast one had room is a loss no later decision undoes, and which backend
// enumerates the faster card first varies by machine. probe times one
// representative decode matvec per device, a few milliseconds each.
//
// TuneOff keeps the given order, for tests that want a deterministic placement
// (and for fakeTier, whose instant answers would order it by noise).
func order(devs []backend.Device, tune TuneMode) []backend.Device {
	if len(devs) < 2 || tune == TuneOff {
		return devs
	}
	_, rates := probe(devs)
	idx := make([]int, len(devs))
	for i := range idx {
		idx[i] = i
	}
	// Stable, so devices that time the same -- or that both failed to time at
	// all, which is rates 0 -- keep enumeration order rather than swapping run
	// to run.
	sort.SliceStable(idx, func(a, b int) bool { return rates[idx[a]] > rates[idx[b]] })
	out := make([]backend.Device, len(devs))
	for i, j := range idx {
		out[i] = devs[j]
	}
	if backend.Verbose() {
		for i, j := range idx {
			fmt.Fprintf(os.Stderr, "jitllm: order %d %-6s %-40s %8.1f Gmac/s\n",
				i, devs[j].API(), devs[j].Name(), rates[j])
		}
	}
	return out
}

// Name is every device in the tier, fastest first.
func (g *GPU) Name() string {
	names := make([]string, 0, len(g.devs))
	for _, d := range g.devs {
		names = append(names, d.dev.Name()+" ["+d.dev.API()+"]")
	}
	return strings.Join(names, " + ")
}

// Bytes is how much device memory the resident weights occupy, over every
// device. Pools() is what says whether two of those numbers are the same bytes.
func (g *GPU) Bytes() uint64 {
	var n uint64
	for _, d := range g.devs {
		n += d.Bytes()
	}
	return n
}

// Err reports the last reason a device operation declined, for diagnostics.
func (g *GPU) Err() string {
	g.mu.Lock()
	e := g.rowsErr
	g.mu.Unlock()
	if e != "" {
		return e
	}
	return g.Stats().LastErr
}

// resetRecSteps zeroes every device's count at the head of a rows call, as
// layersCall does for its own: a step refused before it ran must read zero,
// not the count of whatever call came before it, or a caller deciding whether
// the rows may run again (the server's step loop) would refuse a sound retry.
func (g *GPU) resetRecSteps(sid uint64) {
	for i := 0; ; i++ {
		d := g.dev(i)
		if d == nil {
			return
		}
		d.setRecSteps(sid, 0)
	}
}

// dev is device i, or nil past the last: a walk over the devices that takes
// g.mu per step instead of copying the list, which allocated on every call.
func (g *GPU) dev(i int) *devTier {
	g.mu.Lock()
	defer g.mu.Unlock()
	if i >= len(g.devs) {
		return nil
	}
	return g.devs[i]
}

// RecSteps is how many linear blocks advanced across every device during the
// last Layers or rows call made on the GPU itself (session 0). A State steps on
// a session of its own and counts there: ask the State
// (model.State.RecurrenceAdvanced), never this. See devTier.recSteps.
func (g *GPU) RecSteps() int {
	return g.recStepsOf(0)
}

// recStepsOf is RecSteps for session sid: the count is per session, as the
// call it counts is.
func (g *GPU) recStepsOf(sid uint64) int {
	n := 0
	for i := 0; ; i++ {
		d := g.dev(i)
		if d == nil {
			return n
		}
		n += d.recStepsOf(sid)
	}
}

// SetActWindow tells every device which activation amax window this session
// uses, so its host-side quantize matches the CPU tier's byte for byte.
func (g *GPU) SetActWindow(w int) {
	for _, d := range g.devs {
		d.SetActWindow(w)
	}
}

// Close closes every device. A tier whose second device failed to open never
// reaches here; New closes the whole list itself.
func (g *GPU) Close() {
	g.settle()
	for _, d := range g.devs {
		d.Close()
	}
	if g.xdev != nil {
		g.xdev.Close()
	}
	// The arena last: releasing it unmaps the host memory the device buffers
	// were uploaded from, and a queued transfer out of those pages would fault
	// inside the driver. Closing the devices first drains them.
	g.Config.arena.Close()
}

// SetArena sets what the packed arena may retain, in host bytes. Zero, the
// default, retains nothing and every pack is transient. Above zero a block's
// packed bytes outlive the device buffer they were uploaded into, so paging it
// back in is a DMA instead of a repack. Lowering the limit stops growth and does
// not evict, because an entry that is out is a pointer someone holds.
func (g *GPU) SetArena(bytes uint64) { g.Config.arena.SetLimit(bytes) }

// DropArena gives back the host memory holding blocks [lo, hi) in the device
// layout, and returns the bytes freed.
//
// A unified device imports these pages rather than copying them, so unmapping
// them under a resident block would leave its launches reading unmapped memory
// (a fault inside the driver, far from the cause). So a block resident anywhere
// with wrapped weights is paged out first, the same demotion a budget cut makes:
// it keeps its owner and KV history and the next token repacks it.
//
// ReleaseLayers deliberately does not drop the pack: that would turn the next
// page-in into a repack.
func (g *GPU) DropArena(lo, hi int) uint64 {
	var freed uint64
	for li := lo; li < hi; li++ {
		for _, d := range g.devs {
			d.dropImported(li)
		}
		freed += g.Config.arena.Release(li)
	}
	return freed
}

// ArenaStats is what the packed arena has done. Unlike Stats it is not summed
// per device: devices share the arena's entries, and a per-device count would
// count one pack twice.
func (g *GPU) ArenaStats() kernels.ArenaStats { return g.Config.arena.Stats() }

// DropStage frees every pack that has not been uploaded, on every device.
func (g *GPU) DropStage() {
	for _, d := range g.devs {
		d.DropStage()
	}
}

// Reserve sizes the per-call scratch on every device (each card needs its own
// staging buffers; a few MiB buys never growing one on the serving path).
//
// False means at least one device could not reserve. That device still works
// through its lazy grow paths, so the caller has nothing to do about it.
func (g *GPU) Reserve(maxRows, maxK int) bool {
	ok := len(g.devs) > 0
	for _, d := range g.devs {
		ok = d.Reserve(maxRows, maxK) && ok
	}
	return ok
}

// MatVec is matVec for the zero session: a caller that never attached.
func (g *GPU) MatVec(out []float32, t quant.Type, w []byte, x []float32, nrows, k int) bool {
	return g.matVec(0, out, t, w, x, nrows, k)
}

// matVec serves one matvec, asking each device in turn.
//
// noKernel and tooSmall are the same answer on every device (oldkernels.MatVec is
// device-independent, and mvPays judges the model, not a card), so they stop
// the walk; a full card or a failed compile or allocation asks the next one.
//
// The walk is not round-robin: prepare() skips the activation upload when x is
// unchanged, per device, and the first device that accepts a tensor keeps it.
func (g *GPU) matVec(sid uint64, out []float32, t quant.Type, w []byte, x []float32, nrows, k int) bool {
	return false
}

// PrewarmLayer packs a whole block off the main loop, for the device that would
// be asked for it first.
//
// It guesses, and a wrong guess costs host memory rather than correctness: the
// pack is filed under the device that will upload it, so if the cursor moves on
// before PrepLayer arrives the block is packed again. DropStage frees the loser.
func (g *GPU) PrewarmLayer(li int, p *nn.LayerPlan, w *nn.LayerWeights) bool {
	if g.foreign(p.Model) {
		return false
	}
	d := g.next()
	return d != nil && d.PrewarmLayer(li, p, w)
}

// next is the device that will be offered the next block, or nil when every
// device has already declined one.
func (g *GPU) next() *devTier {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.cur >= len(g.devs) {
		return nil
	}
	return g.devs[g.cur]
}

// PrepLayer is prepLayer for the zero session: a caller that never attached.
func (g *GPU) PrepLayer(li int, p *nn.LayerPlan, w *nn.LayerWeights) bool {
	return g.prepLayer(0, li, p, w)
}

// prepLayer offers block li to each device in turn and takes the first that
// accepts.
//
// A device's blocks form one contiguous run. The residual never returns to the
// host inside a submission, but two devices cannot pass it between them, so
// every change of device costs a readback and an upload. Contiguous
// runs cost at most (devices - 1) crossings for the whole model; scattering
// blocks would cost more than a second card gives back.
//
// The cursor enforces it: it only moves forward while a placement grows, so a
// device that declined is not asked again (ReleaseLayers moves it back), and it
// moves only on success, so a block no device takes leaves it where it was.
//
// Two passes: the first asks every device whether the block fits, the second,
// only when none has room, lets one of them page. The other order would have
// the fastest card page the whole model through two slots while the next card
// sat empty: the same tokens at a much worse rate.
func (g *GPU) prepLayer(sid uint64, li int, p *nn.LayerPlan, w *nn.LayerWeights) bool {
	g.settle()
	done, ok := g.admit(p.Model, fmt.Sprintf("block %d", li))
	if !ok {
		return false
	}
	defer done()
	// A block already on a device is offered there first. The weights are the
	// tier's, not the State's, so a second State re-offers blocks the first
	// placed; starting from the cursor (left on the last device) would upload
	// duplicates onto the tail card until it filled.
	g.mu.Lock()
	start, end := g.cur, len(g.devs)
	if o := g.own[li]; o != nil {
		// Only there: the route is shared, and another session's history for
		// this block is on o. A second copy elsewhere would take the route from
		// under it. A session o has no room for runs the block on the host.
		start = slices.Index(g.devs, o)
		end = start + 1
	}
	// A spill's mark lasts until a later device takes the block, the block
	// goes home (ReleaseLayers) or another block is offered: the model can
	// offer one block twice in an attempt (a held block's plan-only offer,
	// then the full one), and a mark the first refusal cleared sent the second
	// back onto the card it was leaving -- which the model read as a spill,
	// and relocateUntil spun on it.
	old := g.spillFrom
	if old != nil && g.spillLi == li {
		start, end = slices.Index(g.devs, old)+1, len(g.devs)
	} else {
		old = nil
		g.spillFrom = nil
	}
	if g.own[li] == nil && old == nil && li < len(g.share) && g.share[li] > start {
		start = g.share[li]
	}
	// Priced on the second block, not the first: the first block admitted to a
	// device also pays one-time charges (the unpack staging buffer), which
	// overstates the per-block cost and spreads the plan over a device too
	// many. The first two blocks land on the first device under any plan.
	first := g.planN > 1 && g.share == nil && len(g.own) == 1
	// A streamed plan is priced on the first streamed block after the one it
	// measures from, not on a dense lead resident whole: Kimi-K3's block 0
	// priced every base at its own size and the plan fell back to fill-first.
	if g.spreadAll && g.planN > 1 && !g.streamPlanned {
		d0 := g.devs[0]
		d0.mu.Lock()
		_, auto := d0.autoStream[li]
		d0.mu.Unlock()
		first = (auto || g.StreamExperts) && g.streamSeen > 0
		if auto || g.StreamExperts {
			g.streamSeen++
		}
		if first {
			g.share, g.streamPlanned = nil, true
		}
	}
	g.mu.Unlock()
	for _, mayPage := range [2]bool{false, true} {
		for i := start; i < end; i++ {
			d := g.devs[i]
			d.mu.Lock()
			before := d.used
			d.mu.Unlock()
			if !d.prepLayer(sid, li, p, w, mayPage) {
				continue
			}
			g.mu.Lock()
			g.own[li], g.cur = d, i
			g.geo[li] = geoOf(p)
			g.split = g.split || p.GeomSplit
			if old != nil {
				g.spillFrom = nil
			}
			if first {
				d.mu.Lock()
				per := d.used - before
				// The first block also paid the device's fixed costs (the
				// scratch, the session's seat); every device a plan uses
				// pays them once.
				var fixed uint64
				if base := g.planBase[i]; before > base+per && !g.streamPlanned {
					fixed = before - base - per
				}
				g.planShares(per, fixed)
				d.mu.Unlock()
			}
			g.mu.Unlock()
			if old != nil {
				old.ReleaseLayers(li, li+1)
			}
			return true
		}
	}
	return false
}

// HoldsBlock reports that block li is placed on a device, where offering it
// again shares the weights already there (nn.BlockHolder).
func (g *GPU) HoldsBlock(li int) bool {
	g.mu.Lock()
	d := g.own[li]
	g.mu.Unlock()
	if d == nil {
		return false
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	l := d.layers[li]
	return l != nil && l.ok
}

// PlanBlocks tells the router how many blocks a placement is about to offer,
// so that a model which fits across the devices is spread over them in
// proportion to their budgets instead of filling the first device to its last
// megabyte. Without a plan (or when the model does not fit) placement is
// fill-first. A full card has no room for the batched prefill scratch, which
// is allocated at the first prompt and is not in the budget; without it every
// batched matvec is refused and the prompt runs row by row.
//
// The per-block cost is measured on an admitted block (what that device
// actually charged) and assumed for the rest; a device that refuses a block it
// was given still passes it on, so a model whose blocks differ in size
// degrades to fill-first rather than failing.
func (g *GPU) PlanBlocks(n int, extra uint64) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.planN, g.share, g.planExtra = n, nil, extra
	g.spreadAll = g.StreamExperts && g.StreamCacheSlots == 0
	g.streamPlanned, g.streamSeen = false, 0
	if g.Config.FillFirst {
		g.planN = 0
	}
	g.planBase = g.planBase[:0]
	for _, d := range g.devs {
		d.mu.Lock()
		g.planBase = append(g.planBase, d.used)
		d.mu.Unlock()
	}
}

// planShares turns PlanBlocks' count and one block's measured cost into the
// device each block is steered to. per is one block's charge and fixed what
// a device pays once (scratch, the session's seat). Callers hold g.mu.
func (g *GPU) planShares(per, fixed uint64) {
	// Over the fewest devices that hold it, not all of them: every device a
	// placement touches is one more crossing of the residual a token, and a
	// measured three-way spread lost to two cards that hold the model. What
	// they must hold is the blocks AND what runs on them: each device's fixed
	// cost, the history the context will take and the output projection
	// (planExtra). Counting the weights alone left a device at its budget, so
	// a long prompt evicted its own history and went a row at a time.
	avail := func(i int) uint64 {
		if l, b := g.devs[i].limit, g.planBase[i]; l > b {
			return l - b
		}
		return 0
	}
	// A tenth of what the chosen devices have past their fixed costs is left
	// over, because a pool grows in steps and a scratch grows past its
	// reservation (a width it did not reserve, a deeper plan). With the need
	// met exactly (43.38 GiB of 43.38 on that 70B) the third card still
	// filled. The batched prefill scratch was that slack's largest part until
	// it was reserved at placement and priced in fixed (scratch.go), so the
	// tenth is not taken of fixed: priced bytes need no margin. Measure the
	// tenth again before narrowing it.
	need := func(k int) uint64 { return per*uint64(g.planN) + fixed*uint64(k) + g.planExtra }
	fits := func(sum uint64, k int) bool {
		f := fixed * uint64(k)
		return sum >= f && sum-(sum-f)/10 >= need(k)
	}
	var sum uint64
	k := 0
	for k < len(g.devs) {
		sum += avail(k)
		k++
		// Streamed blocks spread over every device: what the bases leave is
		// their expert caches (sizeAutoCaches), and a card left empty is
		// cache nobody uses -- fill-first packed Kimi-K3's 93 bases onto six
		// of eight V100s, and only the seven blocks on the sixth got a cache.
		if fits(sum, k) && !g.spreadAll {
			break
		}
	}
	if per == 0 || sum < need(k) {
		return // does not fit: fill-first
	}
	g.share = make([]int, g.planN)
	var acc uint64
	dev := 0
	for li := range g.share {
		// Block li goes to the device whose cumulative budget share covers
		// the middle of the block, so the split lands in proportion.
		mid := (2*uint64(li) + 1) * sum / uint64(2*g.planN)
		for dev < k-1 && mid >= acc+avail(dev) {
			acc += avail(dev)
			dev++
		}
		g.share[li] = dev
	}
}

// SpillAfter arms the next PrepLayer of block li to place it on a device after
// the one holding it, and reports whether there is one. When a later device
// takes it the old copy is freed; when none does, nothing has changed.
//
// It is how a full card grows its KV history without demoting the model:
// moving the first card's highest block onto the next shifts the seam and keeps
// both runs contiguous, so it costs no crossing.
func (g *GPU) SpillAfter(li int) bool { return g.spillAfter(li, 0) }

// spillAfter is SpillAfter for session sid. A block another session has
// history on does not move: the move frees the old copy, and that session's
// history with it.
func (g *GPU) spillAfter(li int, sid uint64) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	o := g.own[li]
	if o == nil || slices.Index(g.devs, o) == len(g.devs)-1 {
		return false
	}
	o.mu.Lock()
	shared := o.sharedBeyond(li, sid)
	o.mu.Unlock()
	if shared {
		return false
	}
	g.spillFrom, g.spillLi = o, li
	return true
}

// Stream marks block li's weights as streaming through a device's slots
// rather than staying resident, before it is offered (PrepLayer): a streamed
// block that does not fit is admitted by paging other streamed blocks out,
// where one that is not is declined and runs on the host. The mark goes to
// every device, since which one takes the block is decided at the offer.
// Off, a streamed block becomes resident where it is when it next fits and is
// no longer a victim. It reports whether any device is attached.
func (g *GPU) Stream(li int, on bool) bool {
	for _, d := range g.devs {
		d.mu.Lock()
		if d.stream == nil {
			d.stream = map[int]bool{}
		}
		if on {
			d.stream[li] = true
		} else {
			delete(d.stream, li)
		}
		d.mu.Unlock()
	}
	return len(g.devs) > 0
}

// DeclineSize refuses a block whose resident bytes exceed every device's
// whole budget, so placement does not read gigabytes to be told so by
// PrepLayer. A device that streams the block (Config.StreamExperts, or the
// block marked by Stream) keeps neither the bank resident nor necessarily the
// block, so it never refuses here; PrepLayer stays the real decision.
func (g *GPU) DeclineSize(li int, total, bank uint64) string {
	var widest uint64
	for _, d := range g.devs {
		d.mu.Lock()
		_, auto := d.autoStream[li]
		streams := d.StreamExperts || d.stream[li] || auto
		lim := d.limit
		d.mu.Unlock()
		if streams || total <= lim {
			return ""
		}
		widest = max(widest, lim)
	}
	if len(g.devs) == 0 {
		return ""
	}
	// No block index or byte count in the text: placement groups identical
	// reasons, and ninety-two lines that differ by a number say one thing.
	// A block with no routed bank has nothing to stream, so its reason names
	// neither experts nor a streaming placement.
	if bank == 0 {
		return fmt.Sprintf("the block is above every device's whole budget (the largest %.2f GiB), "+
			"so it runs on the host", float64(widest)/(1<<30))
	}
	return fmt.Sprintf("the block with its routed experts is above every device's whole budget "+
		"(the largest %.2f GiB), so it runs on the host unread by the device "+
		"(a placement that streams it -- JITLLM_GPU_STREAM, -placement N=DEV~ -- puts it on a card)",
		float64(widest)/(1<<30))
}

// AutoStream marks block li streamed on every device when its base -- the
// block without its routed bank -- fits one of them. It is the default for a
// mixture no card can hold: on Kimi-K3 over eight V100s the streamed blocks,
// cached, prefilled 1.5x faster than the host and decoded at its rate
// (placement.md 16c). Config.NoAutoStream turns it off and leaves such blocks
// on the host.
//
// The blocks are placed with the plain bank, so every one fits before any
// cache takes room; sizeAutoCaches gives them their caches from what each card
// really has left once placement ends. blocks and meanBase are kept for the
// report.
func (g *GPU) AutoStream(li int, total, bank uint64, nExpert, nUsed, blocks int, meanBase uint64) bool {
	if g.NoAutoStream || len(g.devs) == 0 || bank >= total || nExpert <= nUsed || nUsed < 2 {
		return false
	}
	base := total - bank
	var widest uint64
	for _, d := range g.devs {
		d.mu.Lock()
		widest = max(widest, d.limit)
		d.mu.Unlock()
	}
	if base > widest {
		return false
	}
	g.mu.Lock()
	g.spreadAll = g.spreadAll || g.StreamCacheSlots == 0
	g.mu.Unlock()
	for _, d := range g.devs {
		d.mu.Lock()
		if d.autoStream == nil {
			d.autoStream = map[int]expertMode{}
		}
		if _, set := d.autoStream[li]; !set {
			d.autoStream[li] = expAuto
		}
		d.AutoMeanBase = meanBase
		d.mu.Unlock()
	}
	return true
}

// expertMode is where a streamed block's routed experts run.
type expertMode int

const (
	// expAuto is GPU.AutoStream's mark: the block's experts run where the
	// tier's defaults say (on the host unless Config.NoHybrid).
	expAuto expertMode = iota
	// expHost runs them on the host (hybrid), whatever the defaults.
	expHost
	// expCard sends their sheets to the card each token.
	expCard
)

// PlaceExperts is nn.ExpertPlacer: it places block li's routed experts off
// the card -- "host" runs them on the host (hybrid), "card" streams their
// sheets to it -- on every device, so the block keeps the choice wherever it
// moves; "" clears the choice. It reports false for a word it does not know.
func (g *GPU) PlaceExperts(li int, where string) bool {
	var m expertMode
	switch where {
	case "host":
		m = expHost
	case "card":
		m = expCard
	case "":
	default:
		return false
	}
	// A block placed with its experts off the card spreads like an
	// auto-streamed one: packed onto the first cards, it left them no room for
	// a prompt's batched scratch, and Kimi-K3's first chunk failed there.
	if where != "" {
		g.mu.Lock()
		g.spreadAll = true
		g.mu.Unlock()
	}
	for _, d := range g.devs {
		d.mu.Lock()
		if d.autoStream == nil {
			d.autoStream = map[int]expertMode{}
		}
		if where == "" {
			delete(d.autoStream, li)
		} else {
			d.autoStream[li] = m
		}
		d.mu.Unlock()
	}
	return len(g.devs) > 0
}

// SetExpertMode re-decides where every marked block's experts run, for the
// trial: "host", "card", or "" for the defaults. A block takes it the next
// time it is placed. It reports how many blocks it touched.
func (g *GPU) SetExpertMode(where string) int {
	m := expAuto
	switch where {
	case "host":
		m = expHost
	case "card":
		m = expCard
	}
	n := 0
	for i, d := range g.devs {
		d.mu.Lock()
		for li := range d.autoStream {
			d.autoStream[li] = m
			if i == 0 {
				n++
			}
		}
		d.mu.Unlock()
	}
	return n
}

// sizeAutoCaches gives every streamed block on its plain bank an expert
// cache once placement has ended: on each card, the budget less what the
// card holds now, split evenly among the card's streamed blocks in device
// sheet bytes. Config.StreamCacheSlots states the size instead (negative:
// no cache). Measured rather than estimated -- the scratch,
// the head and the shared bank are whatever they came to -- and sized per
// card, so a card that took the head gives smaller caches than one that did
// not. The kernels need nothing new: an indexed matvec addresses a sheet by
// slot times its stride whatever the bank's size, so the slot ids the cache
// writes over the selection index the bigger bank as they did the compact
// one. A cache the card cannot allocate leaves the block on its plain bank.
func (g *GPU) sizeAutoCaches() {
	if g.StreamCacheSlots != 0 {
		return
	}
	for _, d := range g.devs {
		d.mu.Lock()
		d.sizeAutoCaches()
		d.mu.Unlock()
	}
}

// cacheBlock gives block l a cache of n sheets, or reports false and leaves
// it on its plain bank. Callers hold g.mu.
func (g *devTier) cacheBlock(l *layer, n, k int) bool {
	var got [3]*resident
	for m, r := range [3]*resident{l.gate, l.up, l.down} {
		if r == nil {
			continue // an ungated bank has no gate
		}
		nr, _, ok := g.residentCompact(r.t, r.nrows/k, r.k, n)
		if !ok {
			for _, r := range got {
				if r != nil {
					g.refund(r.bytes())
					g.freeResident(r)
				}
			}
			return false
		}
		got[m] = nr
	}
	for m, dst := range [3]**resident{&l.gate, &l.up, &l.down} {
		if got[m] != nil {
			l.stream.plain[m] = *dst
			*dst = got[m]
		}
	}
	l.stream.initCache(n, l.stream.nExpert)
	g.StreamCacheSize = max(g.StreamCacheSize, n)
	g.StreamCacheSlots += n
	g.StreamCacheBlocks++
	return true
}

// dropCaches gives back expert caches sized after placement (cacheBlock),
// one block at a time from the last, until n more bytes fit: a cache holds
// copies of host pages and is the first thing to go when history or a block
// needs the card. A block without its cache runs on the plain bank it was
// placed on. It reports whether n fits. Callers hold g.mu, outside a
// submission.
func (g *devTier) dropCaches(n uint64) bool {
	for li := len(g.layers) - 1; li >= 0 && !g.room(n); li-- {
		l := g.layers[li]
		if l == nil || l.stream == nil || !l.stream.cached() || l.stream.plain[2] == nil {
			continue
		}
		if err := g.quiesce(); err != nil {
			g.LastErr = err.Error()
			return false
		}
		for m, dst := range [3]**resident{&l.gate, &l.up, &l.down} {
			if l.stream.plain[m] == nil {
				continue
			}
			g.refund((*dst).bytes())
			g.freeResident(*dst)
			*dst, l.stream.plain[m] = l.stream.plain[m], nil
		}
		l.stream.uncache()
		g.dropGraph()
		g.CachesDropped++
	}
	return g.room(n)
}

// sizeAutoCaches is GPU.sizeAutoCaches on one device. Callers hold g.mu.
func (g *devTier) sizeAutoCaches() {
	var ls []*layer
	var sheet uint64
	k := 0
	for _, l := range g.layers {
		// A block whose experts carry biases keeps its plain bank: see
		// cacheSlotsFor.
		if l == nil || l.stream == nil || l.stream.cached() || l.stream.hybrid && g.NoHybridCache || l.down == nil ||
			l.expGateB != nil || l.expUpB != nil || l.expDownB != nil {
			continue
		}
		var s uint64
		for m := range l.stream.sh {
			for pl := range l.stream.sh[m] {
				s += uint64(l.stream.sh[m][pl])
			}
		}
		sheet = max(sheet, s)
		k = len(l.stream.sel)
		ls = append(ls, l)
	}
	// The budget already leaves the driver its headroom.
	room := g.limit
	if len(ls) == 0 || sheet == 0 || g.used >= room {
		return
	}
	n := int((room - g.used) / (uint64(len(ls)) * sheet))
	// The cache must hold more than a token's selection to keep anything.
	if n <= k {
		return
	}
	for _, l := range ls {
		// A card's allocations round, so the even share can be a sheet too
		// many for the last blocks: those step down rather than go without.
		for ; n > k; n-- {
			if g.cacheBlock(l, min(n, l.stream.nExpert), k) {
				break
			}
			g.StreamCacheShort++
		}
		if n <= k {
			return
		}
	}
}

// SetBudget retargets every device's weight budget while the model is running,
// and reports how many blocks still have their weights on a card. It is how a
// running model makes room for another (or hands a card back).
//
// A demotion does not move a token: it does not change which device owns a
// block, only whether its weights are on the card at this instant. A budget too
// small to page at all takes blocks away, and that reaches model/ as Layers
// returning false, the existing demotion path.
//
// The pools are retargeted first and the devices trimmed second, the order
// that leaves the two accounts agreeing.
//
// A budget below what a device's pinned blocks charge is refused before
// anything changes: a pin says the block stays where it is (GPU.Pin).
func (g *GPU) SetBudget(bytes uint64) (int, error) {
	for _, d := range g.devs {
		d.mu.Lock()
		pb := d.pinBytes()
		d.mu.Unlock()
		if pb > bytes {
			return 0, fmt.Errorf("tier: %s holds %.2f GiB of pinned blocks, and a %.2f GiB budget "+
				"would evict them; unpin them or ask for more", label(d.dev, d.ord), float64(pb)/(1<<30), float64(bytes)/(1<<30))
		}
	}
	for _, p := range g.pools {
		p.SetLimit(bytes)
	}
	n := 0
	for _, d := range g.devs {
		n += d.SetBudget(bytes)
	}
	return n, nil
}

// device is the device a placement names, or an error naming the ones there
// are.
func (g *GPU) device(name string) (*devTier, error) {
	want := DeviceName(name)
	var have []string
	for _, d := range g.devs {
		if d.name != "" && d.name == want {
			return d, nil
		}
		if d.name != "" {
			have = append(have, d.name)
		}
	}
	return nil, fmt.Errorf("tier: no device named %q is attached (named devices: %v); "+
		"a placement names the devices -devices named", name, have)
}

// PrepLayerOn is prepLayerOn for the zero session: a caller that never attached.
func (g *GPU) PrepLayerOn(name string, li int, p *nn.LayerPlan, w *nn.LayerWeights) (bool, error) {
	return g.prepLayerOn(0, name, li, p, w)
}

// prepLayerOn offers block li to the named device alone: an explicit placement
// rather than the router's choice. False means the device declined (its budget,
// or a graph it does not run); an error means no such device is attached.
func (g *GPU) prepLayerOn(sid uint64, name string, li int, p *nn.LayerPlan, w *nn.LayerWeights) (bool, error) {
	g.settle()
	d, err := g.device(name)
	if err != nil {
		return false, err
	}
	done, ok := g.admit(p.Model, fmt.Sprintf("block %d", li))
	if !ok {
		return false, nil
	}
	defer done()
	g.mu.Lock()
	old := g.own[li]
	g.mu.Unlock()
	// Moving the block frees the old copy with every session's history on it,
	// so a block another session is using stays where it is. (On the named
	// device already, prepLayer only adds this session's history.)
	if old != nil && old != d {
		old.mu.Lock()
		shared := old.sharedBeyond(li, sid)
		old.mu.Unlock()
		if shared {
			return false, fmt.Errorf("tier: block %d is on %s and another session is using it, "+
				"so it cannot move to %s", li, label(old.dev, old.ord), name)
		}
	}
	for _, mayPage := range [2]bool{false, true} {
		if d.prepLayer(sid, li, p, w, mayPage) {
			g.mu.Lock()
			g.own[li], g.geo[li] = d, geoOf(p)
			g.split = g.split || p.GeomSplit
			g.mu.Unlock()
			if old != nil && old != d {
				old.ReleaseLayers(li, li+1)
			}
			return true, nil
		}
	}
	return false, nil
}

// PrepHeadOn places the output projection on the named device.
func (g *GPU) PrepHeadOn(name string, h *nn.Head) (bool, error) {
	g.settle()
	d, err := g.device(name)
	if err != nil {
		return false, err
	}
	done, ok := g.admit(h.Model, "output projection")
	if !ok {
		return false, nil
	}
	defer done()
	g.mu.Lock()
	t := g.tail()
	g.mu.Unlock()
	if d != t && t != nil {
		t.mu.Lock()
		var plan *nn.LayerPlan
		if t.bs != nil {
			p := t.bs.p
			plan = &p
		}
		t.mu.Unlock()
		if plan == nil || !d.headScratch(plan) {
			return false, nil
		}
	}
	if !d.PrepHead(h) {
		return false, nil
	}
	g.mu.Lock()
	prev := g.head
	g.head = d
	if g.stagedHead == d {
		g.stagedHead = nil
	}
	// The copy the head moved from is freed when no block is left on that
	// device to share its scratch (devTier.dropHead), with a move's other
	// leftovers (RetireMoved).
	if prev != nil && prev != d {
		g.retire = append(g.retire, retiree{prev, -1})
	}
	g.mu.Unlock()
	g.RetireMoved()
	return true, nil
}

// DeviceOf is the name of the device holding block li ("" when none does, or
// the one that does has no name), and whether one holds it.
func (g *GPU) DeviceOf(li int) (string, bool) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if d := g.own[li]; d != nil {
		return d.name, true
	}
	return "", false
}

// HeadDeviceName is the name of the device holding the projection, and whether
// one does.
func (g *GPU) HeadDeviceName() (string, bool) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.head != nil {
		return g.head.name, true
	}
	return "", false
}

// Pin keeps block li where it is: the pager never picks it as a victim, and a
// budget below what the pinned blocks charge is refused. False when no device
// holds the block.
func (g *GPU) Pin(li int, on bool) bool {
	g.mu.Lock()
	d := g.own[li]
	g.mu.Unlock()
	if d == nil {
		return false
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.pinned == nil {
		d.pinned = map[int]bool{}
	}
	if on {
		d.pinned[li] = true
	} else {
		delete(d.pinned, li)
	}
	return true
}

// run is one contiguous stretch of blocks on one device: one submission.
// geo is its blocks' scratch set (gemma4.go) and split says the model has two
// attention geometries (Gemma 4), so the run takes one rotary table (tables).
type run struct {
	dev    *devTier
	lo, hi int
	geo    geoKey
	split  bool
}

// tables is the rotary tables run r is handed. A split model's runs are one
// geometry each, built with one table apiece (scratchPlanOf): the sliding
// layers' run takes the local table where the global one goes.
func (r run) tables(cs, csSWA []float32) ([]float32, []float32) {
	switch {
	case !r.split:
		return cs, csSWA
	case r.geo&geoSliding != 0:
		return csSWA, nil
	}
	return cs, nil
}

// runs cuts [lo, hi) into one submission per device. Callers hold g.mu.
//
// It reports false when a block in the range is on no device at all, which the
// caller must treat as "run the whole range on the host" -- the same answer a
// single device gives when it has not taken a block.
func (g *GPU) runs(lo, hi int) ([]run, bool) { return g.runsInto(nil, lo, hi) }

// runsInto is runs appending to out[:0].
func (g *GPU) runsInto(out []run, lo, hi int) ([]run, bool) {
	out = out[:0]
	for li := lo; li < hi; li++ {
		d := g.own[li]
		if d == nil {
			return nil, false
		}
		// A change of geometry is cut as a change of device is: one
		// submission is one scratch set (gemma4.go).
		geo := g.geo[li]
		if n := len(out); n > 0 && out[n-1].dev == d && out[n-1].geo == geo {
			out[n-1].hi = li + 1
			continue
		}
		out = append(out, run{dev: d, lo: li, hi: li + 1, geo: geo, split: g.split})
	}
	return out, true
}

// Layers is layersFor for the zero session: a caller that never attached.
func (g *GPU) Layers(lo, hi, pos, n int, x, cs, csSWA []float32, head *nn.Head) bool {
	return g.layersFor(0, lo, hi, pos, n, x, cs, csSWA, head)
}

// layersFor runs blocks [lo, hi) as one submission per device. The residual
// crosses the host at every change of device (no peer-to-peer copies are
// attempted); PrepLayer's contiguity rule keeps that at (devices - 1).
//
// The projection rides the last submission when it is on that device, which
// is the placement PrepHead makes; elsewhere it becomes its own submission, one
// more crossing and still worth it for the densest unit of a large model.
func (g *GPU) layersFor(sid uint64, lo, hi, pos, n int, x, cs, csSWA []float32, head *nn.Head) bool {
	ok := g.layersCall(sid, lo, hi, pos, n, x, cs, csSWA, head)
	// A refusal before any device ran leaves the head device's promise
	// unconsumed; it lives for one call (devTier.layersCall).
	return ok
}

// PipelineDepth is how many devices hold blocks (nn.Pipeliner): a prompt chunk
// crosses each once, in order.
func (g *GPU) PipelineDepth() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	seen := map[*devTier]bool{}
	for _, d := range g.own {
		if d != nil {
			seen[d] = true
		}
	}
	return len(seen)
}

// nonCausalAt reports whether block li is a non-causal block, whose rows --
// a picture's patches, each attending to every other -- go in one submission
// however many there are, and so are never cut into pieces.
func (d *devTier) nonCausalAt(li int) bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	l := d.layers[li]
	return l != nil && l.nonCausal
}

// pipeline runs rows wider than one device chunk in pieces of batchWidth: the
// run on device r takes piece k once run r-1 has finished it, so device 0
// starts piece k+1 while device 1 runs piece k. One after another, a long
// prompt on a dense model split over cards left every card but one idle.
//
// Each piece's rows are their own slice of x, and a run takes its pieces in
// order, so every piece finds the history of the pieces before it on its
// device. The head rides the last piece through the last run.
func (g *GPU) pipeline(sid uint64, rs []run, pos, n int, x, cs, csSWA []float32, tail *nn.Head) bool {
	e, r := len(x)/n, len(cs)/n
	rw := 0
	if len(csSWA) > 0 {
		rw = len(csSWA) / n
	}
	pieces := (n + batchWidth - 1) / batchWidth
	// next[i] is how run i hears that a piece cleared the run before it.
	next := make([]chan bool, len(rs)+1)
	for i := range next {
		next[i] = make(chan bool, pieces)
	}
	for range pieces {
		next[0] <- true
	}
	for i, rn := range rs {
		go func() {
			for k := range pieces {
				ok := <-next[i]
				if ok {
					lo := k * batchWidth
					cnt := min(batchWidth, n-lo)
					var h *nn.Head
					if i == len(rs)-1 && k == pieces-1 {
						h = tail
					}
					var sw []float32
					if rw > 0 {
						sw = csSWA[lo*rw : (lo+cnt)*rw]
					}
					c0, c1 := rn.tables(cs[lo*r:(lo+cnt)*r], sw)
					ok = rn.dev.layersFor(sid, rn.lo, rn.hi, pos+lo, cnt, x[lo*e:(lo+cnt)*e], c0, c1, h)
					if ok {
						rn.dev.mu.Lock()
						rn.dev.Stats.PipelinePieces++
						rn.dev.mu.Unlock()
					}
				}
				next[i+1] <- ok
			}
		}()
	}
	ok := true
	for range pieces {
		ok = <-next[len(rs)] && ok
	}
	return ok
}

// pieces is pipeline in order: each batchWidth piece of the chunk through
// every run before the next piece starts, the head on the last piece of the
// last run.
func (g *GPU) pieces(sid uint64, rs []run, pos, n int, x, cs, csSWA []float32, tail *nn.Head) bool {
	e, r := len(x)/n, len(cs)/n
	rw := 0
	if len(csSWA) > 0 {
		rw = len(csSWA) / n
	}
	for lo := 0; lo < n; lo += batchWidth {
		cnt := min(batchWidth, n-lo)
		var sw []float32
		if rw > 0 {
			sw = csSWA[lo*rw : (lo+cnt)*rw]
		}
		for i, rn := range rs {
			var h *nn.Head
			if i == len(rs)-1 && lo+cnt == n {
				h = tail
			}
			c0, c1 := rn.tables(cs[lo*r:(lo+cnt)*r], sw)
			if !rn.dev.layersFor(sid, rn.lo, rn.hi, pos+lo, cnt, x[lo*e:(lo+cnt)*e], c0, c1, h) {
				return false
			}
		}
	}
	return true
}

// LayeredChunks is how many device chunks a prompt chunk holds when a device
// pages its blocks (nn.LayerGrouper): the wider the chunk, the fewer times
// each block's weights cross per prompt. 1 when no device pages.
func (g *GPU) LayeredChunks() int {
	g.mu.Lock()
	seen := map[*devTier]int{}
	for _, d := range g.own {
		if d != nil {
			seen[d]++
		}
	}
	g.mu.Unlock()
	for d, owned := range seen {
		d.mu.Lock()
		sl := d.slots()
		d.mu.Unlock()
		if owned > sl {
			return layeredChunks
		}
	}
	return 1
}

// layeredChunks is how many device chunks a layered prompt chunk holds.
const layeredChunks = 4

// blockGroups cuts the runs into groups of consecutive blocks on one device,
// none holding more blocks than the device has slots: what a device pages in
// together. A run wider than the slots is cut; runs of different geometries
// that fit together stay one group.
func blockGroups(rs []run) [][]run {
	var out [][]run
	var cur []run
	held := 0
	for _, rn := range rs {
		step := rn.dev.slotCount()
		for lo := rn.lo; lo < rn.hi; {
			if len(cur) > 0 && (cur[0].dev != rn.dev || held >= step) {
				out, cur, held = append(out, cur), nil, 0
			}
			hi := min(rn.hi, lo+step-held)
			sub := rn
			sub.lo, sub.hi = lo, hi
			cur, held, lo = append(cur, sub), held+hi-lo, hi
		}
	}
	if len(cur) > 0 {
		out = append(out, cur)
	}
	return out
}

// slotCount is slots() under the device's lock, at least 1.
func (d *devTier) slotCount() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return max(1, d.slots())
}

// layered runs a wide chunk block group by block group: a group is as many
// blocks as the device has slots, and every batchWidth piece of the chunk goes
// through it before the next group pages in. Pieces of a stream-major residual
// are cut by streams (cutRows), and each piece's per-row inputs with them.
func (g *GPU) layered(sid uint64, groups [][]run, pos, n int, x, cs, csSWA []float32) bool {
	r := len(cs) / n
	rw := 0
	if len(csSWA) > 0 {
		rw = len(csSWA) / n
	}
	piece := func(grp []run, lo int) bool {
		d := grp[0].dev
		ns := max(1, d.streams())
		e := len(x) / (n * ns)
		cnt := min(batchWidth, n-lo)
		var sw []float32
		if rw > 0 {
			sw = csSWA[lo*rw : (lo+cnt)*rw]
		}
		part := d.cutRows(x, ns, e, n, lo, cnt)
		was := d.cutInputs(sid, n, lo, cnt)
		ok := true
		for _, rn := range grp {
			c0, c1 := rn.tables(cs[lo*r:(lo+cnt)*r], sw)
			if ok = d.layersFor(sid, rn.lo, rn.hi, pos+lo, cnt, part, c0, c1, nil); !ok {
				break
			}
		}
		d.restoreInputs(sid, was)
		d.uncutRows(x, part, ns, e, n, lo, cnt)
		if ok {
			d.mu.Lock()
			d.Stats.LayeredPieces++
			d.mu.Unlock()
		}
		return ok
	}
	if g.rowsFaulted(rowsPieceouter) {
		// The violation: the pieces outer and the groups inner, as the chunked
		// path pages, so every piece pages every group in again.
		for lo := 0; lo < n; lo += batchWidth {
			for _, grp := range groups {
				if !piece(grp, lo) {
					return false
				}
			}
		}
		return true
	}
	for _, grp := range groups {
		for lo := 0; lo < n; lo += batchWidth {
			if !piece(grp, lo) {
				return false
			}
		}
	}
	return true
}

func (g *GPU) layersCall(sid uint64, lo, hi, pos, n int, x, cs, csSWA []float32, head *nn.Head) bool {
	// The runs slice is g.runBuf, taken for the call so a decode token
	// allocates none; a concurrent call finds it taken and makes its own.
	g.mu.Lock()
	rs, ok := g.runsInto(g.runBuf, lo, hi)
	g.runBuf = nil
	hd := g.head
	g.mu.Unlock()
	defer g.putRuns(rs)
	if !ok {
		return false
	}
	if head != nil && hd == nil {
		return false
	}
	if len(rs) == 0 {
		// An empty range with a head is a real call, not a no-op: model/ makes
		// it under a partial seam (blocks on the host, projection on a device)
		// and reads true as "the logits are filled". Returning true without
		// running would leave the previous token's logits in place.
		if head == nil {
			return true
		}
		return hd.layersFor(sid, lo, hi, pos, n, x, cs, csSWA, head)
	}
	tail := head
	if head != nil && (rs[len(rs)-1].dev != hd || rs[len(rs)-1].geo != 0) {
		// It does not belong to the last run -- another device, or the
		// sliding layers' scratch set, not the one holding the head -- and
		// gets its own submission.
		tail = nil
	}
	var groups [][]run
	if n > batchWidth && !rs[0].dev.nonCausalAt(lo) {
		groups = blockGroups(rs) // a decode token never reaches it
	}
	if len(groups) > 1 {
		// A run that pages its blocks reads every block's weights once per
		// call, so the chunk's pieces go through one block group at a time
		// instead of each piece paging the whole run.
		if !g.layered(sid, groups, pos, n, x, cs, csSWA) {
			return false
		}
		if head != nil {
			return hd.layersFor(sid, hi, hi, pos, n, x, cs, csSWA, head)
		}
		return true
	}
	// The pieces cut x by rows, which AltUp's stream-major residual is not
	// (altup.go): such a chunk is refused whole.
	if n > batchWidth && rs[0].dev.streams() > 1 {
		return false
	}
	if n > batchWidth && !rs[0].dev.nonCausalAt(lo) && g.split {
		// The pieces go one after another: two runs of one device at two
		// geometries share that device's tier, which the pipeline would drive
		// from two goroutines at once.
		if !g.pieces(sid, rs, pos, n, x, cs, csSWA, tail) {
			return false
		}
		if head != nil && tail == nil {
			return hd.layersFor(sid, hi, hi, pos, n, x, cs, nil, head)
		}
		return true
	}
	if n > batchWidth && !rs[0].dev.nonCausalAt(lo) {
		if !g.pipeline(sid, rs, pos, n, x, cs, csSWA, tail) {
			return false
		}
		if head != nil && tail == nil {
			return hd.layersFor(sid, hi, hi, pos, n, x, cs, csSWA, head)
		}
		return true
	}
	for i, r := range rs {
		h := (*nn.Head)(nil)
		if i == len(rs)-1 {
			h = tail
		}
		c0, c1 := r.tables(cs, csSWA)
		if !r.dev.layersFor(sid, r.lo, r.hi, pos, n, x, c0, c1, h) {
			return false
		}
	}
	if head != nil && tail == nil {
		// A head-only submission: the blocks are done, x is back on the host,
		// and the device holding the projection takes it from there. A split
		// model's head runs in the global layers' set, with their table.
		if g.split {
			csSWA = nil
		}
		return hd.layersFor(sid, hi, hi, pos, n, x, cs, csSWA, head)
	}
	return true
}

// putRuns hands layersCall's runs slice back for the next call.
func (g *GPU) putRuns(rs []run) {
	g.mu.Lock()
	g.runBuf = rs
	g.mu.Unlock()
}

// PrepHead uploads the output norm and the vocabulary projection onto the
// device that holds the last block, so that it can ride that block's
// submission instead of paying a crossing of its own.
//
// The tail first, then any other device, and never the host while a card has
// room: when the tail is full, Layers issues the head as its own submission on
// whichever device holds it, one more crossing against a vocab x NEmbd matvec
// on the CPU. Later devices are asked before earlier ones, since the head runs
// after the last block.
func (g *GPU) PrepHead(h *nn.Head) bool {
	if h == nil {
		return false
	}
	done, ok := g.admit(h.Model, "output projection")
	if !ok {
		return false
	}
	defer done()
	// Placed already, by another State of this model: one head serves them all,
	// and placing a second on another card would orphan the first.
	g.mu.Lock()
	hd := g.head
	g.mu.Unlock()
	if hd != nil && hd.HeadResident() {
		return hd.joinHead(h)
	}
	g.mu.Lock()
	t := g.tail()
	var order []*devTier
	if t != nil {
		order = append(order, t)
		at := -1
		for i, d := range g.devs {
			if d == t {
				at = i
			}
		}
		order = append(order, g.devs[at+1:]...)
		order = append(order, g.devs[:at]...)
	}
	g.mu.Unlock()
	var plan *nn.LayerPlan
	if t != nil {
		t.mu.Lock()
		if t.bs != nil {
			p := t.bs.p
			plan = &p
		}
		t.mu.Unlock()
	}
	for _, d := range order {
		if d != t && (plan == nil || !d.headScratch(plan)) {
			continue
		}
		if d.PrepHead(h) {
			g.mu.Lock()
			g.head = d
			g.mu.Unlock()
			return true
		}
	}
	return false
}

// EmbedRows gathers token rows on the device that holds the head
// (nn.EmbedDevice); false when no device holds one that is also the embedding.
//
// The head's device gathers inside its next submission, which is the chunk's
// first only when it also holds block 0; on any other placement the rows are
// made real on the host at once and uploaded as always.
func (g *GPU) EmbedRows(ids []int32, dst []float32) bool {
	return g.embedRows(0, ids, dst)
}

// embedRows is EmbedRows for session sid: the promise is the session's, and
// only its next submission gathers it.
func (g *GPU) embedRows(sid uint64, ids []int32, dst []float32) bool {
	g.mu.Lock()
	hd := g.head
	g.mu.Unlock()
	if hd == nil {
		return false
	}
	v := hd.as(sid)
	if v == nil {
		return false
	}
	defer v.done()
	if !v.EmbedRows(ids, dst) {
		return false
	}
	return true
}

// tail is the device holding the highest-numbered block. Callers hold g.mu.
func (g *GPU) tail() *devTier {
	best, at := (*devTier)(nil), -1
	for li, d := range g.own {
		if li > at {
			best, at = d, li
		}
	}
	return best
}

// HeadDevice is the index into Placed() of the device holding the output
// projection, or -1 when it is on the host. It lets a gate assert which
// device, which HeadResident cannot.
func (g *GPU) HeadDevice() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.head == nil {
		return -1
	}
	return g.head.ord
}

// HeadResident reports whether the output projection is on a device.
func (g *GPU) HeadResident() bool {
	g.mu.Lock()
	d := g.head
	g.mu.Unlock()
	return d != nil && d.HeadResident()
}

// MigrateRec is migrateRec for the zero session: a caller that never attached.
func (g *GPU) MigrateRec(li int, conv, state []float32, toDevice bool) bool {
	return g.migrateRec(0, li, conv, state, toDevice)
}

// migrateRec routes block li's recurrent summary to the device that holds it.
func (g *GPU) migrateRec(sid uint64, li int, conv, state []float32, toDevice bool) bool {
	g.mu.Lock()
	d := g.own[li]
	g.mu.Unlock()
	if d == nil {
		return false // the block is on the host; there is nothing to move
	}
	return d.migrateRec(sid, li, conv, state, toDevice)
}

// PublishKV names block li's page of the zero session by key, on the device
// holding the block (nn.KVShareDevice).
func (g *GPU) PublishKV(li, page int, key string) bool { return g.publishKV(0, li, page, key) }

// MapKV puts the page key names into the zero session's page of block li.
func (g *GPU) MapKV(li, page int, key string, end int) bool { return g.mapKV(0, li, page, key, end) }

// SnapRec, HasRec and LoadRec are the zero session's restore points of
// block li, on the device holding it (nn.RecShareDevice).
func (g *GPU) SnapRec(li int, key string) bool { return g.recShare(0, li, key, 0) }
func (g *GPU) HasRec(li int, key string) bool  { return g.recShare(0, li, key, 1) }
func (g *GPU) LoadRec(li int, key string) bool { return g.recShare(0, li, key, 2) }

// SnapRecs and PublishKVs are the zero session's batches (nn.RecShareDevice,
// nn.KVShareDevice).
func (g *GPU) SnapRecs(lis []int32, key string) bool { return g.snapRecs(0, lis, key) }
func (g *GPU) PublishKVs(lis, pages []int32, keys []string) bool {
	return g.publishKVsBatch(0, lis, pages, keys)
}

// ownerOf is the device holding every block of lis, nil when they are
// spread over more than one (or one is on the host).
func (g *GPU) ownerOf(lis []int32) *devTier {
	g.mu.Lock()
	defer g.mu.Unlock()
	var d *devTier
	for _, li := range lis {
		o := g.own[int(li)]
		if o == nil || d != nil && o != d {
			return nil
		}
		d = o
	}
	return d
}

func (g *GPU) snapRecs(sid uint64, lis []int32, key string) bool {
	if d := g.ownerOf(lis); d != nil {
		return d.snapRecs(sid, lis, key)
	}
	for _, li := range lis {
		if !g.recShare(sid, int(li), key, 0) {
			return false
		}
	}
	return true
}

func (g *GPU) publishKVsBatch(sid uint64, lis, pages []int32, keys []string) bool {
	if d := g.ownerOf(lis); d != nil {
		return d.publishKVsBatch(sid, lis, pages, keys)
	}
	ok := true
	for i := range lis {
		if !g.publishKV(sid, int(lis[i]), int(pages[i]), keys[i]) {
			ok = false
		}
	}
	return ok
}

func (g *GPU) recShare(sid uint64, li int, key string, op int) bool {
	g.mu.Lock()
	d := g.own[li]
	g.mu.Unlock()
	return d != nil && d.recShareCall(sid, li, key, op)
}

func (g *GPU) publishKV(sid uint64, li, page int, key string) bool {
	g.mu.Lock()
	d := g.own[li]
	g.mu.Unlock()
	return d != nil && d.publishKVs(sid, li, page, key)
}

func (g *GPU) mapKV(sid uint64, li, page int, key string, end int) bool {
	g.mu.Lock()
	d := g.own[li]
	g.mu.Unlock()
	return d != nil && d.mapKVs(sid, li, page, key, end)
}

// MigrateKV is migrateKV for the zero session: a caller that never attached.
func (g *GPU) MigrateKV(li int, k, v []float32, pos int, toDevice bool) bool {
	return g.migrateKV(0, li, k, v, pos, toDevice)
}

// migrateKV moves block li's attention history between host and device. Only
// the device holding that block can answer.
func (g *GPU) migrateKV(sid uint64, li int, k, v []float32, pos int, toDevice bool) bool {
	g.mu.Lock()
	d := g.own[li]
	g.mu.Unlock()
	if d == nil {
		return false // the block is on the host; there is nothing to move
	}
	return d.migrateKV(sid, li, k, v, pos, toDevice)
}

// PerSequenceKV reports that every device holding a block pages its history
// (nn.SeqKVDevice); a mixed tier answers false and migrates in the flat layout.
func (g *GPU) PerSequenceKV() bool {
	g.mu.Lock()
	// On the stack for any machine's count of devices.
	var arr [16]*devTier
	ds := arr[:0]
	// g.own is by block: each device once.
	for _, d := range g.own {
		if d != nil && !slices.Contains(ds, d) {
			ds = append(ds, d)
		}
	}
	g.mu.Unlock()
	for _, d := range ds {
		if !d.PerSequenceKV() {
			return false
		}
	}
	return len(ds) > 0
}

// MigrateKVSeq is migrateKVSeq for the zero session: a caller that never attached.
func (g *GPU) MigrateKVSeq(li, base int, k, v []float32, pos int, toDevice bool) bool {
	return g.migrateKVSeq(0, li, base, k, v, pos, toDevice)
}

// migrateKVSeq routes to the device holding block li; see devTier.MigrateKVSeq.
func (g *GPU) migrateKVSeq(sid uint64, li, base int, k, v []float32, pos int, toDevice bool) bool {
	g.mu.Lock()
	d := g.own[li]
	g.mu.Unlock()
	if d == nil {
		return false
	}
	return d.migrateKVSeq(sid, li, base, k, v, pos, toDevice)
}

// MigrateEnt is migrateEnt for the zero session: a caller that never attached.
func (g *GPU) MigrateEnt(li, base int, ent []float32, n int, toDevice bool) bool {
	return g.migrateEnt(0, li, base, ent, n, toDevice)
}

// migrateEnt forwards to the device holding block li; see devTier.MigrateEnt.
func (g *GPU) migrateEnt(sid uint64, li, base int, ent []float32, n int, toDevice bool) bool {
	g.mu.Lock()
	d := g.own[li]
	g.mu.Unlock()
	if d == nil {
		return false
	}
	return d.migrateEnt(sid, li, base, ent, n, toDevice)
}

// owners appends to buf every distinct device that holds a block, in block
// order. A decode step reserves through it every token once blocks span two
// devices, so the caller passes a stack array and nothing reaches the heap
// below that many devices.
func (g *GPU) owners(buf []*devTier) []*devTier {
	g.mu.Lock()
	defer g.mu.Unlock()
	for _, d := range g.own {
		if d != nil && !slices.Contains(buf, d) {
			buf = append(buf, d)
		}
	}
	return buf
}

// ownersStack is the stack array owners fills: more devices than this
// allocate, fewer do not.
const ownersStack = 16

// ReserveKVSeqs is reserveKVSeqs for the zero session: a caller that never attached.
func (g *GPU) ReserveKVSeqs(bases, ends []int) bool {
	return g.reserveKVSeqs(0, bases, ends)
}

// reserveKVSeqs asks every device holding a block; see devTier.ReserveKVSeqs.
func (g *GPU) reserveKVSeqs(sid uint64, bases, ends []int) bool {
	var buf [ownersStack]*devTier
	for _, d := range g.owners(buf[:0]) {
		if !d.reserveKVSeqs(sid, bases, ends) {
			return false
		}
	}
	return true
}

// ReserveKV is reserveKV for the zero session: a caller that never attached.
func (g *GPU) ReserveKV(pos int) bool {
	return g.reserveKV(0, pos)
}

// reserveKV makes every device that holds a block able to take `pos` positions.
// It is all or nothing: the caller is about to migrate a whole prefix, and a
// partial reservation would leave one sequence split across two tiers at two
// different depths.
func (g *GPU) reserveKV(sid uint64, pos int) bool {
	var buf [ownersStack]*devTier
	var refused *devTier
	for _, d := range g.owners(buf[:0]) {
		if !d.reserveKV(sid, pos) {
			refused = d
			break
		}
	}
	g.mu.Lock()
	g.refused = refused
	g.mu.Unlock()
	return refused == nil
}

// TrimKV is trimKV for the zero session: a caller that never attached.
func (g *GPU) TrimKV(pos int) bool {
	return g.trimKV(0, pos)
}

// trimKV shrinks the history on every device that holds a block to the smallest
// capacity that holds pos, where the caller is that device's only session. See
// devTier.TrimKV.
func (g *GPU) trimKV(sid uint64, pos int) bool {
	g.mu.Lock()
	ds := map[*devTier]bool{}
	for _, d := range g.own {
		if d != nil {
			ds[d] = true
		}
	}
	g.mu.Unlock()
	trimmed := false
	for d := range ds {
		trimmed = d.trimKV(sid, pos) || trimmed
	}
	return trimmed
}

// RoomGen moves whenever any device may have gained room -- memory given back,
// or a budget raised -- so a caller holding blocks on the host knows when
// offering them again could succeed, without offering them every token.
func (g *GPU) RoomGen() uint64 {
	var n uint64
	for _, d := range g.devs {
		n += d.roomGen.Load()
	}
	return n
}

// Refused lists the blocks held by the device whose last ReserveKV refused, or
// nothing when it did not, so a caller making room takes from that device
// rather than from whichever card holds the highest block.
func (g *GPU) Refused() []int {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.refused == nil {
		return nil
	}
	var out []int
	for li, d := range g.own {
		if d == g.refused {
			out = append(out, li)
		}
	}
	return out
}

// ReleaseLayers is releaseLayers for the zero session: a caller that never attached.
func (g *GPU) ReleaseLayers(lo, hi int) {
	g.releaseLayers(0, lo, hi)
}

// releaseLayers hands blocks [lo, hi) back to the host for the calling
// session. A block another session still has history on stays where it is,
// with its route: only this session's history goes (devTier.leaveLayers).
//
// Only the devices that owned something in the range are told, because a
// release drops the captured recording and an untouched device would pay a
// re-capture for nothing. The cursor goes back to the device holding the
// highest remaining block (or the fastest), since the release freed room.
func (g *GPU) releaseLayers(sid uint64, lo, hi int) {
	g.settle()
	g.mu.Lock()
	if g.spillFrom != nil && g.spillLi >= lo && g.spillLi < hi {
		g.spillFrom = nil
	}
	touched := map[*devTier]bool{}
	for li := lo; li < hi; li++ {
		if d := g.own[li]; d != nil {
			touched[d] = true
		}
	}
	g.mu.Unlock()
	var gone []int
	for d := range touched {
		gone = append(gone, d.leaveLayers(sid, lo, hi)...)
	}
	g.mu.Lock()
	for _, li := range gone {
		delete(g.own, li)
		delete(g.geo, li)
	}
	// Split is a property of the blocks held: with none left it goes, or the
	// next model's runs are cut as a geometry-split model's and lose their
	// local rotary table (run.tables).
	if len(g.own) == 0 {
		g.split = false
	}
	g.cur = 0
	if d := g.tail(); d != nil {
		g.cur = d.ord
	}
	if g.head != nil && touched[g.head] && !g.head.HeadResident() {
		g.head = nil
	}
	g.mu.Unlock()
}

// Placed reports how many blocks each device holds, fastest device first.
// It is not Stats.Blocks, which counts blocks run and grows with every token.
func (g *GPU) Placed() []int {
	out := make([]int, len(g.devs))
	g.mu.Lock()
	defer g.mu.Unlock()
	for _, d := range g.own {
		out[d.ord]++
	}
	return out
}

// Crossings is how many times a token's residual stream has to come back to the
// host because the next block is on a different device: (runs - 1).
func (g *GPU) Crossings() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	hi := -1
	for li := range g.own {
		if li > hi {
			hi = li
		}
	}
	rs, ok := g.runs(0, hi+1)
	if !ok {
		return 0
	}
	if len(rs) == 0 {
		return 0
	}
	return len(rs) - 1
}

// ReleasesHostPage asks whichever device took block li. A block no device owns
// is not released.
func (g *GPU) ReleasesHostPage(li int) bool {
	g.mu.Lock()
	d := g.own[li]
	g.mu.Unlock()
	if d == nil {
		return false
	}
	return d.ReleasesHostPage(li)
}

// HoldsExperts is nn.ExpertHolder: whether the device holding block li holds
// its routed bank, so the host may give the bank's expert pages back. A
// streamed or hybrid block holds none of it.
func (g *GPU) HoldsExperts(li int) bool {
	g.mu.Lock()
	d := g.own[li]
	g.mu.Unlock()
	if d == nil {
		return false
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	l, ok := d.layers[li]
	return ok && l.stream == nil
}

// RopeRows reads back the rotary table the first device's width-row batched
// scratch built last: row r at r*NRot, the layout nn.Rope.Table writes. It is
// for a gate that checks the table itself, since logit tolerances hide a table
// built for the wrong positions (model.TestRopeTablePrefillChunkMatchesTheUpload).
func (g *GPU) RopeRows(width int) ([]float32, error) {
	g.mu.Lock()
	d := g.devs[0]
	g.mu.Unlock()
	d.mu.Lock()
	defer d.mu.Unlock()
	bs := d.bbs[width]
	if bs == nil || bs.cs == nil {
		return nil, fmt.Errorf("tier: no %d-row batched scratch with a rotary table", width)
	}
	raw := make([]byte, width*bs.p.RopeW()*4)
	if err := bs.cs.Read(raw); err != nil {
		return nil, err
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&raw[0])), width*bs.p.RopeW()), nil
}

// CacheCoverage is, for every block with an expert cache, the share of its
// counted selections that its hottest slots' worth of experts carry, at the
// cache's size and at twice it: how much each block's routing a cache of
// that size can hold, and what a bigger one would add. A measurement for
// sizing caches by block; nil without caches. Not on a token's path.
func (g *GPU) CacheCoverage() (at, twice []float64) {
	for _, d := range g.devs {
		d.mu.Lock()
		for _, l := range d.layers {
			if l == nil || l.stream == nil || !l.stream.cached() || len(l.stream.heat) == 0 {
				continue
			}
			h := append([]uint32(nil), l.stream.heat...)
			slices.SortFunc(h, func(a, b uint32) int { return int(b) - int(a) })
			var tot, c1, c2 float64
			n := len(l.stream.owner)
			for i, v := range h {
				tot += float64(v)
				if i < n {
					c1 += float64(v)
				}
				if i < 2*n {
					c2 += float64(v)
				}
			}
			if tot > 0 {
				at, twice = append(at, c1/tot), append(twice, c2/tot)
			}
		}
		d.mu.Unlock()
	}
	return at, twice
}
