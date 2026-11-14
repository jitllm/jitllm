package gpu

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/nn"
)

// A vision tower's set is sized to the picture it encodes.
//
// A non-causal set (geoNonCausal) runs one picture per call, and every kernel
// it builds bakes its row count, so the set is built for a number of rows,
// visCap, which capped hands every tower plan as its MaxSeq: the scratch, the
// score rows, the shared k/v pair (transientkv.go) and the batched twins all
// follow it. A tower's own MaxSeq is the largest grid it takes, which a
// dynamic-resolution tower puts far past any picture a session encodes --
// 65536 patches on HunyuanOCR. The set used to be built for that at
// placement: 5,669,846,908 bytes of scratch on HunyuanOCR, allocated before
// the budget (2.3 GB on an integrated GPU, host RAM there) was asked, and every
// call ran all 65536 rows for a picture of a thousand.
//
// Placement builds the set for visStart rows, or the tower's MaxSeq when that
// is smaller (a fixed-grid tower's picture). ReserveRows -- before a
// picture's blocks run, outside any submission, as ReserveKV is for a
// history -- moves it to the picture's rows rounded up to an eighth of the
// power of two above them (visRowsFor): up when the picture has more rows than the set, after asking the
// budget for the build's bytes, extrapolated from the set it replaces; down
// when visShrinkCalls calls in a row have needed half the set or less, so a
// large picture once does not make every later one run its rows, and calls
// whose sizes alternate (an encoder's questions, short and long) do not
// rebuild the set -- and compile its kernels -- at every call. A picture the budget cannot hold is
// refused with nothing allocated, and the instance takes tower blocks to the CPU until
// it fits (engine.Instance.reserveTowerRows): the picture encodes on the host,
// never out of memory. The last tower block to leave takes the set with it
// (dropIdleTower), and so does a first block the budget refused.

// visStart is the rows a tower's set is built for at placement.
const visStart = 1024

// visShrinkCalls is how many calls in a row must need half the set or less
// before ReserveRows shrinks it.
const visShrinkCalls = 8

// visState is one non-causal set's sizing and the k/v pair its blocks share
// (transientkv.go). cap is the rows the set is built for and max the largest
// grid a block placed in it takes; keys the keys its k/v pairs and score rows
// hold when it is row-chunked (LayersPhase), a picture of keys rows running
// cap rows a call, 0 when it takes a picture whole; small the calls in a row
// it was more than twice what they needed (ReserveRows), which shrinks it at
// visShrinkCalls; first the rows it was first built for, every width a
// picture's set takes doubling from it up to max (widthFor), each formed at
// the placement (prewarmTowerWidths); chunk, for a width whose whole set did
// not fit there, the chunk its row-chunked set was formed at; warm the
// largest grid and the count of its blocks the widths were last formed for,
// so a placement that changes neither forms nothing; tkv the pair and
// tkvRefs how many hold it.
type visState struct {
	cap, max, keys, small, first int
	chunk                        map[int]int
	warm                         [2]int
	tkv                          *kvPair
	tkvRefs                      int
}

// visOf is set k's sizing, made on first use. Callers hold g.mu.
func (g *devTier) visOf(k geoKey) *visState {
	if g.vis == nil {
		g.vis = map[geoKey]*visState{}
	}
	v := g.vis[k]
	if v == nil {
		v = &visState{}
		g.vis[k] = v
	}
	return v
}

// vs is the sizing of the set in use. Callers hold g.mu.
func (g *devTier) vs() *visState { return g.visOf(g.geoCur) }

// towerVis is the sizing of the device's widest non-causal set that is not a
// decision head's, a zero one for none: what Stats reports and a chunked call
// reads. Callers hold g.mu.
func (g *devTier) towerVis() *visState {
	out := &visState{}
	for k, v := range g.vis {
		if k.flags&geoNonCausal != 0 && k.flags&geoNCHead == 0 && v.cap > out.cap {
			out = v
		}
	}
	return out
}

// towerSets is the keys of the non-causal sets this device's blocks run in,
// a decision head's aside (it moves with its encoder's). Callers hold g.mu.
func (g *devTier) towerSets() []geoKey {
	var out []geoKey
	for _, l := range g.layers {
		if l != nil && l.geo.flags&geoNonCausal != 0 && l.geo.flags&geoNCHead == 0 && !slices.Contains(out, l.geo) {
			out = append(out, l.geo)
		}
	}
	return out
}

// holdsSet reports whether a block of this device runs in set k. Callers
// hold g.mu.
func (g *devTier) holdsSet(k geoKey) bool {
	for _, l := range g.layers {
		if l != nil && l.geo == k {
			return true
		}
	}
	return false
}

// visGrain is the finest step a tower's rows move in.
const visGrain = 64

// visRowsFor is the rows a tower's set is built for to take a call of n rows:
// n rounded up to an eighth of the power of two at or above it, at least
// visGrain -- at most a quarter more rows than the picture has, and a few
// widths an octave, each its own set of batched twins -- and never past most,
// the tower's largest grid.
func visRowsFor(n, most int) int {
	step := visGrain
	for step*8 < n {
		step *= 2
	}
	return min(roundUp(max(n, 1), step), most)
}

// widthFor is the width a picture of n rows takes the set at: its first,
// doubled until it holds n, never past its largest grid. Every one is formed
// at the placement (prewarmTowerWidths), so a picture's encode allocates its
// set and composes nothing.
func (v *visState) widthFor(n int) int {
	w := max(v.first, 1)
	for w < n && w < v.max {
		w *= 2
	}
	return min(w, v.max)
}

// prewarmTowerWidths forms each non-causal set's parts at every width a
// picture's set takes (widthFor), building the set at each in turn and then
// as it was: the composer runs at the placement, and an encode only
// allocates. A width whose whole set does not fit forms the row-chunked set a
// picture of it takes (chunkTower), its chunk kept in the set's chunk.
// Callers hold g.mu.
func (g *devTier) prewarmTowerWidths() {
	for _, k := range g.towerSets() {
		g.prewarmSetWidths(k)
	}
}

// prewarmSetWidths is prewarmTowerWidths for set k. Callers hold g.mu.
func (g *devTier) prewarmSetWidths(k geoKey) {
	home := g.geoCur
	g.useGeom(k)
	defer g.useGeom(home)
	if g.NoPromptPrewarm || g.vs().first == 0 {
		return
	}
	towers := 0
	for _, l := range g.layers {
		if l != nil && l.geo == k {
			towers++
		}
	}
	if g.vs().warm == [2]int{g.vs().max, towers} {
		return
	}
	g.vs().warm = [2]int{g.vs().max, towers}
	if g.bs == nil {
		return
	}
	plan := g.bs.p
	plan.MaxSeq = g.vs().max
	first, keys, why := g.vs().cap, g.vs().keys, g.LastErr
	was := g.prewarming
	g.prewarming = true
	defer func() { g.prewarming = was }()
	back := func() bool {
		g.dropGraph()
		g.vs().keys = 0
		return g.towerComposedAt(&plan, first)
	}
	for w := g.vs().first; ; w = min(2*w, g.vs().max) {
		if w != first || keys != 0 {
			g.dropGraph()
			g.vs().keys = 0
			if !g.towerComposedAt(&plan, w) || !g.headSetAt(w) {
				if !back() {
					break
				}
				g.chunkTower(w, w)
			}
			g.formTables(g.bs)
		}
		if w >= g.vs().max {
			break
		}
	}
	if (g.vs().cap != first || g.vs().keys != 0) && !back() {
		g.LastErr = fmt.Sprintf("%s; and the tower's set at its first %d rows could not be rebuilt after its widths formed: %s",
			why, first, g.LastErr)
		return
	}
	if keys != 0 {
		g.chunkTower(keys, keys)
	}
	g.LastErr = why
}

// buildTowerScratch builds the set in use's scratch from plan p at visCap
// rows. Callers hold g.mu with the non-causal set in use.
func (g *devTier) buildTowerScratch(p *nn.LayerPlan) bool {
	cp := g.capped(p)
	rows := cp.MaxSeq
	if g.vs().keys > 0 {
		rows = g.vs().cap // a row-chunked set: chunk rows, every key's width
	}
	if g.bs = g.initScratch(&cp, rows); g.bs == nil {
		return false
	}
	// Token columns per thread, as prepBatch picks for a prompt.
	tok := BatchTok
	for tok > 1 && g.bs.rows%tok != 0 {
		tok /= 2
	}
	g.bs.tok = tok
	return true
}

// towerPlanesAt is towerPlanes for a set of rows rows whose score rows are
// keys wide: a row-chunked set's (LayersPhase), where keys is the picture's.
func (g *devTier) towerPlanesAt(p *nn.LayerPlan, rows, keys int) (int, int64) {
	q := *p
	q.MaxSeq = keys
	_, rc := visionChunk(rows, q.NHead, scoreStride(&q, rows), g.tiles.qtile, g.tiles.atile, g.planeBudget())
	qt, at := g.tiles.qtile, g.tiles.atile
	for rc%qt != 0 {
		qt /= 2
	}
	for rc%at != 0 {
		at /= 2
	}
	ss := scoreStride(&q, rc)
	ar, _ := visionChunk(rc, q.NHead, ss, qt, at, g.planeBudget())
	planes := int64(2)
	if q.AttnSoftcap != 0 {
		planes = 3
	}
	return rc, planes * int64(ar*q.NHead*ss*4)
}

// planeBytes is what bs's score planes hold.
func planeBytes(bs *blockScratch) int64 {
	n := int64(2)
	if bs.p.AttnSoftcap != 0 {
		n = 3
	}
	return n * int64(bs.arows*bs.p.NHead*bs.sstride*4)
}

// towerKV is every tower k/v pair of the non-causal set in use: the shared
// one (the encoder's set's) and any a block holds of its own (transientKV
// could not share). Callers hold g.mu.
func (g *devTier) towerKV() []*kvPair {
	var out []*kvPair
	if g.vs().tkv != nil {
		out = append(out, g.vs().tkv)
	}
	for _, l := range g.layers {
		if l == nil || l.geo != g.geoCur {
			continue
		}
		for _, kvp := range l.kv {
			if kvp != nil && !kvp.shared && !kvp.paged {
				out = append(out, kvp)
			}
		}
	}
	return out
}

// towerGrowth is what moving the set from old to rows rows adds to the budget,
// the old set freed first: the scratch's bytes extrapolated by the rows from
// what old measured, its score planes priced exactly (they are capped, not
// linear: visionChunk), and the k/v pairs at the new rows. An estimate, asked
// before anything is allocated; the build itself is charged by measurement
// and checked again.
func (g *devTier) towerGrowth(old *blockScratch, rows int) int64 {
	return g.towerGrowthAt(old, rows, rows)
}

// towerGrowthAt is towerGrowth for a set of rows rows over keys keys: the
// scratch scales with the rows, the score planes and k/v pairs with the keys.
func (g *devTier) towerGrowthAt(old *blockScratch, rows, keys int) int64 {
	p := old.p
	p.MaxSeq = keys
	rc, planes := g.towerPlanesAt(&p, rows, keys)
	lin := old.bytes - planeBytes(old)
	need := max(lin, 0)*int64(rc)/int64(max(old.rows, 1)) + planes - old.bytes
	cache, vcache := g.kvCacheBytes(&p, keys)
	for _, kvp := range g.towerKV() {
		need += int64(cache+vcache) - int64(kvp.bytes)
	}
	// The tower blocks' composed parts the set holds scale with its rows as
	// its scratch does.
	if parts := old.partBytes(); parts > 0 {
		need += parts*int64(rc)/int64(max(old.rows, 1)) - parts
	}
	return need
}

// partBytes is what bs holds of its composed parts: their instances at its
// rows and the arena they share.
func (bs *blockScratch) partBytes() int64 {
	var n int64
	for _, in := range bs.prompt {
		n += int64(in.bytes)
	}
	if a := bs.arena; a != nil {
		for _, m := range a.n {
			n += int64(m)
		}
	}
	return n
}

// ReserveRows makes the set block li runs in able to take a call of n rows
// (nn.RowsReserver): it grows to n, or shrinks to it when it has been twice
// what the calls needed for visShrinkCalls calls in a row, and false -- with
// the set as it was -- is a budget that cannot hold it. A device not holding
// block li as a non-causal block has nothing to size. A segment whose blocks
// attend at more than one geometry runs a set of each in one call, so every
// one of them is sized. Callers must NOT hold g.mu: a resize drops the
// captured graph and compiles.
func (g *devTier) ReserveRows(li, n int) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	l := g.layers[li]
	if l == nil || l.geo.flags&geoNonCausal == 0 {
		return true
	}
	home := g.geoCur
	defer g.useGeom(home)
	// Block li's set, then every other set of its segment on this device --
	// a tower's, or a non-causal text model's, one per geometry its blocks
	// attend at.
	var ks [4]geoKey
	sets := append(ks[:0], l.geo)
	for _, o := range g.layers {
		if o != nil && o.geo.flags == l.geo.flags && !slices.Contains(sets, o.geo) {
			sets = append(sets, o.geo)
		}
	}
	for _, k := range sets {
		g.useGeom(k)
		if !g.reserveSetRows(n) {
			return false
		}
	}
	return true
}

// segmentSplit says set k's segment runs in more than one set on this
// device: its blocks attend at more than one geometry. Callers hold g.mu.
func (g *devTier) segmentSplit(k geoKey) bool {
	for _, l := range g.layers {
		if l != nil && l.geo.flags == k.flags && l.geo != k {
			return true
		}
	}
	return false
}

// reserveSetRows is ReserveRows for the set in use. Callers hold g.mu.
func (g *devTier) reserveSetRows(n int) bool {
	if n > g.vs().max {
		g.LastErr = fmt.Sprintf("a call of %d rows on a tower whose largest grid is %d", n, g.vs().max)
		return false
	}
	want := g.vs().widthFor(n)
	if g.vs().keys == 0 && n <= g.vs().cap && !g.towerHeld() {
		// The whole set's composed parts are not held beside the blocks
		// placed: the picture takes a chunked set.
		g.vs().small = 0
		return g.chunkTower(n, want)
	}
	if g.vs().keys == 0 && (want == g.vs().cap || n <= g.vs().cap && 2*want > g.vs().cap) {
		g.vs().small = 0
		return true
	}
	if g.vs().keys == 0 && n <= g.vs().cap {
		if g.vs().small++; g.vs().small < visShrinkCalls {
			return true
		}
	}
	g.vs().small = 0
	// A row-chunked set holds a picture its keys cover, for as long as a
	// whole set still does not fit: tried first, since a picture taken
	// whole is fewer submissions.
	if g.resizeTower(n, want) {
		return true
	}
	if g.vs().keys >= n && g.vs().keys < 2*want {
		return true
	}
	return g.chunkTower(n, want)
}

// TowerChunk is the rows a row-chunked tower set takes a call (LayersPhase),
// 0 when the set takes a picture whole. Read after ReserveRows.
func (g *devTier) TowerChunk() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	if v := g.towerVis(); v.keys != 0 {
		return v.cap
	}
	return 0
}

// towerKeep is the room a streamed tower's set must leave: minSlots of the
// widest block to swap through, 0 for a resident tower. Callers hold g.mu.
func (g *devTier) towerKeep() int64 {
	for li, l := range g.layers {
		if l != nil && l.geo == g.geoCur && g.stream[li] {
			return int64(minSlots) * int64(g.widest)
		}
	}
	return 0
}

// towerWindowed reports a tower block that masks keys by row windows, which
// a row-chunked call cannot: the windows are rows of the whole picture.
// Callers hold g.mu.
func (g *devTier) towerWindowed() bool {
	for _, l := range g.layers {
		if l != nil && l.geo == g.geoCur && l.windowed {
			return true
		}
	}
	return false
}

// chunkTower builds the non-causal set row-chunked when no whole set for a
// picture of n rows fits: k/v pairs and score rows for want keys, the rest
// of the scratch for a chunk of rows, the widest chunk that fits, halving
// down to visGrain. The tower then runs a picture block by block in two
// phases a chunk at a time (LayersPhase). Refused for windowed towers, an
// encoder with a decision head's set, a segment of more than one geometry
// (each set would have to take the one chunk TowerChunk reports), and where
// even the narrowest chunk does not fit. Callers hold g.mu with the set in use.
func (g *devTier) chunkTower(n, want int) bool {
	if g.NoTowerChunks || g.towerWindowed() || g.geoCur.flags&geoTower == 0 && g.hasHeadSet() ||
		g.segmentSplit(g.geoCur) {
		return false
	}
	old := g.bs
	if old == nil {
		return false
	}
	keep := g.towerKeep()
	plan := old.p
	plan.MaxSeq = g.vs().max
	prevCap, prevKeys, why := g.vs().cap, g.vs().keys, g.LastErr
	// A call takes the chunk its width was formed at (prewarmTowerWidths);
	// the placement searches for the widest that fits and keeps it.
	top, formed := g.vs().chunk[want]
	if !g.prewarming && !formed {
		g.LastErr = fmt.Sprintf("%s; and no row-chunked set of %d keys was formed at the placement", why, want)
		return false
	}
	if !formed {
		top = roundUp(want/2, visGrain)
	}
	for c := top; c >= visGrain; c = roundUp(c/2, visGrain) {
		need := g.towerGrowthAt(old, c, want) + keep
		if need > 0 && !g.room(uint64(need)) {
			g.reclaim(uint64(need), -1, 0, 0)
		}
		if !g.room(uint64(max(need, 0))) {
			if c == visGrain {
				break
			}
			continue
		}
		g.dropGraph()
		g.vs().keys = want
		if g.towerComposedAt(&plan, c) && (keep == 0 || g.slots() >= minSlots) {
			g.TowerChunked++
			if g.prewarming {
				if g.vs().chunk == nil {
					g.vs().chunk = map[int]int{}
				}
				g.vs().chunk[want] = c
			}
			return true
		}
		g.vs().keys = prevKeys
		if !g.towerAt(&plan, prevCap) {
			g.LastErr = fmt.Sprintf("%s; and the set's %d rows could not be rebuilt: %s", why, prevCap, g.LastErr)
			return false
		}
		if c == visGrain || !g.prewarming {
			break
		}
	}
	g.LastErr = fmt.Sprintf("%s; and no chunk of %d rows down to %d fits beside the picture's %d keys",
		why, top, visGrain, want)
	return false
}

// resizeTower moves the set in use to want rows for a call of n. Callers
// hold g.mu with the set in use.
func (g *devTier) resizeTower(n, want int) bool {
	old := g.bs
	if old == nil {
		g.LastErr = "a tower block with no scratch set"
		return false
	}
	// A streamed tower swaps its blocks through the device's slots, so
	// the set may take the room the resident ones hold but must leave
	// minSlots of the widest block to swap through; a set that leaves
	// fewer is refused as before, and the blocks run on the CPU.
	keep := g.towerKeep()
	if want > g.vs().cap || g.vs().keys > 0 {
		// Asked before anything is allocated. The step may not fit where the
		// picture's own rows would.
		need := g.towerGrowth(old, want) + keep
		// A set that does not fit takes its attention in narrower passes
		// first: the score planes are the part of it that does not scale
		// with the rows' activations, and a pass more is cheap beside a
		// picture the device refuses.
		for need > 0 && !g.room(uint64(need)) && g.planeBudget()/2 >= minPlaneBudget {
			g.planeCap = g.planeBudget() / 2
			need = g.towerGrowth(old, want) + keep
		}
		// A streamed block resident in a slot is room: it pages out, and in
		// again when its turn in the call comes (paging is the design). A
		// resident placement's blocks are not victims (victim, streamed).
		if need > 0 && !g.room(uint64(need)) {
			g.reclaim(uint64(need), -1, 0, 0)
		}
		if !g.room(uint64(max(need, 0))) {
			g.LastErr = fmt.Sprintf("a picture of %d rows: the tower's set at %d rows needs about %d bytes "+
				"more than it holds, and %d of %d are used", n, want, need, g.used, g.limit)
			return false
		}
	}
	plan := old.p
	plan.MaxSeq = g.vs().max
	prev, prevKeys := g.vs().cap, g.vs().keys
	g.dropGraph()
	g.vs().keys = 0 // a whole set
	if g.towerComposedAt(&plan, want) {
		// The estimate is an estimate: what the set really charged must
		// still leave a streamed tower its slot, or the first page-in of
		// the call fails the device mid-picture.
		if keep == 0 || g.slots() >= minSlots {
			return g.headSetAt(want)
		}
		g.LastErr = fmt.Sprintf("a picture of %d rows: the tower's set at %d rows leaves %d block slot(s), "+
			"below the %d a swap needs", n, want, g.slots(), minSlots)
	}
	why := g.LastErr
	// What fitted before fits again, its blocks' composed parts with it: the
	// old rows, or nothing at all, and the caller moves blocks to the CPU.
	g.vs().keys = prevKeys
	if !g.towerComposedAt(&plan, prev) {
		g.LastErr = fmt.Sprintf("%s; and the set's %d rows could not be rebuilt: %s", why, prev, g.LastErr)
		return false
	}
	g.LastErr = why
	return false
}

// towerAt rebuilds the non-causal set, which is in use, at rows rows from plan
// p (its MaxSeq the tower's largest grid): the scratch, the k/v pairs and
// every tower block's batched twins. The old set is freed first, so a tight
// budget can hold the new one where it could not hold both; a build the
// budget does not hold is freed again. Callers hold g.mu and dropped the graph.
func (g *devTier) towerAt(p *nn.LayerPlan, rows int) bool {
	// The build's own window, closed before the budget is read: the scratch
	// is charged when it settles.
	ok := func() bool {
		defer g.scratchWin().close()
		for w := range g.bbs {
			g.dropBatch(w)
		}
		g.dropScratch(g.bs)
		g.bs = nil
		g.vs().cap = rows
		return g.buildTowerScratch(p) && g.towerKVAt(p)
	}()
	if ok && g.overBudget() {
		g.LastErr = fmt.Sprintf("the tower's set at %d rows passes the budget: %d of %d used", rows, g.used, g.limit)
		ok = false
	}
	if !ok && g.bs != nil {
		g.dropScratch(g.bs)
		g.bs = nil
	}
	return ok
}

// towerHeld reports that every tower block's composed part and its
// instance at the set's rows are held (prewarmTower), forming them where they
// fit. Callers hold g.mu.
func (g *devTier) towerHeld() bool {
	for li, l := range g.layers {
		if l != nil && l.nonCausal && g.heldWith(l.geo) && g.prewarmTower(li) != "" {
			return false
		}
	}
	return true
}

// heldWith says a block of set k is held beside the set in use: its own, or
// a decision head's beside its encoder's, which moves with it (headSetAt).
// Callers hold g.mu.
func (g *devTier) heldWith(k geoKey) bool {
	return k == g.geoCur || g.geoCur.flags&geoTower == 0 && k.flags&geoNCHead != 0
}

// towerComposedAt is towerAt holding every tower block's composed part at
// the set's rows too, as their placement does (prewarmTower): a set whose
// parts the budget does not hold is refused here, so a picture takes a
// chunked set rather than meeting no room at its call. Callers hold g.mu
// and dropped the graph.
func (g *devTier) towerComposedAt(p *nn.LayerPlan, rows int) bool {
	if !g.towerAt(p, rows) {
		return false
	}
	if !g.towerHeld() {
		g.dropScratch(g.bs)
		g.bs = nil
		return false
	}
	return true
}

// towerKVAt reallocates every tower k/v pair for visCap rows, moving its
// charge by the difference. Callers hold g.mu.
func (g *devTier) towerKVAt(p *nn.LayerPlan) bool {
	cp := g.capped(p)
	cache, vcache := g.kvCacheBytes(p, cp.MaxSeq)
	for _, kvp := range g.towerKV() {
		for _, b := range []*backend.Buf{&kvp.kc, &kvp.vc} {
			if *b != nil {
				(*b).Free()
				*b = nil
			}
		}
		g.refund(kvp.bytes)
		g.KVBytes -= min(g.KVBytes, kvp.bytes)
		kvp.bytes = 0
		var err error
		if kvp.kc, err = g.dev.Alloc(cache); err == nil && vcache > 0 {
			kvp.vc, err = g.dev.Alloc(vcache)
		}
		if err != nil {
			g.LastErr = fmt.Sprintf("the tower's k/v pair at %d rows: %v", cp.MaxSeq, err)
			return false
		}
		kvp.bytes = uint64(cache + vcache)
		g.charge(kvp.bytes)
		g.KVBytes += kvp.bytes
	}
	for _, l := range g.layers {
		if l != nil && l.geo == g.geoCur && l.kvBytes > 0 {
			l.kvBytes = uint64(cache + vcache)
		}
	}
	return true
}

// dropIdleTower frees each non-causal set no block is left in: its last one
// released, or a first one the budget refused after its set was built.
// Callers hold g.mu.
func (g *devTier) dropIdleTower() {
	home := g.geoCur
	for k, v := range g.vis {
		if k.flags&geoNonCausal == 0 || v.cap == 0 || g.holdsSet(k) {
			continue
		}
		g.useGeom(k)
		if g.bs != nil || len(g.bbs) > 0 {
			g.dropGraph()
			for w := range g.bbs {
				g.dropBatch(w)
			}
			g.dropScratch(g.bs)
			g.bs = nil
		}
		v.cap, v.max, v.first, v.chunk, v.warm = 0, 0, 0, nil, [2]int{}
	}
	g.useGeom(home)
}

// hasHeadSet reports a decision head's own non-causal set on this device
// (geoNCHead). Callers hold g.mu.
func (g *devTier) hasHeadSet() bool {
	for _, l := range g.layers {
		if l != nil && l.geo.flags&geoNCHead != 0 {
			return true
		}
	}
	return false
}

// headSetAt moves a decision head's own set to rows rows, as resizeTower
// moved the encoder's: one call runs both, so both hold its rows. A device
// with no head set has nothing to move. Callers hold g.mu and dropped the
// graph.
func (g *devTier) headSetAt(rows int) bool {
	if g.geoCur.flags&geoTower != 0 || !g.hasHeadSet() {
		return true
	}
	home := g.geoCur
	defer g.useGeom(home)
	// One set per geometry the head's blocks attend at.
	var ks [4]geoKey
	sets := ks[:0]
	for _, l := range g.layers {
		if l != nil && l.geo.flags&geoNCHead != 0 && !slices.Contains(sets, l.geo) {
			sets = append(sets, l.geo)
		}
	}
	for _, k := range sets {
		g.useGeom(k)
		if g.bs == nil {
			continue
		}
		plan := g.bs.p
		plan.MaxSeq = g.vs().max
		if !g.towerAt(&plan, rows) {
			g.LastErr = fmt.Sprintf("the decision head's set at %d rows: %s", rows, g.LastErr)
			return false
		}
	}
	return true
}

// ReserveRows asks every device (nn.RowsReserver); one not holding block li
// says yes.
func (g *GPU) ReserveRows(li, n int) bool {
	g.resetCause(0, true)
	g.mu.Lock()
	ds := g.devs
	g.mu.Unlock()
	for _, d := range ds {
		if !d.ReserveRows(li, n) {
			return false
		}
	}
	return true
}

// ReserveRows forwards under the session's call locks: a resize rebuilds the
// set another session's call would run in.
func (s *gpuSession) ReserveRows(li, n int) bool {
	defer s.leave(s.enter())
	return s.g.ReserveRows(li, n)
}

// TowerChunk is the rows every device's row-chunked tower set takes a call,
// 0 when every set takes a picture whole (nn.RowChunker). Devices that
// disagree report the narrowest: a phase call is cut to it on all of them.
func (g *GPU) TowerChunk() int {
	g.mu.Lock()
	ds := g.devs
	g.mu.Unlock()
	c := 0
	for _, d := range ds {
		if k := d.TowerChunk(); k > 0 && (c == 0 || k < c) {
			c = k
		}
	}
	return c
}

// LayersPhase is Layers over one phase of a row-chunked non-causal block
// (nn.RowChunker): rows [pos, pos+n) of a picture of keys rows. Phase 1
// writes the rows' k and v at their positions and leaves x as it was;
// phase 2 runs their queries over all keys rows' k and v and the rest of
// the blocks, writing their x. Every chunk's phase 1 runs before any
// chunk's phase 2, block by block.
func (g *GPU) LayersPhase(lo, hi, pos, n, keys, phase int, x, cs, csSWA []float32) bool {
	return g.layersPhase(0, lo, hi, pos, n, keys, phase, x, cs, csSWA)
}

func (g *GPU) layersPhase(sid uint64, lo, hi, pos, n, keys, phase int, x, cs, csSWA []float32) bool {
	g.resetCause(sid, true)
	if phase != 1 && phase != 2 || keys < pos+n {
		return false
	}
	g.mu.Lock()
	ds := g.devs
	g.mu.Unlock()
	set := func(ph, k int) {
		for _, d := range ds {
			d.mu.Lock()
			ss := d.sessOf(sid)
			ss.ncPhase, ss.ncKeys = ph, k
			if ph != 0 {
				d.TowerPhases++
			}
			d.mu.Unlock()
		}
	}
	set(phase, keys)
	defer set(0, 0)
	return g.layersFor(sid, lo, hi, pos, n, x, cs, csSWA, nil)
}

// TowerChunk and LayersPhase forward under the session's locks.
func (s *gpuSession) TowerChunk() int {
	defer s.leave(s.enter())
	return s.g.TowerChunk()
}

func (s *gpuSession) LayersPhase(lo, hi, pos, n, keys, phase int, x, cs, csSWA []float32) bool {
	defer s.leave(s.inputsTo(s.step()))
	return s.g.layersPhase(s.sid, lo, hi, pos, n, keys, phase, x, cs, csSWA)
}
