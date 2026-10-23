package tier

import (
	"fmt"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
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
// refused with nothing allocated, and the State takes tower blocks home until
// it fits (model.State.reserveTowerRows): the picture encodes on the host,
// never out of memory. The last tower block to leave takes the set with it
// (dropIdleTower), and so does a first block the budget refused.

// visStart is the rows a tower's set is built for at placement.
const visStart = 1024

// visShrinkCalls is how many calls in a row must need half the set or less
// before ReserveRows shrinks it.
const visShrinkCalls = 8

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

// holdsTower reports whether a tower block (a non-causal block that runs in
// the non-causal set, not a convolutional one) is on this device. Callers
// hold g.mu.
func (g *devTier) holdsTower() bool {
	for _, l := range g.layers {
		if l != nil && l.geo&geoNonCausal != 0 {
			return true
		}
	}
	return false
}

// buildTowerScratch builds the set in use's scratch from plan p at visCap
// rows. Callers hold g.mu with the non-causal set in use.
func (g *devTier) buildTowerScratch(p *nn.LayerPlan) bool {
	cp := g.capped(p)
	rows := cp.MaxSeq
	if g.visKeys > 0 {
		rows = g.visCap // a row-chunked set: chunk rows, every key's width
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

// towerPlanes is the rows a tower scratch for rows rows is built at, and the
// bytes of its score planes: initScratch's arithmetic, without building it.
func (g *devTier) towerPlanes(p *nn.LayerPlan, rows int) (int, int64) {
	return g.towerPlanesAt(p, rows, rows)
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
	if g.tkv != nil && g.geoCur == geoNonCausal {
		out = append(out, g.tkv)
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
	return need
}

// ReserveRows makes this device's tower blocks able to take a call of n rows
// (nn.RowsReserver): their set grows to n, or shrinks to it when it has been
// twice what the calls needed for visShrinkCalls calls in a row, and false --
// with the set as it was -- is a budget that cannot hold it. A device holding no tower block has nothing to size. Callers must
// NOT hold g.mu: a resize drops the captured graph and compiles.
func (g *devTier) ReserveRows(n int) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.holdsTower() {
		return true
	}
	if n > g.visMax {
		g.LastErr = fmt.Sprintf("a call of %d rows on a tower whose largest grid is %d", n, g.visMax)
		return false
	}
	want := visRowsFor(n, g.visMax)
	if g.visKeys == 0 && (want == g.visCap || n <= g.visCap && 2*want > g.visCap) {
		g.visSmall = 0
		return true
	}
	if g.visKeys == 0 && n <= g.visCap {
		if g.visSmall++; g.visSmall < visShrinkCalls {
			return true
		}
	}
	g.visSmall = 0
	// A row-chunked set holds a picture its keys cover, for as long as a
	// whole set still does not fit: tried first, since a picture taken
	// whole is fewer submissions.
	if g.resizeTower(n, want) {
		return true
	}
	if g.visKeys >= n && g.visKeys < 2*want {
		return true
	}
	return g.chunkTower(n, want)
}

// TowerChunk is the rows a row-chunked tower set takes a call (LayersPhase),
// 0 when the set takes a picture whole. Read after ReserveRows.
func (g *devTier) TowerChunk() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.visKeys == 0 {
		return 0
	}
	return g.visCap
}

// towerKeep is the room a streamed tower's set must leave: minSlots of the
// widest block to swap through, 0 for a resident tower. Callers hold g.mu.
func (g *devTier) towerKeep() int64 {
	for li, l := range g.layers {
		if l != nil && l.geo == geoNonCausal && g.stream[li] {
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
		if l != nil && l.geo == geoNonCausal && l.windowed {
			return true
		}
	}
	return false
}

// chunkTower builds the non-causal set row-chunked when no whole set for a
// picture of n rows fits: k/v pairs and score rows for want keys, the rest
// of the scratch for a chunk of rows, the widest chunk that fits, halving
// down to visGrain. The tower then runs a picture block by block in two
// phases a chunk at a time (LayersPhase). Refused for windowed towers and
// where even the narrowest chunk does not fit. Callers hold g.mu.
func (g *devTier) chunkTower(n, want int) bool {
	if g.NoTowerChunks || g.towerWindowed() || g.hasHeadSet() {
		return false
	}
	home := g.geoCur
	g.useGeom(geoNonCausal)
	defer g.useGeom(home)
	old := g.bs
	if old == nil {
		return false
	}
	keep := g.towerKeep()
	plan := old.p
	plan.MaxSeq = g.visMax
	prevCap, prevKeys, why := g.visCap, g.visKeys, g.LastErr
	for c := roundUp(want/2, visGrain); c >= visGrain; c = roundUp(c/2, visGrain) {
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
		g.visKeys = want
		if g.towerAt(&plan, c) && (keep == 0 || g.slots() >= minSlots) {
			g.TowerChunked++
			return true
		}
		g.visKeys = prevKeys
		if !g.towerAt(&plan, prevCap) {
			g.LastErr = fmt.Sprintf("%s; and the set's %d rows could not be rebuilt: %s", why, prevCap, g.LastErr)
			return false
		}
		if c == visGrain {
			break
		}
	}
	g.LastErr = fmt.Sprintf("%s; and no chunk of %d rows down to %d fits beside the picture's %d keys",
		why, want/2, visGrain, want)
	return false
}

// resizeTower moves the non-causal set to want rows for a call of n. Callers
// hold g.mu.
func (g *devTier) resizeTower(n, want int) bool {
	home := g.geoCur
	g.useGeom(geoNonCausal)
	defer g.useGeom(home)
	old := g.bs
	if old == nil {
		g.LastErr = "a tower block with no scratch set"
		return false
	}
	// A streamed tower swaps its blocks through the device's slots, so
	// the set may take the room the resident ones hold but must leave
	// minSlots of the widest block to swap through; a set that leaves
	// fewer is refused as before, and the blocks run at home.
	keep := g.towerKeep()
	if want > g.visCap || g.visKeys > 0 {
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
			exact := min(roundUp(n, visGrain), g.visMax)
			if exact >= want || !g.room(uint64(max(g.towerGrowth(old, exact)+keep, 0))) {
				g.LastErr = fmt.Sprintf("a picture of %d rows: the tower's set at %d rows needs about %d bytes "+
					"more than it holds, and %d of %d are used", n, want, need, g.used, g.limit)
				return false
			}
			want = exact
		}
	}
	plan := old.p
	plan.MaxSeq = g.visMax
	prev, prevKeys := g.visCap, g.visKeys
	g.dropGraph()
	g.visKeys = 0 // a whole set
	if g.towerAt(&plan, want) {
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
	// What fitted before fits again: the old rows, or nothing at all, and the
	// caller takes blocks home.
	g.visKeys = prevKeys
	if !g.towerAt(&plan, prev) {
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
		g.visCap = rows
		return g.buildTowerScratch(p) && g.towerKVAt(p) && g.towerTwins()
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

// towerTwins builds every tower block's batched twins at the scratch's rows,
// which PrepLayer builds for a placed block and a resize must build for all:
// a Compile from inside a submission deadlocks. Callers hold g.mu.
func (g *devTier) towerTwins() bool {
	for li, l := range g.layers {
		if l == nil || l.geo != g.geoCur {
			continue
		}
		for _, m := range []*mv{&l.mvq, &l.mvk, &l.mvv, &l.mvo, &l.mvg, &l.mvu, &l.mvd, &l.mvSG, &l.mvBA, &l.mvSO} {
			if m.kern == nil {
				continue
			}
			if _, ok := g.batchMV(*m, g.bs.rows, g.bs.tok); !ok {
				g.LastErr = fmt.Sprintf("non-causal block %d: no batched twin at %d rows", li, g.bs.rows)
				return false
			}
		}
		// A block's own activation bakes the set's rows too.
		if l.actK != nil {
			l.actK.Close()
			l.actK = nil
		}
		if !g.ownActKern(l) {
			return false
		}
	}
	return true
}

// ownActKern builds non-causal block l's own activation kernel at the set's
// rows, where its MLP's gating or activation is not the set's (layer.actK):
// the set's bs.actMul bakes the first block's. Callers hold g.mu with the
// non-causal set in use.
func (g *devTier) ownActKern(l *layer) bool {
	bs := g.bs
	if bs == nil || (l.ungated == ungatedFFN(&bs.p) && l.act == bs.p.Act) {
		return true
	}
	k, err := kernels.ActMul(bs.rows*l.nffn, l.act)
	if l.ungated {
		k, err = kernels.Act(bs.rows*l.nffn, l.act)
	}
	if err == nil {
		l.actK, err = g.dev.Compile(k)
	}
	if err != nil {
		g.LastErr = fmt.Sprintf("a non-causal block's own %s activation: %v", l.act, err)
		return false
	}
	return true
}

// dropIdleTower frees the non-causal set when no tower block is left on the
// device: the last one released, or a first one the budget refused after its
// set was built. Callers hold g.mu.
func (g *devTier) dropIdleTower() {
	if g.visCap == 0 || g.holdsTower() {
		return
	}
	home := g.geoCur
	for _, k := range []geoKey{geoNonCausal, geoNonCausal | geoNCHead} {
		g.useGeom(k)
		if g.bs != nil || len(g.bbs) > 0 {
			g.dropGraph()
			for w := range g.bbs {
				g.dropBatch(w)
			}
			g.dropScratch(g.bs)
			g.bs = nil
		}
	}
	g.useGeom(home)
	g.visCap, g.visMax = 0, 0
}

// hasHeadSet reports a decision head's own non-causal set on this device
// (geoNCHead). Callers hold g.mu.
func (g *devTier) hasHeadSet() bool {
	for _, l := range g.layers {
		if l != nil && l.geo == geoNonCausal|geoNCHead {
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
	if !g.hasHeadSet() {
		return true
	}
	home := g.geoCur
	g.useGeom(geoNonCausal | geoNCHead)
	defer g.useGeom(home)
	if g.bs == nil {
		return true
	}
	plan := g.bs.p
	plan.MaxSeq = g.visMax
	if !g.towerAt(&plan, rows) {
		g.LastErr = fmt.Sprintf("the decision head's set at %d rows: %s", rows, g.LastErr)
		return false
	}
	return true
}

// ReserveRows asks every device (nn.RowsReserver); one holding no tower block
// says yes.
func (g *GPU) ReserveRows(n int) bool {
	g.mu.Lock()
	ds := g.devs
	g.mu.Unlock()
	for _, d := range ds {
		if !d.ReserveRows(n) {
			return false
		}
	}
	return true
}

// ReserveRows forwards under the session's call locks: a resize rebuilds the
// set another session's call would run in.
func (s *gpuSession) ReserveRows(n int) bool {
	defer s.leave(s.enter())
	return s.g.ReserveRows(n)
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
