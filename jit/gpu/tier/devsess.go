package tier

import (
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
)

// A device serves several sessions at once. What it holds for all of them --
// the blocks, the kernels, the paged history, the recurrent pools, the budget
// -- is its devShared. What a session's calls write is the session's own: the
// per-call fields below (devSess), and the scratch a call runs in (a lane). A
// devTier is one session's view of one device: the three together, and the
// counters its writes go to.
//
// The scratch is the expensive part, so a lane is held for a call, not for a
// session's life. lane0 is the scratch placement builds; a call takes it when
// it is free, and otherwise the lane its session ran in last, any other free
// lane, or a new clone of lane0 (takeLane) -- built from lane0's plans and
// charged to the budget as scratch. Sessions that step one after another run
// in one lane, as they did when the scratch was the device's; each session
// stepping at the same time as another costs one more.
//
// lane0 is what placement, PrepHead, a capacity growth and every other change
// of shape act on, through the device's own view (devShared.dv). A clone is
// not kept in step with it: laneGen moves when lane0 changes shape, and a
// clone built at an older generation is dropped rather than run.
//
// A vision tower's set is not cloned: it is one per device (visrows.go), as
// the tower's k/v pair is (transientkv.go), and a call whose lane is a clone
// runs a tower's range in lane0 (lane0View).

// devSess is one session's per-call state on one device.
type devSess struct {
	sid uint64
	// cur is the lane the session's current call holds, nil between calls,
	// and depth how many entries deep that call is (as, done); last is the
	// lane its previous call ran in, which the next prefers -- its
	// recordings are there. view is the session's devTier, over cur during a
	// call; v0 its devTier over lane0, for the ranges only lane0 runs.
	cur, last *lane
	depth     int
	view, v0  *devTier
	// ncPhase and ncKeys run the session's non-causal call as one phase of a row-chunked
	// block (LayersPhase): 1 writes the chunk's k and v at its rows and runs
	// nothing after them, 2 runs the chunk's queries over the ncKeys keys
	// phase 1 wrote and the rest of the block. 0 is a whole call.
	ncPhase, ncKeys int
	// inSub says the session's submission is running outside g.mu: what it
	// builds on the way (a kernel, a grown staging buffer) takes g.mu for the
	// build (subLock; subHeld while it does), and nothing it calls may
	// quiesce. Its counters go to acc meanwhile, added to the device's when it
	// completes (layersOnce).
	inSub, subHeld bool
	acc            Stats
	// prologueHeld says layersOnce is between its waits and its submission:
	// what it read into the submission's descriptors must not change, so
	// nothing it calls may let g.mu go (quiesce refuses).
	prologueHeld bool
	// q is the session's queue on a device that runs sessions at once
	// (backend.Queued), made at its first submission; qTried says that was
	// asked, so a device that refused is not asked every token. Without one
	// the session's submissions take the device's Session.
	q      backend.Queue
	qTried bool
	// qd is the device as a backend.Queued, asserted once with the queue:
	// the assertion on every submission filled the runtime's type-assertion
	// cache inside a timed decode, one allocation the first time.
	qd backend.Queued
	// runF is the submission runSub is running, and ranFn runSub's wrapper
	// around it, a method value made once so a submission allocates nothing.
	runF  func(backend.Session)
	ranFn func(backend.Session)
	// bare is the session's view with no lane, for a call that reads no
	// scratch (reserveKV, trimKV): such a call must not cost a clone.
	bare *devTier
	// winRuns is the windowed key runs SetKeyRuns gave this session, staged
	// into the tower's set at each of its calls over a tower (setWindows):
	// the set is the device's, and another session's picture may have been
	// staged in it since.
	winRuns []nn.KeyRun
	// seqRuns is the packed sequences SetSeqRuns gave this session, staged
	// with the windows at each of its non-causal calls.
	seqRuns []nn.KeyRun
	// winPerRow says winRuns is one window a row (SetRowWindows: a symmetric
	// sliding window, whose rows' windows overlap) rather than runs
	// partitioning the rows.
	winPerRow bool
	// embPend is the prompt chunk EmbedRows promised and the next submission
	// gathers (embed.go). Guarded by mu.
	embPend *embPend
	// embBuf is what embPend points at, kept so a prompt chunk's promise
	// allocates nothing past the first.
	embBuf embPend
	// bidir is the full key runs the next causal Layers calls honour
	// (keyruns.go). Guarded by mu.
	bidir []nn.KeyRun
	// winSegs is setWindows' [lo, hi) pairs, kept so a picture's windows
	// allocate nothing past the first. Guarded by mu.
	winSegs []int
	// winBuf is the windows' upload, one [lo, hi) word pair a row, kept for
	// the same reason: a warm call stages them again and allocates nothing.
	winBuf []byte
	// pleHost is the rows' per-layer embedding inputs for the next call
	// (SetLayerInputs) and pleStage their padded staging.
	pleHost, pleStage []float32
	// tokIDs is the rows' token ids for the next call (SetTokenIDs), which a
	// DeepSeek V4 block routing by token reads (ds4.go).
	tokIDs []int32
	// ngHost is the rows' n-gram table rows for the next call (SetNGramRows),
	// which Qwen3.8-Flash-Next's n-gram block reads (q4e.go).
	ngHost []float32
	// qsaMarks is the rows' rotary marks for the next call (SetRopeMarks),
	// which a QSA block's pooled keys turn by (qsa.go).
	qsaMarks []int32
	// prompt says the session's calls are a prompt's (SetPrompt): the
	// composer serves a step's parts and none of a prompt's (Config.Composed).
	prompt bool
	// recSteps counts the linear blocks whose emitLinear completed with the
	// submission still good, for the current Layers call: the measured answer
	// to "may a failed submission be restarted on the host". Positional proxies
	// (the run's start index, or "a linear block is placed") are wrong in one
	// direction or the other.
	recSteps int
	// sub is the submission layersOnce hands layersSession, and subBytes its
	// staging bytes; see submitArgs and stagingBytes.
	sub      submitArgs
	subBytes stagingBytes
	// launchTo is the submission's launch target; see launcher.
	launchTo launchTo
	// recDescW is recDesc's words between submissions.
	recDescW []uint32
	// stepRS is the ragStep GPU.layersRows hands this device, its slices
	// reused from step to step.
	stepRS ragStep
	// rag is the rows of an in-flight LayersRows; see rows.go.
	rag *ragStep
	// recSess and rec are recorderOf's cache.
	recSess backend.Session
	rec     backend.Recorder
	// pgSeqs is pagedAppendRows' and ReserveKVSeqs' staging, reused so a
	// token allocates nothing.
	pgSeqs []seqEnd
	pgRows []pagedRow
	// graphOff is the tier deciding, not the caller: this backend has no
	// capture, or a capture failed once. Either way, do not keep asking.
	graphOff bool
}

// lane is a scratch: the block scratches of every geometry set, the batched
// ones by width, the staging a submission launches against, and the
// recordings made over them. Every kernel in it bakes its buffers, so a call
// holds the lane it runs in until it returns.
type lane struct {
	bs *blockScratch
	// served are the parts the composer serves on this lane, by their key,
	// each the lane's instance of a device's form (served.go); capturing says
	// the lane's submission is being recorded, so a part not formed yet is
	// not formed inside it (errServedCold). formErr is why a part whose
	// entry is nil did not form.
	served    map[servedKey]*servedPart
	formErr   map[servedKey]error
	capturing bool
	// bbs is the batched-prefill scratch, one per chunk width: a prompt asks
	// for the full chunk and then a narrower tail, and each width bakes its own
	// row count into every kernel. See prepBatch.
	bbs map[int]*blockScratch
	// geoSets is the scratch sets of the geometries not in use and geoCur the
	// one in use (gemma4.go). promptW is the prompt width reserved at
	// placement, 0 where none is; stepW the ragged step's.
	geoSets        map[geoKey]geomSet
	geoCur         geoKey
	promptW, stepW int
	// exactWs are the prompt chunk widths built for a chunk's own width
	// (layersFor), least recently used first (keepExact).
	exactWs []int
	// exactDropped is the exact widths given back for room (keepExact):
	// one asked for again runs in a kept wider width rather than giving back
	// another, so prompts of more lengths than the lane keeps settle.
	exactDropped map[int]bool
	// noRoom is the batched widths prepBatch built and then refused for the
	// budget, by geometry set, with the budget as it stood (noRoomFor).
	noRoom map[noRoomKey]budgetAt
	// planeCap narrows the vision score planes below planeBudget's when a
	// picture's set does not fit the budget at the default (resizeTower);
	// 0 is no cap.
	planeCap        int
	partBuf, f16Buf backend.Buf
	partCap, f16Cap int
	// argmaxOut is the device argmax's one word (PrepHead).
	argmaxOut backend.Buf
	// The device sampler's buffers (sample.go), made the first time a call
	// in this lane samples: the history, the penalized row, the first
	// pass's candidates and the k that come home.
	sampleArgs, samplePen, sampleV1, sampleI1, sampleV, sampleI backend.Buf
	// The label pick's ids and the logits it gathers (pick.go), made the
	// first time a call in this lane picks, and the ids' host staging.
	pickIDs, pickOut backend.Buf
	pickHost         []byte
	// recs is a map, not one slot, because a token can issue more than one
	// sequence (blocks under {0,gl} and a head-only call under {gl,gl}); one
	// slot would have them evict each other, a capture per call. A recording
	// bakes this lane's buffers and its session's pages: it is keyed by the
	// session (graphKey.sid) and kept with the lane.
	recs map[graphKey]backend.Recording
	// recUsed is when each recording last ran, on the recTick clock, so the
	// per-session bound (maxGraphs) retires the least recently used rather
	// than every one. An entry whose recording is gone is pruned there.
	recUsed map[graphKey]uint64
	recTick uint64
	// recParts is the composer's parts each recording holds, by part, which
	// a replay of it runs again (Stats.PartsReplayed).
	recParts map[graphKey][numParts]int
	// stale holds recordings that were retired from inside a Session and so
	// could not be destroyed there; see dropGraph.
	stale []backend.Recording
	// clone says the lane was built from lane0 (cloneLane), at lane0's
	// generation gen; held says a call runs in it. A clone borrows lane0's
	// head (shareHead) and owns everything else.
	clone, held bool
	gen         uint64
}

// devTier is one session's view of one device: the device's shared state, the
// session's per-call state, the lane its call runs in and the counters its
// writes go to. The device's own view (devShared.dv) is lane0's and the zero
// session's.
type devTier struct {
	*devShared
	*devSess
	*lane
	*Stats
	// subFn and mvcFn are layersSession and matVecSession bound to this view,
	// built once (layersOnce, matVec).
	subFn, mvcFn func(backend.Session)
}

// newDevice builds the device's view over sh: lane0, the zero session, and
// the device's counters.
func newDevice(sh *devShared) *devTier {
	l0 := &lane{bbs: map[int]*blockScratch{}}
	ds := &devSess{}
	dv := &devTier{devShared: sh, devSess: ds, lane: l0, Stats: &sh.tot}
	sh.lane0, sh.dv = l0, dv
	sh.sess = map[uint64]*devSess{0: ds}
	sh.idle.L = &sh.mu
	return dv
}

// as is session sid's view of this device for one call, holding a lane; the
// caller pairs it with done. A call already holding one (an entry reached
// from inside another, for the same session) keeps it. nil when no lane can
// be had -- every lane busy and a clone does not fit the budget: the caller
// refuses, and the State runs on the host. Callers must not hold g.mu.
func (g *devTier) as(sid uint64) *devTier {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.asLocked(sid)
}

// asLocked is as with g.mu held.
func (g *devTier) asLocked(sid uint64) *devTier {
	ds := g.sessOf(sid)
	if ds.cur != nil {
		ds.depth++
		return ds.view
	}
	l := g.takeLane(ds)
	if l == nil {
		return nil
	}
	ds.cur, ds.last, ds.depth = l, l, 1
	if ds.view == nil {
		ds.view = &devTier{devShared: g.devShared, devSess: ds, Stats: &g.tot}
	}
	ds.view.lane = l
	return ds.view
}

// done ends what as began, giving the lane back when the outermost entry
// returns. Callers must not hold g.mu.
func (g *devTier) done() {
	g.mu.Lock()
	defer g.mu.Unlock()
	ds := g.devSess
	if ds.depth--; ds.depth > 0 {
		return
	}
	ds.cur.held = false
	// A call may be waiting for a lane (takeLane), or for lane0 to run a
	// tower's range (borrowLane0).
	g.idle.Broadcast()
	ds.cur = nil
}

// bareView is session sid's view for a call that reads no scratch: its
// per-call state and an empty lane, so no lane is taken or cloned for it.
// Callers must not hold g.mu.
func (g *devTier) bareView(sid uint64) *devTier {
	g.mu.Lock()
	defer g.mu.Unlock()
	ds := g.sessOf(sid)
	if ds.bare == nil {
		ds.bare = &devTier{devShared: g.devShared, devSess: ds, lane: &lane{}, Stats: &g.tot}
	}
	return ds.bare
}

// borrowLane0 is the session's view over lane0 for a range only lane0 runs (a
// tower's), waiting until no other call holds lane0. The call keeps the lane
// it holds as well; returnLane0 gives lane0 back. A call holding lane0
// already gets its own view. Callers hold g.mu, outside any submission.
func (g *devTier) borrowLane0() *devTier {
	if g.lane == g.lane0 {
		return g
	}
	// Waiting, lane0 is no new call's to take (freeLane), or a session
	// stepping in it call after call would take it back every time.
	g.lane0Want++
	for g.lane0.held {
		g.idle.Wait()
	}
	g.lane0Want--
	g.lane0.held = true
	return g.lane0View()
}

// returnLane0 ends what borrowLane0 began. Callers hold g.mu.
func (g *devTier) returnLane0(v *devTier) {
	if v == g {
		return
	}
	g.lane0.held = false
	g.idle.Broadcast()
}

// subLock takes g.mu for a build reached from inside this session's
// submission -- a kernel compiled on first use, a staging buffer grown --
// and reports whether it did; subUnlock gives it back. Outside a submission
// the caller holds g.mu already and nothing is taken. The pair is written
// defer g.subUnlock(g.subLock()).
func (g *devTier) subLock() bool {
	if !g.inSub || g.subHeld {
		return false
	}
	g.mu.Lock()
	g.subHeld = true
	return true
}

// subUnlock gives back what subLock took.
func (g *devTier) subUnlock(took bool) {
	if took {
		g.subHeld = false
		g.mu.Unlock()
	}
}

// runSub runs f as one of the session's submissions: on the session's own
// queue where the device runs sessions at once (backend.Queued), so it runs
// beside other sessions' submissions; on the device's Session otherwise.
// Callers hold g.mu when the submission runs alone, and not otherwise; the
// queue is made under g.mu by the caller (sessQueue).
func (g *devTier) runSub(f func(backend.Session)) {
	if g.ranFn == nil {
		g.ranFn = g.ran
	}
	g.runF = f
	g.lastSub.Store(time.Now().UnixNano())
	if g.q != nil {
		g.qd.SessionOn(g.q, g.ranFn)
	} else {
		g.dev.Session(g.ranFn)
	}
	g.lastSub.Store(time.Now().UnixNano())
	g.runF = nil
}

// ran is the submission inside the backend's session, counting whether
// another's was running there when it began (Stats.RanBeside).
func (g *devTier) ran(s backend.Session) {
	if g.onDev.Add(1) > 1 {
		g.ranBeside.Add(1)
	}
	defer g.onDev.Add(-1)
	g.runF(s)
}

// sessQueue makes the session's queue if the device has them and it has not
// been asked. A refusal leaves the session on the device's Session. Callers
// hold g.mu.
func (g *devTier) sessQueue() {
	if g.qTried {
		return
	}
	g.qTried = true
	qd, ok := g.dev.(backend.Queued)
	if !ok {
		return
	}
	q, err := qd.NewQueue()
	if err != nil {
		g.LastErr = "no queue of its own for a session: " + err.Error()
		return
	}
	g.q, g.qd = q, qd
}

// closeQueue closes the session's queue. Callers hold g.mu, with no
// submission of the session's running.
func (ds *devSess) closeQueue() {
	ds.launchTo.closePersist()
	if ds.q != nil {
		ds.q.Close()
		ds.q = nil
	}
	ds.qTried = false
}

// sessOf is session sid's state on this device, made at its first call.
// Callers hold g.mu.
func (g *devTier) sessOf(sid uint64) *devSess {
	if ds := g.sess[sid]; ds != nil {
		return ds
	}
	ds := &devSess{sid: sid}
	g.sess[sid] = ds
	return ds
}

// takeLane holds a lane for one of ds's calls: the lane it ran in last, if
// free and current, then lane0, then any free clone, then a new clone. A clone
// lane0 has moved past is dropped on the way. When every lane is held and the
// budget has no room for another, the call waits for one to come free: the
// sessions go one after another, as they did when the scratch was the
// device's, rather than one of them to the host. Callers hold g.mu.
//
// The waits are first come, first served (laneNext, laneServe): a session
// that gives its lane back and calls again joins the line behind one already
// waiting, rather than taking the lane back before the other wakes.
func (g *devTier) takeLane(ds *devSess) *lane {
	if g.laneNext == g.laneServe {
		if l := g.freeLane(ds); l != nil {
			return l
		}
	}
	my := g.laneNext
	g.laneNext++
	g.LanesWaited++
	for {
		if my == g.laneServe {
			if l := g.freeLane(ds); l != nil {
				g.laneServe++
				// The next in line may find a lane too.
				g.idle.Broadcast()
				return l
			}
		}
		g.idle.Wait()
	}
}

// freeLane holds a lane for ds if one is free or can be built, nil if not.
// Callers hold g.mu.
func (g *devTier) freeLane(ds *devSess) *lane {
	pick := func(l *lane) *lane {
		l.held = true
		return l
	}
	g.tidyLanes()
	if l := ds.last; l != nil && !l.held && g.liveLane(l) && (l != g.lane0 || g.lane0Want == 0) {
		return pick(l)
	}
	if !g.lane0.held && g.lane0Want == 0 {
		return pick(g.lane0)
	}
	for _, l := range g.lanes {
		if !l.held {
			return pick(l)
		}
	}
	// A clone that did not fit is not tried again until the device may have
	// room (roomGen) or lane0 changed shape (laneGen).
	if room := g.roomGen.Load(); !g.noClone || room != g.noCloneRoom || g.laneGen != g.noCloneGen {
		if l, ok := g.dv.cloneLane(); ok {
			g.noClone = false
			g.lanes = append(g.lanes, l)
			return pick(l)
		}
		g.noClone, g.noCloneRoom, g.noCloneGen = true, g.roomGen.Load(), g.laneGen
	}
	return nil
}

// liveLane reports that l is lane0 or one of its current clones. Callers hold
// g.mu.
func (g *devTier) liveLane(l *lane) bool {
	if l == g.lane0 {
		return true
	}
	for _, c := range g.lanes {
		if c == l {
			return true
		}
	}
	return false
}

// tidyLanes drops every free clone built before lane0's last change of shape.
// A held one is dropped when its call gives it back and the next call comes
// here. Callers hold g.mu.
func (g *devTier) tidyLanes() {
	kept := g.lanes[:0]
	for _, l := range g.lanes {
		if !l.held && l.gen != g.laneGen {
			g.dv.dropLane(l)
			continue
		}
		kept = append(kept, l)
	}
	clear(g.lanes[len(kept):])
	g.lanes = kept
	for _, ds := range g.sess {
		if ds.last != nil && !g.liveLane(ds.last) {
			ds.last = nil
		}
	}
}

// lane0View is the session's view over lane0, for a range only lane0 runs: a
// vision tower's, whose set is one per device. It is the session's own view
// when its call holds lane0. Callers hold g.mu.
func (g *devTier) lane0View() *devTier {
	if g.lane == g.lane0 {
		return g
	}
	if g.v0 == nil {
		g.v0 = &devTier{devShared: g.devShared, devSess: g.devSess, lane: g.lane0, Stats: &g.tot}
	}
	return g.v0
}

// shapeChanged records that lane0 changed shape -- a scratch built, rebuilt
// or dropped, the head placed or dropped -- so no clone of the old shape runs
// again. A change to a clone moves nothing. Callers hold g.mu.
func (g *devTier) shapeChanged() {
	if g.lane == g.lane0 {
		g.laneGen++
	}
}

// homeSet is lane l's home geometry set's decode scratch: the one holding
// the head.
func homeSet(l *lane) *blockScratch {
	if l.geoCur == 0 {
		return l.bs
	}
	return l.geoSets[0].bs
}

// cloneLane builds a lane from lane0's plans: every causal set's decode
// scratch, the batched widths lane0 holds, the staging at lane0's sizes, and
// lane0's head shared rather than copied. A vision tower's set is not cloned
// (lane0View). Everything it allocates is charged as scratch. g is the
// device's view. Callers hold g.mu.
func (g *devTier) cloneLane() (*lane, bool) {
	defer g.scratchWin().close()
	l0 := g.lane0
	nl := &lane{bbs: map[int]*blockScratch{}, clone: true, gen: g.laneGen}
	v := &devTier{devShared: g.devShared, devSess: g.devSess, lane: nl, Stats: g.Stats}
	fail := func() (*lane, bool) {
		g.dropLane(nl)
		return nil, false
	}
	sets := map[geoKey]geomSet{l0.geoCur: {bs: l0.bs, bbs: l0.bbs, promptW: l0.promptW, stepW: l0.stepW}}
	for k, s := range l0.geoSets {
		sets[k] = s
	}
	for k, s := range sets {
		if s.bs == nil || k&geoNonCausal != 0 {
			continue
		}
		v.useGeom(k)
		if v.bs = v.initScratch(&s.bs.p, 1); v.bs == nil {
			return fail()
		}
		for w, src := range s.bbs {
			if src == nil || !v.prepBatch(w) || !v.reservePaged(v.bbs[w]) {
				return fail()
			}
			if src.ragLogits != nil && !v.reserveRowsHead(v.bbs[w], &src.p) {
				return fail()
			}
		}
		v.promptW, v.stepW = s.promptW, s.stepW
	}
	v.useGeom(0)
	if h := homeSet(l0); h != nil && h.head != nil && nl.bs != nil {
		if !v.shareHead(nl.bs, h) {
			return fail()
		}
	}
	if l0.partCap > 0 && !v.sizePart(l0.partCap) {
		return fail()
	}
	if l0.f16Cap > 0 && !v.sizeF16(l0.f16Cap) {
		return fail()
	}
	if l0.argmaxOut != nil {
		b, err := g.dev.Alloc(4)
		if err != nil {
			g.LastErr = "a lane's argmax word: " + err.Error()
			return fail()
		}
		nl.argmaxOut = b
	}
	if err := g.cloneParts(nl); err != nil {
		g.LastErr = "a lane's composed parts: " + err.Error()
		return fail()
	}
	return nl, true
}

// shareHead gives dst the head src holds: the projection, its norm and its
// cap are the model's and stay src's, and dst gets logits of its own to write.
// Callers hold g.mu inside a scratch window.
func (g *devTier) shareHead(dst, src *blockScratch) bool {
	dst.head, dst.mvHead, dst.headGraph = src.head, src.mvHead, src.headGraph
	dst.hNorm, dst.hNormB, dst.headCap = src.hNorm, src.hNormB, src.headCap
	dst.headEmbeds, dst.embScale = src.headEmbeds, src.embScale
	dst.headShared = true
	n := src.mvHead.rows
	var err error
	if dst.logits, err = g.dev.Alloc(n * 4); err != nil {
		g.LastErr = "a lane's logits: " + err.Error()
		return false
	}
	if src.logitsCap != nil {
		if dst.logitsCap, err = g.dev.Alloc(n * 4); err != nil {
			g.LastErr = "a lane's capped logits: " + err.Error()
			return false
		}
	}
	dst.hRaw = make([]byte, n*4)
	dst.hOut = unsafe.Slice((*float32)(unsafe.Pointer(&dst.hRaw[0])), n)
	return true
}

// unshareHead forgets a borrowed head, so freeing dst frees only what is
// dst's own.
func unshareHead(dst *blockScratch) {
	if dst == nil || !dst.headShared {
		return
	}
	dst.head, dst.mvHead, dst.headGraph = nil, mv{}, nil
	dst.hNorm, dst.hNormB, dst.headCap = nil, nil, nil
	dst.headShared = false
}

// dropLane frees a clone, its recordings and its scratch, and refunds it. g is
// the device's view. Callers hold g.mu, outside any Session.
func (g *devTier) dropLane(l *lane) {
	if l == nil || !l.clone {
		return
	}
	defer g.scratchWin().close()
	v := &devTier{devShared: g.devShared, devSess: g.devSess, lane: l, Stats: g.Stats}
	v.dropLaneGraph()
	g.dropLaneParts(l)
	for _, s := range l.geoSets {
		unshareHead(s.bs)
	}
	unshareHead(l.bs)
	v.dropLaneScratch()
	if l.argmaxOut != nil {
		l.argmaxOut.Free()
		l.argmaxOut = nil
	}
	l.freeSample()
	l.freePick()
}

// dropLanes frees every clone: what a device dropping its scratch, or closing,
// does. Callers hold g.mu, outside any Session, with no call holding a clone.
func (g *devTier) dropLanes() {
	for _, l := range g.lanes {
		g.dv.dropLane(l)
	}
	clear(g.lanes)
	g.lanes = g.lanes[:0]
	for _, ds := range g.sess {
		if ds.last != nil && ds.last != g.lane0 {
			ds.last = nil
		}
	}
}

// forgetSess forgets session sid on this device: its recordings in every
// lane, and its state. The zero session is the device's and stays. Callers
// hold g.mu, outside any Session.
func (g *devTier) forgetSess(sid uint64) {
	g.eachLane(func(l *lane) {
		for k, r := range l.recs {
			if k.sid == sid {
				l.stale = append(l.stale, r)
				delete(l.recs, k)
				delete(l.recUsed, k)
			}
		}
		freeStaleOf(l)
	})
	if ds := g.sess[sid]; ds != nil {
		ds.closeQueue()
	}
	if sid != 0 {
		delete(g.sess, sid)
	}
	// A clone exists for sessions stepping at once; with one fewer session it
	// may be one too many, and a free one goes back to the budget. The next
	// call that finds every lane taken builds one again.
	kept := g.lanes[:0]
	for _, l := range g.lanes {
		if l.held {
			kept = append(kept, l)
			continue
		}
		g.dv.dropLane(l)
	}
	clear(g.lanes[len(kept):])
	g.lanes = kept
	for _, ds := range g.sess {
		if ds.last != nil && !g.liveLane(ds.last) {
			ds.last = nil
		}
	}
}

// eachLane calls f with lane0 and every clone. Callers hold g.mu.
func (g *devTier) eachLane(f func(*lane)) {
	f(g.lane0)
	for _, l := range g.lanes {
		f(l)
	}
}

// dropLaneGraph retires the recordings made in this view's lane, destroying
// them unless the call is inside its submission, where the next call outside
// one does (see dropGraph). A lane is its call's alone, so nothing in flight
// names them. Callers hold g.mu.
func (g *devTier) dropLaneGraph() {
	for k, r := range g.recs {
		g.stale = append(g.stale, r)
		delete(g.recs, k)
	}
	if !g.inSub {
		freeStaleOf(g.lane)
	}
}

// freeStaleOf destroys a lane's retired recordings. Callers must be outside a
// Session.
func freeStaleOf(l *lane) {
	for i, r := range l.stale {
		r.Free()
		l.stale[i] = nil
	}
	l.stale = l.stale[:0]
}
