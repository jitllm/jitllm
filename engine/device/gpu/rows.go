package gpu

import (
	"errors"
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/nn"
)

// ragStep is one LayersRows call's rows, read by layersOnce while it runs.
type ragStep struct {
	pos, slot []int
	// sid, when set, is each row's session: a step whose rows belong to
	// different sessions (gpuSession.LayersSessions). Nil is self's.
	sid []uint64
	// self is the session making the call: every row's when sid is nil, and
	// the one a device-wide choice made for the call (eviction, the
	// recording's key) is made for.
	self uint64
	// seqLen is the positions one sequence of a session's batch spans
	// (nn.RowsDevice): row r is sequence (slot[r]-pos[r])/seqLen of its
	// session.
	seqLen int
	// runs is the rows grouped by sequence, each run's rows in position order,
	// with each run's session and seat row (groupRuns), and sess the step's
	// distinct sessions: what a linear block's recurrent descriptor is built
	// from. Set for a step that runs a linear block or crosses sessions.
	runs    [][]int
	runSid  []uint64
	runSeat []int
	sess    []uint64
	// keys is groupRuns' scratch, kept with the rest.
	keys []runKey
	// again says an earlier run of this step ran on the same device (a cut at
	// a change of geometry, geomsets.go), so the step's rows are counted once.
	again bool
}

// oneSequence reports whether the step's rows are one sequence's, in
// position order.
func (r *ragStep) oneSequence() bool {
	for i := range r.pos {
		if i == 0 {
			continue
		}
		if r.pos[i] <= r.pos[i-1] || r.sid != nil && r.sid[i] != r.sid[0] ||
			r.seqLen > 0 && (r.slot[i]-r.pos[i])/r.seqLen != (r.slot[0]-r.pos[0])/r.seqLen {
			return false
		}
	}
	return true
}

// LayersRows runs one step for len(pos) rows, every block and the head, and
// fills head.Tokens with each row's greedy token and, with head.RowLogits, reads
// each row's logits into head.Logits (see nn.RowsDevice). The rows may be
// independent sequences or one sequence's prompt.
//
// It is batched prefill with different attention: the batched scratch runs the
// matvecs and writes each row's K and V at its own offset, and each row then
// reads its own sequence's keys (kernels.AttnScoresRagged/AttnAccRagged),
// softmaxes over its own count and ends in its own argmax. N sequences cost
// one pass over the weights instead of N.
func (g *devTier) LayersRows(lo, hi int, pos, slot []int, seqLen int, x, cs, csSWA []float32, head *nn.Head) bool {
	v := g.as(0)
	if v == nil {
		return false
	}
	defer v.done()
	rs := &ragStep{pos: pos, slot: slot, seqLen: seqLen}
	return v.layersRows(lo, hi, rs, x, cs, csSWA, head)
}

// rowGroups is rowsInGroups' scratch: the order a step's rows run in, a
// group's rows in it, and the group's step, head and rows gathered.
type rowGroups struct {
	rs                           ragStep
	head                         nn.Head
	ord, gi, pos, slot           []int
	sid                          []uint64
	x, cs, sw, ple, cut, lg, hid []float32
	tok, toks                    []int32
}

// layersRows is LayersRows for rs's rows, each of its own session where
// rs.sid is set and of rs.self's otherwise.
func (g *devTier) layersRows(lo, hi int, rs *ragStep, x, cs, csSWA []float32, head *nn.Head) bool {
	fail := func(f string, a ...any) bool {
		g.mu.Lock()
		g.LastErr = fmt.Sprintf("rows: "+f, a...)
		g.mu.Unlock()
		return false
	}
	pos, slot, sid := rs.pos, rs.slot, rs.self
	n := len(pos)
	// Gemma 4: the range's geometry's scratch set (geomsets.go), as layersCall.
	g.mu.Lock()
	prev := g.geoCur
	if g.geoUsed {
		g.useGeom(g.geomFor(lo, hi))
	}
	g.mu.Unlock()
	defer g.restoreGeom(prev)
	switch {
	case g.rowsFaulted("noswa"):
		csSWA = nil
	}
	if n < 1 || len(slot) != n || lo < 0 || lo > hi || (lo == hi && head == nil) {
		return fail("%d positions, %d slots, blocks [%d,%d)", n, len(slot), lo, hi)
	}
	g.mu.Lock()
	grp := g.pagedCall(lo, hi, n)
	if g.rowsComposed(lo, hi) || lo == hi && g.headReserved() {
		grp = min(grp, max(g.promptW, 1))
	}
	g.mu.Unlock()
	if grp < n {
		return g.rowsInGroups(lo, hi, grp, rs, x, cs, csSWA, head)
	}
	if g.NoBatch {
		return fail("NoBatch is set")
	}
	// A sequence's rows -- a prompt chunk beside the decoding rows -- must be
	// its next positions in order, which a linear block's run chains through
	// and which a step across sessions checks for its attention too: a repeat
	// would write one position twice.
	g.mu.Lock()
	linear := g.linearIn(lo, hi)
	var err error
	if linear || rs.sid != nil {
		err = g.groupRuns(sid, rs)
	}
	g.mu.Unlock()
	if err != nil {
		return fail("%v", err)
	}
	// Each row's sequence takes its own pages.
	g.pgRows = g.pagedRows(sid, g.pgRows, n, n, 0, rs)
	if !g.pagedAppendRows(sid, g.pgRows) {
		return false
	}
	g.mu.Lock()
	// A step the composer serves whole runs in the reserved width, its rows
	// the call's, as a prompt does (rowsComposed): no width is built,
	// formed or compiled for its count of rows, and a wider step ran as
	// steps of that width (rowsInGroups). So does a head over the rows on a
	// device that holds none of the step's blocks.
	composed := g.rowsComposed(lo, hi) || lo == hi && g.headReserved()
	w := g.promptW
	if !composed {
		w = g.rowsWidth(lo, n)
	}
	if w > batchWidth {
		g.mu.Unlock()
		return fail("%d rows, more than the batch width %d", n, batchWidth)
	}
	for li := lo; li < hi; li++ {
		l := g.layers[li]
		switch {
		case l == nil || !l.ok:
			g.mu.Unlock()
			return fail("block %d is not on this device", li)
		case l.nonCausal:
			g.mu.Unlock()
			return fail("block %d is non-causal", li)
		}
	}
	// The step's own width, or the reserved one when that does not fit.
	ok := g.bs != nil && (head == nil || g.bs.head != nil && g.bs.head.ok)
	if ok {
		if !composed {
			w = g.batchFor(w)
		}
		ok = w > 0 && g.prepRagged(g.bbs[w], head, n)
	}
	bs := g.bbs[w]
	if ok {
		g.RowsSteps++
		g.RowsPadded += w - n
	}
	if ok && linear {
		ok = g.prepRecRows(sid, lo, hi, rs)
	}
	g.rag = rs
	g.mu.Unlock()
	defer func() {
		g.mu.Lock()
		g.rag = nil
		g.mu.Unlock()
	}()
	if !ok {
		return false
	}
	if head != nil && len(head.Tokens) < n {
		head.Tokens = make([]int32, n)
	}
	if !g.submit(sid, bs, lo, hi, 0, x, cs, csSWA, head) {
		return false
	}
	g.mu.Lock()
	if rs.sid != nil && !rs.again {
		g.SessionRows += n
	}
	if linear {
		nl := 0
		for li := lo; li < hi; li++ {
			if g.layers[li].linear {
				nl++
			}
		}
		g.RecChainRows += (n - len(rs.runs)) * nl
	}
	// Counted here rather than in emitMLARows, which a replayed recording
	// does not run.
	for li := lo; li < hi; li++ {
		// l.mla is the model's: a hybrid's linear blocks carry it too.
		if l := g.layers[li]; l.mla && !l.linear {
			g.RowsLatent++
		}
	}
	g.mu.Unlock()
	return true
}

// rowsInGroups runs rs's rows as steps of grp rows each. A step's rows are
// sequences of their own or a sequence's next positions, each row's history
// written before any row attends: so the groups take the rows in position
// order, a row's earlier positions in a group before its own, and in each
// group the rows wanting the head lead, as they lead the step. Each group's
// rows, their side inputs among them, are gathered into the session's
// scratch and its outputs scattered back.
func (g *devTier) rowsInGroups(lo, hi, grp int, rs *ragStep, x, cs, csSWA []float32, head *nn.Head) bool {
	n := len(rs.pos)
	xw, rw, sw := len(x)/n, len(cs)/n, len(csSWA)/n
	want, vocab, embd := 0, 0, 0
	if head != nil {
		if len(head.Tokens) < n {
			head.Tokens = make([]int32, n)
		}
		want = head.WantedRows(n)
	}
	g.mu.Lock()
	if g.bs != nil {
		embd = g.bs.p.NEmbd
		if g.bs.head != nil {
			vocab = g.bs.head.nrows
		}
	}
	// The rows are gathered from x, so a promised gather is made real first.
	if pe := g.embPend; pe != nil {
		g.embPend = nil
		g.materializeLocked(pe)
	}
	ple, cut, toks := g.pleHost, g.cutHost, g.stepToks
	g.mu.Unlock()
	defer func() {
		g.mu.Lock()
		g.pleHost, g.cutHost, g.stepToks = ple, cut, toks
		g.mu.Unlock()
	}()
	pw, cw, tw := 0, 0, 0
	if len(ple)%n == 0 {
		pw = len(ple) / n
	}
	if len(cut)%n == 0 {
		cw = len(cut) / n
	}
	if len(toks) >= n {
		tw = 1
	}
	q := &g.grp
	q.ord = q.ord[:0]
	for r := range n {
		i := len(q.ord)
		q.ord = append(q.ord, r)
		for ; i > 0 && rs.pos[q.ord[i-1]] > rs.pos[r]; i-- {
			q.ord[i] = q.ord[i-1]
		}
		q.ord[i] = r
	}
	for a := 0; a < n; a += grp {
		b := min(a+grp, n)
		gi := q.gi[:0]
		for _, r := range q.ord[a:b] {
			if r < want {
				gi = append(gi, r)
			}
		}
		k := len(gi)
		for _, r := range q.ord[a:b] {
			if r >= want {
				gi = append(gi, r)
			}
		}
		q.gi = gi
		m := len(gi)
		q.pos, q.slot, q.sid = q.pos[:0], q.slot[:0], q.sid[:0]
		q.x, q.cs, q.sw, q.ple, q.toks = q.x[:0], q.cs[:0], q.sw[:0], q.ple[:0], q.toks[:0]
		for _, r := range gi {
			q.pos, q.slot = append(q.pos, rs.pos[r]), append(q.slot, rs.slot[r])
			if rs.sid != nil {
				q.sid = append(q.sid, rs.sid[r])
			}
			q.x = append(q.x, x[r*xw:(r+1)*xw]...)
			q.cs = append(q.cs, cs[r*rw:(r+1)*rw]...)
			q.sw = append(q.sw, csSWA[r*sw:(r+1)*sw]...)
			q.ple = append(q.ple, ple[r*pw:(r+1)*pw]...)
			q.toks = append(q.toks, toks[r*tw:(r+1)*tw]...)
		}
		q.cut = slices.Grow(q.cut[:0], m*cw)[:m*cw]
		runs, runSid, runSeat, sess, keys := q.rs.runs[:0], q.rs.runSid[:0], q.rs.runSeat[:0], q.rs.sess[:0], q.rs.keys[:0]
		q.rs = ragStep{pos: q.pos, slot: q.slot, self: rs.self, seqLen: rs.seqLen, again: rs.again,
			runs: runs, runSid: runSid, runSeat: runSeat, sess: sess, keys: keys}
		if rs.sid != nil {
			q.rs.sid = q.sid
		}
		var h *nn.Head
		if head != nil && k > 0 {
			q.head = *head
			q.tok = slices.Grow(q.tok[:0], m)[:m]
			q.head.Tokens, q.head.LogitRows, q.head.Logits, q.head.Hidden = q.tok, k, nil, nil
			if head.RowLogits && vocab > 0 {
				q.lg = slices.Grow(q.lg[:0], k*vocab)[:k*vocab]
				q.head.Logits = q.lg
			}
			if head.Hidden != nil && embd > 0 {
				q.hid = slices.Grow(q.hid[:0], k*embd)[:k*embd]
				q.head.Hidden = q.hid
			}
			h = &q.head
		}
		g.mu.Lock()
		if pw > 0 {
			g.pleHost = q.ple
		}
		if cw > 0 {
			g.cutHost = q.cut
		}
		if tw > 0 {
			g.stepToks = q.toks
		}
		g.mu.Unlock()
		if !g.layersRows(lo, hi, &q.rs, q.x, q.cs, q.sw, h) {
			return false
		}
		for i, r := range gi {
			copy(x[r*xw:(r+1)*xw], q.x[i*xw:(i+1)*xw])
			copy(cut[r*cw:(r+1)*cw], q.cut[i*cw:(i+1)*cw])
		}
		if h == nil {
			continue
		}
		head.Applied = h.Applied
		for i, r := range gi[:k] {
			head.Tokens[r] = q.tok[i]
			if h.Logits != nil && len(head.Logits) >= (r+1)*vocab {
				copy(head.Logits[r*vocab:(r+1)*vocab], q.lg[i*vocab:(i+1)*vocab])
			}
			if h.Hidden != nil && len(head.Hidden) >= (r+1)*embd {
				copy(head.Hidden[r*embd:(r+1)*embd], q.hid[i*embd:(i+1)*embd])
			}
		}
	}
	return true
}

// rowsWidth is the width a ragged step of n rows from block lo runs at: n
// rounded up to the row grain, the matrix tile's 32 where the composer forms
// for one (composedMatrix) and 8 elsewhere, since padding to 32 would do 32
// rows of work. Callers hold g.mu.
func (g *devTier) rowsWidth(lo, n int) int {
	grain := batchGrain
	if g.composedMatrix() == ([3]int{}) {
		grain = 8
	}
	return (n + grain - 1) / grain * grain
}

// prepRagged builds bs's ragged-decode set once, outside any session: Compile
// and Alloc are driver calls a capture forbids. Callers hold g.mu. The head's
// part (the batched projection and the per-row argmax) is built only for a
// step that runs the head, and over the rows it wants (nn.Head.LogitRows).
func (g *devTier) prepRagged(bs *blockScratch, rowsHead *nn.Head, n int) bool {
	if bs == nil {
		return false
	}
	if g.roomFaulted("nobuild") || g.roomFaulted("nobuild-room") {
		g.LastErr = "a ragged step's twins do not build (roomFault)"
		return false
	}
	defer g.scratchWin().close()
	return g.prepRowsHead(bs, rowsHead, n)
}

// prepRowsHead readies bs's per-row head -- the composer's head over the
// rows a step wants, their logits and tokens -- once, outside any session.
// A ragged step needs it, and so does a chunk whose head wants every row
// (nn.Head.RowLogits on Layers). Callers hold g.mu.
func (g *devTier) prepRowsHead(bs *blockScratch, rowsHead *nn.Head, n int) bool {
	defer g.scratchWin().close()
	R := bs.rows
	fail := func(err error) bool { g.LastErr = "rows: " + err.Error(); return false }
	// A batch's rows read their history through the scratch's paged set
	// (pagedDecode); what is left to build is the head.
	if rowsHead == nil {
		return true
	}
	head := g.bs.head
	if need := g.rowsHeadNeed(bs, head.nrows, rowsHead.Softcap); need > 0 && !g.room(need) {
		return fail(fmt.Errorf("the logits of %d rows need %d bytes, %d of %d used", R, need, g.used, g.limit))
	}
	// The composer's head over rows serves any count of them at bs's width
	// (servedHead), formed here, outside any step; a head it does not serve
	// is refused at the step, so the tier builds no projection of its own.
	if g.bs.headServes(g.bs.hNorm, false, false) != partServes {
		return fail(errors.New("the composer does not serve the head over rows"))
	}
	_, err := g.servedHead(g.bs, g.bs.hNorm, R)
	if err != nil {
		return fail(fmt.Errorf("the composed head over %d rows: %w", R, err))
	}
	if !bs.ragBuilt {
		if !g.ragSampleServes() {
			return fail(errors.New("the composer states no sample for the head"))
		}
		// reserveRowsHead may have allocated them at placement, from the
		// plan's vocabulary; a head of another width replaces them.
		if bs.ragLogits != nil && bs.ragVocab != head.nrows {
			freeRagOut(bs)
		}
		if bs.ragLogits == nil {
			if bs.ragLogits, err = g.dev.Alloc(R * head.nrows * 4); err == nil {
				bs.ragVocab = head.nrows
				bs.ragTok, err = g.dev.Alloc(R * 4)
			}
		}
		if err != nil {
			return fail(err)
		}
		bs.ragBuilt = true
	}
	// The composed sample of the rows' tokens (ragTokens), formed and held
	// here at the width: the step forms nothing, and the reservation counts
	// it.
	_, err = g.servedSampleAt(g.bs.sampleGraph, R)
	if err != nil {
		return fail(fmt.Errorf("the composed sample over %d rows: %w", R, err))
	}
	return true
}

// rowsHeadNeed is what prepRowsHead allocates for bs's per-row logits at a
// head of nrows rows, the final softcap's twin included: nothing where they
// are held at that width.
func (g *devTier) rowsHeadNeed(bs *blockScratch, nrows int, softcap float32) uint64 {
	if bs == nil {
		return 0
	}
	var n uint64
	fresh := bs.ragLogits == nil || bs.ragVocab != nrows
	if fresh {
		n += g.footprint(bs.rows*nrows*4) + g.footprint(bs.rows*4)
	}
	return n
}

// prepRowsReserved builds the reserved width's per-row head where a ragged
// step's rows run there (rowsReserved) -- the per-row argmax and the final
// softcap, the composer's head writing the logits they read -- at placement
// and at PrepHead, so no step compiles them. Callers hold g.mu.
func (g *devTier) prepRowsReserved() {
	bb := g.reservedFor(nn.ComposedBlock)
	if bb == nil || g.bs == nil || g.bs.head == nil || !g.bs.head.ok || !g.rowsHeld() {
		return
	}
	was := g.prewarming
	g.prewarming = true
	defer func() { g.prewarming = was }()
	if !g.prepRowsHead(bb, &nn.Head{}, bb.rows) {
		g.LastErr = "the reserved width's per-row head: " + g.LastErr
	}
}

// ragTokens writes the greedy token of each of a step's n wanted rows of
// logits into bs.ragTok: the composer's sample at the scratch's rows, its
// rows the call's (headCalls).
func (g *devTier) ragTokens(lc *launcher, bs, hb *blockScratch, n int, served bool) error {
	if !served {
		return errors.New("rows: the composed sample was not asked for the step's tokens")
	}
	sv, err := g.servedSampleAt(hb.sampleGraph, bs.rows)
	if err != nil {
		return err
	}
	sv.rows, sv.sl.tables[tableKey(composer.TableCall)] = n, bs.headCalls
	return sv.run(g, lc, sv.inst, []backend.Buf{bs.ragOut()}, []backend.Buf{bs.ragTok})
}

// rowsCall is what a rows call keeps between its calls: its runs slice,
// reused so a step allocates none, and why its last call failed, naming the
// device (the devices' LastErr summed by Stats is whichever wrote last, not
// the one that refused). The zero session's is the GPU's, a session's its own:
// steps led by different sessions at once -- the server's decode groups --
// share none of it.
type rowsCall struct {
	runs []run
	err  string
	// in is the session whose inputs (gpuSession.inputsTo) each device takes
	// as the call reaches it, nil for none.
	in *gpuSession
}

// LayersRows runs blocks [lo, hi) for the rows, one call per device run, the
// residual crossing to the next device through x. Each row's history for a
// block lives in the cache of the device that holds the block. The head rides
// the last block's step where it is on that device, and is a step of its own
// on the device holding it otherwise.
func (g *GPU) LayersRows(lo, hi int, pos, slot []int, seqLen int, x, cs, csSWA []float32, head *nn.Head) bool {
	return g.layersRows(&g.rows, 0, lo, hi, pos, slot, nil, seqLen, x, cs, csSWA, head)
}

// layersRows is LayersRows for session self, with each row's session, or nil
// for self's; c is self's rows call.
func (g *GPU) layersRows(c *rowsCall, self uint64, lo, hi int, pos, slot []int, sid []uint64, seqLen int, x, cs, csSWA []float32, head *nn.Head) bool {
	beside := g.rowsCalls.Add(1) > 1
	defer g.rowsCalls.Add(-1)
	g.mu.Lock()
	rs, ok := g.runsInto(c.runs, lo, hi)
	c.runs = nil
	why := ""
	switch {
	case lo == hi && head != nil && g.head != nil:
		// The head alone (a hyper-connected model's, after the host's
		// collapse): a step of its own on the head's device.
		rs = append(rs[:0], run{dev: g.head, lo: hi, hi: hi, split: g.split})
	case !ok || len(rs) == 0:
		why = fmt.Sprintf("rows: blocks [%d,%d) are not all on devices", lo, hi)
	case head != nil && g.head == nil:
		why = "rows: no device holds the head"
	}
	// A head on another device than the last block's -- placed where the
	// tail device had no room for it (PrepHead) -- or a last run outside the
	// home scratch set (geomsets.go) leaves the projection to a step of its
	// own on the head's device, the residual crossing through x, as Layers
	// does.
	if why == "" && head != nil {
		if r := rs[len(rs)-1]; r.dev != g.head || r.geo != (geoKey{}) {
			rs = append(rs, run{dev: g.head, lo: hi, hi: hi, split: g.split})
		}
	}
	c.err = why
	g.mu.Unlock()
	defer g.keepRuns(c, rs)
	// The call's counts and causes start at zero, and the session's inputs
	// go to each device, as the call reaches it (stageBegin): a device runs
	// one step's stage at a time holding its lock, so touching a device
	// another step is on would hold this one at its head until that stage
	// ended, and steps led by different sessions would never overlap.
	g.startRows(self, rs, !beside)
	if why != "" {
		g.startRest(self, rs, 0)
		return false
	}
	for i, r := range rs {
		var h *nn.Head
		if i == len(rs)-1 {
			h = head
		}
		// A step of its own on each device: its rows' grouping is the
		// device's. The ragStep is the session's on that device, its slices
		// reused.
		g.stageBegin(r.dev)
		if !slices.ContainsFunc(rs[:i], func(o run) bool { return o.dev == r.dev }) {
			r.dev.startCall(self)
			if c.in != nil {
				c.in.inputsOn(r.dev)
			}
		}
		v := r.dev.as(self)
		if v == nil {
			g.stageEnd(r.dev)
			g.startRest(self, rs, i+1)
			why = fmt.Sprintf("%s: no scratch for this session: %s", label(r.dev.dev, r.dev.ord), r.dev.stats().LastErr)
			g.mu.Lock()
			c.err = why
			g.mu.Unlock()
			return false
		}
		st := &v.stepRS
		st.pos, st.slot, st.sid, st.self, st.seqLen = pos, slot, sid, self, seqLen
		st.again = slices.ContainsFunc(rs[:i], func(o run) bool { return o.dev == r.dev })
		c0, c1 := r.tables(cs, csSWA)
		ok := v.layersRows(r.lo, r.hi, st, x, c0, c1, h)
		v.done()
		// The runs go too: a step that groups none (no linear block in its
		// range, one session) would otherwise hand recDesc the last step's,
		// whose rows index past this one's.
		st.pos, st.slot, st.sid = nil, nil, nil
		st.runs, st.runSid, st.runSeat, st.sess = st.runs[:0], st.runSid[:0], st.runSeat[:0], st.sess[:0]
		g.stageEnd(r.dev)
		if !ok {
			g.startRest(self, rs, i+1)
			why = fmt.Sprintf("%s, blocks [%d,%d): %s", label(r.dev.dev, r.dev.ord), r.lo, r.hi, r.dev.stats().LastErr)
			g.mu.Lock()
			c.err = why
			g.mu.Unlock()
			return false
		}
	}
	return true
}

// startRows starts a rows call of session self over rs on the devices it
// does not run on, which no step's stage holds: their counts and causes for
// self go to zero, and every device's own cause too when no other rows call
// is in flight (shared). The devices of rs start as the call reaches them.
func (g *GPU) startRows(self uint64, rs []run, shared bool) {
	for i := 0; ; i++ {
		d := g.dev(i)
		if d == nil {
			return
		}
		if shared {
			d.cause.Store(0)
		}
		if !slices.ContainsFunc(rs, func(o run) bool { return o.dev == d }) {
			d.startCall(self)
		}
	}
}

// startRest starts the call on the devices of rs from rs[from] on that a
// failed call never reached, so a step refused part way reads no count left
// by an earlier one.
func (g *GPU) startRest(self uint64, rs []run, from int) {
	for j := from; j < len(rs); j++ {
		d := rs[j].dev
		if !slices.ContainsFunc(rs[:j], func(o run) bool { return o.dev == d }) {
			d.startCall(self)
		}
	}
}

// keepRuns hands a rows call's runs slice back to c for its next call.
func (g *GPU) keepRuns(c *rowsCall, rs []run) {
	g.mu.Lock()
	c.runs = rs
	g.mu.Unlock()
}

// stageBegin waits for d to finish any other call's rows stage and marks one
// running there, counting an overlap on d when another device is running
// one: a call runs its stages one after another, so that stage is another
// call's. A card runs one step's stage at a time, so steps led by different
// sessions -- the server's decode groups -- that reach a card together go
// through it one after the other and leave it offset by a stage, each on its
// own card from then on, where a lane each would run them on the same card
// at once and in step, and the other cards idle.
func (g *GPU) stageBegin(d *devTier) {
	d.stageMu.Lock()
	g.stageMu.Lock()
	other := g.stages > d.stages
	g.stages++
	d.stages++
	g.stageMu.Unlock()
	if other {
		d.mu.Lock()
		d.Stats.StageOverlaps++
		d.mu.Unlock()
	}
}

// stageEnd ends what stageBegin began.
func (g *GPU) stageEnd(d *devTier) {
	g.stageMu.Lock()
	g.stages--
	d.stages--
	g.stageMu.Unlock()
	d.stageMu.Unlock()
}

// RowsWidth is the widest step of at most want rows every device holding a
// block can run (nn.RowsWidther): each device's batched scratch for it held
// or built now (batchFor), halving the width until one fits; on a device
// whose every block the composer serves in the reserved width, that width
// (rowsReserved). 0 when none does, or no device holds a block.
func (g *GPU) RowsWidth(want int) int {
	w := min(want, batchWidth)
	if g.roomFaulted("narrow-lie") {
		return w
	}
	for _, d := range g.devs {
		d.mu.Lock()
		held := false
		for _, l := range d.layers {
			if l != nil && l.ok {
				held = true
				break
			}
		}
		switch {
		case held && d.rowsReserved():
			// The composer serves every block in the reserved width at any
			// count of rows up to it: nothing is built for the step.
			w = min(w, d.promptW)
			if d.roomFaulted("narrow") {
				w = min(w, 64)
			}
		case held && d.bs != nil:
			for w >= 8 {
				r := (w + 7) / 8 * 8
				if b := d.batchFor(r); b >= r {
					break
				}
				w /= 2
			}
		}
		d.mu.Unlock()
		if w < 8 {
			return 0
		}
	}
	return w
}

// rowsRoom is RowsRoom for session sid: on every device holding a block, the
// step's scratch (stepRoom) and, where a linear block is placed, the session's
// recurrent seat for seqs sequences (growSeat); with head, the device holding
// the head has its per-row head too. A scratch that does not build is an
// error, not a shortage of room.
func (g *GPU) rowsRoom(sid uint64, rows, seqs int, head *nn.Head) (bool, error) {
	g.resetCause(sid, true)
	if g.roomFaulted("noroom-lie") || g.roomFaulted("narrow-lie") {
		return true, nil
	}
	if g.roomFaulted("noroom") {
		return false, nil
	}
	g.mu.Lock()
	hd := g.head
	g.mu.Unlock()
	for _, d := range g.devs {
		v := d.as(sid)
		if v == nil {
			continue
		}
		h := head
		if d != hd {
			h = nil
		}
		v.mu.Lock()
		lo, linear := -1, false
		for li, l := range v.layers {
			if l != nil && l.ok {
				if lo < 0 {
					lo = li
				}
				linear = linear || l.linear
			}
		}
		ok, err := lo < 0 && h == nil, error(nil)
		if !ok && v.bs != nil {
			ok, err = v.stepRoom(lo, rows, h)
			ok = ok && (!linear || v.growSeat(sid, seqs))
		}
		v.mu.Unlock()
		v.done()
		if err != nil {
			return false, fmt.Errorf("%s: %w", label(d.dev, d.ord), err)
		}
		if !ok {
			return false, nil
		}
	}
	return true, nil
}

// errStepBuild is a ragged step's scratch that did not build: a kernel the
// device refused, a buffer the driver did not give, the head not on the
// device. No relocation gives it room, so the step is refused by name.
type errStepBuild struct{ why string }

func (e errStepBuild) Error() string { return "the step's scratch does not build: " + e.why }

// stepRoom reports whether a ragged step of n rows from block lo has its
// scratch here: the batched scratch for the width it runs at (rowsWidth,
// batchFor) and the ragged set the step builds in it (prepRagged), with
// head's per-row head. They are built now, outside any session, and held to
// the budget as a width built after placement is (prepBatch): a step whose
// scratch would pass it has no room, and the caller relocates blocks to make
// some rather than the device charging past its budget. The per-row head is
// priced before it is asked for, a padded step's head being the scratch's
// rows of logits. A set that does not build is an errStepBuild. Callers hold
// g.mu.
func (g *devTier) stepRoom(lo, n int, head *nn.Head) (bool, error) {
	var w int
	switch {
	case g.rowsReserved() || lo < 0 && g.headReserved():
		// The step runs in the reserved width (layersRows): nothing is
		// built for its rows.
		if n > g.promptW {
			return false, errStepBuild{fmt.Sprintf("%d rows, more than the reserved width %d the composer serves", n, g.promptW)}
		}
		w = g.promptW
	default:
		if w = g.rowsWidth(lo, n); w > batchWidth {
			return false, nil
		}
		if w = g.batchFor(w); w == 0 {
			return false, nil
		}
	}
	bs := g.bbs[w]
	if head != nil {
		if g.bs == nil || g.bs.head == nil || !g.bs.head.ok {
			return false, errStepBuild{"the step's head is not on this device"}
		}
		if !g.rowsHeadRoom(bs, head) {
			return false, nil
		}
	}
	if !g.prepRagged(bs, head, n) {
		if g.roomFaulted("nobuild-room") {
			return false, nil
		}
		return false, errStepBuild{g.LastErr}
	}
	if g.NoScratchReserve || g.kvCompact(g.promisedBytes(), nil) {
		return true, nil
	}
	g.LastErr = fmt.Sprintf("a %d-row step's scratch passes the budget (%d of %d bytes charged)", w, g.used, g.limit)
	return false, nil
}

// rowsHeadRoom reports whether bs's per-row head for head fits the budget:
// the logits every row of the scratch projects into, the tokens their argmax
// writes and a final softcap's capped copy, as prepRowsHead allocates them,
// priced as the card spends them beside the history promised to admitted
// rows. What is built costs nothing, and a width reserved at placement had
// them reserved with it (reserveRowsHead). Callers hold g.mu, with the head
// on this device.
func (g *devTier) rowsHeadRoom(bs *blockScratch, head *nn.Head) bool {
	if g.NoScratchReserve {
		return true
	}
	R, vocab := bs.rows, g.bs.head.nrows
	// A head of another width than the one reserved gives back every buffer
	// (prepRowsHead's freeRagOut) and asks for them all again.
	fresh := !bs.ragBuilt && bs.ragLogits != nil && bs.ragVocab != vocab
	var need uint64
	if !bs.ragBuilt && (bs.ragLogits == nil || fresh) {
		need += g.footprint(R*vocab*4) + g.footprint(R*4)
	}
	if need == 0 || g.kvCompact(need+g.promisedBytes(), nil) {
		return true
	}
	g.LastErr = fmt.Sprintf("a %d-row step's per-row head (%d bytes) does not fit the budget (%d of %d bytes charged)",
		R, need, g.used, g.limit)
	return false
}

// RowsRoom is rowsRoom for the zero session (nn.RowsRoomer).
func (g *GPU) RowsRoom(rows, seqs int, head *nn.Head) (bool, error) {
	return g.rowsRoom(0, rows, seqs, head)
}

// RowsRoom is the session's (nn.RowsRoomer): its recurrent seat is its own.
func (s *gpuSession) RowsRoom(rows, seqs int, head *nn.Head) (bool, error) {
	return s.g.rowsRoom(s.sid, rows, seqs, head)
}

// RowsWidth forwards the tier's (nn.RowsWidther): scratch is the device's,
// not a session's.
func (s *gpuSession) RowsWidth(want int) int { return s.g.RowsWidth(want) }

func (s *gpuSession) LayersRows(lo, hi int, pos, slot []int, seqLen int, x, cs, csSWA []float32, head *nn.Head) bool {
	defer s.leave(s.inputsTo(s.step()))
	s.rows.in = s
	return s.g.layersRows(&s.rows, s.sid, lo, hi, pos, slot, nil, seqLen, x, cs, csSWA, head)
}

// LayersSessions runs one step for rows of different sessions on this GPU, row
// i being sess[i]'s sequence at pos[i] (nn.SessionStepper): every block's
// weights are read once for all of them, where one session after another
// reads them once each. Each row's history is its own session's. A session may
// take several rows -- a prompt chunk beside the others' decoding -- which
// must be its next positions, each once (devTier.groupRuns refuses a gap or a
// repeat); every row's K and V are written before any row attends, so a
// chunk's rows see each other causally.
func (s *gpuSession) LayersSessions(lo, hi int, sess []nn.LayerRunner, pos []int, x, cs, csSWA []float32, head *nn.Head) bool {
	// The session's own lists, reused: a session steps from one goroutine.
	s.sidBuf = slices.Grow(s.sidBuf[:0], len(sess))[:len(sess)]
	s.slotBuf = slices.Grow(s.slotBuf[:0], len(sess))[:len(sess)]
	sid, slot := s.sidBuf, s.slotBuf
	for i, d := range sess {
		o, ok := d.(*gpuSession)
		if !ok || o.g != s.g {
			s.g.resetRecSteps(s.sid)
			s.g.mu.Lock()
			s.rows.err = fmt.Sprintf("sessions: row %d is not a session of this GPU", i)
			s.g.mu.Unlock()
			return false
		}
		sid[i], slot[i] = o.sid, pos[i]
	}
	defer s.leave(s.inputsTo(s.step()))
	s.rows.in = s
	// A session is one sequence whose slots are its positions, so the span a
	// sequence takes is no matter: every row's sequence is its session's 0.
	return s.g.layersRows(&s.rows, s.sid, lo, hi, pos, slot, sid, 0, x, cs, csSWA, head)
}

// ResetRecRows is resetRecRows for the zero session: a caller that never attached.
func (g *devTier) ResetRecRows(rows []int) bool {
	return g.resetRecRows(0, rows)
}

// resetRecRows zeroes the recurrent state (the conv window and the delta
// state, in every half) of the given rows in every linear block of this
// session: a row a new sequence takes must not start from the last one's
// summary. It is a write per buffer, outside any submission.
func (g *devTier) resetRecRows(sid uint64, rows []int) bool {
	v := g.as(sid)
	if v == nil {
		return false
	}
	defer v.done()
	g = v
	g.mu.Lock()
	defer g.mu.Unlock()
	st, seated := g.recSeats[sid]
	if !seated || len(rows) == 0 {
		return true
	}
	ls := g.recLs[:0]
	for _, l := range g.layers {
		if l != nil && l.linear && l.recOf(sid) != nil {
			ls = append(ls, l)
		}
	}
	g.recLs = ls
	defer clear(g.recLs)
	for _, row := range rows {
		if row < 0 || row >= st.rows {
			continue // a row the batch has not grown to holds nothing yet
		}
		if err := g.recZeroLayers(ls, st.base+row, 1); err != nil {
			g.LastErr = "rows: " + err.Error()
			return false
		}
	}
	return true
}

// RecMark is recMark for the zero session: a caller that never attached.
func (g *GPU) RecMark() {
	g.recMark(0)
}

// recMark marks the session's state on every device (nn.RecRewinder).
func (g *GPU) recMark(sid uint64) {
	for _, d := range g.devs {
		d.recMark(sid)
	}
}

// RecRewind is recRewind for the zero session: a caller that never attached.
func (g *GPU) RecRewind() bool {
	return g.recRewind(0)
}

// recRewind rewinds every device or none: every device is checked before any
// flips, so a refusal on the second cannot leave the first rewound.
func (g *GPU) recRewind(sid uint64) bool {
	for _, d := range g.devs {
		if _, ok := d.rewindable(sid); !ok {
			return false
		}
	}
	for _, d := range g.devs {
		if !d.recRewind(sid) {
			return false // checked above; a session's own state does not move between
		}
	}
	return true
}

func (s *gpuSession) RecMark() {
	defer s.leave(s.enter())
	s.g.recMark(s.sid)
}

func (s *gpuSession) RecRewind() bool {
	defer s.leave(s.enter())
	return s.g.recRewind(s.sid)
}

// ResetRecRows is resetRecRows for the zero session: a caller that never attached.
func (g *GPU) ResetRecRows(rows []int) bool {
	return g.resetRecRows(0, rows)
}

// resetRecRows resets the rows on every device; see devTier.ResetRecRows.
func (g *GPU) resetRecRows(sid uint64, rows []int) bool {
	ok := true
	for _, d := range g.devs {
		ok = d.resetRecRows(sid, rows) && ok
	}
	return ok
}

func (s *gpuSession) ResetRecRows(rows []int) bool {
	defer s.leave(s.enter())
	return s.g.resetRecRows(s.sid, rows)
}

// ragSampleServes says the composer writes a ragged step's tokens: asked
// for the token and a prompt's rows, over the sample its head states.
func (g *devTier) ragSampleServes() bool {
	return g.bs != nil && g.bs.sampleGraph != nil
}
