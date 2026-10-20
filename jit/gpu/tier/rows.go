package tier

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
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
	// a change of geometry, gemma4.go), so the step's rows are counted once.
	again bool
}

// LayersRows runs one step for len(pos) rows, every block and the head, and
// fills head.Tokens with each row's greedy token and, with head.RowLogits, reads
// each row's logits into head.Logits (see nn.RowsDevice). The rows may be
// independent sequences or one sequence's prompt.
//
// It is batched prefill with different attention: the batched scratch runs the
// matvecs and writes each row's K and V at its own offset, and each row then
// reads its own sequence's keys,
// softmaxes over its own count and ends in its own argmax. N sequences cost
// one pass over the weights instead of N.
func (g *devTier) LayersRows(lo, hi int, pos, slot []int, seqLen int, x, cs, csSWA []float32, head *nn.Head) bool {
	v := g.as(0)
	if v == nil {
		return false
	}
	defer v.done()
	return v.layersRows(lo, hi, &ragStep{pos: pos, slot: slot, seqLen: seqLen}, x, cs, csSWA, head)
}

// LayersRows runs blocks [lo, hi) for the rows, one call per device run, the
// residual crossing to the next device through x. Each row's history for a
// block lives in the cache of the device that holds the block. The head rides
// the last block's step where it is on that device, and is a step of its own
// on the device holding it otherwise.
func (g *GPU) LayersRows(lo, hi int, pos, slot []int, seqLen int, x, cs, csSWA []float32, head *nn.Head) bool {
	return g.layersRows(0, lo, hi, pos, slot, nil, seqLen, x, cs, csSWA, head)
}

// RowsWidth is the widest step of at most want rows every device holding a
// block can run (nn.RowsWidther): each device's batched scratch for it held
// or built now (batchFor), halving the width until one fits. 0 when none
// does, or no device holds a block.
func (g *GPU) RowsWidth(want int) int {
	w := min(want, batchWidth)
	if g.roomFaulted(roomNarrowLie) {
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
		if held && d.bs != nil {
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
// batched scratch for rows rows (batchFor) and, where a linear block is
// placed, the session's recurrent seat for seqs sequences (growSeat).
func (g *GPU) rowsRoom(sid uint64, rows, seqs int) bool {
	if g.roomFaulted(roomNoseatLie) || g.roomFaulted(roomNarrowLie) {
		return true
	}
	r := (rows + 7) / 8 * 8
	for _, d := range g.devs {
		v := d.as(sid)
		if v == nil {
			continue
		}
		v.mu.Lock()
		held, linear := false, false
		for _, l := range v.layers {
			if l != nil && l.ok {
				held = true
				linear = linear || l.linear
			}
		}
		ok := !held || v.bs != nil && r <= batchWidth && v.batchFor(r) >= r && (!linear || v.growSeat(sid, seqs))
		v.mu.Unlock()
		v.done()
		if !ok {
			return false
		}
	}
	return true
}

// RowsRoom is rowsRoom for the zero session (nn.RowsRoomer).
func (g *GPU) RowsRoom(rows, seqs int) bool { return g.rowsRoom(0, rows, seqs) }

// RowsRoom is the session's (nn.RowsRoomer): its recurrent seat is its own.
func (s *gpuSession) RowsRoom(rows, seqs int) bool { return s.g.rowsRoom(s.sid, rows, seqs) }

// RowsWidth forwards the tier's (nn.RowsWidther): scratch is the device's,
// not a session's.
func (s *gpuSession) RowsWidth(want int) int { return s.g.RowsWidth(want) }

func (s *gpuSession) LayersRows(lo, hi int, pos, slot []int, seqLen int, x, cs, csSWA []float32, head *nn.Head) bool {
	defer s.leave(s.inputsTo(s.step()))
	return s.g.layersRows(s.sid, lo, hi, pos, slot, nil, seqLen, x, cs, csSWA, head)
}

// LayersSessions runs one step for rows of different sessions on this GPU, row
// i being sess[i]'s sequence at pos[i] (nn.SessionStepper): every block's
// weights are read once for all of them, where one session after another
// reads them once each. Each row's history is its own session's. A session may
// take several rows -- a prompt chunk beside the others' decoding -- which
// must be its next positions, each once (devTier.groupRuns refuses a gap or a
// repeat); every row's K and V are written before any row attends, so a
// chunk's rows see each other causally.
func (s *gpuSession) LayersSessions(lo, hi int, sess []nn.LayerDevice, pos []int, x, cs, csSWA []float32, head *nn.Head) bool {
	s.g.resetRecSteps(s.sid)
	// The session's own lists, reused: a session steps from one goroutine.
	s.sidBuf = slices.Grow(s.sidBuf[:0], len(sess))[:len(sess)]
	s.slotBuf = slices.Grow(s.slotBuf[:0], len(sess))[:len(sess)]
	sid, slot := s.sidBuf, s.slotBuf
	for i, d := range sess {
		o, ok := d.(*gpuSession)
		if !ok || o.g != s.g {
			s.g.mu.Lock()
			s.g.rowsErr = fmt.Sprintf("sessions: row %d is not a session of this GPU", i)
			s.g.mu.Unlock()
			return false
		}
		sid[i], slot[i] = o.sid, pos[i]
	}
	defer s.leave(s.inputsTo(s.step()))
	// A session is one sequence whose slots are its positions, so the span a
	// sequence takes is no matter: every row's sequence is its session's 0.
	return s.g.layersRows(s.sid, lo, hi, pos, slot, sid, 0, x, cs, csSWA, head)
}

// ragKey names a ragged step's compiled matvec in devTier.ragK: the kind of
// kernel and the shape fields it is keyed on. A struct, where a formatted
// string was built for every lookup -- every matvec of every step.
type ragKey struct {
	kind                               string
	q                                  kernels.Quant
	k, rows, ntok, r, tok, rowt, split int
	off                                int // a tail launch's first token
	tile                               [5]int
	grp, bias                          bool
	// epi and act are a few-sequence matvec's epilogue (groupKern).
	epi groupEpi
	act kernels.ActKind
}

// headMV is the decode head projection as groupMV reads it: its split, its
// format and its k.
func (g *devTier) headMV() mv {
	m := g.bs.mvHead
	m.q, m.k = g.bs.head.t, g.bs.p.NEmbd
	return m
}

// groupTokMax is the widest step groupMV takes: one thread carries every
// token, so its dot products grow with the step and past four sequences the
// tiled twins win: on an 8B Q4_K_M model groupMV leads at two and four
// sessions and loses at eight (gpu-kernels.md, "Measurements once cited in
// jit/gpu/tier's comments").
const groupTokMax = 4

// groupEpi is the epilogue a groupMV launch carries, decode's own fusions
// over every token of the step: the plain row, the residual added per token
// (decode's mvres: the projection writes x + W*a), or act(gate)*up (decode's
// mvgate). Without them a step pays an Add or an ActMul launch per block that
// a decode token does not.
type groupEpi int

const (
	groupPlain groupEpi = iota
	groupRes
	groupGate
)

// segShapeKey names a few-sequence step's q/k/v launch (groupQKV) by its three
// shapes, the step and the split.
type segShapeKey struct {
	t                 [3]kernels.Quant
	rows              [3]int
	bias              [3]bool
	k, ntok, r, split int
	grp               bool
}

// rowClass is the executed row-width class (graph.RowClass) of a launch of
// ntok rows: one row is a decode; a prompt chunk is at least half the
// reserved prompt width, the width the prompt's verdicts were timed at; and
// anything between (a speculation's verification, a ragged step, a prompt's
// tail) is the few-row class.
func (g *devTier) rowClass(ntok int) graph.RowClass {
	switch {
	case ntok <= 1:
		return graph.RowsDecode
	case ntok*2 >= g.promptW:
		return graph.RowsChunk
	}
	return graph.RowsFew
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

// fusedKern is a prompt chunk's fused operand kernel under key, compiled on
// first use from build, or nil where the generator or the device refuses it
// -- remembered, so the caller keeps its two launches without asking again.
func (g *devTier) fusedKern(key ragKey, build func() (*ir.Kernel, error)) backend.Kernel {
	// A build reached from inside a submission takes g.mu (subLock).
	defer g.subUnlock(g.subLock())
	if k, ok := g.ragK[key]; ok {
		return k
	}
	ker, err := build()
	var c backend.Kernel
	if err == nil {
		if c, err = g.dev.Compile(ker); err != nil {
			c = nil
		}
	}
	if g.ragK == nil {
		g.ragK = map[ragKey]backend.Kernel{}
	}
	g.ragK[key] = c
	return c
}

// fillSplit is the k-split that brings threads up to fill: doubled while the
// grid is short of it and the halves still divide the trips, at most 8.
func fillSplit(trips, threads, fill int) int {
	split := 1
	for threads*split < fill && trips%(2*split) == 0 && split < 8 {
		split *= 2
	}
	return split
}

// kb0Split is GemmVolta's k-split for a grid of groups workgroups over trips
// k-trips: doubled while under 64 workgroups (an 80-SM Volta card holds two
// of these an SM) and while the halves still divide the trips.
func kb0Split(trips, groups int) int {
	split := 1
	for groups*split < 64 && trips%(2*split) == 0 && split < 8 {
		split *= 2
	}
	return split
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
	// Gemma 4: the range's geometry's scratch set (gemma4.go), as layersCall.
	g.mu.Lock()
	prev := g.geoCur
	if g.geoUsed {
		g.useGeom(g.geomFor(lo, hi))
	}
	g.mu.Unlock()
	defer g.restoreGeom(prev)
	switch {
	case g.rowsFaulted(rowsSwatab) && csSWA != nil:
		csSWA = cs
	case g.rowsFaulted(rowsNoswa):
		csSWA = nil
	}
	if n < 1 || len(slot) != n || lo < 0 || lo > hi || (lo == hi && head == nil) {
		return fail("%d positions, %d slots, blocks [%d,%d)", n, len(slot), lo, hi)
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
	grain := g.rowGrain()
	w := (n + grain - 1) / grain * grain
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
		case l.nonCausal || l.stream != nil && !l.stream.hybrid:
			g.mu.Unlock()
			return fail("block %d is non-causal or streamed", li)
		}
	}
	// The step's own width, or the reserved one when that does not fit.
	ok := g.bs != nil
	if ok {
		w = g.batchFor(w)
		ok = w > 0
	}
	bs := g.bbs[w]
	if ok && head != nil && !g.ensureRowsHead(bs, head) {
		g.mu.Unlock()
		return fail("the head has no composed per-row form: %s", g.LastErr)
	}
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

// layersRows is LayersRows for session self, with each row's session, or nil
// for self's.
func (g *GPU) layersRows(self uint64, lo, hi int, pos, slot []int, sid []uint64, seqLen int, x, cs, csSWA []float32, head *nn.Head) bool {
	g.resetRecSteps(self)
	g.mu.Lock()
	rs, ok := g.runsInto(g.runBuf, lo, hi)
	g.runBuf = nil
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
	// home scratch set (gemma4.go) leaves the projection to a step of its
	// own on the head's device, the residual crossing through x, as Layers
	// does.
	if why == "" && head != nil {
		if r := rs[len(rs)-1]; r.dev != g.head || r.geo != 0 {
			rs = append(rs, run{dev: g.head, lo: hi, hi: hi, split: g.split})
		}
	}
	g.rowsErr = why
	g.mu.Unlock()
	defer g.putRuns(rs)
	if why != "" {
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
		v := r.dev.as(self)
		if v == nil {
			why = fmt.Sprintf("%s: no scratch for this session: %s", label(r.dev.dev, r.dev.ord), r.dev.stats().LastErr)
			g.mu.Lock()
			g.rowsErr = why
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
		if !ok {
			why = fmt.Sprintf("%s, blocks [%d,%d): %s", label(r.dev.dev, r.dev.ord), r.lo, r.hi, r.dev.stats().LastErr)
			g.mu.Lock()
			g.rowsErr = why
			g.mu.Unlock()
			return false
		}
	}
	return true
}

// prewarmPrompt compiles, ahead of the first prompt, the fused binary16
// operand kernels of every placed block at the reserved prompt width, which a
// submission would otherwise compile on first use (the head's two prompt
// kernels are prewarmHeadKernels'). A kernel a block turns out not to launch
// is a compile spent at load, never a wrong answer. Callers hold g.mu.
func (g *devTier) prewarmPrompt() {
	if g.bs == nil || g.promptW <= 0 || g.bbs[g.promptW] == nil {
		return
	}
	if g.NoPromptPrewarm {
		return
	}
	g.prewarmFused(g.bbs[g.promptW])
}

// prewarmFused forms every placed block's program for a step of bs's rows.
// Callers hold g.mu.
func (g *devTier) prewarmFused(bs *blockScratch) {
	g.ensureFused(0, len(g.layers), bs.rows, bs, nil)
}

// anyNarrow reports whether a placed block or the head holds a matrix
// narrower than f32 -- quantized, or a two-byte float: what a matrix
// instruction would be asked for. Callers hold g.mu.
func (g *devTier) anyNarrow() bool {
	narrow := func(q kernels.Quant) bool { return !kernels.IsFloat(q) || q.Info().Float == 2 }
	if g.bs != nil && g.bs.head != nil && g.bs.head.t != 0 && narrow(g.bs.head.t) {
		return true
	}
	for _, l := range g.layers {
		if l == nil || !l.ok {
			continue
		}
		for _, m := range l.ms {
			if m != nil && m.rows > 0 && narrow(m.q) {
				return true
			}
		}
	}
	return false
}

// rowGrain is a step over rows' grain: the matrix instruction's 32 rows
// where the device lowers it, or where the model asks it for nothing, and 8
// where the model would ask it -- a narrow matrix (quantized, or a two-byte
// float a staged GEMM would take), a mixture's banks, or a prefill attention
// tile at a head a whole number of 32 wide where no collective tile serves
// it -- and the device does not: the dp4a twins and every attention tile
// take 8. Callers hold g.mu.
func (g *devTier) rowGrain() int {
	if g.NoMMA {
		return 8
	}
	if g.intMMA == 0 {
		g.intMMA = -1
		if ok, _ := backend.LowersIntMMA(g.dev); ok {
			g.intMMA = 1
		}
	}
	if g.intMMA > 0 || g.bs == nil {
		return batchGrain
	}
	// What any geometry's blocks ask, the tier's grain being one: a
	// mixture's banks, or a prefill attention tile at a head a whole number
	// of 32 wide where no collective tile serves it.
	asks := func(p *nn.LayerPlan) bool {
		attn := g.composeTarget().API != compose.APIMSL && p.HeadDim > 0 && p.HeadDim%32 == 0
		return p.NExpert > 0 || attn
	}
	if g.anyNarrow() || asks(&g.bs.p) {
		return 8
	}
	for _, gs := range g.geoSets {
		if gs.bs != nil && asks(&gs.bs.p) {
			return 8
		}
	}
	return batchGrain
}
