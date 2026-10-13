package tier

import (
	"fmt"
	"math"
	"slices"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
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
// reads its own sequence's keys (kernels.AttnScoresRagged/AttnAccRagged),
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
	case rowsFaulted("swatab") && csSWA != nil:
		csSWA = cs
	case rowsFaulted("noswa"):
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
	// The row grain is the matrix instruction's 32 only where there is one;
	// the dp4a twin (ragMV) and every attention tile take 8, and padding to
	// 32 would do 32 rows of work. One compile settles which this device has.
	grain := batchGrain
	if l := g.layers[lo]; l != nil && l.ok {
		// The block's first projection: a latent block with a query latent
		// has no mvq, and a linear block none of the attention's.
		for _, m := range []mv{l.mvq, l.mvQA, l.mvKVA, l.mvSO} {
			if m.kern != nil {
				g.batchMV(m, batchGrain, 1)
				break
			}
		}
	}
	if g.mmaOff || g.NoMMA {
		grain = 8
	}
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
	// A final softcap (gemma2) caps every row the head projects, as decode's
	// head caps its one (prepRowsHead builds it from the step's head). A head
	// whose cap disagrees with the one the device was given would be a
	// different function from the sessions' decode.
	if head != nil && g.bs != nil && (head.Softcap != 0) != (g.bs.headCap != nil) {
		g.mu.Unlock()
		return fail("the step's head has a final softcap of %g and the device's head was built "+
			"with a cap %v", head.Softcap, g.bs.headCap != nil)
	}
	// The step's own width, or the reserved one when that does not fit.
	ok := g.bs != nil && (head == nil || g.bs.head != nil && g.bs.head.ok)
	if ok {
		w = g.batchFor(w)
		ok = w > 0 && g.prepRagged(g.bbs[w], head, n)
	}
	bs := g.bbs[w]
	if ok && linear {
		ok = g.prepRagLinear(bs, n) && g.prepRecRows(sid, lo, hi, rs)
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

// prepRagged builds bs's ragged-decode set once, outside any session: Compile
// and Alloc are driver calls a capture forbids. Callers hold g.mu. The head's
// part (the batched projection and the per-row argmax) is built only for a
// step that runs the head, and over the rows it wants (nn.Head.LogitRows).
func (g *devTier) prepRagged(bs *blockScratch, rowsHead *nn.Head, n int) bool {
	if bs == nil {
		return false
	}
	defer g.scratchWin().close()
	// Every block's matvecs, on every call: a map lookup once compiled, and
	// the one place they can be compiled (Compile is refused inside a session).
	for _, l := range g.layers {
		if l == nil || !l.ok {
			continue // a block of a split placement that another tier runs
		}
		// A block of another scratch set is built when its set's step is
		// (gemma4.go): this set's batched scratch may not run it at all (a
		// mixture block beside MiniMax-M3's dense lead).
		if g.geoUsed && l.geo != g.geoCur {
			continue
		}
		// A linear block's projections and the shared expert's too: every
		// matvec a ragged block can launch, as prepBatch's list.
		// A latent block's query, latent and output projections too; its
		// absorb banks run grouped (prepBatch builds them).
		for _, m := range []mv{l.mvq, l.mvk, l.mvv, l.mvo, l.mvg, l.mvu, l.mvd,
			l.mvsg, l.mvsu, l.mvsd, l.mvSG, l.mvBA, l.mvSO, l.mvFA, l.mvFB, l.mvGA, l.mvGB,
			l.mvpg, l.mvpp,
			l.mvQA, l.mvQB, l.mvKVA,
			l.mvIQ, l.mvIK, l.mvIW, l.mvLL, l.mvLR, l.mvMQ,
			l.mvCKV, l.mvCG, l.mvICKV, l.mvICG} {
			if m.kern == nil || m.slots > 0 {
				continue
			}
			if _, ok := g.ragStepMV(m, bs.rows, n); !ok {
				return false
			}
		}
		if !g.ds4Twins(l, bs.rows) {
			return false
		}
		// Decode's fusions over the step, where decode has them: built here
		// because a session cannot compile, and declined (the separate
		// launches serve) where a shape refuses.
		g.groupKern(l.mvo, n, bs.rows, groupRes, 0)
		g.groupKern(l.mvd, n, bs.rows, groupRes, 0)
		g.groupKern(l.mvu, n, bs.rows, groupGate, bs.p.Act)
		g.groupQKV(l, n, bs.rows)
	}
	return g.prepRowsHead(bs, rowsHead, n)
}

// prepRowsHead builds bs's per-row head -- the batched projection over the
// rows a step wants and a per-row argmax -- once, outside any session. A
// ragged step needs it, and so does a chunk whose head wants every row
// (nn.Head.RowLogits on Layers). Callers hold g.mu.
func (g *devTier) prepRowsHead(bs *blockScratch, rowsHead *nn.Head, n int) bool {
	defer g.scratchWin().close()
	p := &bs.p
	R := bs.rows
	fail := func(err error) bool { g.LastErr = "rows: " + err.Error(); return false }
	comp := func(k *ir.Kernel, err error) backend.Kernel {
		if err != nil {
			fail(err)
			return nil
		}
		c, err := g.dev.Compile(k)
		if err != nil {
			fail(err)
			return nil
		}
		return c
	}
	// A batch's rows read their history through the scratch's paged set
	// (pagedDecode); what is left to build is the head.
	if rowsHead == nil {
		return true
	}
	head := g.bs.head
	g.groupMV(g.headMV(), rowsHead.WantedRows(n), R)
	g.headOneMV(R)
	if bs.ragArgmax == nil {
		amk := comp(kernels.ArgmaxRows(head.nrows, R))
		if amk == nil {
			return false
		}
		hm, ok := g.ragMV(mv{q: head.t, k: p.NEmbd, rows: head.nrows}, R)
		if !ok {
			return fail(fmt.Errorf("no batched head matvec: %s", g.LastErr))
		}
		// reserveRowsHead may have allocated them at placement, from the
		// plan's vocabulary; a head of another width replaces them.
		if bs.ragLogits != nil && bs.ragVocab != head.nrows {
			freeRagOut(bs)
		}
		var err error
		if bs.ragLogits == nil {
			if bs.ragLogits, err = g.dev.Alloc(R * head.nrows * 4); err == nil {
				bs.ragVocab = head.nrows
				bs.ragTok, err = g.dev.Alloc(R * 4)
			}
		}
		if err != nil {
			return fail(err)
		}
		bs.ragArgmax, bs.ragHead = amk, hm
	}
	// The final softcap over every row the head can project: the kernel is
	// sized for R rows and launched over the wanted ones.
	if c := rowsHead.Softcap; c != 0 && (bs.ragCapK == nil || bs.ragCapC != c) {
		k := comp(kernels.Softcap(R*head.nrows, c))
		if k == nil {
			return false
		}
		if bs.ragCap == nil {
			b, err := g.dev.Alloc(R * head.nrows * 4)
			if err != nil {
				closeKern(k)
				return fail(err)
			}
			bs.ragCap = b
		}
		if bs.ragCapK != nil {
			bs.ragCapK.Close()
		}
		bs.ragCapK, bs.ragCapC = k, c
	}
	return true
}

// LayersRows runs blocks [lo, hi) for the rows, one call per device run, the
// residual crossing to the next device through x. Each row's history for a
// block lives in the cache of the device that holds the block. A head is only
// taken on the device that runs the last block; a head-only step (the blocks
// elsewhere, the projection here) is refused rather than run.
func (g *GPU) LayersRows(lo, hi int, pos, slot []int, seqLen int, x, cs, csSWA []float32, head *nn.Head) bool {
	return g.layersRows(0, lo, hi, pos, slot, nil, seqLen, x, cs, csSWA, head)
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
	case !ok || len(rs) == 0:
		why = fmt.Sprintf("rows: blocks [%d,%d) are not all on devices", lo, hi)
	case head != nil && rs[len(rs)-1].dev != g.head:
		why = "rows: the head is not on the device that runs the last block"
	}
	// A last run outside the home scratch set (gemma4.go) leaves the head to
	// a step of its own on the same device, as Layers does.
	if why == "" && head != nil && rs[len(rs)-1].geo != 0 {
		r := rs[len(rs)-1]
		rs = append(rs, run{dev: r.dev, lo: hi, hi: hi, split: r.split})
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

// ragMV is a ragged step's batched twin of m, compiled on first use -- which
// prepRagged makes happen outside any session, so inside one it is a lookup.
//
// On a device with a matrix instruction it is the prefill twin. Without one it
// is not: the prefill twin's dp4a tile is sized for 128-row chunks and leaves a
// small step's grid nearly empty. Here every token of the step is in one
// thread, so a weight is read once, and k is split until the grid fills the
// card, with a Reduce after.
func (g *devTier) ragMV(m mv, ntok int) (mv, bool) {
	if _, _, ok := g.mmaTile(m.rows, ntok); ok && !kernels.IsFloat(m.q) {
		return g.intBatchMV(m, ntok, 1)
	}
	if bm, ok := g.voltaMV(m, ntok); ok {
		return bm, true
	}
	// Tok 8; Rowt 2 below 64 rows and 4 from there, swept on a Volta-class card
	// (Tok 4 and 16 lost at every width).
	tok, rowt := min(ntok, 8), 2
	if ntok >= 64 {
		rowt = 4
	}
	if b := g.kb.batch; b.RagTok > 0 && ntok%b.RagTok == 0 {
		tok = b.RagTok
	}
	if b := g.kb.batch; b.RagRowT > 0 {
		rowt = b.RagRowT
	}
	for ntok%tok != 0 {
		tok /= 2
	}
	for rowt > 1 && m.rows%rowt != 0 {
		rowt /= 2
	}
	return g.dot4Split(m, ntok, tok, rowt, g.kb.batch.RagSplit)
}

// headMV is the decode head projection as groupMV reads it: its split, its
// format and its k.
func (g *devTier) headMV() mv {
	m := g.bs.mvHead
	m.q, m.k = g.bs.head.t, g.bs.p.NEmbd
	return m
}

// headOneMV is decode's head matvec -- its split and its Reduce -- reading the
// first of R quantized rows, which is the one row of a ragged step that wants
// the head (rows that want logits lead): one projection where a batched twin
// would project every row of the step to read back one. The activation's
// sums sit after every row's scales (kernels.MatVecShape.ActRows), which is
// the only way it differs from decode's own kernel. Compiled on first use,
// which prepRagged makes outside any session; false where the shape is
// refused, and the batched twin serves.
func (g *devTier) headOneMV(R int) (mv, bool) {
	// A build reached from inside a submission takes g.mu (subLock).
	defer g.subUnlock(g.subLock())
	m := g.bs.mvHead
	if g.NoRagHeadOne || m.kern == nil {
		return mv{}, false
	}
	q, k := g.bs.head.t, g.bs.p.NEmbd
	key := ragKey{kind: "headone", q: q, k: k, rows: m.rows, r: R, split: m.split}
	c, ok := g.ragK[key]
	if !ok {
		ker, err := kernels.MatVec(kernels.MatVecShape{Center: g.center, T: q, K: k, Rows: m.rows,
			Split: m.split, ActRows: R})
		if err == nil {
			c, err = g.dev.Compile(ker)
		}
		if err != nil {
			g.LastErr = "rows: the one-row head: " + err.Error()
			c = nil
		}
		if g.ragK == nil {
			g.ragK = map[ragKey]backend.Kernel{}
		}
		g.ragK[key] = c
	}
	if c == nil {
		return mv{}, false
	}
	m.kern, m.alt = c, nil
	return m, true
}

// ragStepMV is the matvec a ragged step of n live rows in a scratch of R
// runs: the decode matvec over the n rows where it applies (groupMV), else
// the batched twin over all R, padding included.
func (g *devTier) ragStepMV(m mv, R, n int) (mv, bool) {
	if bm, ok := g.groupMV(m, n, R); ok {
		return bm, true
	}
	return g.ragMV(m, R)
}

// groupTokMax is the widest step groupMV takes: one thread carries every
// token, so its dot products grow with the step and past four sequences the
// tiled twins win: on an 8B Q4_K_M model groupMV leads at two and four
// sessions and loses at eight (gpu-kernels.md, "Measurements once cited in
// jit/gpu/tier's comments").
const groupTokMax = 4

// groupMV is m's decode matvec carrying every token of a small step in one
// thread: each weight word is read once for all the step's tokens. That is
// llama.cpp's mul_mat_vec_q at ncols > 1, and on a card with no integer
// matrix instruction it is what a step of a few sequences wants -- the
// tensor-core and dp4a tiles read the weights at three quarters the rate,
// pad to their tile and add a Reduce (and an f16 conversion) per matvec.
//
// It takes decode's own split, so each token sums its k segments as decode
// does: unsplit where decode is unsplit (the head), and otherwise reduced in
// the group whether decode reduces there or in a second launch -- the
// partials are added in segment order either way, and the launch is saved. A
// split the group cannot hold is declined, and the tiled twin serves it. The
// activation holds R quantized rows, of which the step's ntok come first.
func (g *devTier) groupMV(m mv, ntok, R int) (mv, bool) {
	return g.groupKern(m, ntok, R, groupPlain, 0)
}

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

// groupKern is groupMV with an epilogue; act is the gate's activation. The
// residual form goes where decode built one (m.res: the attention output and
// the FFN down projection, with no bias of their own). The gate form goes
// wherever the gate is SiLU or GELU and the up projection has no bias: decode
// fuses it only where its split writes the final row, and this kernel always
// does, reducing its split in the group.
func (g *devTier) groupKern(m mv, ntok, R int, epi groupEpi, act kernels.ActKind) (mv, bool) {
	// A build reached from inside a submission takes g.mu (subLock).
	defer g.subUnlock(g.subLock())
	if ntok < 2 || ntok > groupTokMax || m.slots > 0 || kernels.IsFloat(m.q) || g.NoRagGroup {
		return mv{}, false
	}
	switch {
	case epi == groupRes && (m.res == nil || m.bias != nil || g.NoRagFuse),
		epi == groupGate && (m.bias != nil || g.NoRagFuse || (act != kernels.ActSiLU && act != kernels.ActGELU)):
		return mv{}, false
	}
	split := max(m.split, 1)
	grp := split > 1
	if grp && !kernels.GroupSplitOK(m.rows, split) {
		return mv{}, false
	}
	key := ragKey{kind: "grp", q: m.q, k: m.k, rows: m.rows, ntok: ntok, r: R, split: split, grp: grp,
		bias: m.bias != nil, epi: epi, act: act}
	k, ok := g.ragK[key]
	if !ok {
		ker, err := kernels.MatVec(kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows, NTok: ntok,
			Tok: ntok, ActRows: R, Split: split, GroupSplit: grp, Bias: m.bias != nil || epi == groupRes,
			BiasRows: epi == groupRes, Gate: epi == groupGate, GateAct: act})
		if err == nil {
			k, err = g.dev.Compile(ker)
		}
		if err != nil {
			k = nil
		}
		if g.ragK == nil {
			g.ragK = map[ragKey]backend.Kernel{}
		}
		g.ragK[key] = k
	}
	if k == nil {
		return mv{}, false
	}
	if epi == groupPlain {
		g.RagGroupMV++
	}
	return mv{kern: k, rows: m.rows, split: split, groups: 1, group: grp, bias: m.bias, q: m.q, k: m.k}, true
}

// groupQKV is block l's q, k and v for a step of ntok rows in a scratch of R
// as one kernels.MatVecSegments launch with every token in one thread --
// decode's l.qkv over the step, at q's split as there, reduced in the group
// -- or nil where the three cannot share one (fuseQKV's conditions, less the
// one on decode's own split: this kernel always reduces in the group).
func (g *devTier) groupQKV(l *layer, ntok, R int) (segLaunch, bool) {
	// A build reached from inside a submission takes g.mu (subLock).
	defer g.subUnlock(g.subLock())
	if ntok < 2 || ntok > groupTokMax || g.NoRagGroup || g.NoRagFuse || g.NoSegFuse ||
		l.mla || l.linear || l.nonCausal || l.wq == nil || l.wk == nil || l.wv == nil {
		return segLaunch{}, false
	}
	split := max(l.mvq.split, 1)
	grp := split > 1
	ms := [3]mv{l.mvq, l.mvk, l.mvv}
	sk := segShapeKey{k: l.mvq.k, ntok: ntok, r: R, split: split, grp: grp}
	for i, m := range ms {
		if m.kern == nil || m.slots > 0 || m.k != l.mvq.k || kernels.IsFloat(m.q) {
			return segLaunch{}, false
		}
		sk.t[i], sk.rows[i], sk.bias[i] = m.q, m.rows, m.bias != nil
	}
	// A lookup on a struct, since an encode runs this every block of every
	// step and a decode token allocates nothing.
	if sl, ok := g.ragSeg[sk]; ok {
		return sl, sl.kern != nil
	}
	var segs []kernels.MatVecShape
	for _, m := range ms {
		segs = append(segs, kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows,
			Bias: m.bias != nil, NTok: ntok, Tok: ntok, ActRows: R})
	}
	if g.ragSeg == nil {
		g.ragSeg = map[segShapeKey]segLaunch{}
	}
	blocks, w, err := kernels.SegmentsGrid(segs, split, grp)
	if err != nil {
		g.ragSeg[sk] = segLaunch{}
		return segLaunch{}, false
	}
	key := fmt.Sprintf("%v/%d/%v", segs, split, grp)
	k, ok := g.segKerns[key]
	if !ok {
		ker, err := kernels.MatVecSegments(segs, split, grp)
		if err == nil {
			k, err = g.dev.Compile(ker)
		}
		if err != nil {
			g.LastErr = err.Error()
			k = nil
		}
		if g.segKerns == nil {
			g.segKerns = map[string]backend.Kernel{}
		}
		g.segKerns[key] = k
	}
	n := 0
	for _, b := range blocks {
		n += b
	}
	sl := segLaunch{kern: k, blocks: n, w: w}
	g.ragSeg[sk] = sl
	return sl, k != nil
}

// segShapeKey names a few-sequence step's q/k/v launch (groupQKV) by its three
// shapes, the step and the split.
type segShapeKey struct {
	t                 [3]kernels.Quant
	rows              [3]int
	bias              [3]bool
	k, ntok, r, split int
	grp               bool
}

// dot4Split is the dp4a batched twin of m with k split until the grid holds
// ~65536 threads (or split pinned when > 0), and the Reduce that sums it. The
// Tok x Rowt tile divides the grid, so without the split a narrow projection
// (1024 rows) runs nearly as long as one fourteen times its work.
func (g *devTier) dot4Split(m mv, ntok, tok, rowt, split int) (mv, bool) {
	// A build reached from inside a submission takes g.mu (subLock).
	defer g.subUnlock(g.subLock())
	if split < 1 {
		split = 1
		for m.rows/rowt*(ntok/tok)*split < 1<<16 && split < 64 {
			split *= 2
		}
	}
	var c backend.Kernel
	// A refused split is the descent working, not a failure: only the last
	// refusal is recorded, and only when no split compiled. Recorded as it
	// went, a narrow projection (Gemma 3n's LAuReL, k=64) left
	// "Split=4 does not divide 2 sub-blocks" in LastErr on a model that ran
	// fine.
	refusal := ""
	for ; split >= 1; split /= 2 {
		key := ragKey{kind: "dot4", q: m.q, k: m.k, rows: m.rows, ntok: ntok, tok: tok, rowt: rowt, split: split,
			bias: m.bias != nil}
		k, ok := g.ragK[key]
		if !ok {
			// A split that does not divide the row's sub-blocks is refused
			// by MatVec, and the next one down is tried.
			ker, err := kernels.MatVec(kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows, NTok: ntok,
				Tok: tok, Rowt: rowt, Split: split, Bias: m.bias != nil})
			if err == nil {
				k, err = g.dev.Compile(ker)
			}
			if err != nil {
				refusal = err.Error()
				k = nil
			}
			if g.ragK == nil {
				g.ragK = map[ragKey]backend.Kernel{}
			}
			g.ragK[key] = k
		}
		if c = k; c != nil {
			break
		}
	}
	if c == nil {
		if refusal != "" {
			g.LastErr = refusal
		}
		return mv{}, false
	}
	out := mv{kern: c, rows: m.rows / rowt, split: split, groups: ntok / tok, bias: m.bias,
		q: m.q, k: m.k, redN: m.rows * ntok}
	if split > 1 {
		if out.red = g.reduceKernel(m.rows*ntok, split); out.red == nil ||
			!g.sizePart(m.rows*ntok*split) {
			return mv{}, false
		}
	}
	return out, true
}

// voltaMV is m's batched twin on sm_70's f16 tensor cores (kernels.MatVecMMA70)
// and the ActF16 conversion it reads, or false where it does not apply: a card
// with an integer matrix instruction takes that instead (batchMV asks first),
// and a backend that does not lower the m8n8k4 shape refuses the compile, which
// is remembered. On a Volta-class card it is ~2.3x the dp4a twin (MT=2 NT=4,
// swept).
func (g *devTier) voltaMV(m mv, ntok int) (mv, bool) {
	// A build reached from inside a submission takes g.mu (subLock).
	defer g.subUnlock(g.subLock())
	if g.NoVolta || m.slots > 0 || !kernels.GemmVoltaOK(m.q) {
		return mv{}, false
	}
	// A two-byte float weight takes the staged GEMM where the card lowers no
	// m16n8 instruction (sm_70): the integer MMA, which decides mmaOff, never
	// asks for a float, so it would wait on a refusal that does not come.
	// Elsewhere it keeps its own batched kernel, unchanged.
	if kernels.IsFloat(m.q) {
		if g.f16GemmK() != 0 {
			return mv{}, false
		}
	} else if !(g.mmaOff || g.NoMMA) {
		return mv{}, false
	}
	if bm, ok := g.voltaGemm(m, ntok); ok {
		return bm, true
	}
	// A float weight has the staged GEMM alone (GemmVoltaOK).
	if !kernels.Volta70OK(m.q) {
		return mv{}, false
	}
	mt, nt := 2, 4
	for nt > 1 && ntok%(8*nt) != 0 {
		nt /= 2
	}
	for mt > 1 && m.rows%(32*mt) != 0 {
		mt /= 2
	}
	if ntok%(8*nt) != 0 || m.rows%(32*mt) != 0 {
		return mv{}, false
	}
	// Split k until the grid holds ~1280 warps (sixteen an SM on an 80-SM
	// card): the tile alone left the narrow projections 64-256 of them.
	warps := (m.rows / (32 * mt)) * (ntok / (8 * nt))
	split := g.kb.batch.Split
	if split < 1 {
		split = 1
		for warps*split < 1280 && split < 16 {
			split *= 2
		}
	}
	var c backend.Kernel
	for ; split >= 1; split /= 2 {
		key := ragKey{kind: "volta", q: m.q, k: m.k, rows: m.rows, ntok: ntok, tok: mt, rowt: nt, split: split,
			bias: m.bias != nil}
		k, ok := g.ragK[key]
		if !ok {
			// A split that does not divide the sub-blocks is refused and the
			// next one down tried; a backend without the shape refuses them all.
			ker, err := kernels.MatVecMMA70(kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows, NTok: ntok,
				MT: mt, NT: nt, Split: split, Bias: m.bias != nil})
			if err == nil {
				k, err = g.dev.Compile(ker)
			}
			if err != nil {
				k = nil
			}
			if g.ragK == nil {
				g.ragK = map[ragKey]backend.Kernel{}
			}
			g.ragK[key] = k
		}
		if c = k; c != nil {
			break
		}
	}
	if c == nil {
		return mv{}, false
	}
	akey := ragKey{kind: "actf16", q: m.q, ntok: ntok, k: m.k}
	a, ok := g.ragK[akey]
	if !ok {
		ker, err := kernels.ActF16(m.q, ntok, m.k)
		if err == nil {
			a, err = g.dev.Compile(ker)
		}
		if err != nil {
			a = nil
		}
		g.ragK[akey] = a
	}
	if a == nil || !g.sizeF16(ntok*m.k*2) {
		return mv{}, false
	}
	out := mv{kern: c, act: a, actN: ntok * m.k / 4, rows: m.rows, split: split, bias: m.bias,
		q: m.q, k: m.k, threads: warps * split * 32, redN: m.rows * ntok}
	if split > 1 {
		if out.red = g.reduceKernel(m.rows*ntok, split); out.red == nil ||
			!g.sizePart(m.rows*ntok*split) {
			return mv{}, false
		}
	}
	g.VoltaMV++
	return out, true
}

// tileGemms are GemmTile's blockings, widest first, all four subgroups (runBatched
// launches 128-thread workgroups); the narrower ones exist for a row or token
// count the wider blocks do not divide.
//
// 64x64 leads 64x32 (llama.cpp's kernel_mul_mm block): a few percent faster on
// Apple Silicon by TestGemmTileSpeed and in pp512.
var tileGemms = []kernels.TileGemm{
	{MT: 4, NT: 4, WM: 2, WN: 2}, {MT: 4, NT: 2, WM: 2, WN: 2}, {MT: 2, NT: 2, WM: 2, WN: 2}, {MT: 1, NT: 2, WM: 2, WN: 2},
}

// tileMV is m's batched twin as kernels.GemmTile (binary16 tiles on the
// collective matrix unit, staged through workgroup memory) with the ActF16T
// conversion it reads; false where the device does not lower it or no tile
// divides the shape, and the next twin is asked.
//
// Whether the device lowers the collective tiles at all is asked once
// (tilesOK) rather than by a refused compile per shape, and a refusal of a
// built kernel is recorded once (tileOff): a property of the device, not the
// shape. Where the device has the integer matrix instruction (MatVecMMA), that
// runs first and this is not reached.
func (g *devTier) tileMV(m mv, ntok int) (mv, bool) {
	// A build reached from inside a submission takes g.mu (subLock).
	defer g.subUnlock(g.subLock())
	if g.NoVolta || m.slots > 0 || !kernels.Volta70OK(m.q) || !g.tilesOK() {
		return mv{}, false
	}
	sub, _, _, _ := kernels.Layout(m.q)
	build := func(tl kernels.TileGemm, n int) (backend.Kernel, kernels.MatVecShape, bool) {
		s := kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows, NTok: n, Bias: m.bias != nil}
		key := ragKey{kind: "gemmtile", q: m.q, k: m.k, rows: m.rows, ntok: n, off: tl.TokOff,
			tile: [5]int{tl.MT, tl.NT, tl.WM, tl.WN, tl.KB}, bias: s.Bias}
		c, ok := g.ragK[key]
		if !ok {
			ker, err := kernels.GemmTile(s, tl)
			if err == nil {
				c, err = g.dev.Compile(ker)
			}
			if err != nil {
				g.tileOff, g.TileWhy = true, err.Error()
				return nil, s, false
			}
			if g.ragK == nil {
				g.ragK = map[ragKey]backend.Kernel{}
			}
			g.ragK[key] = c
		}
		return c, s, true
	}
	act := func() (backend.Kernel, bool) {
		akey := ragKey{kind: "actf16t", q: m.q, ntok: ntok, k: m.k}
		a, ok := g.ragK[akey]
		if !ok {
			ker, err := kernels.ActF16T(m.q, ntok, m.k)
			if err == nil {
				a, err = g.dev.Compile(ker)
			}
			if err != nil {
				a = nil
			}
			if g.ragK == nil {
				g.ragK = map[ragKey]backend.Kernel{}
			}
			g.ragK[akey] = a
		}
		return a, a != nil && g.sizeF16(ntok*m.k*2)
	}
	tall := g.tileTall()
	for wi, tl := range tileGemms {
		tl.KB = max(32/sub, 1)
		tl.Tall = tall && tl.MT%2 == 0
		if m.rows%tl.Rows() != 0 {
			continue
		}
		// The widest block over the leading tokens and the first narrower
		// one that divides the rest over the tail, rather than the narrower
		// block over the whole prompt (416 tokens: 384 at 64 wide, 32 at 32).
		lead, rest := ntok, 0
		if ntok%tl.Toks() != 0 {
			if g.NoTileTail || ntok < tl.Toks() {
				continue
			}
			lead, rest = ntok-ntok%tl.Toks(), ntok%tl.Toks()
		}
		var tail *mvTail
		if rest > 0 {
			for _, tt := range tileGemms[wi+1:] {
				tt.KB, tt.TokOff = tl.KB, lead
				tt.Tall = tall && tt.MT%2 == 0
				if m.rows%tt.Rows() != 0 || rest%tt.Toks() != 0 {
					continue
				}
				tk, ts, ok := build(tt, rest)
				if !ok {
					return mv{}, false
				}
				tail = &mvTail{kern: tk, threads: kernels.GemmTileGroups(ts, tt) * tt.Threads()}
				break
			}
			if tail == nil {
				continue
			}
		}
		c, s, ok := build(tl, lead)
		if !ok {
			return mv{}, false
		}
		a, ok := act()
		if !ok {
			return mv{}, false
		}
		g.TileMV++
		if tail != nil {
			g.TileTail++
		}
		return mv{kern: c, act: a, actN: ntok * m.k / sub, rows: m.rows, split: 1, bias: m.bias,
			q: m.q, k: m.k, threads: kernels.GemmTileGroups(s, tl) * tl.Threads(), redN: m.rows * ntok, tail: tail}, true
	}
	return mv{}, false
}

// voltaTiles are GemmVolta's blockings, widest first. All are four warps
// (runBatched launches 128-thread workgroups), all four along the tokens: a
// 2x2 warp grid dequantizes twice the weights per MMA and measured slower on a
// Volta-class card. The 96-row tiles serve matrices no 128-row block divides
// (gpt-oss's 2880 rows).
var voltaTiles = []kernels.VoltaTile{
	{MT: 4, NT: 4, WM: 1, WN: 4}, {MT: 4, NT: 2, WM: 1, WN: 4}, {MT: 4, NT: 1, WM: 1, WN: 4},
	{MT: 3, NT: 4, WM: 1, WN: 4}, {MT: 3, NT: 2, WM: 1, WN: 4},
	// A 16-token block, for a prompt chunk that is an odd number of 16s
	// (promptGrain on sm_70): one warp a workgroup.
	{MT: 4, NT: 2, WM: 1, WN: 1}, {MT: 3, NT: 2, WM: 1, WN: 1},
}

// voltaGemm is m's batched twin as kernels.GemmVolta, the shared-memory-staged
// form of MatVecMMA70, with the ActF16T conversion it reads; false where no
// tile divides the shape, and voltaMV then takes MatVecMMA70.
//
// k is staged 32 elements a trip (one sub-block of a 32-element format, two of
// Q6_K's 16): four 16-byte chunks a row, the width the swizzle is built for,
// keeping a double-buffered 128x128 block at 32 KiB. k is split only where the
// grid is thin (kb0Split); a well-filled grid is fastest unsplit.
func (g *devTier) voltaGemm(m mv, ntok int) (mv, bool) {
	// A build reached from inside a submission takes g.mu (subLock).
	defer g.subUnlock(g.subLock())
	if bm, ok := g.stagedGemm(m, ntok, voltaTiles, 0, 0); ok || !kernels.IsFloat(m.q) {
		return bm, ok
	}
	// A float weight has no MatVecMMA70 behind this, so a row count no 96-
	// or 128-row block divides (ModernBERT-large's 2624) takes a 64-row one.
	return g.stagedGemm(m, ntok, voltaTiles64, 0, 0)
}

// voltaTiles64 are GemmVolta's 64-row blockings, for the float shapes
// voltaTiles leave to a slower kernel.
var voltaTiles64 = []kernels.VoltaTile{{MT: 2, NT: 4, WM: 1, WN: 4}, {MT: 2, NT: 2, WM: 1, WN: 4}}

// f16Tiles are GemmVolta's m16n8 blockings, widest first: four warps as 2x2,
// each four m-tiles of 16 rows by up to eight n-tiles of 8 tokens, so a
// 128-row block meets 128, 64 or 32 tokens. The 96- and 32-row blocks serve
// matrices no 128-row block divides.
var f16Tiles = []kernels.VoltaTile{
	{MT: 4, NT: 4, WM: 2, WN: 2}, {MT: 4, NT: 2, WM: 2, WN: 2},
	{MT: 3, NT: 4, WM: 2, WN: 2}, {MT: 3, NT: 2, WM: 2, WN: 2},
	{MT: 2, NT: 4, WM: 2, WN: 2}, {MT: 2, NT: 2, WM: 2, WN: 2}, {MT: 1, NT: 2, WM: 2, WN: 2},
}

// f16GemmK is the k of the m16n8 binary16 instruction this device lowers
// (16 from sm_80, 8 on sm_75), or 0 where it lowers neither. Probed once by a
// compile: the PTX target decides, and a refusal is the device's, not the
// shape's. Callers hold g.mu.
func (g *devTier) f16GemmK() int {
	if g.f16K == 0 {
		g.f16K = -1
		if g.dev.API() == "ptx" {
			for _, k := range []int{16, 8} {
				ker, err := kernels.GemmVolta(kernels.MatVecShape{Center: g.center, T: kernels.Q8_0, K: 32, Rows: 32, NTok: 32},
					kernels.VoltaTile{MT: 1, NT: 2, WM: 2, WN: 2, KB: 1, F16K: k})
				if err != nil {
					continue
				}
				if c, err := g.dev.Compile(ker); err == nil {
					c.Close()
					g.f16K = k
					break
				}
			}
		}
	}
	return max(g.f16K, 0)
}

// f16Gemm is m's batched twin as GemmVolta on the m16n8 binary16
// instruction (sm_75 on): the weights dequantized once a workgroup into
// shared memory with their scales folded in, the activations ActF16T's, and
// the dot accumulated in float32 over the whole k. MatVecMMA reads both
// operands from global memory inside its loop, re-dequantizes a weight for
// every 32 tokens and pays a float epilogue per sub-block per instruction;
// this form pays neither. False where the device lowers no m16n8 shape, the
// format has no binary16 dequant, or no tile divides the shape. NoVolta
// refuses it, as it refuses every binary16-activation twin.
func (g *devTier) f16Gemm(m mv, ntok int) (mv, bool) {
	// A build reached from inside a submission takes g.mu (subLock).
	defer g.subUnlock(g.subLock())
	if g.NoVolta || m.slots > 0 || !kernels.Volta70OK(m.q) {
		return mv{}, false
	}
	k := g.f16GemmK()
	if k == 0 {
		return mv{}, false
	}
	// About a workgroup an SM: a sixteenth of the threads the card holds
	// resident (Slots, asked of the driver; 1536 or 2048 an SM, 128 a
	// workgroup). The card decides it, not the shape: the 128x64 block gives
	// a 1024x1024 projection at 512 rows 64 workgroups, three an SM of an
	// RTX 3050 Ti's 20 and fewer than one an SM of a 142-SM card. The bar is
	// sm_70's measured one (kb0Split: a V100's 128-row blocks were fastest
	// unsplit from 64 workgroups on 80 SMs), not full residency: a staged
	// block's shared memory holds an SM to a few of them anyway.
	deep := g.Config.GemmDeep
	if deep == 0 {
		deep = gemmDeepDefault
	}
	if deep > 0 || deep == 0 && g.gemmDeep[splitKey{m.q, m.rows, m.k}] {
		if bm, ok := g.stagedGemm(m, ntok, f16DeepTiles, k, g.dev.Slots()/16); ok {
			g.Stats.GemmDeep++
			return bm, true
		}
	}
	return g.stagedGemm(m, ntok, f16Tiles, k, g.dev.Slots()/16)
}

// f16DeepTiles are the m16n8 blockings in llama.cpp MMQ's shape: eight warps
// a workgroup and 64 elements of k a trip in one buffer (KB counts 32-element
// steps here). They win on some shapes and lose on others, by card, so
// tuneGemm times them against f16Tiles' pick per shape and gemmDeep records
// where they won.
var f16DeepTiles = []kernels.VoltaTile{
	{MT: 4, NT: 4, WM: 2, WN: 4, KB: 2, Single: true}, {MT: 4, NT: 2, WM: 2, WN: 4, KB: 2, Single: true},
}

// gemmDeepDefault stands in for a Config.GemmDeep of 0; only a jitllmtest
// build changes it (ForceGemmDeepForTest).
var gemmDeepDefault int

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

// rmsF16For is the RMSNorm that also writes m's binary16 operand
// (kernels.RMSNormF16TRows; with add, x+y and its norm as bs.addRms), and
// the conversion it stands in for: m's batched twin's ActF16T kernel, which
// the caller names in the conversion memo so the GEMM finds its operand
// written. nil where m's twin reads int8 (no conversion to save), where the
// norm is not RMSNorm's one launch, or NoOperandFuse.
func (g *devTier) rmsF16For(bs *blockScratch, m mv, ntok int, add bool) (norm, conv backend.Kernel) {
	if g.NoOperandFuse || m.kern == nil || bs.rms == nil || bs.normVar != nil {
		return nil, nil
	}
	bm, ok := g.batchMV(m, ntok, bs.tok)
	if !ok || bm.act == nil || m.k != bs.p.NEmbd {
		return nil, nil
	}
	kind := "rmsf16t"
	if add {
		kind = "addrmsf16t"
	}
	p := &bs.p
	k := g.fusedKern(ragKey{kind: kind, q: bm.q, k: p.NEmbd, rows: bs.rows}, func() (*ir.Kernel, error) {
		return kernels.RMSNormF16TRows(p.NEmbd, bs.rows, float32(p.RMSEps), false, add, bm.q, g.warpNorm())
	})
	if k == nil {
		return nil, nil
	}
	return k, bm.act
}

// actF16For is the gated activation that writes m's binary16 operand
// (kernels.ActMulF16T), the conversion it stands in for, and its thread
// count; nil as rmsF16For.
func (g *devTier) actF16For(bs *blockScratch, m mv, ntok int) (act, conv backend.Kernel, threads int) {
	if g.NoOperandFuse || m.kern == nil {
		return nil, nil, 0
	}
	bm, ok := g.batchMV(m, ntok, bs.tok)
	if !ok || bm.act == nil {
		return nil, nil, 0
	}
	// ntok only clamps the last thread (RULE 13). Where a token's words are
	// whole workgroups the launch has no surplus thread at any ntok, so one
	// kernel at the scratch's width serves every chunk and is compiled once,
	// ahead of the first prompt (prewarmPrompt); elsewhere it is per ntok.
	kn := ntok
	if (m.k/2)%128 == 0 && ntok <= bs.rows {
		kn = bs.rows
	}
	k := g.fusedKern(ragKey{kind: "actmulf16t", q: bm.q, k: m.k, ntok: kn, act: bs.p.Act}, func() (*ir.Kernel, error) {
		return kernels.ActMulF16T(bm.q, kn, m.k, bs.p.Act)
	})
	if k == nil {
		return nil, nil, 0
	}
	return k, bm.act, kernels.ActMulF16TThreads(ntok, m.k)
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

// prewarmFused compiles bs's fused operand kernels for every placed block:
// prewarmPrompt's at the reserved width, prewarmPromptWidths' at each exact
// prompt width. Callers hold g.mu.
func (g *devTier) prewarmFused(bs *blockScratch) {
	R := bs.rows
	// The staged attention's span variants (prefillVariantAt), every span a
	// chunk can reach at this scratch's capacity: a chunk compiles the one
	// its history needs otherwise.
	if pg := bs.pkv; pg != nil && pg.pf != nil && pg.pf.flash == nil {
		hi, _ := g.prefillPass()
		if m := bs.p.MaxSeq; m > 0 && m < hi {
			hi = m
		}
		for c := 64; ; c *= 2 {
			if _, err := g.prefillVariantAt(bs, pg.pf, c, 0); err != nil || c >= hi {
				break
			}
		}
	}
	for _, l := range g.layers {
		if l == nil || !l.ok {
			continue
		}
		g.rmsF16For(bs, l.mvq, R, false)
		first := l.mvg
		if first.kern == nil {
			first = l.mvu
		}
		g.rmsF16For(bs, first, R, true)
		g.actF16For(bs, l.mvd, R)
	}
}

// int8GemmOn reports whether the device lowers GemmInt8, probed once by a
// compile (the int8 m16n8k32 instruction is sm_80's). Callers hold g.mu.
func (g *devTier) int8GemmOn() bool {
	if g.i8Gemm == 0 {
		g.i8Gemm = -1
		if g.dev.API() == "ptx" {
			ker, err := kernels.GemmInt8(kernels.MatVecShape{Center: g.center, T: kernels.Q8_0, K: 32, Rows: 32, NTok: 32},
				kernels.Int8Tile{MT: 1, NT: 2, WM: 2, WN: 2})
			if err == nil {
				if c, err := g.dev.Compile(ker); err == nil {
					c.Close()
					g.i8Gemm = 1
				}
			}
		}
	}
	return g.i8Gemm > 0
}

// int8Gemm is m's batched twin as kernels.GemmInt8: MatVecMMA's int8
// arithmetic and its quantized activations, staged through shared memory as
// GemmVolta stages, so a workgroup decodes a trip of its rows once for all
// its tokens. Its blockings are f16Tiles' (the same 16-row m-tiles and 8-token
// n-tiles), chosen by the same fill. False where the device lacks the
// instruction, the format has no int8 decode, or no tile divides the shape.
// Only Config.PreferGemmInt8 asks for it; NoMMA refuses it.
func (g *devTier) int8Gemm(m mv, ntok int) (mv, bool) {
	// A build reached from inside a submission takes g.mu (subLock).
	defer g.subUnlock(g.subLock())
	if !g.PreferGemmInt8 || g.NoMMA || m.slots > 0 || !kernels.Int8OK(m.q) || !g.int8GemmOn() {
		return mv{}, false
	}
	fill := g.dev.Slots() / 16
	first := fillTile(f16Tiles, 16, m.rows, ntok, fill)
	for i := -1; i < len(f16Tiles); i++ {
		if i < 0 && first < 0 || i >= 0 && i == first {
			continue
		}
		idx := i
		if i < 0 {
			idx = first
		}
		vt := f16Tiles[idx]
		tl := kernels.Int8Tile{MT: vt.MT, NT: vt.NT, WM: vt.WM, WN: vt.WN}
		if m.rows%tl.Rows() != 0 || ntok%tl.Toks() != 0 {
			continue
		}
		trips := m.k / 32
		split := fillSplit(trips, (m.rows/tl.Rows())*(ntok/tl.Toks())*tl.Threads(), fill)
		if f := g.kb.batch.Split; f >= 1 {
			split = f
		}
		var c backend.Kernel
		for ; split >= 1; split /= 2 {
			s := kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows, NTok: ntok, Split: split, Bias: m.bias != nil}
			key := ragKey{kind: "gemmint8", q: m.q, k: m.k, rows: m.rows, ntok: ntok,
				tile: [5]int{tl.MT, tl.NT, tl.WM, tl.WN}, split: split, bias: s.Bias}
			k, ok := g.ragK[key]
			if !ok {
				ker, err := kernels.GemmInt8(s, tl)
				if err == nil {
					k, err = g.dev.Compile(ker)
				}
				if err != nil {
					k = nil
				}
				if g.ragK == nil {
					g.ragK = map[ragKey]backend.Kernel{}
				}
				g.ragK[key] = k
			}
			if c = k; c != nil {
				break
			}
		}
		if c == nil {
			continue
		}
		s := kernels.MatVecShape{Rows: m.rows, NTok: ntok, Split: split}
		out := mv{kern: c, rows: m.rows, split: split, bias: m.bias, q: m.q, k: m.k,
			threads: kernels.GemmInt8Groups(s, tl) * tl.Threads(), width: tl.Threads(), redN: m.rows * ntok}
		if split > 1 {
			if out.red = g.reduceKernel(m.rows*ntok, split); out.red == nil ||
				!g.sizePart(m.rows*ntok*split) {
				return mv{}, false
			}
		}
		g.GemmInt8++
		return out, true
	}
	return mv{}, false
}

// fillTile is the tile stagedGemm tries first, on instruction f16k, for a grid of at least fill
// threads: the first of tiles (widest first) that divides the shape and
// reaches it, else the dividing tile with the most workgroups, whose grid a
// k-split (fillSplit) then widens. -1 where none divides.
func fillTile(tiles []kernels.VoltaTile, f16k, rows, ntok, fill int) int {
	best, most := -1, 0
	for i, tl := range tiles {
		tl.F16K = f16k // an m-tile's rows depend on it
		if rows%tl.Rows() != 0 || ntok%tl.Toks() != 0 {
			continue
		}
		groups := (rows / tl.Rows()) * (ntok / tl.Toks())
		if groups*tl.Threads() >= fill {
			return i
		}
		if groups > most {
			best, most = i, groups
		}
	}
	return best
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

// stagedGemm builds m's GemmVolta twin from the first of tiles that divides
// the shape and compiles, with instruction f16k (VoltaTile.F16K). With fill
// > 0 the tile and split are chosen for a grid of at least fill threads
// (fillTile, fillSplit) and the others are the fallbacks; with 0, tiles'
// order and kb0Split decide. Callers hold g.mu.
func (g *devTier) stagedGemm(m mv, ntok int, tiles []kernels.VoltaTile, f16k, fill int) (mv, bool) {
	sub, _, _, _ := kernels.Layout(m.q)
	kbN := max(32/sub, 1)
	first := -1
	if fill > 0 {
		first = fillTile(tiles, f16k, m.rows, ntok, fill)
	}
	// i = -1 is the fill's tile, then the list in order without it.
	for i := -1; i < len(tiles); i++ {
		if i < 0 && first < 0 || i >= 0 && i == first {
			continue
		}
		idx := i
		if i < 0 {
			idx = first
		}
		tl := tiles[idx]
		// A list's KB is the trip in 32-element steps (0 is one).
		tl.KB, tl.F16K = kbN*max(tl.KB, 1), f16k
		s := kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows, NTok: ntok, Bias: m.bias != nil}
		if m.rows%tl.Rows() != 0 || ntok%tl.Toks() != 0 {
			continue
		}
		groups := (m.rows / tl.Rows()) * (ntok / tl.Toks())
		split := kb0Split(m.k/sub/kbN, groups)
		if fill > 0 {
			split = fillSplit(m.k/sub/kbN, groups*tl.Threads(), fill)
		}
		if f := g.kb.batch.Split; f >= 1 {
			split = f
		}
		var c backend.Kernel
		for ; split >= 1; split /= 2 {
			s.Split = split
			key := ragKey{kind: "gemmvolta", q: m.q, k: m.k, rows: m.rows, ntok: ntok,
				tile: [5]int{tl.MT, tl.NT, tl.WM, tl.WN, tl.KB + 1000*b2i(tl.Single)}, split: split, bias: s.Bias, r: f16k}
			k, ok := g.ragK[key]
			if !ok {
				ker, err := kernels.GemmVolta(s, tl)
				if err == nil {
					k, err = g.dev.Compile(ker)
				}
				if err != nil {
					k = nil
				}
				// A tile the compiler could not fit in the register file is
				// passed over for the next that divides the shape: on sm_70
				// q6_k's m4n4 k2 tile hits 255 registers and spills.
				if r, ok := backend.KernelResourcesOf(k); k != nil && ok && r.SpillBytes > 0 && !g.SpillingTiles {
					k.Close()
					k = nil
					g.SpillRefused++
				}
				if g.ragK == nil {
					g.ragK = map[ragKey]backend.Kernel{}
				}
				g.ragK[key] = k
			}
			if c = k; c != nil {
				break
			}
		}
		if c == nil {
			continue
		}
		akey := ragKey{kind: "actf16t", q: m.q, ntok: ntok, k: m.k}
		a, ok := g.ragK[akey]
		if !ok {
			ker, err := kernels.ActF16T(m.q, ntok, m.k)
			if err == nil {
				a, err = g.dev.Compile(ker)
			}
			if err != nil {
				a = nil
			}
			g.ragK[akey] = a
		}
		if a == nil || !g.sizeF16(ntok*m.k*2) {
			return mv{}, false
		}
		out := mv{kern: c, act: a, actN: ntok * m.k / sub, rows: m.rows, split: split, bias: m.bias,
			q: m.q, k: m.k, threads: kernels.GemmVoltaGroups(s, tl) * tl.Threads(), width: tl.Threads(), redN: m.rows * ntok}
		if split > 1 {
			if out.red = g.reduceKernel(m.rows*ntok, split); out.red == nil ||
				!g.sizePart(m.rows*ntok*split) {
				return mv{}, false
			}
		}
		if f16k != 0 {
			g.GemmF16++
			return out, true
		}
		g.VoltaMV++
		g.VoltaGemm++
		return out, true
	}
	return mv{}, false
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

// prepRagLinear builds bs's linear-block kernels for a step of n rows, once per
// size and outside any session. Callers hold g.mu.
//
// The rows are the step's runs (kernels' run forms): each sequence's rows at
// consecutive positions chain through its one state, so a prompt chunk riding
// beside decoding rows steps once a row, as a prefill would. Sized to the real
// rows, not the padded width: the padding never reaches a state, and the next
// states (recNext, in the shared form) are one a run.
func (g *devTier) prepRagLinear(bs *blockScratch, n int) bool {
	r := bs.rec
	if r.Conv == 0 || bs.ragLin[n] != nil {
		return true
	}
	fail := func(err error) bool { g.LastErr = "rows: " + err.Error(); return false }
	// The scan's lanes are the device's promise: a lane group a state row
	// where the subgroup is guaranteed, and where it is not -- llvmpipe, a
	// Vulkan driver that will not pin the width -- one thread a row, the whole
	// row in its registers and both dots summed serially. That is the same
	// generator at one lane, with no shuffle in it, so it lowers on every
	// backend. ScalarLinearRows takes it on a device that has the subgroup.
	lanes := bs.qkLanes
	if g.ScalarLinearRows {
		lanes = 1
	}
	// A short convolution (LFM2) has no rule to scan: its run is the
	// convolution and the shift alone.
	var d kernels.DeltaScan
	rk := &ragLinear{}
	if !r.ShortConv {
		d = deltaScanOf(r, n, lanes)
		d.Runs = true
		rk.perRun, rk.lanes = d.Threads()/n, d.ScanLanes()
	}
	eps := bs.p.RMSEps
	if r.ChanDecay {
		eps = 1e-6
	}
	var err error
	for _, k := range []struct {
		dst *backend.Kernel
		mk  func() (*ir.Kernel, error)
	}{
		{&rk.delta, func() (*ir.Kernel, error) {
			if r.ShortConv {
				return nil, nil
			}
			if r.SSD {
				return kernels.GatedDeltaFused(d, kernels.DeltaFuse{Chans: r.Chans, SSD: true})
			}
			if r.Mamba1 {
				return kernels.GatedDeltaFused(d, kernels.DeltaFuse{Mamba1: true})
			}
			return kernels.GatedDeltaFused(d, kernels.DeltaFuse{Chans: r.Chans,
				Eps: float32(eps / float64(r.KDim)), Scale: float32(1 / math.Sqrt(float64(r.KDim))),
				BARep: r.VHeads / r.KHeads, Bound: r.DecayBound})
		}},
		{&rk.conv, func() (*ir.Kernel, error) { return kernels.Conv1dRowsRuns(r.Conv, r.Chans, n) }},
		{&rk.shift, func() (*ir.Kernel, error) { return kernels.Conv1dShiftRuns(r.Conv, r.Chans, n) }},
	} {
		var kk *ir.Kernel
		if kk, err = k.mk(); err == nil && kk != nil {
			*k.dst, err = g.dev.Compile(kk)
		}
		if err != nil {
			for _, c := range []backend.Kernel{rk.conv, rk.shift, rk.delta} {
				if c != nil {
					closeKern(c)
				}
			}
			return fail(err)
		}
	}
	if bs.ragLin == nil {
		bs.ragLin = map[int]*ragLinear{}
	}
	bs.ragLin[n] = rk
	return true
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
	var ls []*layer
	for _, l := range g.layers {
		if l != nil && l.linear && l.recOf(sid) != nil {
			ls = append(ls, l)
		}
	}
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
