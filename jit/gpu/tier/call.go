package tier

import (
	"cmp"
	"encoding/binary"
	"fmt"
	"slices"
	"time"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// Layers is layersFor for the zero session: a caller that never attached.
func (g *devTier) Layers(lo, hi, pos, n int, x, cs, csSWA []float32, head *nn.Head) bool {
	return g.layersFor(0, lo, hi, pos, n, x, cs, csSWA, head)
}

// layersFor runs blocks [lo, hi) at position pos as one submission.
//
// The read at the end is the only synchronise: consecutive launches on one
// stream are already ordered, so the whole prefix issues without the device
// draining. Those launches are captured once and replayed (a CUDA graph pays the
// per-launch cost once); see graphKey for why a decode token's sequence is
// stable enough to capture, and cuda/graph.go for the mechanism.
//
// It runs in session sid's view of the device, holding a lane for the call
// (devsess.go); a call reached from inside another keeps the lane it holds.
func (g *devTier) layersFor(sid uint64, lo, hi, pos, n int, x, cs, csSWA []float32, head *nn.Head) bool {
	v := g.as(sid)
	if v == nil {
		return false
	}
	defer v.done()
	ok := v.layersCall(sid, lo, hi, pos, n, x, cs, csSWA, head)
	v.dropEmbed()
	return ok
}

// layersCall is Layers. The wrapper exists because an EmbedRows promise lives
// for exactly one Layers call: a call that refuses before submitting leaves the
// caller to fill x, and a surviving promise would be gathered into a later
// token's x.
func (g *devTier) layersCall(sid uint64, lo, hi, pos, n int, x, cs, csSWA []float32, head *nn.Head) bool {
	// Fault injection, present only in a `jitllmfault` build; see tier/fault.go.
	// In a release build injectedFail is `return false` and inlines away.
	if g.injectedFail() {
		return false
	}
	g.allocFault()
	// A tower's range runs in lane0, whose tower set is the device's own
	// (visrows.go): a call holding a clone borrows lane0 for it.
	g.mu.Lock()
	if l := g.layers[lo]; lo < hi && l != nil && l.nonCausal && g.lane != g.lane0 {
		v0 := g.borrowLane0()
		g.mu.Unlock()
		ok := v0.layersCall(sid, lo, hi, pos, n, x, cs, csSWA, head)
		g.mu.Lock()
		g.returnLane0(v0)
		g.mu.Unlock()
		return ok
	}
	// Reset per call, because the caller reads it immediately after a false to
	// decide whether a host restart is sound.
	g.recSteps = 0
	// Gemma 4: the range's geometry's scratch set, and the caller's set again
	// on the way out (gemma4.go). GPU.runsInto hands one geometry per call.
	prev := g.geoCur
	if g.geoUsed {
		g.useGeom(g.geomFor(lo, hi))
	}
	g.mu.Unlock()
	defer g.restoreGeom(prev)
	// KV growth happens before anything is submitted, because it drops the
	// captured graph and dropGraph may not be called inside a backend Session
	// (a deadlock on CUDA). A non-causal range is decided first: its n is the
	// run's rows, not a position, and must not grow the causal capacity. It is
	// one whole run in its own geometry's scratch set, which the switch above
	// made current.
	// An empty range is the text head alone (PrepHead's call): its lo is the
	// block past the text model's last, which is a tower's first when a tower
	// follows, and says nothing about the call.
	if l := g.layers[lo]; lo < hi && l != nil && l.nonCausal {
		g.mu.Lock()
		bs := g.bs
		// The windows staged in the set are whichever session's picture
		// came last: this session's go in again.
		wok := g.setWindowsLocked(g.winRuns, g.winPerRow) && g.setSeqsLocked(g.seqRuns)
		g.mu.Unlock()
		if !wok {
			return false
		}
		if bs == nil || n > bs.rows || head != nil {
			g.LastErr = fmt.Sprintf("non-causal n=%d scratch=%v head=%v", n, bs != nil, head != nil)
			return false
		}
		return g.submit(sid, bs, lo, hi, pos, x, cs, csSWA, nil)
	}
	if n == 1 {
		g.retuneDecode()
	}
	if !g.ensureKVCap(pos + n) {
		return false
	}
	// Paged history takes the chunk's pages now, before any block is paged in
	// for the submission.
	g.pgRows = g.pagedRows(sid, g.pgRows, n, n, pos, nil)
	if !g.pagedAppendRows(sid, g.pgRows) {
		return false
	}
	// A prefill chunk, padded to a fixed width: every kernel bakes its row
	// count, so a ragged tail runs the full width. The surplus rows write K and V
	// at positions not yet reached, which decode overwrites, and no earlier row
	// attends to them because the scores mask past each row's causal count.
	if n != 1 {
		// Config.NoBatch makes the caller go one device call per prompt token:
		// the way to tell a batching bug from an f32 tie, and the A/B control.
		if g.NoBatch {
			g.LastErr = "batched prefill disabled"
			return false
		}
		bw := batchWidth
		if n > bw {
			g.LastErr = fmt.Sprintf("batch n=%d width=%d head=%v", n, bw, head != nil)
			return false
		}
		// The width is the chunk's rounded up to the grain, so a short tail does
		// not run the full grid. A scratch per width costs VRAM, so the narrow one
		// is used only if it fits; otherwise the wide one pads.
		g.mu.Lock()
		grain := g.promptGrain() // f16GemmK probes once, under g.mu
		g.mu.Unlock()
		w := (n + grain - 1) / grain * grain
		if w > bw {
			return false
		}
		if g.NoTail {
			w = bw
		}
		g.mu.Lock()
		// The chunk's own width, or the width reserved at placement when the
		// narrow one does not fit (scratch.go); with no reservation, the full
		// width.
		// The chunk's own grain width first: batchFor rounds a width it has
		// not built to a power of two, which is right for a step loop whose
		// row count moves every step but ran a 415-token prompt over 512
		// rows, a quarter of its arithmetic on padding. prepBatch refuses a
		// width the budget has no room for, and the chunk then pads as before.
		//
		// Each exact width is budgeted beside the reservation. A few are
		// kept, the least recently used given back when the budget has no
		// room for the next or the lane holds maxExactWidths (keepExact);
		// a dropped width's kernels stay in the caches.
		ww := 0
		if !g.NoExactPrompt && w != bw {
			if ww = g.keepExact(w); ww != 0 {
				g.ExactPrompts++
			}
		}
		if ww == 0 {
			ww = g.batchFor(w)
		}
		if ww == 0 && w != bw && g.prepBatch(bw) {
			ww = bw
		}
		ok := ww > 0
		if ok {
			w = ww
		}
		bbs, dbs, sub := g.bbs[w], g.bs, max(g.promptW, nn.MaxPrefillChunk)
		if g.roomFaulted(roomNoscratch) || g.roomFaulted(roomNoscratchSplit) {
			ok, bbs = false, nil
		}
		if g.roomFaulted(roomNoscratchSplit) {
			sub = 32 // the width a starved device reserved: a chunk goes in 32-row parts
		}
		// A head over every row (a speculative verification) runs only in
		// the one-submission path, with the per-row head built for it: every
		// fallback below splits the chunk and projects the last row alone.
		if ok && bbs != nil && head != nil && head.RowLogits {
			if dbs == nil || dbs.head == nil || !g.prepRowsHead(bbs, head, n) {
				ok = false
			} else if len(head.Tokens) < n {
				head.Tokens = make([]int32, n)
			}
		}
		// The fold copies the last row into the decode scratch and projects it
		// there (layersOnce), which a resident float head cannot take. Only a
		// resident head: the split runs the chunk before the head, and a head
		// that then failed would send a hybrid's advanced rows round again.
		// Never for a head over every row, which the per-row head above serves
		// and the split, projecting the last row alone, cannot.
		noFold := dbs != nil && dbs.head != nil && dbs.head.ok && bbs != nil &&
			!(head != nil && head.RowLogits) &&
			(kernels.IsFloat(dbs.head.t) || dbs.p.NEmbd != bbs.p.NEmbd)
		g.mu.Unlock()
		if head != nil && head.RowLogits && (!ok || bbs == nil) {
			if g.LastErr == "" {
				g.LastErr = fmt.Sprintf("a %d-row chunk wanting every row's logits has no batched scratch", n)
			}
			return false
		}
		// Every fallback below cuts x by rows. A stream-major residual (AltUp's,
		// DeepSeek V4's, Kimi-K3's) holds a row's streams n rows apart, so its
		// cut is rows [a, b) of every stream (cutRows), not a slice.
		// A head rides only the one-submission path. Every fallback splits the
		// chunk, so there the head runs on the last row as its own submission;
		// so does a head the fold cannot take, rather than the whole chunk
		// going a row at a time.
		if head != nil && (!ok || bbs == nil || noFold) {
			// A head-only call (lo == hi: the blocks ran in a submission of
			// their own, GPU.layersCall's tail) has no blocks to run first;
			// asking for the empty range without the head is a refusal, which
			// sent a chunk whose blocks had all run round again a row at a
			// time (MiniMax-M3's selecting blocks are a geometry of their own,
			// so its float head always came here).
			if dbs == nil || lo < hi && !g.layersFor(sid, lo, hi, pos, n, x, cs, csSWA, nil) {
				return false
			}
			e, nr, ns := dbs.p.NEmbd, dbs.p.RopeW(), dbs.p.Streams()
			var rcs, rswa []float32
			if nr > 0 {
				rcs = cs[(n-1)*nr : n*nr]
				if len(csSWA) >= n*nr {
					rswa = csSWA[(n-1)*nr : n*nr]
				}
			}
			last := g.cutRows(x, ns, e, n, n-1, 1)
			defer g.uncutRows(x, last, ns, e, n, n-1, 1)
			was := g.cutInputs(sid, n, n-1, 1)
			defer g.restoreInputs(sid, was)
			// The head's call resets recSteps (layersCall), and the chunk's
			// blocks have advanced by then: a head that fails must still
			// report them, or the caller restarts a hybrid from the embeddings
			// and applies its linear blocks twice.
			g.mu.Lock()
			adv := g.recSteps
			g.mu.Unlock()
			if !g.layersFor(sid, hi, hi, pos+n-1, 1, last, rcs, rswa, head) {
				g.mu.Lock()
				g.recSteps += adv
				g.mu.Unlock()
				return false
			}
			return true
		}
		if (!ok || bbs == nil) && n > sub && dbs != nil {
			// A device that cannot hold the wide scratch takes the chunk in narrower
			// submissions -- the width it reserved, when it reserved one --
			// consecutive sub-chunks being what the caller's own chunk loop would
			// have issued.
			sw := sub
			nr, ns := dbs.p.RopeW(), dbs.p.Streams()
			g.mu.Lock()
			g.PromptSplits++
			g.mu.Unlock()
			for off := 0; off < n; off += sw {
				m := min(sw, n-off)
				var rcs, rswa []float32
				if nr > 0 {
					rcs = cs[off*nr : (off+m)*nr]
					if len(csSWA) >= (off+m)*nr {
						rswa = csSWA[off*nr : (off+m)*nr]
					}
				}
				e := dbs.p.NEmbd
				part := g.cutRows(x, ns, e, n, off, m)
				was := g.cutInputs(sid, n, off, m)
				ran := g.layersFor(sid, lo, hi, pos+off, m, part, rcs, rswa, nil)
				g.restoreInputs(sid, was)
				g.uncutRows(x, part, ns, e, n, off, m)
				if !ran {
					return false
				}
			}
			return true
		}
		if (!ok || bbs == nil) && dbs == nil {
			return false
		}
		if (!ok || bbs == nil) && bidirCheck(g.bidir, pos, 1, &dbs.p, true) != "" && g.attendsIn(lo, hi) {
			// One row at a time cuts a bidirectional run the chunk holds,
			// which attention cannot read row by row: refused before any
			// block runs (nothing has advanced), so the chunk restarts on
			// the host whole.
			g.LastErr = "a bidirectional chunk this device has no batched scratch for"
			return false
		}
		if !ok || bbs == nil {
			g.mu.Lock()
			g.PromptSplits++
			g.mu.Unlock()
			// A batch this device cannot prepare is not a device failure. prepBatch
			// refuses before submitting anything, so x is untouched and the rows go
			// one at a time. Returning false would restart the chunk on the host,
			// which on a hybrid re-runs linear blocks whose state already advanced.
			ns := dbs.p.Streams()
			for i := 0; i < n; i++ {
				row := g.cutRows(x, ns, dbs.p.NEmbd, n, i, 1)
				var rcs, rswa []float32
				if nr := dbs.p.RopeW(); nr > 0 {
					rcs = cs[i*nr : (i+1)*nr]
					if len(csSWA) >= (i+1)*nr {
						rswa = csSWA[i*nr : (i+1)*nr]
					}
				}
				// Each row is a token of the recurrence, and each linear block's
				// halves flip with it, so row 1's recording is keyed apart from
				// row 0's (rangeParity).
				was := g.cutInputs(sid, n, i, 1)
				ran := g.submit(sid, dbs, lo, hi, pos+i, row, rcs, rswa, nil)
				g.restoreInputs(sid, was)
				g.uncutRows(x, row, ns, dbs.p.NEmbd, n, i, 1)
				if !ran {
					return false
				}
			}
			return true
		}
		return g.submit(sid, bbs, lo, hi, pos, x, cs, csSWA, head)
	}
	if g.PerLayerSubmit && (hi-lo > 1 || head != nil) {
		// The other arm of the submission comparison: one Session and one Read
		// per block. The head becomes one more (empty-range) submission. submit
		// honours PerLayerSubmit itself, since it is also what pages a block in.
		if !g.submit(sid, g.bs, lo, hi, pos, x, cs, csSWA, nil) {
			return false
		}
		if head != nil {
			return g.layersOnce(sid, g.bs, hi, hi, pos, x, cs, csSWA, head)
		}
		return true
	}
	return g.submit(sid, g.bs, lo, hi, pos, x, cs, csSWA, head)
}

// parityMask is the halves a submission's linear blocks read, one bit per
// linear block of its range in order: what a recording of that range is keyed
// by, since a recording names one half as source and the other as
// destination. Four words cover 256 linear blocks; a range with more is never
// recorded (rangeParity).
type parityMask [4]uint64

// rangeParity is the halves the linear blocks in [lo, hi) read next -- the
// running step's (stepPar), so a step across sessions keys on the half its
// rows were aligned to. It is the range's own rather than the device's: a
// device holding two runs of one model (an explicit placement) submits twice a
// token, and a device-wide flip per submission would come back to the same
// parity every token while each block's halves alternate -- replaying a
// recording that reads the stale half. Every block, not only the first: a
// block placed later, or a session that stepped apart from the others, can
// read a half its neighbours do not. ok is false for a range too long to key.
// Callers hold g.mu.
func (g *devTier) rangeParity(sid uint64, lo, hi int) (m parityMask, ok bool) {
	bit := 0
	for li := lo; li < hi; li++ {
		l := g.layers[li]
		if l == nil || !l.linear {
			continue
		}
		if bit == 64*len(m) {
			return m, false
		}
		if par, has := g.stepPar(sid, l); has && par == 1 {
			m[bit/64] |= 1 << (bit % 64)
		}
		bit++
	}
	return m, true
}

// replayRecurrent is emitLinear's bookkeeping for a submission that replayed a
// recording instead of emitting. The flip lives in Go, not in the graph, so a
// replay must swap each linear block's halves by hand; otherwise the same
// recording is chosen every token and the recurrence freezes. A token that
// records ran emit during the capture and has already flipped.
func (g *devTier) replayRecurrent(sid uint64, lo, hi int) {
	for li := lo; li < hi; li++ {
		l := g.layers[li]
		if l == nil || !l.linear {
			continue
		}
		if _, has := g.stepPar(sid, l); has {
			g.stepFlip(sid, l)
			g.recSteps++
			if g.rag != nil && g.rag.sid != nil {
				g.SessionLinear++
			}
		}
	}
}

// hasRecurrent reports whether any block in [lo, hi) keeps a recurrent state.
func (g *devTier) hasRecurrent(lo, hi int) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	for li := lo; li < hi; li++ {
		if l := g.layers[li]; l != nil && l.rec != nil {
			return true
		}
	}
	return false
}

// graphKey is everything about a token that decides which launch sequence it
// issues. Two tokens with the same key issue byte-identical submissions, so one
// captured graph serves both.
//
// It is short because the sequence barely varies: every buffer a decode launch
// names is allocated once and never moves, and every per-token scalar (the
// position, the KV offset, the rotary table) reaches the device inside a
// buffer. The only grid that changes is the attention scores', over pos+1
// positions.
type graphKey struct {
	lo, hi   int
	scoreN   int // the rounded position count the scores grid was sized for
	rows     int // the batch width; 1 for decode
	head     bool
	tblSplit bool
	// The softmax arm: the two kernels' grids differ, and verify -ab softmax
	// flips it every round.
	scalarSM bool
	flash    bool
	// The packed V cache changes copyV's grid and AttnAcc's kernel entirely, so
	// a recording made under one arm cannot replay under the other.
	kvF16 bool
	// Whether the rotary table is built on the device, which adds launches at
	// the head of the sequence; verify -ab ropetab flips it every round.
	ropeDev bool
	// argmax says the head ends in the device argmax (nn.Head.ArgmaxOnly):
	// one more launch, so a recording without it cannot serve a call with it.
	argmax bool
	// sampleK is the device sampler's k (nn.Head.SampleK), 0 for none: its
	// launches follow the head, so a recording without them cannot serve a
	// call with them, nor one of another k.
	sampleK int
	// samplePen says the sampler's penalty launch runs: with no history it
	// is skipped and the first top-k pass reads the logits.
	samplePen bool
	// pick says the head ends in the label pick's gather (nn.Head.Pick).
	pick bool
	// rag is a LayersRows step's sequence count, 0 for any other call: ragged
	// attention and a per-row head, and on a hybrid the per-count linear
	// kernels (ragLin), which a recording names by address.
	rag int
	// ragRuns is how many runs a hybrid's ragged step groups its rows into:
	// the linear kernels' grids are sized by it (ragLinear.perRun).
	ragRuns int
	// The recurrent parity: a linear block's state deliberately moves between
	// two halves, so a recording names one as source and the other as
	// destination. Replayed at the other parity it would read the stale half
	// and the state would never advance. Two recordings alternate instead,
	// as a partial seam alternates two lo/hi keys. One bit per linear block
	// (rangeParity).
	recParity parityMask
	// pagedVar is the paged attention's variant: the decode plan a recording
	// launched, which the keys' count chose and nCap may not.
	pagedVar *pagedVariant
	// ragHead is how many of a ragged step's rows the head projects
	// (nn.Head.LogitRows): it picks the head's matvec, which a recording
	// names, and a step of the same rows wanting fewer logits differs only
	// here.
	ragHead int
	// ragMap is the ragged call's run-table width (raggedprefill.go): it
	// picks the run-table launch, which a recording names, and 0 is the
	// decode plan for every row.
	ragMap int
	// hnorm is which norm the head reads (headNormFor): a recording names its
	// buffer by address.
	hnorm int
	// persist says the recording holds composed persistent kernels
	// (Config.Persist), which a per-op recording does not.
	persist bool
	// persistDeps says those kernels are ordered by dependency counters
	// (Config.PersistDeps) rather than grid barriers.
	persistDeps bool
	// gateUp says the dense FFNs ran gate and up as one launch
	// (Config.GateUpFuse).
	gateUp bool
	// gateUp2 says the dense FFNs ran gate and up as one Gate2 launch
	// (Config.GateUp2).
	gateUp2 bool
	// gateUp2Force says Gate2 ran whatever the tuner chose
	// (Config.GateUp2Force).
	gateUp2Force bool
	// noAddNorm says a prompt chunk's block-end adds ran alone
	// (Config.NoAddNormFold).
	noAddNorm bool
	// sid is the session the recording was made for: it bakes that session's
	// page descriptors and recurrent seats, so another session never replays
	// it. Keying on it, rather than dropping every recording when the current
	// session changes, is what lets two sessions that take turns -- a
	// speculator's trunk and draft, two requests stepped one after another --
	// each keep replaying instead of recording again at every turn.
	sid uint64
	// noFuse is Config.NoFuse: the fused kernels and their parts are two
	// launch sequences, and verify -ab fuse flips it every round.
	noFuse bool
	// noActFuse is Config.NoActFuse, verify -ab actfuse's switch.
	noActFuse bool
	// ffnForm is Config.FFNForm, verify -ab ffn's switch.
	ffnForm FFNForm
}

// canRecord reports whether this device's sessions can capture a launch
// sequence, cached because it costs an ownership hop to ask.
func (g *devTier) canRecord() bool {
	if g.recCap == 0 {
		g.recCap = -1
		g.dev.Session(func(s backend.Session) {
			if _, ok := s.(backend.Recorder); ok {
				g.recCap = 1
			}
		})
	}
	return g.recCap == 1
}

// scoreGrain rounds the score grid's position count up to a multiple of itself.
//
// The rounding keeps the sequence stable for a run of tokens, re-capturing when
// it rolls over, as llama.cpp pads its KV length. It costs a little arithmetic
// and no correctness: AttnScores clamps its thread index with n read from a
// buffer, so surplus groups recompute and store the same last score. It rounds
// on both arms of the graph A/B so they do the same device work.
const scoreGrain = 128

func (g *devTier) layersOnce(sid uint64, bs *blockScratch, lo, hi, pos int, x, cs, csSWA []float32, head *nn.Head) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	// Every wait comes first. The prologue below reads what other sessions
	// change -- this session's pages, whether any are at home, the blocks --
	// into the submission's descriptors, and a wait after it would let one of
	// them change under the submission (another session's page-in or
	// eviction). Alone on the device: wait for every submission in flight.
	// Beside the others: wait for a quiesce in progress to have its turn. From
	// here to the submission nothing lets g.mu go (prologueHeld).
	// The question is asked again after every wait: another session's
	// prologue may have sent this session's history home meanwhile.
	var excl bool
	for {
		if excl = g.exclusiveRange(lo, hi, g.rag, bs) || g.evictedFor(sid, g.rag); excl {
			// Running alone is right whatever changes while it waits.
			if err := g.quiesce(); err != nil {
				g.LastErr = err.Error()
				return false
			}
			break
		}
		if g.quiet == 0 {
			break
		}
		g.idle.Wait()
	}
	g.prologueHeld = true
	defer func() { g.prologueHeld = false }()
	// An empty range is legal only with a head: the projection under a partial
	// seam. Every refusal records why, since the State's answer to false is a
	// silent restart on the host.
	refuse := func(f string, a ...any) bool {
		g.LastErr = fmt.Sprintf(f, a...)
		return false
	}
	switch {
	case bs == nil:
		return refuse("no scratch for this submission")
	case lo > hi || (lo == hi && head == nil):
		return refuse("an empty block range [%d,%d) with no head", lo, hi)
	case pos >= bs.p.MaxSeq:
		return refuse("position %d is past this device's capacity of %d (asked for %d)",
			pos, bs.p.MaxSeq, g.maxSeqAsked)
	case head != nil && lo < hi && bs.p.Streams() > 1:
		// AltUp's head reads the streams' mean and DeepSeek V4's their
		// collapse, which the host takes (altup.go, ds4.go): the blocks and
		// the head go as two calls.
		return refuse("a head folded into %d blocks of %d streams", hi-lo, bs.p.Streams())
	}
	// The head reads one residual vector and writes one logit row. A ragged
	// step's head is the decode scratch's; the batched one has none.
	rag := g.rag
	hb := bs
	// A chunk projecting every row (nn.Head.RowLogits) runs the ragged step's
	// per-row head, which reads the resident head through the decode scratch.
	if rag != nil || bs.rows > 1 && head != nil && head.RowLogits {
		hb = g.bs
	}
	// A batched chunk may carry the head, which then runs on the last row:
	// prefill wants logits for its final position only. The row is copied into
	// the decode scratch on the card, its norm, quantize and projection run
	// there, and only the logits come home -- no second session, no readback of
	// the chunk.
	// A chunk whose head wants every row's logits (nn.Head.RowLogits: a
	// speculative verification) runs the ragged step's per-row head over the
	// chunk instead of folding its last row; prepRowsHead built it.
	allRows := bs.rows > 1 && head != nil && rag == nil && head.RowLogits
	fold := bs.rows > 1 && head != nil && rag == nil && !allRows
	if allRows && (bs.ragHead.kern == nil || bs.ragLogits == nil) {
		return refuse("a %d-row chunk asked for every row's logits with no per-row head built", bs.rows)
	}
	// The per-row head caps where decode's does: a step's rows end in an
	// argmax and a readback of their own, and uncapped they would be a
	// different function from the session's decode.
	if (rag != nil || allRows) && head != nil && g.bs != nil && g.bs.headCap != nil && bs.ragCapK == nil {
		return refuse("a %d-row step of a model with a final logit softcap, with no per-row softcap built", bs.rows)
	}
	// The head's norm: the one it was built with, or a prediction block's own
	// (headnorm.go). A norm the device was never given is a refusal, never the
	// resident one in its place.
	var hNorm backend.Buf
	hNormID := 0
	if head != nil && g.bs != nil {
		var ok bool
		if hNorm, hNormID, ok = g.headNormFor(head); !ok {
			return refuse("the head brings a norm this device was not given (PrepHead)")
		}
	}
	if fold {
		hb = g.bs
		switch {
		case hb == nil || hb.head == nil || !hb.head.ok:
			return refuse("the head is not resident on this device")
		case kernels.IsFloat(hb.head.t) || hb.p.NEmbd != bs.p.NEmbd:
			return refuse("a batched submission of %d rows with a %v head", bs.rows, hb.head.t)
		}
		if e := g.spanKernel(hb); e != nil {
			return refuse("the folded head's row copy: %v", e)
		}
	}
	if head != nil && (hb == nil || hb.head == nil || !hb.head.ok) {
		return refuse("the head is not resident on this device")
	}
	if rag != nil && len(rag.pos) != len(x)/bs.p.ResidW() {
		return refuse("%d ragged rows for %d rows of x", len(rag.pos), len(x)/bs.p.ResidW())
	}
	for li := lo; li < hi; li++ {
		l := g.layers[li]
		if l == nil || !l.ok {
			return refuse("block %d is not on this device", li)
		}
		// A paged-out block's weight buffers are freed, and launching against one
		// reads whatever now lives there without faulting. submit() pages the
		// range in first; this asserts it, and is why nothing else may call
		// layersOnce.
		if l.pg != nil && !l.pg.in {
			g.LastErr = fmt.Sprintf("block %d is paged out", li)
			return false
		}
		// A latent block without a batched scratch refuses the submission, not
		// the block: Layers then runs the chunk's rows one at a time on this
		// device, keeping decode here. The batched arm is emitMLARows.
		if l.mla && bs.rows > 1 && bs.mlab == nil {
			g.LastErr = fmt.Sprintf("block %d: multi-head latent attention has no batched "+
				"scratch here, so a %d-row chunk goes one row at a time", li, bs.rows)
			return false
		}
		// A ragged step's rows are different sequences, or different positions
		// of one, each reading its own history through its descriptor: only the
		// paged latent cache has one (the contiguous cache is one sequence's).
		if l.mla && rag != nil && bs.pkv == nil {
			return refuse("block %d: a ragged step over a latent block needs the paged cache", li)
		}
		// A windowed vision block over the whole image is fluent and wrong, so
		// it runs only under windows that cover exactly this image's rows.
		if l.windowed && (bs.winMask == nil || bs.vwinRows != len(x)/bs.p.NEmbd) {
			return refuse("block %d attends inside windows and the device holds windows for %d "+
				"rows, not %d (SetKeyRuns)", li, bs.vwinRows, len(x)/bs.p.NEmbd)
		}
		// Packed sequences over other rows than the call's would let a row
		// read another sequence's keys.
		if l.nonCausal && bs.vsegRows != 0 && bs.vsegRows != len(x)/bs.p.NEmbd {
			return refuse("block %d: the device holds packed sequences over %d rows, not %d (SetSeqRuns)",
				li, bs.vsegRows, len(x)/bs.p.NEmbd)
		}
	}
	g.Submits++
	p := &bs.p
	// The cache row; see initScratch.
	kvDim := p.KVRow()
	qdim := p.NHead * p.HeadDim

	// Which softmax this call runs, resolved once rather than per block.
	sm, smG, smW := bs.softmax, bs.smGroups, bs.smWidth
	if g.ScalarSoftmax && bs.softmaxAlt != nil {
		sm, smG, smW = bs.softmaxAlt, bs.smAltGroups, bs.smAltWidth
	}

	R := bs.rows
	// The caller's row count comes from the slice and the grid does not: one
	// scratch serves one width, so a ragged chunk runs the full width with the
	// surplus rows zeroed (harmless; see Layers).
	nrow := len(x) / p.ResidW()
	if nrow < 1 || nrow > R || pos+nrow > p.MaxSeq {
		return refuse("%d rows at position %d on a scratch of %d rows and capacity %d",
			nrow, pos, R, p.MaxSeq)
	}
	hx := bs.hx[:R*p.ResidW()]
	// A promised embedding is gathered into the scratch, not uploaded (see
	// EmbedRows); the gather writes all R rows of bs.x, padding as zeros. Its
	// last workgroup must not run past R rows; otherwise the promise is made real
	// on the host.
	pe := g.takeEmbedLocked(x)
	var gatherN int
	if pe != nil {
		hr := g.bs.head
		gatherN = kernels.GetRowsThreads(hr.t, hr.k, R)
		if !g.embedKernel(g.bs) || hr.k != p.NEmbd || len(pe.ids) > R || gatherN%128 != 0 {
			g.materializeLocked(pe)
			pe = nil
		}
	}
	if pe == nil {
		for i := range hx {
			hx[i] = 0
		}
		// AltUp's streams are stream-major over the caller's nrow rows and
		// over the scratch's R (altup.go); one stream is a prefix either way.
		streamRows(hx, x, p, R, nrow)
	}
	// pN carries a header at R > 1: element 0 is the uniform score width every
	// grid is sized for, elements 1..R each row's own causal key count. At R == 1
	// it is the single count.
	n := uint32(pos + R)
	if p.NonCausal && g.ncPhase == 2 {
		// A row-chunked block's second phase reads every key the first
		// phase wrote, whatever chunk its queries are.
		n = uint32(g.ncKeys)
	}
	if int(n) > p.MaxSeq {
		// The score row is MaxSeq wide, so the uniform width cannot exceed it.
		// Clamping is safe: a real row needs at most pos+nrow of it.
		n = uint32(p.MaxSeq)
	}
	// The staging bytes are the devTier's, reused: they are written to the
	// device before the call returns and read by nothing after.
	pn := append(g.subBytes.pn[:0], u32one(n)...)
	kst := g.kStride(p)
	koffs := g.subBytes.koffs[:0]
	kposs := g.subBytes.kposs[:0]
	// A refusal returns before the submission below keeps these; a call that
	// refuses every time would otherwise regrow all of them every time.
	var rposs []byte
	defer func() {
		g.subBytes.pn, g.subBytes.koffs, g.subBytes.kposs = pn, koffs, kposs
		if rposs != nil {
			g.subBytes.rposs = rposs
		}
	}()
	for r := 0; r < R; r++ {
		at := pos + r
		if r >= nrow {
			at = p.MaxSeq // the trash slot; see PrepLayer
		}
		koffs = append(koffs, u32one(uint32(at*kvDim))...)
		kposs = append(kposs, u32one(uint32(at))...)
	}
	if R == 1 {
		n = uint32(pos + 1)
		pn = binary.LittleEndian.AppendUint32(pn[:0], n)
	}
	// Only a backend that can record pays the rounding; elsewhere it would be
	// surplus attention work for nothing.
	nCap := int(n)
	if g.canRecord() && R == 1 {
		nCap = (int(n) + scoreGrain - 1) / scoreGrain * scoreGrain
	}
	// A call with no attention block in it (the head alone, or a hybrid's
	// linear blocks, which scan their rows in order) cuts no run: a run is
	// a key count, and only attention reads one.
	if len(g.bidir) > 0 && !p.NonCausal && g.attendsIn(lo, hi) {
		if why := bidirCheck(g.bidir, pos, nrow, p, g.pagedPlan(p)); why != "" {
			return refuse("rows [%d,%d): %s", pos, pos+nrow, why)
		}
	}
	if R > 1 {
		// The per-row counts go in after the cap, which is what the rounding
		// above would otherwise invalidate -- so a batched call does not round.
		pn = pn[:4]
		binary.LittleEndian.PutUint32(pn, uint32(nCap))
		for r := 0; r < R; r++ {
			// A padded row takes the last real row's count. It must be clamped,
			// because AttnAcc walks each row's own count and an unclamped one runs
			// past the cache (a fault on one model, a neighbour's buffer on
			// another). It must not be smaller than its neighbours', because
			// AttnAccTiled walks the count of its tile's last row, so a short one
			// truncates every real row in the tile.
			n := pos + r + 1
			if r >= nrow {
				n = pos + nrow
			}
			if p.NonCausal {
				// The whole call is one run: the mask is a per-row key count, so
				// every row gets the same count and needs no second score kernel. A
				// padded row still takes nrow, for the tile reason above.
				n = nrow
				if g.ncPhase == 2 {
					n = g.ncKeys
				}
			} else if r < nrow {
				// A row of a bidirectional run counts keys to the run's end
				// (bidir.go); bidirCheck has kept the run inside these rows.
				if e := bidirCount(g.bidir, pos+r); e > 0 && !p.BidirOff {
					n = e
				}
			}
			pn = append(pn, u32one(uint32(n))...)
		}
	}

	// A ragged step's rows are different sequences (LayersRows), each at its
	// own position with its own pages (pagedRows); what is staged here is the
	// longest history, which sizes the recording's key.
	if rag != nil {
		pn, koffs, kposs = pn[:4], koffs[:0], kposs[:0]
		mc := 1
		for r := 0; r < R; r++ {
			cnt := 1
			if r < nrow {
				if rag.slot[r]-rag.pos[r] < 0 {
					return refuse("row %d: slot %d at position %d", r, rag.slot[r], rag.pos[r])
				}
				cnt = rag.pos[r] + 1
			}
			mc = max(mc, cnt)
			koffs = append(koffs, u32one(0)...)
			kposs = append(kposs, u32one(0)...)
			pn = append(pn, u32one(uint32(cnt))...)
		}
		// Rounded as decode's is, so a recording serves scoreGrain steps.
		nCap = mc
		if g.canRecord() {
			nCap = min((mc+scoreGrain-1)/scoreGrain*scoreGrain, p.MaxSeq)
		}
		binary.LittleEndian.PutUint32(pn, uint32(nCap))
	}

	// Paged history: every row's descriptor and the tables, staged with the
	// other per-call writes. The pages were taken before the submission
	// (pagedAppend); here nothing may allocate one, since growing the pool can
	// page weights out from under the range about to run.
	var pagedVar *pagedVariant
	if pk := bs.pkv; pk != nil {
		pk.rows = g.pagedRows(sid, pk.rows, R, nrow, pos, rag)
		if e := g.pagedPrep(sid, bs, pk.rows); e != nil {
			return refuse("paged attention: %v", e)
		}
		pk.st = nil
		if pk.cur != nil {
			g.pagedStage(pk, p, pk.rows)
			// A call over evicted history streams (pagedstream.go), through
			// the staged decode kernels rather than the prefill ones.
			pf := pk.pf != nil && rag == nil && R > 1 && nrow > 0
			if e := g.pagedStreamPrep(sid, bs, pk.rows, pf); e != nil {
				return refuse("paged attention: %v", e)
			}
			// The lightning indexer scores every position through the
			// table, which a streamed pass does not hold.
			if pk.st != nil && pk.idx != nil {
				return refuse("the lightning indexer reads the whole history and part of it is evicted")
			}
			// So do MiniMax-M3's and QSA's block selections (msa.go, qsa.go).
			if pk.st != nil && (pk.msa != nil || pk.qsa != nil) {
				return refuse("the block selection reads the whole history and part of it is evicted")
			}
		}
		pk.ragRows = 0
		if rag != nil && pk.cur != nil {
			g.ragPrep(bs)
		}
		pk.prefill = pk.pf != nil && pk.cur != nil && pk.st == nil && rag == nil && R > 1 && nrow > 0
		if pk.prefill {
			if e := g.prefillPrep(bs, nrow); e != nil {
				return refuse("paged prefill attention: %v", e)
			}
		}
		pagedVar = pk.cur
	}

	// A plan with a local rotary table (two bases, gemma3) needs it for every
	// row the global one has: short, the local layers would rotate by whatever
	// the buffer last held -- a table of other positions -- and stay fluent.
	// The plan decides, not the scratch: a card's scratch outlives the model
	// that sized it, and one base's model must not be held to another's table.
	if p.SWAPeriod > 0 && len(csSWA) < len(cs) {
		return refuse("the local layers' rotary table has %d floats against the global one's %d",
			len(csSWA), len(cs))
	}
	hcs := cs
	if R > 1 {
		hcs = bs.hcs[:R*p.RopeW()]
		for i := range hcs {
			hcs[i] = 0
		}
		copy(hcs, cs)
	}
	// The local table gets the identical treatment, and its own staging buffer:
	// sharing hcs would have the second copy overwrite the first.
	hcsSWA := csSWA
	if R > 1 && csSWA != nil {
		hcsSWA = bs.hcsSWA[:R*p.RopeW()]
		for i := range hcsSWA {
			hcsSWA[i] = 0
		}
		copy(hcsSWA, csSWA)
	}

	// rrows is how many rows the caller filled. The rotary kernel is clamped to
	// it, so the padded rows of a ragged chunk (a short prompt is all ragged)
	// keep the zeros initRopeTable wrote rather than a real rotation. The
	// position bound is checked here, where the kernel is launched. NRot is zero
	// on a vision block, so the division is guarded.
	rrows := 0
	if p.NRot > 0 {
		rrows = R
		if v := len(cs) / p.RopeW(); v < rrows {
			rrows = v
		}
		if bs.csSWA != nil {
			if v := len(csSWA) / p.RopeW(); v < rrows {
				rrows = v
			}
		}
	}
	top := pos + rrows
	if rag != nil {
		top = 0
		for _, v := range rag.pos {
			top = max(top, v+1)
		}
	}
	ropeDev := bs.ropeTable != nil && !g.RopeTableHost && rrows >= 1 &&
		pos >= 0 && top <= kernels.RopeTabMaxPos
	// QSA's marks, each row's pictures (qsa.go).
	if bs.pkv != nil {
		if e := g.qsaStage(bs.pkv.qsa, R); e != nil {
			return refuse("QSA: %v", e)
		}
	}
	// DeepSeek V4's entries rotate at positions of their own, which only the
	// device's table builds; its keys are read through the paged pool.
	if bs.ds4 != nil && lo < hi {
		pk := bs.pkv
		switch {
		case !ropeDev:
			return refuse("DeepSeek V4's entries rotate at positions of their own, which needs the " +
				"device's rotary table")
		case pk == nil || pk.cur == nil:
			return refuse("DeepSeek V4 reads its keys through the paged pool, and this call holds none")
		case pk.st != nil:
			return refuse("DeepSeek V4's window and entries are read through their tables and part " +
				"of the history is evicted")
		}
		if e := g.ds4Stage(bs, lo, hi, pk.rows); e != nil {
			return refuse("%v", e)
		}
	}
	if ropeDev {
		rposs = append(g.subBytes.rposs[:0], u32one(uint32(rrows))...)
		for r := 0; r < R; r++ {
			at := pos + r
			if r >= rrows {
				// Never read (the kernel clamps at rrows), but kept inside the
				// range the decomposition covers.
				at = pos
			} else if rag != nil {
				at = rag.pos[r]
			}
			rposs = append(rposs, u32one(uint32(at))...)
		}
		g.RopeTables += rrows
		if bs.csSWA != nil {
			g.RopeTables += rrows
		}
	} else if bs.cs != nil {
		g.RopeTableUploads += R
	}

	// Retire recordings before the session opens: freeing one is a driver call
	// that cannot nest inside a session. See dropGraph.
	//
	// Retire on nCap, keep on lo/hi. nCap only increases, so a key differing
	// only there is dead once a longer one appears; lo/hi/head is a placement,
	// and a partial seam alternates two of them every token.
	argmax := head != nil && head.ArgmaxOnly && g.argmaxK != nil && R == 1
	sampleK, samplePen := 0, false
	if head != nil && !argmax && head.SampleK > 0 && R == 1 && rag == nil && !allRows && !fold && g.prepSample(head) {
		sampleK, samplePen = head.SampleK, head.SampleArgs[0] > 0
	}
	// The label pick: one row's logits, a decode's or a folded chunk's last.
	pick := head != nil && !argmax && sampleK == 0 && len(head.Pick) > 0 && rag == nil && !allRows &&
		(R == 1 || fold) && g.prepPick(head)
	ragN, ragRuns, ragHead, ragMap := 0, 0, 0, 0
	if bs.pkv != nil {
		// The width, and whether the decode plan runs beside it (ragAlone):
		// two launch sequences a recording names.
		ragMap = bs.pkv.ragKey()
	}
	if rag != nil {
		ragN, ragRuns = len(rag.pos), len(rag.runs)
		if head != nil {
			ragHead = head.WantedRows(ragN)
		}
	}
	// Counted here rather than in emit, which a replayed recording skips.
	if ragHead == 1 {
		if _, ok := g.headOneMV(R); ok {
			g.RagHeadOne++
		}
	}
	par, parOK := g.rangeParity(sid, lo, hi)
	if allRows {
		ragHead = head.WantedRows(nrow)
	}
	// The persistent path: decode only, and not over history streamed from
	// home, whose uploads sit between launches.
	persistOn := R == 1 && rag == nil && !(bs.pkv != nil && bs.pkv.st != nil) && g.ensurePersist()
	key := graphKey{lo, hi, nCap, R, head != nil, g.TableSplit, g.ScalarSoftmax, g.flashOn(), g.KVF16,
		ropeDev, argmax, sampleK, samplePen, pick, ragN, ragRuns, par, pagedVar, ragHead, ragMap, hNormID, persistOn, persistOn && g.PersistDeps, g.GateUpFuse, g.GateUp2, g.GateUp2Force, g.NoAddNormFold, sid, g.NoFuse, g.NoActFuse, g.FFNForm}
	if lo == hi {
		// The head alone reads no history, so the capacity is not part of
		// what it records: keyed by it, prompts on either side of a growth
		// would retire each other's recording on every call.
		key.scoreN = 0
	}
	if _, live := g.recs[key]; !live {
		for k, r := range g.recs {
			older := k
			older.scoreN = key.scoreN
			if older != key {
				continue // a different placement, still live
			}
			g.stale = append(g.stale, r)
			delete(g.recs, k)
		}
	}
	// A backstop, not the policy above, and per session: several sessions
	// taking turns each hold their own few keys, and a session over the bound
	// loses its least recently used.
	if bound := g.graphBound(); len(g.recs) > bound {
		var mine []graphKey
		for k := range g.recs {
			if k.sid == sid {
				mine = append(mine, k)
			}
		}
		if len(mine) > bound {
			slices.SortFunc(mine, func(a, b graphKey) int { return cmp.Compare(g.recUsed[a], g.recUsed[b]) })
			for _, k := range mine[:len(mine)-bound] {
				g.stale = append(g.stale, g.recs[k])
				delete(g.recs, k)
			}
		}
		for k := range g.recUsed {
			if _, live := g.recs[k]; !live {
				delete(g.recUsed, k)
			}
		}
	}
	g.freeStale()

	g.subBytes.pn, g.subBytes.koffs, g.subBytes.kposs = pn, koffs, kposs
	if rposs != nil {
		g.subBytes.rposs = rposs
	}
	// Every block's program for this step, its kernels generated from its
	// ops: here, outside the submission, where compiling is allowed.
	g.ensureFused(lo, hi, R, bs, rag)
	t0 := time.Now()
	g.sub = submitArgs{
		bs: bs, hb: hb, head: head, lo: lo, hi: hi, pos: pos, x: x,
		hx: hx, hcs: hcs, hcsSWA: hcsSWA, pn: pn, koffs: koffs, kposs: kposs, rposs: rposs,
		p: p, rag: rag, pe: pe, key: key, R: R, nrow: nrow, kvDim: kvDim, qdim: qdim,
		kst: kst, nCap: nCap, gatherN: gatherN, rrows: rrows, sm: sm, smG: smG, smW: smW,
		fold: fold, ropeDev: ropeDev, argmax: argmax, sampleK: sampleK, samplePen: samplePen, pick: pick, ragHead: ragHead, parOK: parOK,
		allRows: allRows, hNorm: hNorm,
	}
	if g.subFn == nil {
		g.subFn = g.layersSession
	}
	// Read here, under g.mu: a page-in another session makes changes it.
	g.sub.paging = g.pagesIn < len(g.layers)
	g.sub.persist = persistOn
	g.sessQueue()
	// The prologue streams this call's history only when some of it was at
	// home before the prologue began (evictedFor), which made the call one
	// to run alone; nothing can have gone home since, so this is a check of
	// that reasoning and not a path.
	if !excl && g.exclusiveRange(lo, hi, rag, bs) {
		return refuse("a submission that must run alone was prepared to run beside others")
	}
	g.prologueHeld = false
	if excl {
		// Alone on the device, as every submission was before sessions ran
		// at once: nothing else is in flight (the wait was before the
		// prologue), and nothing starts while g.mu is held across it.
		g.runSub(g.subFn)
	} else {
		// Beside any other session's: g.mu is let go for the submission, so
		// the bookkeeping of the others goes on meanwhile. What the
		// submission builds on the way takes g.mu for the build (subLock),
		// and its counters are its own until it completes.
		t, err := g.beginSub()
		if err != nil {
			g.sub = submitArgs{}
			return refuse("%v", err)
		}
		g.inSub, g.Stats = true, &g.acc
		g.runUnlocked(g.subFn)
		g.inSub, g.Stats = false, &g.tot
		g.tot.add(g.acc)
		g.acc = Stats{}
		g.endSub(t)
		g.freeStale()
	}
	err, direct := g.sub.err, g.sub.direct
	g.sub = submitArgs{}
	if err != nil {
		g.LastErr = err.Error()
		return false
	}
	if head != nil && !direct {
		for i := range head.Logits {
			head.Logits[i] = hb.hOut[i]
		}
	} else if head == nil && !direct {
		streamRows(x, bs.hout, p, nrow, R)
	}
	g.TLayer += time.Since(t0)
	g.Blocks += hi - lo
	// The submission has completed (its readback waited for it), so pages
	// released before it are no longer read by anything in flight, and the
	// rows it wrote are history eviction may send home.
	if pk := bs.pkv; pk != nil && pk.cur != nil && g.kvp != nil {
		g.kvp.wrote(pk.rows)
	}
	if g.kvp != nil {
		for _, l := range g.kvp.layers {
			l.fence()
		}
	}
	return true
}

// submitArgs is one layersOnce submission's state, handed to layersSession
// through the session's devSess rather than captured by a closure: the
// function a Session runs escapes, so a closure over the submission's locals
// was a heap object -- with every variable it shared with its own closures --
// on every token. A session runs one submission at a time on a device, so one
// set a session serves.
type submitArgs struct {
	// paging says a block of the device is paged out, read before the
	// submission: a recording names device addresses a page-in allocates
	// afresh.
	paging bool
	// persist says the submission composes its decode blocks (persist.go).
	persist                 bool
	bs, hb                  *blockScratch
	head                    *nn.Head
	lo, hi, pos             int
	x, hx, hcs, hcsSWA      []float32
	pn, koffs, kposs, rposs []byte
	p                       *nn.LayerPlan
	rag                     *ragStep
	pe                      *embPend
	key                     graphKey
	R, nrow, kvDim, qdim    int
	kst, nCap, gatherN      int
	rrows                   int
	sm                      backend.Kernel
	smG, smW                int
	ragHead                 int
	fold, ropeDev, argmax   bool
	sampleK                 int
	samplePen, pick         bool
	parOK, allRows          bool
	hNorm                   backend.Buf
	// What the session hands back.
	err    error
	direct bool
}
