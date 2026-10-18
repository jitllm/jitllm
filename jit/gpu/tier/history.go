package tier

import (
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
)

// kvPair is one sequence's attention history for one block.
type kvPair struct {
	kc, vc backend.Buf
	// bytes is what this pair charged, so growing it adjusts the ledger by the
	// difference rather than guessing.
	bytes uint64
	// paged marks history kept in the device's pool (pagedattn.go): kc and vc
	// are nil, the pages are the sequence's, and the pool carries the charge.
	paged bool
	// shared marks the one k/v pair every vision block of the device uses
	// (transientkv.go): it is released by reference count, never by one block.
	shared bool
}

// value is the buffer the accumulate reads. Under MLA it is the key buffer: a
// latent cache row is [latent | rotary key] and the value is its leading
// KVLoraRank floats, so storing them apart would write the same floats twice.
// engine/model/kvpage.go's latent flag is the host's half of the same
// aliasing. vc is left nil rather than set to kc so free walks never
// double-free.
func (p *kvPair) value() backend.Buf {
	if p.vc != nil {
		return p.vc
	}
	return p.kc
}

// recPair is one session's recurrent state on one linear block: the gated
// delta rule's per-head matrices and the causal convolution's window, held at
// the session's seat in the block's pool (recpool.go), each in two halves.
// ir.Validate refuses a param that is both loaded and stored (clamped surplus
// threads would apply the update twice), so the state arrives through one
// half, leaves through the other, and the halves swap. Both are constant in
// the context length, so doubling them is cheap. In the shared form (one
// state per slot and the device's recNext, taken when two halves do not fit)
// only the window has halves.
type recPair struct {
	// cur is the half a submission reads; it writes the other and flips.
	cur int
	// steps counts the flips, so a rewind can tell one step from several
	// (RecRewind); mark is the half and count RecMark saw.
	steps     uint64
	mark      int
	markSteps uint64
	marked    bool
}

// ragLinear is one batch size's linear-block kernels (blockScratch.ragLin).
type ragLinear struct {
	conv, shift, delta backend.Kernel
	// ngConv is Qwen3.8-Flash-Next's n-gram convolution over the step's runs
	// (q4e.go), nil on every other block.
	ngConv backend.Kernel
	// perRun is the scan's threads for one run. The kernels are compiled for
	// as many runs as rows and launched for the step's runs alone: a run's
	// state is in registers for all its rows, and the entries past the last
	// run repeat it (kernels' run descriptor), so a one-sequence step of four
	// rows launched for four runs stepped its whole chain four times over.
	perRun int
	// lanes is how many lanes the scan puts on one state row; 1 is its
	// one-thread form (Stats.LinearRowsScalar).
	lanes int
}

// recOf returns this session's recurrent state for the block, or nil.
func (l *layer) recOf(sid uint64) *recPair {
	if l == nil || l.rec == nil {
		return nil
	}
	return l.rec[sid]
}

// permBytes is what admitting a fresh block charges beyond its weights and
// never gives back while it is resident: this session's history (a KV cache,
// or a linear block's recurrent pair) and the norms. prepLayer prices it into
// the room check, so a block is declined -- and the next device asked -- rather
// than admitted into room its own cache does not have. Callers hold g.mu, after
// initKVCap.
func (g *devTier) permBytes(p *nn.LayerPlan, w *nn.LayerWeights) uint64 {
	n := uint64((2*len(w.AttnNorm) + len(w.PostAttnNorm) + len(w.PostFFNNorm)) * 4)
	if r := p.Recurrent; r.Conv > 0 {
		return n + g.sessions()*recHist(p)
	}
	if g.pagedPlan(p) {
		return n + g.pagedLayerBytes(p)
	}
	cp := g.capped(p)
	cache, vcache := g.kvCacheBytes(p, cp.MaxSeq)
	if p.NonCausal {
		// One pair for every non-causal block, charged once (transientkv.go).
		if g.tkv != nil && g.tkv.bytes >= uint64(cache+vcache) {
			return n
		}
		return n + uint64(cache+vcache)
	}
	return n + g.sessions()*uint64(cache+vcache)
}

// recHist is one session's recurrent pair for a linear block.
func recHist(p *nn.LayerPlan) uint64 {
	r := p.Recurrent
	return 2 * uint64(r.StateLen*4+r.ConvState*4)
}

func (g *devTier) sessions() uint64 { return uint64(max(1, g.Sessions)) }

// reserved is what the device holds back for sessions that have not attached
// yet: for every placed linear block, one recurrent pair per seat of
// Config.Sessions that no live session occupies. room() counts it, so neither
// a block nor a page can spend it; a session taking a seat (addSessionRec)
// spends its own share. An attention history holds no seat: every causal
// block's history is paged (initKVCap sets g.paged with the first text plan),
// and a page is taken as a sequence writes its rows, not when it attaches.
// Callers hold g.mu.
func (g *devTier) reserved() uint64 {
	if g.Sessions <= 1 {
		return 0
	}
	var n uint64
	for _, l := range g.layers {
		if l == nil || !l.ok || l.nonCausal || !l.linear {
			continue
		}
		if free := g.Sessions - len(l.rec); free > 0 {
			n += uint64(free) * l.recHist
		}
	}
	return n
}

// addSessionRec gives session sid the two-half recurrent state for a
// linear block: a seat in the device's pools (one slot, unless its batch has
// grown) and a pool for the block if it has none. Callers hold g.mu. The state
// is zeroed because a summary of garbage is a history the model never saw,
// and every token after it is fluent and wrong (RULE 13).
func (g *devTier) addSessionRec(sid uint64, l *layer, p *nn.LayerPlan) bool {
	r := p.Recurrent
	if r.Conv == 0 {
		return true // not a linear block; nothing to carry
	}
	g.recS, g.recC = r.StateLen, r.ConvState
	need := 2 * uint64(r.StateLen*4+r.ConvState*4)
	l.recHist = need
	if g.recSeats == nil {
		g.recSeats = map[uint64]recSeat{}
	}
	st, seated := g.recSeats[sid]
	if !seated {
		st = recSeat{base: g.seatPlace(sid, 1), rows: 1}
	}
	slots := max(g.recSlots, st.base+st.rows)
	// The device's first pool takes the form Config.SharedRec asks for; a
	// later one takes the pools' form.
	shared := g.recShared
	if g.recSlots == 0 {
		shared = g.SharedRec
	}
	slot := g.recSlotBytes(shared)
	cost := uint64(slots-g.recSlots) * slot * uint64(len(g.recPools()))
	if l.pool == nil {
		cost += uint64(slots) * slot
	}
	if cost > 0 && !g.roomSeat(cost, len(l.rec), need) {
		g.LastErr = "a session's recurrent state does not fit in this device's budget"
		g.SessionDeclines++
		return false
	}
	fail := func(err error) bool { g.LastErr = err.Error(); return false }
	if slots != g.recSlots || shared != g.recShared {
		if err := g.recResize(slots, shared); err != nil {
			return fail(err)
		}
	}
	if !seated {
		g.recSeats[sid] = st
		if err := g.recZero(nil, st.base, st.rows); err != nil {
			return fail(err)
		}
	}
	if l.pool == nil {
		np, err := g.newRecPool(g.recSlots, g.recShared, nil)
		if err != nil {
			return fail(err)
		}
		l.pool = np
		n := uint64(g.recSlots) * slot
		g.charge(n)
		l.recBytes += n
		g.RecBytes += n
		if g.recShared {
			g.RecShared++
		}
	} else if err := g.recZero(l, st.base, st.rows); err != nil {
		// The seat's slots in a block this session left and came back to
		// hold its old summary.
		return fail(err)
	}
	if g.recShared && !g.ensureRecNext(st.rows) {
		return false
	}
	if l.rec == nil {
		l.rec = map[uint64]*recPair{}
	}
	l.rec[sid] = &recPair{}
	return true
}

// ensureRecNext makes the shared form's next-state scratch hold rows
// sequences' states and compiles the copy that moves them home, charging the
// ledger for any growth. It runs outside a session (growSeat, LayersRows
// before submit), where Compile is allowed. Callers hold g.mu.
func (g *devTier) ensureRecNext(rows int) bool {
	fail := func(err error) bool { g.LastErr = "rows: " + err.Error(); return false }
	if g.recS == 0 {
		return true // a short convolution keeps no state beside its window
	}
	if bytes := uint64(rows * g.recS * 4); g.recNextBytes < bytes {
		b, err := g.dev.Alloc(int(bytes))
		if err != nil {
			return fail(err)
		}
		// Every recording, and every submission in flight, names the old one.
		g.dropGraph()
		if g.recNext != nil {
			g.recNext.Free()
			g.refund(g.recNextBytes)
		}
		g.recNext, g.recNextBytes = b, bytes
		g.charge(bytes)
	}
	if _, err := g.slotCopy(g.recS, rows); err != nil {
		return fail(err)
	}
	return true
}

// freeRec releases a block's pool and every session's place in it, which is
// the same contract freeKV has and for the same reason: a release gives the
// block up entirely, so no sequence keeps a summary for it. The callers refund
// l.recBytes; a seat no block uses any more goes at recTidy.
func freeRec(l *layer) {
	if l.pool != nil {
		l.pool.free()
		l.pool = nil
	}
	clear(l.rec)
}

// addSessionKV gives session sid its own attention history for a block
// whose model-resident parts are already on the device. Callers hold g.mu.
func (g *devTier) addSessionKV(sid uint64, l *layer, p *nn.LayerPlan) bool {
	if p.NonCausal {
		if kvp := g.transientKV(p); kvp != nil {
			if l.kv == nil {
				l.kv = map[uint64]*kvPair{}
			}
			l.kv[sid] = kvp
			return true
		}
	}
	g.initKVCap(p)
	if g.pagedPlan(p) {
		// The pages are the sequence's, taken as its rows are written; the
		// block's layer is already in the pool.
		if l.kv == nil {
			l.kv = map[uint64]*kvPair{}
		}
		l.kv[sid] = &kvPair{paged: true}
		return true
	}
	cp := g.capped(p)
	cache, vcache := g.kvCacheBytes(p, cp.MaxSeq)
	need := uint64(cache + vcache)
	if !g.roomSeat(need, len(l.kv), need) {
		g.LastErr = "a second session's KV cache does not fit in this device's budget"
		g.SessionDeclines++
		return false
	}
	g.charge(need)
	kvp := &kvPair{}
	var err error
	// One allocation under MLA, where vcache is zero; see kvPair.value.
	if kvp.kc, err = g.dev.Alloc(cache); err == nil && vcache > 0 {
		kvp.vc, err = g.dev.Alloc(vcache)
	}
	if err != nil {
		if kvp.kc != nil {
			kvp.kc.Free()
		}
		g.refund(need)
		g.LastErr = err.Error()
		return false
	}
	if l.kv == nil {
		l.kv = map[uint64]*kvPair{}
	}
	kvp.bytes = need
	l.kv[sid] = kvp
	// l.kvBytes is what ReleaseLayers refunds, so it moves with every charge;
	// without it a block released after a second session refunds only the first
	// session's share and the budget leaks.
	l.kvBytes += need
	g.KVBytes += need
	return true
}

// freeKV releases every session's history for a block. A release gives the block
// up entirely, so no sequence keeps a cache for it.
func (g *devTier) freeKV(l *layer) {
	for sid, p := range l.kv {
		if p.shared {
			g.transientKVPut()
			delete(l.kv, sid)
			continue
		}
		if p.kc != nil {
			p.kc.Free()
		}
		if p.vc != nil {
			p.vc.Free()
		}
		delete(l.kv, sid)
	}
}

// kvOf returns this session's cache for the block, or nil if it has none yet.
func (l *layer) kvOf(sid uint64) *kvPair {
	if l == nil || l.kv == nil {
		return nil
	}
	return l.kv[sid]
}
