package tier

import (
	"encoding/binary"
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
)

// recorderOf is s as a backend.Recorder, asserted once per session object:
// the backends reuse their sessions, and an interface-to-interface assertion
// on every token caches its answer in a heap object at a random later call.
func (g *devTier) recorderOf(s backend.Session) (backend.Recorder, bool) {
	if g.recSess != s {
		g.recSess = s
		g.rec, _ = s.(backend.Recorder)
	}
	return g.rec, g.rec != nil
}

// stagingBytes is a devTier's host bytes for a submission's small writes,
// reused so a decode token allocates none of them.
type stagingBytes struct {
	pn, koffs, kposs, rposs []byte
	tok                     []byte // a ragged step's tokens, read home
	mlaPOff                 []byte // a batched latent attention's rotary key slots
	embIDs                  []byte // a prompt chunk's ids for the gather on the card (embIDs)
	one                     [4]byte
}

// u32b is v as four little-endian bytes in the devTier's own scratch, for a
// write that completes before the next one: every backend's Write copies p
// before it returns.
func (g *devTier) u32b(v uint32) []byte {
	binary.LittleEndian.PutUint32(g.subBytes.one[:], v)
	return g.subBytes.one[:]
}

func u32one(v uint32) []byte {
	return []byte{byte(v), byte(v >> 8), byte(v >> 16), byte(v >> 24)}
}

// accSplitFor picks how many threads share one attention output: enough to
// fill the device, rounded down to a power of two, capped at 32 (past that the
// partials and the reduction cost more than the occupancy buys). It is derived,
// not tuned, and not context-aware: each thread's trip count is ceil((n-s)/S),
// zero for splits past n, so a constant S is correct at every depth and no
// bucket boundary ever retires a CUDA graph.
//
// pin is WithMatVecAccSplit's value; see there for why.
func accSplitFor(qdim, slots, pin int) int {
	if n := pin; n >= 1 {
		return n
	}
	// 16384 only when the runtime would not say (Vulkan and Metal); CUDA
	// reports the real figure.
	if slots <= 0 {
		slots = 16384
	}
	if qdim <= 0 {
		return 1
	}
	sp := 1
	for sp < 32 && qdim*sp*2 <= slots {
		sp *= 2
	}
	return sp
}

// auxBufs is every per-layer device vector that a page-out and a release both
// drop. It is one list because separate copies drifted repeatedly, each time
// leaking a new kind of vector (vision MLP biases, post-norms, a linear block's
// five, gpt-oss's sinks and biases). Both callers are full teardowns, so
// nothing here is re-read without being uploaded again.
func (l *layer) auxBufs() []backend.Buf {
	return append(append(l.convBufs(),
		l.nAttn, l.nFFN, l.nAttnB, l.nFFNB, l.nQ, l.nK,
		l.nResid, l.nResidB, // BERT's post-norm
		l.nPostAttn, l.nPostFFN, // gemma2/gemma3
		l.nSubAttn, l.nSubFFN, // BitNet
		l.clampB,      // Gemma 4's tower: its clipped linears' bounds
		l.nQA, l.nKVA, // MLA's two latent norms
		l.nIdxK, l.nIdxKB, l.nIdxQ, // the indexers' norms (DeepSeek V3.2, MiniMax-M3)
		l.altRouter, l.altRouterNorm, l.altPredT, l.altCorrT, l.altCorrScale, l.nLaurel, // Gemma 3n
		l.ssmConv1d, l.ssmA, l.ssmDt, l.ssmNorm, l.ssmOnes, // a linear block
		l.ssmD, l.ssmCB, // Mamba-2's and Mamba-1's
		l.m1DtN, l.m1BN, l.m1CN, // Mamba-1's
		l.router,
		l.sinks, l.routerB, l.expGateB, l.expUpB, l.expDownB, // gpt-oss
		l.expSelB,  // DeepSeek-V3
		l.shRouter, // qwen2moe's and qwen3next's shared-expert gate
		l.mvq.bias, l.mvk.bias, l.mvv.bias, l.mvo.bias,
		l.mvu.bias, l.mvd.bias,
		l.outScale,                                         // Gemma 4's output scalar
		l.xielu,                                            // Apertus's activation numbers
		l.nFFN2, l.nPost1, l.nPost2, l.nRouter, l.expScale, // Gemma 4's mixture block
		l.nPLE, l.pleOff, // Gemma 4's per-layer embedding
	), append(append(l.ds4.bufs(), l.k3.bufs()...), l.q4e.bufs()...)...) // DeepSeek V4's, Kimi-K3's, Qwen3.8-Flash-Next's
}

// dropLayerLocal frees the buffers PrepLayer allocates fresh for one block and
// shares with nothing: the KV cache, the norms, the biases and the router. It
// does not touch the weights, which live in g.res keyed by their bytes and may
// be referenced elsewhere. Callers hold g.mu.
func (g *devTier) dropLayerLocal(l *layer) {
	for _, b := range l.auxBufs() {
		if b != nil {
			b.Free()
		}
	}
	g.convLeft(l)
	g.freeKV(l)
	freeRec(l)
	if l.mvRouter != nil {
		l.mvRouter.Close()
	}
	if l.actK != nil {
		l.actK.Close()
		l.actK = nil
	}
	if n := l.recBytes; n > 0 {
		g.refund(n)
		g.RecBytes -= n
		l.recBytes = 0
	}
	if n := l.kvBytes; n > 0 {
		g.refund(n)
		if g.KVBytes >= n {
			g.KVBytes -= n
		}
	}
	if n := l.auxBytes; n > 0 {
		g.refund(n)
	}
}

// ReleaseLayers hands blocks [lo, hi) back to the host and frees what they held
// on the device, which lets placement be undone (demote, park, re-adopt).
//
// It must drop the recording: a captured sequence holds device addresses, and
// replaying it after these buffers are freed reads memory that belongs to
// something else. Kernels are not closed, since g.kerns is shared by every layer
// of the same shape; only the per-layer router kernel is. The packed arena is
// not touched, so re-preparing the block is an upload rather than a repack;
// giving the host bytes back is GPU.DropArena.
//
// It takes the whole block, every session's history included: it is what a
// move between devices and a device's teardown use. A session giving blocks
// back goes through leaveLayers.
func (g *devTier) ReleaseLayers(lo, hi int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.dropGraph()
	for li := lo; li < hi; li++ {
		g.releaseBlock(li)
	}
	g.afterRelease()
}

// leaveLayers is ReleaseLayers for one session: session sid's history
// on blocks [lo, hi) goes, and a block goes with it only when no other
// session has history there -- the weights are the model's, and another
// session is still running against them. It returns the blocks it released.
func (g *devTier) leaveLayers(sid uint64, lo, hi int) []int {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.dropGraph()
	var gone []int
	for li := lo; li < hi; li++ {
		l := g.layers[li]
		if l == nil {
			continue
		}
		if g.sharedBeyond(li, sid) {
			g.dropHistory(li, l, sid)
			continue
		}
		g.releaseBlock(li)
		gone = append(gone, li)
	}
	g.afterRelease()
	return gone
}

// releaseBlock frees block li and every session's history on it. Callers
// hold g.mu and drop the graph.
func (g *devTier) releaseBlock(li int) {
	l := g.layers[li]
	g.freeRecSnaps(li)
	if l == nil {
		return
	}
	// The expert card's banks are the block's: they go with it.
	if l.stream != nil && l.stream.peer != nil {
		l.stream.peer.free()
		l.stream.peer = nil
	}
	// The same free path as a page-out (layer.tensors()); a release also
	// gives up the KV cache and the block itself.
	freed := g.dropTensors(l)
	if l.pg != nil && l.pg.in {
		l.pg.in = false
		g.pagesIn--
		g.pageBytes -= min(g.pageBytes, freed+l.pg.round)
	}
	// The KV cache is charged to the budget at PrepLayer, so releasing has
	// to give it back or the card leaks budget every migration.
	if n := l.kvBytes; n > 0 {
		g.refund(n)
		if g.KVBytes >= n {
			g.KVBytes -= n
		}
	}
	if n := l.auxBytes; n > 0 {
		g.refund(n)
	}
	g.freeKV(l)
	freeRec(l)
	// The block's layer leaves the pool, and with it every session's pages
	// there.
	if g.kvp != nil {
		g.dropKVLayer(g.kvp, li)
	}
	if n := l.recBytes; n > 0 {
		g.refund(n)
		g.RecBytes -= n
		l.recBytes = 0
	}
	for _, b := range l.auxBufs() {
		if b != nil {
			b.Free()
		}
	}
	g.convLeft(l)
	if l.mvRouter != nil {
		l.mvRouter.Close()
	}
	delete(g.layers, li)
	g.layerGen++
}

// afterRelease is what every release ends with. Callers hold g.mu.
func (g *devTier) afterRelease() {
	// A pool no session holds a place in, and the seats of sessions that hold
	// none, go back to the budget.
	g.recTidy()
	// The last tower block out takes its set's scratch with it.
	g.dropIdleTower()
	// The last block out takes the unpack staging with it, or every
	// relocation would leak one widest-tensor allocation.
	if len(g.layers) == 0 {
		g.freeRaw()
		// And a device left with no block and no head holds nothing of its
		// own: the scratches and the staging go too, refunded (the next
		// placement builds and reserves them again).
		if g.bs == nil || g.bs.head == nil {
			g.dropGraph()
			g.dropAllScratch()
		}
	}
	// A decline is cached: resident() keeps a failed entry so a tensor is not
	// re-priced every matvec, and that entry says no forever. Giving budget
	// back is what clears those tombstones, or a seam grown after a shrink is
	// silently refused. Only failures are dropped; a live entry is another
	// block's weights.
	for k, r := range g.res {
		if r != nil && !r.ok {
			delete(g.res, k)
		}
	}
	g.freeStale()
}

// ReserveKV is reserveKV for the zero session: a caller that never attached.
func (g *devTier) ReserveKV(pos int) bool {
	return g.reserveKV(0, pos)
}

// reserveKV gives session sid's sequence the pages pos positions of
// its history need, from outside a submission: growing a layer drops the
// captured graph, which deadlocks CUDA inside a Session. A restored prefix
// arrives with no submission at all.
func (g *devTier) reserveKV(sid uint64, pos int) bool {
	// No scratch is read: the session's pages are all it touches, so it takes
	// no lane.
	g = g.bareView(sid)
	if pos <= 0 {
		return true
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	// No paged history is no attention history here (a tower's keys live for
	// one encode): nothing to reserve.
	if !g.paged {
		return true
	}
	if err := g.pagedHold(sid, pos); err != nil {
		g.LastErr = err.Error()
		return false
	}
	return true
}

// TrimKV is trimKV for the zero session: a caller that never attached.
func (g *devTier) TrimKV(pos int) bool {
	return g.trimKV(0, pos)
}

// trimKV gives back session sid's pages past pos, and reports whether
// any went. The trimmed pages stay in the pool, for this session's next
// positions or anyone's; kvCompact gives them to the budget when something
// needs the room. They are room all the same, so it is announced.
func (g *devTier) trimKV(sid uint64, pos int) bool {
	g = g.bareView(sid)
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.kvp == nil || !g.kvp.trimSeq(seqID{sid, 0}, pos) {
		return false
	}
	g.roomGen.Add(1)
	return true
}

// dropSession frees everything session sid holds on this device: its history
// and its recurrent state on every block.
func (g *devTier) dropSession(sid uint64) {
	g.mu.Lock()
	defer g.mu.Unlock()
	delete(g.hostFns, sid)
	// The crossing sample is this session's too, and session ids never repeat,
	// so a map left to grow is a leak of one small struct per closed State.
	delete(g.mv, sid)
	freed := false
	for li, l := range g.layers {
		freed = g.dropHistory(li, l, sid) || freed
	}
	if freed {
		// A captured launch sequence names the buffers just freed.
		g.dropGraph()
	}
	// Its own recordings name nothing that will be read again either, in
	// whichever lane they were made, and its state on this device goes.
	g.forgetSess(sid)
	// Its seat in the recurrent pools, and a pool it was the last to hold.
	g.recTidy()
	// Its pages are free now and stay in the pool for the next session;
	// kvCompact gives them back when something needs the room. The scratch's
	// bound stays: paged history sizes nothing by context, so resetting it
	// only rebuilt the scratch -- the batched ones with it -- at the next
	// session's first call, inside its prompt: every prompt after a process's
	// first ran about a fifth slower (gpu-kernels.md, "The prompt lost 22%").
}

// dropHistory frees session sid's history on block l -- its KV cache or its
// recurrent pair -- and reports whether it had any. Callers hold g.mu and drop
// the graph when it did.
func (g *devTier) dropHistory(li int, l *layer, sid uint64) bool {
	if l == nil {
		return false
	}
	freed := false
	if kvp := l.kv[sid]; kvp != nil && kvp.paged {
		// Its pages in this layer go and the layer compacts: what the session
		// held here is room again, whoever keeps the block.
		delete(l.kv, sid)
		g.pagedLeave(li, sid, l)
		freed = true
	} else if kvp != nil && kvp.shared {
		delete(l.kv, sid)
		g.transientKVPut()
		freed = true
	} else if kvp != nil {
		if kvp.kc != nil {
			kvp.kc.Free()
		}
		if kvp.vc != nil {
			kvp.vc.Free()
		}
		delete(l.kv, sid)
		g.refund(kvp.bytes)
		l.kvBytes -= min(l.kvBytes, kvp.bytes)
		g.KVBytes -= min(g.KVBytes, kvp.bytes)
		freed = true
	}
	if l.rec[sid] != nil {
		// Its slots in the block's pool are free; recTidy gives the pool, or
		// the slots past the last seat, back to the budget.
		delete(l.rec, sid)
		freed = true
	}
	return freed
}

// MigrateRec is migrateRec for the zero session: a caller that never attached.
func (g *devTier) MigrateRec(li int, conv, state []float32, toDevice bool) bool {
	return g.migrateRec(0, li, conv, state, toDevice)
}

// migrateRec moves block li's recurrent summary between host and device.
// model.State.rconv/rstate are what every host consumer reads (including the
// -kv-cache seal), and a placed linear block never touches them, so without this
// a cached restore would resume from a summary that never saw the prompt. It
// reads half cur, the state as of the last completed token, at the session's
// seat in the block's pool.
func (g *devTier) migrateRec(sid uint64, li int, conv, state []float32, toDevice bool) bool {
	v := g.as(sid)
	if v == nil {
		return false
	}
	defer v.done()
	g = v
	g.mu.Lock()
	defer g.mu.Unlock()
	l := g.layers[li]
	if l == nil {
		return false
	}
	rp := l.recOf(sid)
	if rp == nil {
		// Not a linear block, or no session state: nothing to move is success,
		// exactly as MigrateKV treats a block with no cache.
		return true
	}
	if l.pool == nil || g.recC == 0 {
		return false
	}
	// A batch hands every row's state back to back, row r at r*ConvState and
	// r*StateLen: the seat's own layout, one slot a row.
	rows := len(conv) / g.recC
	if rows < 1 || len(conv) != rows*g.recC || len(state) != rows*g.recS {
		g.LastErr = fmt.Sprintf("MigrateRec: block %d wants a multiple of %d conv and %d state floats, "+
			"the caller offered %d and %d", li, g.recC, g.recS, len(conv), len(state))
		return false
	}
	st := g.recSeats[sid]
	switch {
	case st.rows == rows:
	case st.rows < rows && !toDevice:
		// The seat was never grown to the batch, and it grows at the batch's
		// first step (LayersRows) or before a state is sent up (below): no
		// step has run these rows here, so the host's copy is the current one
		// and there is nothing to bring home.
		return true
	case st.rows < rows:
		// Sent up for a batch the seat does not hold yet: grow it first.
		// growSeat discards the session's states, which is right, since a
		// batch's rows have no history on this device before its first step
		// and every row is written next.
		if !g.growSeat(sid, rows) {
			return false
		}
		st = g.recSeats[sid]
	default:
		g.LastErr = fmt.Sprintf("MigrateRec: block %d holds %d row(s), the caller offered %d",
			li, st.rows, rows)
		return false
	}
	sh := rp.cur
	if g.recShared {
		// One persistent state (s[1] is nil): the step reads s[0] whichever
		// half of the conv window is current, as emitLinear does.
		sh = 0
	}
	if !toDevice {
		rd := func(b backend.Buf, v []float32, w int, name string) bool {
			// A buffer reads from its start, so a seat above slot 0 is first
			// copied to a buffer of its own on the device: reading the pool
			// directly would bring every other session's state below it home
			// too.
			err := func() error {
				if len(v) == 0 {
					// A block with no state of this kind (a short
					// convolution has no delta-rule half): nothing to read,
					// and a zero-byte buffer is no allocation a driver takes.
					return nil
				}
				if st.base == 0 {
					return b.Read(f32b(v))
				}
				tmp, err := g.dev.Alloc(len(v) * 4)
				if err != nil {
					return err
				}
				defer tmp.Free()
				if err := g.dev.Copy(tmp, 0, b, st.base*w*4, len(v)*4); err != nil {
					return err
				}
				return tmp.Read(f32b(v))
			}()
			if err != nil {
				g.LastErr = fmt.Sprintf("MigrateRec: block %d %s: %v", li, name, err)
				return false
			}
			return true
		}
		return rd(l.pool.c[rp.cur], conv, g.recC, "conv") && rd(l.pool.s[sh], state, g.recS, "state")
	}
	// Both halves. Seeding cur alone is already correct (the next submission
	// writes 1-cur before anything reads it); writing both is defence, so no
	// future caller depends on that parity argument.
	for i := 0; i < 2; i++ {
		if err := l.pool.c[i].WriteAt(st.base*g.recC*4, f32b(conv)); err != nil {
			g.LastErr = fmt.Sprintf("MigrateRec: block %d conv half %d: %v", li, i, err)
			return false
		}
		if l.pool.s[i] == nil || len(state) == 0 {
			continue // a shared pool's second half is the device's scratch
		}
		if err := l.pool.s[i].WriteAt(st.base*g.recS*4, f32b(state)); err != nil {
			g.LastErr = fmt.Sprintf("MigrateRec: block %d state half %d: %v", li, i, err)
			return false
		}
	}
	return true
}

// MigrateKV is migrateKV for the zero session: a caller that never attached.
func (g *devTier) MigrateKV(li int, k, v []float32, pos int, toDevice bool) bool {
	return g.migrateKV(0, li, k, v, pos, toDevice)
}

// migrateKV moves block li's attention history between host and device.
// Without it a block that changes sides mid-generation attends over whatever
// was in the pages it arrived at: fluent, wrong text.
//
// k and v are the host arrays for this block, laid out [pos*kvDim + i] in
// float32; only positions [0, pos) are moved (pagedMigrate). A block with no
// paged history here -- a tower, whose keys live for one encode -- has
// nothing to move.
func (g *devTier) migrateKV(sid uint64, li int, k, v []float32, pos int, toDevice bool) bool {
	sv := g.as(sid)
	if sv == nil {
		return false
	}
	defer sv.done()
	g = sv
	g.mu.Lock()
	defer g.mu.Unlock()
	l := g.layers[li]
	kvp := l.kvOf(sid)
	if kvp == nil || !kvp.paged || pos <= 0 {
		return l != nil
	}
	if err := g.pagedMigrate(sid, li, 0, k, v, pos, toDevice); err != nil {
		g.LastErr = fmt.Sprintf("MigrateKV: block %d: %v", li, err)
		return false
	}
	return true
}

// PrewarmLayer packs a whole block off the main loop. See GPU.Prewarm.
func (g *devTier) PrewarmLayer(li int, p *nn.LayerPlan, w *nn.LayerWeights) bool {
	bank := 1
	if w.Router.Data != nil && p.NExpert > 1 {
		bank = p.NExpert
	}
	_ = bank // the bank is packed as one tall matrix, exactly as PrepLayer does
	// A lazy weight is not prewarmable: this must be safe beside a running
	// Layers, and nn.LayerWeights.Ensure reads the file into the pager's frames.
	// Packing without it would pack whatever the frame holds. The upload calls
	// Ensure itself, so only the prewarm is lost.
	if w.Ensure != nil {
		return false
	}
	any := false
	// weightList, not a second copy of it: a matrix left off would silently
	// never prewarm.
	for _, x := range weightList(w) {
		q, ok := quantOf(x.T)
		if !ok || x.Rows <= 0 || x.K <= 0 || len(x.Data) == 0 ||
			(x.K%q.Elems() != 0 && subFor(q, x.K) == 0) {
			continue
		}
		if g.prewarm(li, q, x.Data, x.Rows, x.K) {
			any = true
		}
	}
	return any
}

// weightList is a block's pageable matrices in layer.tensors() order.
//
// One list, for tensors()' reason: prepLayer builds it to decide and upload,
// and pageIn rebuilds it after Ensure re-points the slices. Two copies of the
// order could upload one matrix into another's slot with every size right.
func weightList(w *nn.LayerWeights) []nn.Weight {
	return []nn.Weight{w.Wq, w.Wk, w.Wv, w.Wo, w.Gate, w.Up, w.Down,
		w.ShGate, w.ShUp, w.ShDown,
		w.SSM.Gate, w.SSM.BA, w.SSM.Out,
		w.Wqa, w.Wqb, w.Wkva, w.Wkb, w.Wvb,
		w.SSM.FA, w.SSM.FB, w.SSM.GA, w.SSM.GB,
		w.PLEGate, w.PLEProj,
		w.IdxQB, w.IdxK, w.IdxProj,
		w.LaurelL, w.LaurelR, w.SSM.In,
		w.IdxQ,
		w.WoA, w.CompKV, w.CompGate, w.IdxCompKV, w.IdxCompGate,
		w.MLAGate, w.RoutedDown, w.RoutedUp,
		w.HCAttn.Down, w.HCAttn.Up, w.HCAttn.Inject, w.HCFFN.Down, w.HCFFN.Up, w.HCFFN.Inject,
		w.NGram.Key, w.NGram.Value}
}
